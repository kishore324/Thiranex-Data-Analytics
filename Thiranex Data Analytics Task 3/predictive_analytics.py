"""
Retail Demand Forecasting - an end-to-end predictive analytics project.

Pipeline
  1. Load (or generate) 4 years of daily sales history
  2. Clean and preprocess (missing values, duplicates, outliers, dtypes)
  3. Trend analysis (rolling mean, seasonal decomposition, YoY growth)
  4. Feature engineering (calendar, lags, rolling stats)
  5. Regression models: Linear, Ridge, Random Forest, Gradient Boosting
  6. Time-series models: Seasonal Naive, Holt-Winters, SARIMAX
  7. Time-ordered evaluation (MAE, RMSE, MAPE, R2) + model comparison
  8. Visualise predictions and produce a 90-day future forecast

Run:  python predictive_analytics.py
Outputs go to ./outputs (charts + CSVs).
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SM = True
except ImportError:  # project still runs with the regression models only
    HAS_SM = False
    print("[warn] statsmodels not installed -> skipping Holt-Winters, SARIMAX and "
          "decomposition. Run: pip install statsmodels")

warnings.filterwarnings("ignore")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
TEST_DAYS = 90
FORECAST_DAYS = 90


# --------------------------------------------------------------------------
# 1. DATA
# --------------------------------------------------------------------------
def generate_raw_data(path="sales_history.csv"):
    """Synthetic but realistic daily sales: trend + weekly/yearly seasonality +
    promotions + noise, then deliberately corrupted (NaNs, outliers, dupes)."""
    dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    t = np.arange(len(dates))
    trend = 200 + 0.18 * t
    weekly = 35 * np.sin(2 * np.pi * (dates.dayofweek - 4) / 7) + 20 * (dates.dayofweek >= 5)
    yearly = 50 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)
    promo = (RNG.random(len(dates)) < 0.08).astype(int)
    holiday_boost = np.where((dates.month == 12) & (dates.day >= 15), 60, 0)
    sales = trend + weekly + yearly + 45 * promo + holiday_boost + RNG.normal(0, 14, len(dates))
    df = pd.DataFrame({"date": dates, "sales": sales.round(2), "promo": promo})

    # corrupt the data like real-world exports
    df.loc[RNG.choice(len(df), 40, replace=False), "sales"] = np.nan
    df.loc[RNG.choice(len(df), 6, replace=False), "sales"] *= RNG.choice([4, -1], 6)
    df = pd.concat([df, df.sample(12, random_state=1)]).sample(frac=1, random_state=2)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)
    return path


def load_data(path="sales_history.csv"):
    if not os.path.exists(path):
        generate_raw_data(path)
        print(f"[data] generated synthetic dataset -> {path}")
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# 2. CLEANING
# --------------------------------------------------------------------------
def clean(df):
    report = {"raw_rows": len(df)}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    report["duplicates_removed"] = int(df.duplicated(subset="date").sum())
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")

    # enforce a complete daily index so lags / seasonality are valid
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    report["missing_dates_filled"] = len(full_idx) - len(df)
    df = df.reindex(full_idx)
    df["promo"] = df["promo"].fillna(0).astype(int)

    # outliers: negative values, or > 4 robust-sigmas from a rolling median
    med = df["sales"].rolling(15, center=True, min_periods=5).median()
    mad = (df["sales"] - med).abs().rolling(61, center=True, min_periods=10).median()
    robust_z = (df["sales"] - med).abs() / (1.4826 * mad)
    outliers = (df["sales"] < 0) | (robust_z > 4)
    report["outliers_flagged"] = int(outliers.sum())
    df.loc[outliers, "sales"] = np.nan

    report["missing_values_imputed"] = int(df["sales"].isna().sum())
    df["sales"] = df["sales"].interpolate(method="time").bfill().ffill()
    report["clean_rows"] = len(df)
    df.index.name = "date"
    return df, report


# --------------------------------------------------------------------------
# 3. TREND ANALYSIS
# --------------------------------------------------------------------------
def trend_analysis(df):
    s = df["sales"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(s, alpha=0.35, label="daily")
    axes[0].plot(s.rolling(30).mean(), label="30-day mean")
    axes[0].plot(s.rolling(365).mean(), label="365-day mean", lw=2)
    axes[0].set_title("Sales history and rolling trends"); axes[0].legend()

    monthly = s.resample("MS").sum()
    for y in sorted(set(monthly.index.year)):
        m = monthly[monthly.index.year == y]
        axes[1].plot(m.index.month, m.values, marker="o", label=str(y))
    axes[1].set_title("Monthly sales by year (seasonality)"); axes[1].set_xlabel("month"); axes[1].legend()

    dow = s.groupby(s.index.dayofweek).mean()
    axes[2].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values)
    axes[2].set_title("Average sales by weekday")
    plt.tight_layout(); plt.savefig(f"{OUT}/01_trend_analysis.png", dpi=130); plt.close()

    if HAS_SM:
        dec = seasonal_decompose(s, model="additive", period=365)
        dec.plot().set_size_inches(12, 8)
        plt.tight_layout(); plt.savefig(f"{OUT}/02_decomposition.png", dpi=130); plt.close()

    yearly = s.resample("YS").sum()
    growth = (yearly.pct_change() * 100).round(1).dropna()
    return growth


# --------------------------------------------------------------------------
# 4. FEATURES
# --------------------------------------------------------------------------
def make_features(df):
    d = pd.DataFrame(index=df.index)
    d["y"] = df["sales"]
    d["promo"] = df["promo"]
    d["dow"] = df.index.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["month"] = df.index.month
    d["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)
    d["is_holiday_season"] = ((df.index.month == 12) & (df.index.day >= 15)).astype(int)
    d["t"] = np.arange(len(d))
    for lag in (1, 7, 14, 28, 364):
        d[f"lag_{lag}"] = d["y"].shift(lag)
    shifted = d["y"].shift(1)  # shift first -> no leakage
    d["roll_7"] = shifted.rolling(7).mean()
    d["roll_28"] = shifted.rolling(28).mean()
    return d.dropna()


FEATURES_STATIC = ["promo", "is_weekend", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
                   "is_holiday_season", "t"]


# --------------------------------------------------------------------------
# 5-7. MODELS + EVALUATION
# --------------------------------------------------------------------------
def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": r2_score(y_true, y_pred),
    }


def regression_models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                               n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.04,
                                                       max_depth=3, subsample=0.8,
                                                       random_state=42),
    }


def evaluate_all(df):
    feats = make_features(df)
    train_f, test_f = feats.iloc[:-TEST_DAYS], feats.iloc[-TEST_DAYS:]
    cols = [c for c in feats.columns if c not in ("y",)]
    results, preds = {}, {}

    # --- regression (direct 1-step features from known history; honest for the
    #     test window because lag>=1 uses actuals, so we ALSO report a
    #     lag-free variant that is a true multi-day-ahead forecaster) ---
    for name, model in regression_models().items():
        model.fit(train_f[cols], train_f["y"])
        p = pd.Series(model.predict(test_f[cols]), index=test_f.index)
        preds[f"{name} (1-day ahead)"] = p
        results[f"{name} (1-day ahead)"] = metrics(test_f["y"], p)

    # lag-free, calendar-only models -> genuine long-horizon forecasts
    for name in ("Ridge", "Gradient Boosting"):
        model = regression_models()[name]
        model.fit(train_f[FEATURES_STATIC], train_f["y"])
        p = pd.Series(model.predict(test_f[FEATURES_STATIC]), index=test_f.index)
        preds[f"{name} (90-day horizon)"] = p
        results[f"{name} (90-day horizon)"] = metrics(test_f["y"], p)

    # --- time-series models on the same 90-day horizon ---
    y_train = df["sales"].loc[:train_f.index[-1]]
    y_test = df["sales"].loc[test_f.index]

    # seasonal naive baseline: same day last year
    ly = df["sales"].shift(364).loc[y_test.index]
    preds["Seasonal Naive (364d)"] = ly
    results["Seasonal Naive (364d)"] = metrics(y_test, ly)

    if HAS_SM:
        hw = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=7).fit()
        p = pd.Series(hw.forecast(len(y_test)).values, index=y_test.index)
        preds["Holt-Winters (weekly)"] = p
        results["Holt-Winters (weekly)"] = metrics(y_test, p)

        exog_cols = ["promo", "doy_sin", "doy_cos"]
        exog = make_features(df)[exog_cols]
        sar = SARIMAX(y_train, exog=exog.reindex(y_train.index).bfill(),
                      order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
        p = pd.Series(sar.forecast(len(y_test), exog=exog.loc[y_test.index]).values, index=y_test.index)
        preds["SARIMAX (+promo, yearly)"] = p
        results["SARIMAX (+promo, yearly)"] = metrics(y_test, p)

    table = pd.DataFrame(results).T.sort_values("RMSE").round(3)
    table.to_csv(f"{OUT}/model_comparison.csv")

    # time-series CV for the best calendar-only regressor (robustness check)
    X, y = train_f[FEATURES_STATIC], train_f["y"]
    cv = []
    for tr, va in TimeSeriesSplit(n_splits=5, test_size=60).split(X):
        m = regression_models()["Gradient Boosting"].fit(X.iloc[tr], y.iloc[tr])
        cv.append(np.sqrt(mean_squared_error(y.iloc[va], m.predict(X.iloc[va]))))
    return table, preds, y_test, feats, cols, (np.mean(cv), np.std(cv))


# --------------------------------------------------------------------------
# 8. VISUALS + FUTURE FORECAST
# --------------------------------------------------------------------------
def plot_results(df, table, preds, y_test):
    best = table.index[0]
    shown = [best, "Holt-Winters (weekly)", "SARIMAX (+promo, yearly)",
             "Gradient Boosting (90-day horizon)"]
    shown = [n for n in dict.fromkeys(shown) if n in preds]
    fig, ax = plt.subplots(figsize=(13, 6))
    hist = df["sales"].iloc[-TEST_DAYS - 120:-TEST_DAYS]
    ax.plot(hist, color="grey", alpha=0.6, label="history (train)")
    ax.plot(y_test, color="black", lw=2, label="actual (test)")
    for n in shown:
        ax.plot(preds[n], lw=1.4, label=n)
    ax.set_title("Test-window predictions vs actual"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/03_predictions_vs_actual.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    t = table.sort_values("RMSE", ascending=False)
    ax.barh(t.index, t["RMSE"], color="#4C72B0")
    ax.set_title("Model comparison (lower RMSE is better)"); ax.set_xlabel("RMSE")
    plt.tight_layout(); plt.savefig(f"{OUT}/04_model_comparison.png", dpi=130); plt.close()

    p = preds[best]; r = y_test - p
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(r); axes[0].axhline(0, color="k"); axes[0].set_title(f"Residuals - {best}")
    axes[1].hist(r, bins=25); axes[1].set_title("Residual distribution")
    plt.tight_layout(); plt.savefig(f"{OUT}/05_residuals.png", dpi=130); plt.close()


def feature_importance(feats, cols):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
    m.fit(feats[cols], feats["y"])
    imp = pd.Series(m.feature_importances_, index=cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6)); imp.tail(12).plot.barh(ax=ax)
    ax.set_title("Random Forest feature importance (top 12)")
    plt.tight_layout(); plt.savefig(f"{OUT}/06_feature_importance.png", dpi=130); plt.close()
    return imp.sort_values(ascending=False).head(5)


def future_forecast(df):
    """Refit on ALL data; forecast the next 90 days with Holt-Winters + SARIMAX
    ensemble and a simple residual-based interval."""
    y = df["sales"]
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
    f = pd.DataFrame(index=idx)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["promo"] = 0  # assume no planned promotions

    if HAS_SM:
        hist_exog = pd.DataFrame({"promo": df["promo"],
                                  "doy_sin": np.sin(2 * np.pi * df.index.dayofyear / 365.25),
                                  "doy_cos": np.cos(2 * np.pi * df.index.dayofyear / 365.25)})
        sar = SARIMAX(y, exog=hist_exog, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
        hw = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=7).fit()
        fc = 0.5 * sar.forecast(FORECAST_DAYS, exog=f[["promo", "doy_sin", "doy_cos"]]).values \
            + 0.5 * hw.forecast(FORECAST_DAYS).values
        sigma = np.std(sar.resid.iloc[30:])
    else:  # fallback: calendar-only Gradient Boosting refit on all data
        full = make_features(df)
        gb = regression_models()["Gradient Boosting"].fit(full[FEATURES_STATIC], full["y"])
        fut = f.copy()
        fut["dow"] = idx.dayofweek
        fut["is_weekend"] = (fut["dow"] >= 5).astype(int)
        fut["dow_sin"] = np.sin(2 * np.pi * fut["dow"] / 7)
        fut["dow_cos"] = np.cos(2 * np.pi * fut["dow"] / 7)
        fut["is_holiday_season"] = ((idx.month == 12) & (idx.day >= 15)).astype(int)
        fut["t"] = full["t"].iloc[-1] + 1 + np.arange(FORECAST_DAYS)
        fc = gb.predict(fut[FEATURES_STATIC])
        sigma = np.std(full["y"] - gb.predict(full[FEATURES_STATIC]))
    out = pd.DataFrame({"forecast": fc.round(2),
                        "lower_95": (fc - 1.96 * sigma).round(2),
                        "upper_95": (fc + 1.96 * sigma).round(2)}, index=idx)
    out.index.name = "date"; out.to_csv(f"{OUT}/future_forecast.csv")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y.iloc[-200:], label="history")
    ax.plot(out["forecast"], color="crimson", label="forecast")
    ax.fill_between(out.index, out["lower_95"], out["upper_95"], color="crimson", alpha=0.15, label="95% interval")
    ax.set_title(f"{FORECAST_DAYS}-day sales forecast"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/07_future_forecast.png", dpi=130); plt.close()
    return out


def main():
    raw = load_data()
    df, rep = clean(raw)
    print("\n== CLEANING REPORT =="); [print(f"  {k}: {v}") for k, v in rep.items()]
    df.to_csv(f"{OUT}/clean_sales.csv")

    growth = trend_analysis(df)
    print("\n== YEAR-OVER-YEAR GROWTH (%) =="); print(growth.to_string())

    table, preds, y_test, feats, cols, cv = evaluate_all(df)
    print(f"\n== MODEL COMPARISON (last {TEST_DAYS} days held out) =="); print(table.to_string())
    print(f"\nGradient Boosting calendar-only TimeSeriesSplit CV RMSE: {cv[0]:.2f} +/- {cv[1]:.2f}")

    plot_results(df, table, preds, y_test)
    print("\n== TOP FEATURES =="); print(feature_importance(feats, cols).round(3).to_string())

    fut = future_forecast(df)
    print(f"\n== NEXT {FORECAST_DAYS} DAYS (head) =="); print(fut.head(7).to_string())
    print(f"\nBest model on holdout: {table.index[0]}  |  outputs in ./{OUT}/")


if __name__ == "__main__":
    main()
