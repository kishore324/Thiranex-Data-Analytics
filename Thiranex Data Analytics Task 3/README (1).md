# Retail Demand Forecasting: Predictive Analytics Project

An end-to-end predictive analytics project covering the three outcomes you listed:
regression/time-series prediction, cleaning historical data, and evaluating and visualising forecasts.

Note: the YouTube video could not be fetched (YouTube blocked the request), so this project is built
from the key features and expected outcome you provided, not from the video's own chapters.

## What it does

| Stage | What happens |
|---|---|
| Data | Generates 4 years of daily sales (2022-2025) with trend, weekly and yearly seasonality, promotions, and deliberate defects. Replace `sales_history.csv` with your own `date,sales,promo` file to use real data. |
| Cleaning | Parses dates, removes duplicate days, enforces a complete daily index, flags outliers (rolling median + MAD), interpolates gaps. |
| Trend analysis | Rolling means, monthly seasonality by year, weekday profile, year-over-year growth, seasonal decomposition. |
| Features | Calendar (sin/cos encodings, weekend, holiday season), lags (1, 7, 14, 28, 364), leak-free rolling means. |
| Regression | Linear, Ridge, Random Forest, Gradient Boosting. |
| Time series | Seasonal naive baseline, Holt-Winters, SARIMAX with promo and yearly terms (needs `statsmodels`). |
| Evaluation | Last 90 days held out in time order. MAE, RMSE, MAPE, R2, plus TimeSeriesSplit cross-validation. |
| Forecast | Refits on all data and predicts the next 90 days with 95% intervals. |

## Run

```
pip install -r requirements.txt
python predictive_analytics.py
```

Outputs land in `outputs/`: charts (`01`-`07` PNGs), `model_comparison.csv`, `clean_sales.csv`, `future_forecast.csv`.

## Reading the results

- "1-day ahead" models use yesterday's actual sales as inputs, so they only forecast one step.
- "90-day horizon" models use calendar features only, so they are honest long-range forecasts.
  Compare like with like.
- The data is synthetic, so scores say more about the pipeline than about any real business.

## Ideas to extend

Add holiday calendars, hyperparameter tuning (`GridSearchCV` with `TimeSeriesSplit`), Prophet or LightGBM, or a Streamlit dashboard.
