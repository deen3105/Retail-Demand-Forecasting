# Retail Demand Forecasting

Forecasting daily unit sales across 54 stores and 33 product families for Corporación Favorita, a large Ecuadorian grocery retailer, using classical ML and deep sequence models. Built on the [Kaggle Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset.

## Problem

Retailers need accurate, granular demand forecasts to manage inventory and avoid stockouts or overstock. This project predicts sales for every `(store, product family, date)` combination over a 15-day horizon, using ~3 million rows of historical sales, promotions, store metadata, oil prices, and holiday events.

## Data

| Source | Description |
|---|---|
| `train.csv` / `test.csv` | Daily sales by store & product family |
| `stores.csv` | Store city, state, type, cluster |
| `oil.csv` | Daily oil price (proxy for Ecuador's oil-dependent economy) |
| `holidays_events.csv` | National/regional/local holidays and transfers |
| `transactions.csv` | Daily transaction counts per store |

## Exploratory Findings

- **Strong weekly seasonality** — weekend sales are ~30–40% higher than weekday sales, Thursday is the lowest.
- **New Year's Day collapse** — every January 1st (2013–2017) shows a 95%+ sales drop across all stores, the single strongest anomaly in the data.
- **April 2016 earthquake spike** — a magnitude 7.8 earthquake on April 16, 2016 caused a sharp, week-long spike in relief-goods sales (water, canned food, batteries).
- **Severe scale imbalance** — 31% of rows have zero sales; average daily sales range from <1 unit (Books, Baby Care) to 3,800+ units (Grocery I), which directly informed the modeling and evaluation choices below.

## Feature Engineering

- Calendar features: day-of-week, week-of-year, weekend flag
- `is_holiday` and `is_earthquake_period` binary flags
- Lag features: sales at t-1, t-7, t-14 (per store–family series)
- Rolling mean of sales (7-day, 14-day windows, leakage-safe via `.shift(1)`)
- Promotion momentum: 7-day rolling mean and day-over-day change in `onpromotion`
- Label-encoded categoricals (family, city, state, store type)

## Modeling

Seven models were trained and compared on a strict **time-based** (not random) validation split — the last 15 days of the training period, matching the actual test horizon:

| Model | Validation RMSLE |
|---|---|
| Ridge Regression | 1.28 |
| **Random Forest** | **0.46** |
| XGBoost | 0.62 |
| SimpleRNN | 0.61 |
| LSTM | 0.64 |
| GRU | 0.66 |
| CNN‑LSTM | 0.60 |


**Random Forest was the best overall model.** Among the deep sequence models, CNN‑LSTM performed best, benefiting from a Conv1D layer that extracts short-term local patterns before the LSTM models longer-range temporal dependencies.

### Key debugging finding

An early version of every neural model scored RMSLE > 1.0 (some as high as 2.7) despite reasonable-looking training loss. The root cause: training minimized MSE on a `StandardScaler`-normalized sales target, while the competition evaluates with RMSLE — a **relative-error** metric that heavily penalizes confident non-zero predictions on rows where true sales are 0 (31% of the data). Replacing the target transform with `log1p(sales)` mathematically aligned the training objective with RMSLE and cut every neural model's error by more than half.

A second, more subtle bug was caught during validation-sequence construction: naively building sliding windows over a "train tail + validation" buffer leaked training-period targets into the validation set. Filtering windows by true target date (rather than array position) fixed this before any model comparison was trusted.

## Pipeline

1. Merge and clean 6 source tables (deduplicate holidays, reindex oil prices to a full calendar, fill missing transactions)
2. Exploratory analysis (seasonality, anomalies, scale imbalance)
3. Feature engineering (lags, rolling stats, calendar/promo features)
4. Time-based train/validation split (last 15 days held out)
5. Feature scaling (`StandardScaler`) + `log1p` target transform
6. Train and evaluate 7 models
7. Recursive 15-day forecast on the test set using the best model, feeding each day's predictions forward as next-day lag features

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · TensorFlow / Keras · XGBoost · Matplotlib

## Repository Contents

- `Final_notebook.ipynb` — full pipeline: EDA, feature engineering, model training, evaluation, and forecasting
- `model_comparison.png` — validation RMSLE across all 7 models

## Possible Extensions

- Hyperparameter tuning (KerasTuner / GridSearchCV) on the top models
- Per-family or per-cluster models to address the scale-imbalance problem directly
- Embedding layers for categorical features instead of scaled label encodings
- Prediction intervals via quantile loss
