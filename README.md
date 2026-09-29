# German Electricity Load Forecasting

I built a model that predicts **Germany's hourly electricity demand** from historical
data. It loads six years of real load data, explores the daily and seasonal patterns,
engineers time-based and lag features, and trains a gradient-boosting model — evaluated
with a time-based train/test split.

## Results

| Model | R² | MAE |
|---|---|---|
| **Gradient boosting (mine)** | **0.970** | **1,227 MW** (~2.2% of avg load) |
| Naive baseline (same hour last week) | 0.844 | 2,273 MW |

My model **cuts the error nearly in half** versus the naive baseline — confirming it adds
real predictive value, not just luck.

![Actual vs predicted for one test week](forecast.png)

The model tracks the daily demand wave and the weekend dip closely.

## Approach

1. **Data** — hourly German load, Jan 2015 → Sep 2020 (50,400 rows), from the free
   [Open Power System Data](https://open-power-system-data.org/) time series.
2. **Exploration** — demand is lowest overnight (~44 GW) and peaks mid-morning (~64 GW),
   and is higher in winter than summer. These patterns guided the features.
3. **Features** — hour, day of week, month, weekend flag, **public-holiday flag**
   (German holidays), and two **lag features**: the load 24 hours ago (same hour
   yesterday) and 168 hours ago (a week ago). Lags are the strongest predictors, since
   demand repeats daily and weekly.
4. **Model** — `HistGradientBoostingRegressor` (scikit-learn).
5. **Evaluation** — a **time-based split** (train on the earliest 80%, test on the most
   recent 20%). I avoided a random split on purpose: shuffling would let the model see
   future data during training (data leakage) and overstate its accuracy. I also compared
   against a **naive baseline** (predict the same hour from last week) to prove the model
   earns its keep.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python energy_model.py
```

The first run downloads the dataset (~130 MB) and caches a small CSV, so later runs are
instant. It prints the metrics and saves `forecast.png`.

## What I learned

- Lag features can carry most of the signal in a time series — simple, but powerful.
- For forecasting you must split by time, not randomly, or your scores lie.
- How to go from a raw public dataset to a clean, validated model end to end.

## Possible extensions

Add weather (temperature drives heating/cooling demand), or forecast further ahead than
one hour (multi-step / recursive forecasting).

---
*Tech: Python, pandas, scikit-learn, matplotlib.*
