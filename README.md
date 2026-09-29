# German Electricity Load Forecasting

A model that predicts Germany's hourly electricity demand from past data. It uses six
years of load data, looks at the daily and seasonal patterns, builds time and lag
features, and trains a gradient-boosting model. I tested it with a time-based split so
the scores are realistic.

## Results

| Model | R² | MAE |
|---|---|---|
| Gradient boosting (mine) | 0.970 | 1,227 MW (~2.2% of avg load) |
| Naive baseline (same hour last week) | 0.844 | 2,273 MW |

The model has about half the error of the naive baseline, so it is clearly adding
something over just reusing last week's value.

![Actual vs predicted for one test week](forecast.png)

It follows the daily up-and-down and the weekend dip pretty closely.

## Approach

1. Data: hourly German load from Jan 2015 to Sep 2020 (50,400 rows), from the free
   [Open Power System Data](https://open-power-system-data.org/) set.
2. Exploration: demand is lowest overnight (~44 GW), peaks mid-morning (~64 GW), and is
   higher in winter than summer. I used these patterns to pick the features.
3. Features: hour, day of week, month, a weekend flag, a German public-holiday flag, and
   two lag features (the load 24 hours ago and 168 hours / one week ago). The lags matter
   most, because demand repeats every day and every week.
4. Model: HistGradientBoostingRegressor from scikit-learn.
5. Evaluation: I split by time (train on the first 80%, test on the last 20%) instead of
   doing it randomly. A random split lets the model see future data during training, which makes
   the scores look better than they really are. I also compared against a simple baseline
   (just reuse last week's value) to check the model actually helps.

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

- The lag features do most of the work. Simple, but they help a lot.
- For forecasting you have to split by time, not randomly, or the scores lie.
- How to take a raw public dataset all the way to a working, tested model.

## Possible extensions

Add temperature (it drives heating and cooling demand), or forecast more than one hour
ahead.

---
Tech: Python, pandas, scikit-learn, matplotlib.
