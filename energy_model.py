"""
German electricity-load forecast.

Loads Germany's hourly electricity demand (Open Power System Data), explores the
daily/seasonal patterns, engineers time + lag features, and trains a gradient-
boosting model to predict load. Evaluated with a time-based train/test split.

Run:  python energy_model.py
"""

import os
import holidays
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---- config ----
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "de_load.csv")          
# Open Power System Data - hourly time series (free, no login). ~130 MB one-time download.
OPSD_URL = ("https://data.open-power-system-data.org/time_series/2020-10-06/"
            "time_series_60min_singleindex.csv")
LOAD_COL = "DE_load_actual_entsoe_transparency"      # Germany's actual load (MW)


def load_data():
    """Return a clean hourly table with columns: timestamp, load_mw.

    Downloads the OPSD dataset once, keeps only the German load column, drops
    missing values, and caches a small CSV so later runs are instant.
    """
    if os.path.exists(CACHE):
        print("Loading cached de_load.csv ...")
        df = pd.read_csv(CACHE, parse_dates=["timestamp"])
    else:
        print("First run: downloading OPSD data (~130 MB, one-time)... please wait.")
        raw = pd.read_csv(OPSD_URL, parse_dates=["utc_timestamp"])
        if LOAD_COL not in raw.columns:
            print("German load columns found:", [c for c in raw.columns if "DE_load" in c])
            raise SystemExit("Adjust LOAD_COL to match, then re-run.")
        df = raw[["utc_timestamp", LOAD_COL]].rename(
            columns={"utc_timestamp": "timestamp", LOAD_COL: "load_mw"})
        df = df.dropna().reset_index(drop=True)
        df.to_csv(CACHE, index=False)
        print(f"Cached {len(df)} rows to {CACHE}")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    print(f"{len(df)} hourly rows, {df.timestamp.min()} -> {df.timestamp.max()}")
    return df


def main():
    df = load_data()

    #  Exploratory analysis: demand has strong daily and seasonal rhythms
    df["hour"] = df["timestamp"].dt.hour            # 0-23
    df["dayofweek"] = df["timestamp"].dt.dayofweek  # 0 = Monday ... 6 = Sunday
    df["month"] = df["timestamp"].dt.month          # 1-12
    print("\nAverage load by hour (MW):")
    print(df.groupby("hour")["load_mw"].mean().round(0))
    print("\nAverage load by month (MW):")
    print(df.groupby("month")["load_mw"].mean().round(0))

    # Features
    # Time features plus the two most predictive signals in load forecasting:
    # the load 24 hours ago (same hour yesterday) and 168 hours ago (a week ago).
    de_holidays = holidays.Germany(years=range(2015, 2021))
    df["is_holiday"] = df["timestamp"].dt.date.map(lambda d: int(d in de_holidays))
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["lag_24"] = df["load_mw"].shift(24)
    df["lag_168"] = df["load_mw"].shift(168)
    df = df.dropna()                                # drop early rows with no lag value

    features = ["hour", "dayofweek", "month", "is_holiday", "is_weekend", "lag_24", "lag_168"]
    X = df[features]
    y = df["load_mw"]
    print("\nFeatures:", X.shape, "| target:", y.shape)

    # Model + honest evaluation
    # Split by TIME, not randomly: train on the earliest 80%, test on the most
    # recent 20%. Random shuffling would leak future data and inflate the score.
    n = int(len(df) * 0.8)
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nMAE:", round(mean_absolute_error(y_test, pred), 0), "MW")
    print("R2 :", round(r2_score(y_test, pred), 3))

    #  Baseline: naive forecast = same hour last week
    baseline = X_test["lag_168"]     # naive forecast = same hour last week
    print("\nBaseline (last week) MAE:", round(mean_absolute_error(y_test, baseline), 0), "MW")
    print("Baseline (last week) R2 :", round(r2_score(y_test, baseline), 3))
    # Plot one test week: actual vs predicted
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.values[:168], label="actual")     # 168 hours = one week
    plt.plot(pred[:168], label="predicted")
    plt.title("German electricity load - actual vs predicted (one test week)")
    plt.xlabel("hour")
    plt.ylabel("load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast.png", dpi=120)
    print("Saved forecast.png")


if __name__ == "__main__":
    main()
