"""Feature pipeline: raw hourly data -> features + 24h-ahead targets.

The same build_features() is used for the backfill (with targets) and for the
live hourly run (without targets, on ~72h of recent history).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 24            # forecast horizon in hours
MAX_GAP_HOURS = 3       # gaps up to this length are interpolated
RAIN_THRESHOLD_MM = 0.1

CORE = ["us_aqi", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "uv_index", "precipitation"]
LAGS = (1, 3, 6, 12, 24)


def _fill_short_gaps(s, max_gap):
    """Linearly interpolate NaN runs of at most max_gap hours; leave longer runs empty."""
    isna = s.isna()
    run_id = isna.ne(isna.shift(fill_value=False)).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = s.interpolate(method="time", limit_area="inside")
    return s.where(~(isna & (run_len <= max_gap)), filled)


def impute(raw):
    """Strict hourly grid + documented imputation. Adds a was_imputed flag."""
    df = raw.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    full = pd.date_range(df.index.min().floor("h"), df.index.max().floor("h"), freq="h")
    df = df.reindex(full)
    df.index.name = "time"

    # wind direction is circular, so use sin/cos instead of raw degrees
    rad = np.deg2rad(df["wind_direction_10m"])
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)
    df = df.drop(columns="wind_direction_10m")

    missing = df.isna()
    for col in df.columns:
        df[col] = _fill_short_gaps(df[col], MAX_GAP_HOURS)
    df["was_imputed"] = (missing & df.notna()).any(axis=1)
    return df


def build_features(raw, with_targets=True):
    df = impute(raw)
    idx = df.index
    f = df.drop(columns="was_imputed").copy()          # current values

    # time features
    f["hour"] = idx.hour
    f["day_of_week"] = idx.day_of_week
    f["month"] = idx.month
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

    # lags and rolling means of the six core metrics
    for c in CORE:
        for lag in LAGS:
            f[f"{c}_lag{lag}h"] = df[c].shift(lag)
        f[f"{c}_roll6h_mean"] = df[c].rolling(6).mean()
        f[f"{c}_roll24h_mean"] = df[c].rolling(24).mean()
    f["us_aqi_lag48h"] = df["us_aqi"].shift(48)
    f["us_aqi_roll24h_std"] = df["us_aqi"].rolling(24).std()
    f["wind_speed_roll24h_std"] = df["wind_speed_10m"].rolling(24).std()
    f["pm2_5_roll24h_mean"] = df["pm2_5"].rolling(24).mean()

    # rate of change
    for c in ["us_aqi", "temperature_2m", "relative_humidity_2m", "surface_pressure"]:
        f[f"{c}_change1h"] = df[c].diff(1)
        f[f"{c}_change3h"] = df[c].diff(3)

    f["heat_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    f["was_imputed"] = df["was_imputed"]

    if not with_targets:
        return f

    # targets: values exactly HORIZON hours ahead (NaN until they exist)
    t = pd.DataFrame(index=idx)
    for c in CORE + ["wind_dir_sin", "wind_dir_cos"]:
        t[f"target_{c}_24h"] = df[c].shift(-HORIZON)
    flag = (df["precipitation"] > RAIN_THRESHOLD_MM).astype(float)
    flag = flag.where(df["precipitation"].notna())
    # any rain in hours t+1 ... t+24 (NaN if the full window is not known yet)
    future = flag[::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    t["target_rain_next24h"] = future
    return pd.concat([f, t], axis=1)


def main():
    raw = pd.read_csv("data/raw_history.csv", parse_dates=["time"], index_col="time")
    out = build_features(raw)
    target_cols = [c for c in out.columns if c.startswith("target_")]
    feature_cols = [c for c in out.columns if c not in target_cols]

    ready = out.dropna()
    print(f"Rows total: {len(out)}")
    print(f"Feature columns: {len(feature_cols)}   Target columns: {len(target_cols)}")
    print(f"Rows with all features and targets (usable for training): {len(ready)}")
    print(f"Rows without targets (latest 24h, forecasting only): "
          f"{int(out[target_cols].isna().any(axis=1).sum())}")
    print(f"Rows imputed: {int(out['was_imputed'].sum())}")
    print(f"Rain next 24h share: {ready['target_rain_next24h'].mean() * 100:.2f}%")
    print("\nMissing values per column (top 5):")
    print(out.isna().sum().sort_values(ascending=False).head(5))

    out.to_csv("data/features.csv")
    print("\nSaved to data/features.csv")


def selftest():
    """Remove hours from real data and check the imputation rule."""
    raw = pd.read_csv("data/raw_history.csv", parse_dates=["time"], index_col="time").iloc[:500]
    broken = raw.drop(raw.index[[100, 101, 200, 201, 202, 203, 204, 205]])
    out = impute(broken)
    print("Rows after regridding:", len(out), "(expected 500)")
    print("2-hour gap filled:", bool(out.loc[raw.index[100:102], "us_aqi"].notna().all()))
    print("6-hour gap left empty:", bool(out.loc[raw.index[200:206], "us_aqi"].isna().all()))
    print("Imputed rows flagged:", int(out["was_imputed"].sum()), "(expected 2)")
    short = build_features(raw.iloc[-72:], with_targets=False)
    print("Live mode (72h, no targets) last row has all features:",
          bool(short.iloc[-1].drop("was_imputed").notna().all()))


if __name__ == "__main__":
    selftest() if "--test" in sys.argv else main()