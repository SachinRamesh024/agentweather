"""Backfill hourly history for Hyderabad, Sindh from Open-Meteo -> data/raw_history.csv"""
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

LAT, LON = 25.3960, 68.3578
START = date(2025, 1, 1)
END = date.today() - timedelta(days=1)       # last full day
ARCHIVE_END = END - timedelta(days=7)        # archive data lags a few days behind

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "wind_direction_10m", "surface_pressure"]
AIR_VARS = ["us_aqi", "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "uv_index"]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def get_json(url, params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}")


def to_frame(payload):
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def fetch_weather():
    archive = to_frame(get_json(ARCHIVE_URL, {
        "latitude": LAT, "longitude": LON,
        "start_date": START.isoformat(), "end_date": ARCHIVE_END.isoformat(),
        "hourly": ",".join(WEATHER_VARS), "timezone": "UTC"}))
    # fill the most recent days (not yet in the archive) from the forecast API
    recent = to_frame(get_json(FORECAST_URL, {
        "latitude": LAT, "longitude": LON,
        "hourly": ",".join(WEATHER_VARS),
        "past_days": 10, "forecast_days": 1, "timezone": "UTC"}))
    recent = recent[recent.index > archive.index.max()]
    recent = recent[recent.index < pd.Timestamp(END + timedelta(days=1))]
    return pd.concat([archive, recent])


def fetch_air():
    frames, cur = [], START
    while cur <= END:
        chunk_end = min(cur + timedelta(days=89), END)
        print(f"  air quality {cur} -> {chunk_end}")
        frames.append(to_frame(get_json(AIR_URL, {
            "latitude": LAT, "longitude": LON,
            "start_date": cur.isoformat(), "end_date": chunk_end.isoformat(),
            "hourly": ",".join(AIR_VARS), "timezone": "UTC"})))
        cur = chunk_end + timedelta(days=1)
    return pd.concat(frames)


def main():
    print("Fetching weather...")
    weather = fetch_weather()
    print("Fetching air quality...")
    air = fetch_air()

    df = weather.join(air, how="outer")
    df = df[~df.index.duplicated(keep="first")]
    full = pd.date_range(pd.Timestamp(START),
                         pd.Timestamp(END) + pd.Timedelta(hours=23), freq="h")
    df = df.reindex(full)
    df.index.name = "time"

    print(f"\nRows: {len(df)}  ({df.index.min()} -> {df.index.max()})")
    print("\nMissing values per column:")
    print(df.isna().sum())
    for col in ["us_aqi", "uv_index"]:
        print(f"First valid {col}: {df[col].first_valid_index()}")

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/raw_history.csv")
    print("\nSaved to data/raw_history.csv")


if __name__ == "__main__":
    main()