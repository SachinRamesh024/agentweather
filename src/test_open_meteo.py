import requests
import pandas as pd

LAT, LON = 25.3960, 68.3578  # Hyderabad, Sindh

weather = requests.get(
    "https://api.open-meteo.com/v1/forecast",
    params={
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,uv_index,wind_speed_10m,wind_direction_10m",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "UTC",
    },
    timeout=30,
).json()

air = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        "latitude": LAT,
        "longitude": LON,
        "hourly": "us_aqi,pm10,pm2_5",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "UTC",
    },
    timeout=30,
).json()

df = pd.DataFrame(weather["hourly"]).merge(pd.DataFrame(air["hourly"]), on="time")
print(df.head(10))
print(df.shape)