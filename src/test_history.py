import requests

LAT, LON = 25.3960, 68.3578  # Hyderabad, Sindh
START, END = "2025-01-01", "2025-01-07"

def check(label, url, variables):
    for v in variables:
        try:
            r = requests.get(url, params={
                "latitude": LAT, "longitude": LON,
                "start_date": START, "end_date": END,
                "hourly": v, "timezone": "UTC"}, timeout=30)
            if r.status_code != 200:
                print(label, v, "-> NOT AVAILABLE:", r.text[:100])
                continue
            vals = r.json()["hourly"][v]
            good = sum(x is not None for x in vals)
            print(label, v, "-> OK,", good, "of", len(vals), "values")
        except Exception as e:
            print(label, v, "-> ERROR:", e)

check("archive", "https://archive-api.open-meteo.com/v1/archive",
      ["temperature_2m", "relative_humidity_2m", "precipitation",
       "wind_speed_10m", "wind_direction_10m", "uv_index",
       "precipitation_probability"])

check("air", "https://air-quality-api.open-meteo.com/v1/air-quality",
      ["us_aqi", "pm10", "pm2_5", "uv_index"])