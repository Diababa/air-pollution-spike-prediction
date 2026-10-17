"""Ingestion des données de pollution (PM2.5) et météo pour Paris (48.85, 2.35).

La fenêtre est glissante : HISTORY_DAYS jours se terminant ARCHIVE_DELAY_DAYS
jours avant la date de référence. Le délai laisse le temps à l'archive météo
Open-Meteo d'être consolidée. Une date de référence explicite (fournie par
Airflow) rend chaque exécution reproductible.

Usage : python -m src.data.ingestion
"""

from datetime import date, timedelta

import pandas as pd
import requests

LATITUDE, LONGITUDE = 48.85, 2.35
HISTORY_DAYS = 1095
ARCHIVE_DELAY_DAYS = 7
RAW_PATH = "data/raw/pollution_weather.csv"
TIMEOUT = 60


def date_window(reference=None):
    reference = reference or date.today()
    end = reference - timedelta(days=ARCHIVE_DELAY_DAYS)
    start = end - timedelta(days=HISTORY_DAYS - 1)
    return start, end


def _get_hourly(url, params):
    response = requests.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if "hourly" not in data:
        raise RuntimeError(f"Réponse inattendue de {url} : {data}")
    return data["hourly"]


def fetch_pollution(start, end):
    hourly = _get_hourly(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "pm2_5",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
    )
    return pd.DataFrame({"datetime": hourly["time"], "pm25": hourly["pm2_5"]})


def fetch_weather(start, end):
    hourly = _get_hourly(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
        },
    )
    return pd.DataFrame({
        "datetime": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
    })


def merge_data(reference=None):
    start, end = date_window(reference)
    print(f"Fenêtre d'ingestion : {start} -> {end}")

    df_pollution = fetch_pollution(start, end)
    df_weather = fetch_weather(start, end)

    df_pollution["datetime"] = pd.to_datetime(df_pollution["datetime"])
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])

    # Jointure interne : seules les heures présentes dans les deux sources
    return pd.merge(df_pollution, df_weather, on="datetime", how="inner")


def run(reference=None, path=RAW_PATH):
    df = merge_data(reference)
    df.to_csv(path, index=False)
    print(f"{len(df)} lignes sauvegardées dans {path}")
    return path


if __name__ == "__main__":
    run()
