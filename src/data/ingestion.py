import requests
import pandas as pd

#### Récupétation des donnée de pollution et météo pour Paris (48.85, 2.35) sur les 3 dernières années (1095 jours) ####    
def fetch_pollution():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": 48.85,
        "longitude": 2.35,
        "hourly": "pm2_5",
        "past_days": 1095
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "pm25": data["hourly"]["pm2_5"]
    })

    return df

### Récupération des données météo pour Paris (48.85, 2.35) sur les 3 dernières années (1095 jours) ####
def fetch_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 48.85,
        "longitude": 2.35,
        "start_date": "2023-04-27",
        "end_date": "2026-04-26",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m"
        ]
    }

    response = requests.get(url, params=params)

    print("Weather status:", response.status_code)

    data = response.json()

    if "hourly" not in data:
        print("❌ Erreur météo:", data)
        return None

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "wind_direction": data["hourly"]["wind_direction_10m"]
    })

    return df

### Merge des données de pollution et météo sur la base de la colonne datetime, en utilisant une jointure interne (inner join) pour ne conserver que les lignes où les deux datasets ont des données correspondantes. ####
def merge_data():
    df_pollution = fetch_pollution()
    df_weather = fetch_weather()

    # Conversion datetime
    df_pollution["datetime"] = pd.to_datetime(df_pollution["datetime"])
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])

    # Merge propre
    df = pd.merge(df_pollution, df_weather, on="datetime", how="inner")

    return df


if __name__ == "__main__":
    df = merge_data()

    print(df.head())
    print(df.info())

    df.to_csv("data/raw/pollution_weather.csv", index=False)