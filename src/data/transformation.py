import pandas as pd
import numpy as np


def load_data():
    df = pd.read_csv("data/raw/pollution_weather.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df


def create_features(df):
    # 1. Lags = mémoire du passé
    for lag in [1, 3, 6, 12, 24, 48, 72, 168]:
        df[f"lag_{lag}"] = df["pm25"].shift(lag)

    # 2. Rolling statistics = tendance récente
    df["rolling_mean_6"] = df["pm25"].rolling(6).mean()
    df["rolling_mean_12"] = df["pm25"].rolling(12).mean()
    df["rolling_mean_24"] = df["pm25"].rolling(24).mean()
    df["rolling_mean_48"] = df["pm25"].rolling(48).mean()

    df["rolling_std_24"] = df["pm25"].rolling(24).std()
    df["rolling_max_24"] = df["pm25"].rolling(24).max()
    df["rolling_min_24"] = df["pm25"].rolling(24).min()

    # 3. Interactions météo
    df["temp_x_humidity"] = df["temperature"] * df["humidity"]

    # 4. Variables temporelles
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 5. Encodage cyclique
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def create_target(df):
    # Target régression : PM2.5 dans 48h
    df["target_pm25_48h"] = df["pm25"].shift(-48)

    # Target classification : spike dans 48h
    threshold = df["pm25"].quantile(0.90)
    df["spike_48h"] = (df["target_pm25_48h"] > threshold).astype(int)

    return df


def clean_final(df):
    df = df.dropna()
    return df


if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    df = create_target(df)
    df = clean_final(df)

    print(df.shape)
    print(df.head())
    print(df.isnull().sum())

    df.to_csv("data/processed/train.csv", index=False)
    print("✅ Dataset transformé et sauvegardé")