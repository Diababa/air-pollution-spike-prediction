import pandas as pd


REQUIRED_COLUMNS = [
    "datetime",
    "pm25",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "target_pm25_48h",
]


def validate_dataset(path="data/processed/train.csv"):
    df = pd.read_csv(path)

    print("Shape:", df.shape)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols}")

    if df.isnull().sum().sum() > 0:
        print(df.isnull().sum())
        raise ValueError("Le dataset contient des valeurs manquantes")

    if df.duplicated(subset=["datetime"]).sum() > 0:
        raise ValueError("Le dataset contient des dates dupliquées")

    df["datetime"] = pd.to_datetime(df["datetime"])

    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("Les dates ne sont pas triées")

    if (df["pm25"] < 0).any():
        raise ValueError("PM2.5 contient des valeurs négatives")

    print("✅ Dataset validé avec succès")


if __name__ == "__main__":
    validate_dataset()