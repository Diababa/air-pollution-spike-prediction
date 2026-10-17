import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Colonnes à ne jamais donner au modèle :
# - datetime        : identifiant temporel
# - target_pm25_48h : la cible
# - spike_48h       : dérivée directement de la cible (target > seuil),
#                     donc fuite de cible si on la garde en feature
EXCLUDED_COLUMNS = ["datetime", "target_pm25_48h", "spike_48h"]


def load_data(path="data/processed/train.csv"):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def train_model(df):
    # Découpage chronologique : le test est strictement postérieur
    # à l'entraînement (aucun shuffle, série temporelle)
    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    print(
        f"Train : {len(train)} h "
        f"({train['datetime'].min():%Y-%m-%d} -> {train['datetime'].max():%Y-%m-%d})"
    )
    print(
        f"Test  : {len(test)} h "
        f"({test['datetime'].min():%Y-%m-%d} -> {test['datetime'].max():%Y-%m-%d})"
    )

    X_train = train.drop(columns=EXCLUDED_COLUMNS)
    y_train = train["target_pm25_48h"]

    X_test = test.drop(columns=EXCLUDED_COLUMNS)
    y_test = test["target_pm25_48h"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.3f}")

    # Baseline de persistance sur le même jeu de test, pour situer le modèle
    baseline_mae = mean_absolute_error(y_test, test["pm25"])
    print(f"Baseline persistance MAE : {baseline_mae:.2f}")

    joblib.dump(model, "models/xgboost_pm25_48h.pkl")
    print("Modele sauvegarde dans models/xgboost_pm25_48h.pkl")


if __name__ == "__main__":
    df = load_data()
    train_model(df)
