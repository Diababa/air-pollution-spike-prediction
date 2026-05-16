import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path="data/processed/train.csv"):
    return pd.read_csv(path)


def train_model(df):
    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train.drop(columns=["datetime", "target_pm25_48h"])
    y_train = train["target_pm25_48h"]

    X_test = test.drop(columns=["datetime", "target_pm25_48h"])
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

    joblib.dump(model, "models/xgboost_pm25_48h.pkl")
    print("✅ Modèle sauvegardé dans models/xgboost_pm25_48h.pkl")


if __name__ == "__main__":
    df = load_data()
    train_model(df)