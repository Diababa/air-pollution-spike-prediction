import os

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from src.models.common import (
    CLASSIFIER_PATH,
    REGRESSOR_PATH,
    load_classifier,
    load_regressor,
)


# ---------------------------------------------------
# Charger modèles
# ---------------------------------------------------
# Par défaut : fichiers JSON exportés par src/models/promote.py (déploiement
# Render, sans serveur MLflow). Avec MODEL_SOURCE=mlflow : chargement direct
# de la version « champion » depuis le Model Registry.

def load_model(registered_name, load_fallback):
    if os.getenv("MODEL_SOURCE") == "mlflow":
        import mlflow.xgboost
        from mlflow.exceptions import MlflowException
        try:
            return mlflow.xgboost.load_model(f"models:/{registered_name}@champion")
        except MlflowException as exc:
            # Aucun champion (ex. le modèle n'a jamais battu les baselines) :
            # on garde le dernier modèle exporté plutôt que de bloquer l'API.
            print(f"[{registered_name}] pas de champion MLflow ({exc}), repli sur le fichier exporté")
    return load_fallback()


regression_model = load_model("pm25-48h-regressor", lambda: load_regressor(REGRESSOR_PATH))
classifier_model = load_model("pm25-spike-classifier", lambda: load_classifier(CLASSIFIER_PATH))


# ---------------------------------------------------
# Initialiser API
# ---------------------------------------------------

app = FastAPI(
    title="Air Pollution Prediction API",
    description="Predict PM2.5 and pollution spikes 48h ahead",
    version="1.0"
)


# ---------------------------------------------------
# Input schema
# ---------------------------------------------------

class PollutionFeatures(BaseModel):

    pm25: float
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float

    lag_1: float
    lag_3: float
    lag_6: float
    lag_12: float
    lag_24: float
    lag_48: float
    lag_72: float
    lag_168: float

    rolling_mean_6: float
    rolling_mean_12: float
    rolling_mean_24: float
    rolling_mean_48: float

    rolling_std_24: float
    rolling_max_24: float
    rolling_min_24: float

    temp_x_humidity: float

    hour: int
    day_of_week: int
    month: int
    is_weekend: int

    hour_sin: float
    hour_cos: float

    month_sin: float
    month_cos: float


# ---------------------------------------------------
# Root endpoint
# ---------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Air Pollution ML API is running"
    }


# ---------------------------------------------------
# Regression endpoint
# ---------------------------------------------------

@app.post("/predict-regression")
def predict_regression(features: PollutionFeatures):

    data = pd.DataFrame([features.model_dump()])

    prediction = regression_model.predict(data)[0]

    return {
        "predicted_pm25_48h": round(float(prediction), 2)
    }


# ---------------------------------------------------
# Classification endpoint
# ---------------------------------------------------

@app.post("/predict-spike")
def predict_spike(features: PollutionFeatures):

    data = pd.DataFrame([features.model_dump()])

    probability = classifier_model.predict_proba(data)[0][1]

    prediction = int(probability >= 0.30)

    return {
        "spike_probability": round(float(probability), 3),
        "spike_prediction": prediction
    }