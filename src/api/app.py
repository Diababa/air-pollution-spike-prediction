from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd


# ---------------------------------------------------
# Charger modèles
# ---------------------------------------------------

regression_model = joblib.load(
    "models/xgboost_pm25_48h.pkl"
)

classifier_model = joblib.load(
    "models/xgboost_spike_classifier.pkl"
)


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

    data = pd.DataFrame([features.dict()])

    prediction = regression_model.predict(data)[0]

    return {
        "predicted_pm25_48h": round(float(prediction), 2)
    }


# ---------------------------------------------------
# Classification endpoint
# ---------------------------------------------------

@app.post("/predict-spike")
def predict_spike(features: PollutionFeatures):

    data = pd.DataFrame([features.dict()])

    probability = classifier_model.predict_proba(data)[0][1]

    prediction = int(probability >= 0.30)

    return {
        "spike_probability": round(float(probability), 3),
        "spike_prediction": prediction
    }