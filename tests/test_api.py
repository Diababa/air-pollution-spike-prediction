"""Tests de l'API de prédiction.

Couvre notamment la régression de bug suivante : le modèle de régression
avait été entraîné avec une colonne de plus que le schéma de l'API, ce qui
faisait échouer /predict-regression avec une erreur 500 en production.
"""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_regression_returns_a_value(sample_payload):
    response = client.post("/predict-regression", json=sample_payload)

    assert response.status_code == 200, response.text
    prediction = response.json()["predicted_pm25_48h"]
    assert isinstance(prediction, float)
    assert 0 <= prediction < 500  # concentration PM2.5 plausible


def test_predict_spike_returns_probability_and_label(sample_payload):
    response = client.post("/predict-spike", json=sample_payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["spike_probability"] <= 1.0
    assert body["spike_prediction"] in (0, 1)


def test_missing_feature_is_rejected(sample_payload):
    incomplete = dict(sample_payload)
    incomplete.pop("pm25")

    response = client.post("/predict-spike", json=incomplete)
    assert response.status_code == 422
