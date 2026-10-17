"""Configuration MLflow commune à tout le projet.

En local, sans serveur : base SQLite `mlflow.db` à la racine du projet
(le Model Registry exige une base de données, pas un simple dossier).
Avec le serveur Docker : MLFLOW_TRACKING_URI=http://mlflow:5000.
"""

import os

import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME = "air-pollution-48h"
REGRESSOR_NAME = "pm25-48h-regressor"
CLASSIFIER_NAME = "pm25-spike-classifier"
CHAMPION_ALIAS = "champion"


def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    return MlflowClient()
