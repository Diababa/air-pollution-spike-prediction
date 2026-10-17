"""Éléments partagés par l'entraînement, les baselines et la promotion.

Un seul endroit pour le découpage chronologique, les colonnes exclues
et les métriques : les trois étapes évaluent donc exactement la même chose.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

DATA_PATH = "data/processed/train.csv"
TRAIN_RATIO = 0.8
SPIKE_THRESHOLD_PROBA = 0.30

# Colonnes à ne jamais donner au modèle :
# - datetime        : identifiant temporel
# - target_pm25_48h : la cible
# - spike_48h       : dérivée directement de la cible (fuite de cible)
EXCLUDED_COLUMNS = ["datetime", "target_pm25_48h", "spike_48h"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def chronological_split(df, ratio=TRAIN_RATIO):
    """Test strictement postérieur à l'entraînement : aucun shuffle."""
    split_index = int(len(df) * ratio)
    return df.iloc[:split_index], df.iloc[split_index:]


def feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]


def regression_scores(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_scores(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
