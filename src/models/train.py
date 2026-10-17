"""Entraînement du régresseur PM2.5 à 48 h, tracé dans MLflow.

Chaque exécution :
- enregistre paramètres, période d'entraînement / de test et métriques ;
- calcule les baselines naïves sur le MÊME jeu de test et les journalise
  dans le même run, pour qu'aucune métrique ne soit lue sans référence ;
- enregistre le modèle dans le Model Registry comme nouvelle version
  candidate. La mise en production (alias « champion ») est décidée
  séparément par src/models/promote.py.

Usage : python -m src.models.train
"""

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from xgboost import XGBRegressor

from src.models.baseline import regression_baselines
from src.models.common import (
    chronological_split,
    feature_columns,
    load_data,
    regression_scores,
)
from src.tracking import REGRESSOR_NAME, setup_mlflow

PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}


def _slug(name):
    return name.split(" (")[0].lower().replace(" ", "_")


def train_model(df):
    setup_mlflow()
    train, test = chronological_split(df)
    cols = feature_columns(df)

    X_train, y_train = train[cols], train["target_pm25_48h"]
    X_test, y_test = test[cols], test["target_pm25_48h"]

    with mlflow.start_run(run_name="xgboost-regressor") as run:
        mlflow.set_tags({"task": "regression", "target": "pm25_48h"})
        mlflow.log_params(PARAMS)
        mlflow.log_params({
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": f"{train['datetime'].min():%Y-%m-%d}",
            "train_end": f"{train['datetime'].max():%Y-%m-%d}",
            "test_end": f"{test['datetime'].max():%Y-%m-%d}",
            "n_features": len(cols),
        })

        model = XGBRegressor(**PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        scores = regression_scores(y_test, preds)
        mlflow.log_metrics({f"test_{k.lower()}": v for k, v in scores.items()})

        baselines = regression_baselines(train, test)
        for name, base in baselines.items():
            mlflow.log_metric(f"baseline_{_slug(name)}_mae", base["MAE"])
        best_baseline_mae = min(b["MAE"] for b in baselines.values())
        mlflow.set_tag("beats_best_baseline_mae", str(scores["MAE"] < best_baseline_mae))

        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=infer_signature(X_test, preds),
            input_example=X_test.head(3),
            registered_model_name=REGRESSOR_NAME,
        )

        print(f"MAE {scores['MAE']:.2f} | RMSE {scores['RMSE']:.2f} | R2 {scores['R2']:.3f}")
        print(f"Meilleure baseline MAE : {best_baseline_mae:.2f}")
        print(f"Run MLflow : {run.info.run_id}")

    return model, scores


def run():
    train_model(load_data())


if __name__ == "__main__":
    run()
