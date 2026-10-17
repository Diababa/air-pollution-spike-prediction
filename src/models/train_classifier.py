"""Entraînement du classifieur de pics de pollution à 48 h, tracé dans MLflow.

Même principe que train.py : métriques, baselines sur le même jeu de test,
nouvelle version candidate dans le Model Registry. La promotion en
production est décidée par src/models/promote.py.

Usage : python -m src.models.train_classifier
"""

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from src.models.baseline import classification_baselines
from src.models.common import (
    SPIKE_THRESHOLD_PROBA,
    chronological_split,
    classification_scores,
    feature_columns,
    load_data,
)
from src.tracking import CLASSIFIER_NAME, setup_mlflow

PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}


def _slug(name):
    return name.split(" (")[0].lower().replace(" ", "_").replace('"', "")


def train_classifier(df):
    setup_mlflow()
    train, test = chronological_split(df)
    cols = feature_columns(df)

    X_train, y_train = train[cols], train["spike_48h"]
    X_test, y_test = test[cols], test["spike_48h"]

    # Classes déséquilibrées (~12 % de pics) : on repondère les pics
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    with mlflow.start_run(run_name="xgboost-spike-classifier") as run:
        mlflow.set_tags({"task": "classification", "target": "spike_48h"})
        mlflow.log_params(PARAMS)
        mlflow.log_params({
            "scale_pos_weight": round(float(scale_pos_weight), 3),
            "decision_threshold": SPIKE_THRESHOLD_PROBA,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_end": f"{train['datetime'].max():%Y-%m-%d}",
            "test_end": f"{test['datetime'].max():%Y-%m-%d}",
            "spike_rate_test": round(float(y_test.mean()), 4),
        })

        model = XGBClassifier(**PARAMS, scale_pos_weight=scale_pos_weight)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= SPIKE_THRESHOLD_PROBA).astype(int)

        scores = classification_scores(y_test, preds)
        mlflow.log_metrics({f"test_{k.lower()}": v for k, v in scores.items()})

        baselines = classification_baselines(df, test)
        for name, base in baselines.items():
            mlflow.log_metric(f"baseline_{_slug(name)}_f1", base["F1"])
            mlflow.log_metric(f"baseline_{_slug(name)}_recall", base["Recall"])
        best_baseline_f1 = max(b["F1"] for b in baselines.values())
        mlflow.set_tag("beats_best_baseline_f1", str(scores["F1"] > best_baseline_f1))

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        mlflow.log_metrics({"test_tp": tp, "test_fp": fp, "test_fn": fn, "test_tn": tn})

        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=infer_signature(X_test, probs),
            input_example=X_test.head(3),
            registered_model_name=CLASSIFIER_NAME,
        )

        print(
            f"Precision {scores['Precision']:.3f} | Recall {scores['Recall']:.3f} "
            f"| F1 {scores['F1']:.3f} (meilleure baseline F1 : {best_baseline_f1:.3f})"
        )
        print(f"Run MLflow : {run.info.run_id}")

    return model, scores


def run():
    train_classifier(load_data())


if __name__ == "__main__":
    run()
