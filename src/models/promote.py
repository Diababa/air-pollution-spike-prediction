"""Promotion des modèles : décide quelle version passe en production.

Règle (identique pour les deux modèles) : la dernière version enregistrée
devient « champion » seulement si, sur le jeu de test ACTUEL,
  1. elle bat la meilleure baseline naïve, et
  2. elle bat le champion en place (réévalué sur ce même jeu de test :
     comparer à une métrique enregistrée sur une ancienne fenêtre de test
     serait biaisé, puisque la fenêtre se déplace à chaque réentraînement).

Si elle est promue, elle est aussi exportée en .pkl pour l'API.
Sinon, rien ne change en production et la décision est tracée sur la
version dans le Model Registry (tags promotion_*).

Usage : python -m src.models.promote
"""

import joblib
import mlflow.xgboost

from src.models.baseline import classification_baselines, regression_baselines
from src.models.common import (
    SPIKE_THRESHOLD_PROBA,
    chronological_split,
    classification_scores,
    feature_columns,
    load_data,
    regression_scores,
)
from src.tracking import CHAMPION_ALIAS, CLASSIFIER_NAME, REGRESSOR_NAME, setup_mlflow

EXPORT_PATHS = {
    REGRESSOR_NAME: "models/xgboost_pm25_48h.pkl",
    CLASSIFIER_NAME: "models/xgboost_spike_classifier.pkl",
}


def should_promote(candidate, baseline, champion=None, higher_is_better=True):
    """Décision pure, testable sans MLflow."""
    def better(a, b):
        return a > b if higher_is_better else a < b

    if not better(candidate, baseline):
        return False, "ne bat pas la meilleure baseline"
    if champion is not None and not better(candidate, champion):
        return False, "ne bat pas le champion en place"
    return True, "promu"


def _score(name, model, df, train, test):
    """Renvoie (score du modèle, meilleure baseline, plus haut = mieux ?)."""
    X_test = test[feature_columns(df)]
    if name == REGRESSOR_NAME:
        mae = regression_scores(test["target_pm25_48h"], model.predict(X_test))["MAE"]
        best = min(b["MAE"] for b in regression_baselines(train, test).values())
        return mae, best, False
    preds = (model.predict_proba(X_test)[:, 1] >= SPIKE_THRESHOLD_PROBA).astype(int)
    f1 = classification_scores(test["spike_48h"], preds)["F1"]
    best = max(b["F1"] for b in classification_baselines(df, test).values())
    return f1, best, True


def promote(name, client, df):
    train, test = chronological_split(df)

    latest = client.search_model_versions(
        f"name = '{name}'", order_by=["version_number DESC"], max_results=1
    )
    if not latest:
        print(f"[{name}] aucune version enregistrée, rien à promouvoir")
        return False
    candidate = latest[0]
    cand_model = mlflow.xgboost.load_model(f"models:/{name}/{candidate.version}")
    cand_score, baseline, higher = _score(name, cand_model, df, train, test)

    champion_score = None
    try:
        champion = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        champion = None
    if champion is not None:
        if champion.version == candidate.version:
            print(f"[{name}] v{candidate.version} est déjà champion")
            return False
        champ_model = mlflow.xgboost.load_model(f"models:/{name}@{CHAMPION_ALIAS}")
        champion_score, _, _ = _score(name, champ_model, df, train, test)

    promoted, reason = should_promote(cand_score, baseline, champion_score, higher)

    metric = "F1" if higher else "MAE"
    tags = {
        "promotion_decision": reason,
        f"promotion_candidate_{metric.lower()}": f"{cand_score:.4f}",
        f"promotion_baseline_{metric.lower()}": f"{baseline:.4f}",
    }
    if champion_score is not None:
        tags[f"promotion_champion_{metric.lower()}"] = f"{champion_score:.4f}"
    for key, value in tags.items():
        client.set_model_version_tag(name, candidate.version, key, value)

    champ_txt = "aucun" if champion_score is None else f"{champion_score:.3f}"
    print(
        f"[{name}] v{candidate.version} : {metric} {cand_score:.3f} | "
        f"baseline {baseline:.3f} | champion {champ_txt} -> {reason}"
    )

    if promoted:
        client.set_registered_model_alias(name, CHAMPION_ALIAS, candidate.version)
        joblib.dump(cand_model, EXPORT_PATHS[name])
        print(f"[{name}] exporté vers {EXPORT_PATHS[name]}")
    return promoted


def run(names=(REGRESSOR_NAME, CLASSIFIER_NAME)):
    client = setup_mlflow()
    df = load_data()
    return {name: promote(name, client, df) for name in names}


if __name__ == "__main__":
    run()
