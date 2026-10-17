"""
### Réentraînement hebdomadaire — prédiction de pics de pollution PM2.5 à 48 h

1. **ingest** : télécharge 3 ans de PM2.5 et de météo (fenêtre glissante
   calée sur la date logique du run, donc rejouable à l'identique).
2. **transform** : features de séries temporelles et cibles à 48 h.
3. **validate** : contrôles qualité (colonnes, valeurs manquantes, doublons,
   ordre chronologique). Le pipeline s'arrête ici si les données sont mauvaises.
4. **train_regressor / train_classifier** (en parallèle) : runs MLflow,
   nouvelles versions candidates dans le Model Registry.
5. **promote_regressor / promote_classifier** : alias « champion » donné
   uniquement si la version bat les baselines ET le champion en place ;
   le champion est exporté pour l'API.
6. **baseline_report** : rapport de comparaison aux baselines, généré APRÈS
   la promotion pour décrire les modèles réellement servis par l'API.

Hebdomadaire plutôt que quotidien : une semaine de données nouvelles sur
3 ans d'historique (environ 0,6 %) ne justifie pas un réentraînement par jour.
"""

import os
from datetime import datetime, timedelta

from airflow.sdk import chain, dag, task

PROJECT_DIR = os.getenv("AIR_POLLUTION_PROJECT_DIR", "/opt/airflow/project")


def _in_project():
    """Les modules du projet utilisent des chemins relatifs à sa racine."""
    os.chdir(PROJECT_DIR)


@dag(
    dag_id="air_pollution_retraining",
    schedule="@weekly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["ml", "mlflow", "air-pollution"],
    doc_md=__doc__,
)
def air_pollution_retraining():

    @task(retries=3, retry_delay=timedelta(minutes=10))
    def ingest(logical_date=None):
        _in_project()
        from src.data.ingestion import run

        reference = (logical_date or datetime.now()).date()
        return run(reference=reference)

    @task
    def transform():
        _in_project()
        from src.data.transformation import run

        return run()

    @task(retries=0)
    def validate():
        _in_project()
        from src.data.validation import validate_dataset

        validate_dataset()

    @task
    def train_regressor():
        _in_project()
        from src.models.train import run

        run()

    @task
    def train_classifier():
        _in_project()
        from src.models.train_classifier import run

        run()

    @task
    def baseline_report():
        _in_project()
        from src.models.baseline import run

        run()

    @task
    def promote_regressor():
        _in_project()
        from src.models.promote import run
        from src.tracking import REGRESSOR_NAME

        return run(names=(REGRESSOR_NAME,))

    @task
    def promote_classifier():
        _in_project()
        from src.models.promote import run
        from src.tracking import CLASSIFIER_NAME

        return run(names=(CLASSIFIER_NAME,))

    chain(
        ingest(),
        transform(),
        validate(),
        [train_regressor(), train_classifier()],
        [promote_regressor(), promote_classifier()],
        baseline_report(),
    )


air_pollution_retraining()
