"""Fenêtre d'ingestion et structure du DAG Airflow."""

from datetime import date

import pytest

from src.data.ingestion import ARCHIVE_DELAY_DAYS, HISTORY_DAYS, date_window


def test_date_window_is_reproducible_and_complete():
    start, end = date_window(date(2026, 10, 17))
    assert end == date(2026, 10, 10)  # 7 jours de délai d'archive
    assert (end - start).days + 1 == HISTORY_DAYS
    assert (date(2026, 10, 17) - end).days == ARCHIVE_DELAY_DAYS


def test_dag_structure():
    pytest.importorskip("airflow")
    from airflow.models.dagbag import DagBag

    bag = DagBag(dag_folder="dags", include_examples=False)
    assert bag.import_errors == {}
    dag = bag.dags["air_pollution_retraining"]

    upstream = {t.task_id: t.upstream_task_ids for t in dag.tasks}
    assert upstream["validate"] == {"transform"}
    assert upstream["train_regressor"] == {"validate"}
    assert upstream["train_classifier"] == {"validate"}
    # Chaque promotion attend l'entraînement de SON modèle
    assert upstream["promote_regressor"] == {"train_regressor"}
    assert upstream["promote_classifier"] == {"train_classifier"}
    # Le rapport décrit les modèles servis : il vient APRÈS la promotion
    assert upstream["baseline_report"] == {"promote_regressor", "promote_classifier"}
