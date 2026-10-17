"""Tests de cohérence des modèles entraînés.

Deux invariants protègent des deux bugs déjà rencontrés :
1. les features attendues par chaque modèle = le schéma exposé par l'API ;
2. aucun modèle n'utilise une colonne dérivée de la cible (fuite de cible).
"""

import pytest

from src.api.app import PollutionFeatures
from src.models.common import load_classifier, load_regressor

LEAKING_COLUMNS = {"target_pm25_48h", "spike_48h"}

MODEL_LOADERS = [load_regressor, load_classifier]


@pytest.fixture(params=MODEL_LOADERS, ids=["regressor", "classifier"])
def model(request):
    return request.param()


def test_model_features_match_api_schema(model):
    expected = list(PollutionFeatures.model_fields)
    assert model.get_booster().feature_names == expected


def test_model_has_no_target_leakage(model):
    features = set(model.get_booster().feature_names)
    assert features & LEAKING_COLUMNS == set()
