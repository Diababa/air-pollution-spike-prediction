"""Règle de promotion : un modèle ne passe en production que s'il bat
la meilleure baseline ET le champion en place."""

import pytest

from src.models.promote import should_promote


@pytest.mark.parametrize(
    "candidate, baseline, champion, higher, expected",
    [
        # F1 (plus haut = mieux)
        (0.35, 0.30, None, True, True),    # premier modèle, bat la baseline
        (0.28, 0.30, None, True, False),   # ne bat pas la baseline
        (0.35, 0.30, 0.36, True, False),   # moins bon que le champion
        (0.35, 0.30, 0.35, True, False),   # égalité : on garde le champion
        (0.37, 0.30, 0.36, True, True),    # bat baseline et champion
        # MAE (plus bas = mieux)
        (5.90, 5.76, None, False, False),  # cas réel du régresseur actuel
        (5.50, 5.76, 5.60, False, True),
    ],
)
def test_should_promote(candidate, baseline, champion, higher, expected):
    promoted, _ = should_promote(candidate, baseline, champion, higher)
    assert promoted is expected
