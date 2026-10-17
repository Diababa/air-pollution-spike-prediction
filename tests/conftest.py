import pandas as pd
import pytest

from src.api.app import PollutionFeatures


@pytest.fixture(scope="session")
def dataset():
    return pd.read_csv("data/processed/train.csv")


@pytest.fixture(scope="session")
def sample_payload(dataset):
    """Une observation réelle, au format attendu par l'API."""
    row = dataset.iloc[-1]
    return {name: row[name].item() for name in PollutionFeatures.model_fields}
