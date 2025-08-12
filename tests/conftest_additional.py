"""Additional fixtures for integration and unit tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_features():
    """Sample feature matrix for model testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "Age": np.random.normal(30, 15, 100).clip(0, 80),
        "Fare": np.random.lognormal(3, 1, 100).clip(0, 500),
        "Pclass": np.random.choice([1, 2, 3], 100),
        "Sex_male": np.random.randint(0, 2, 100),
        "FamilySize": np.random.poisson(2, 100).clip(1, 8),
        "IsAlone": np.random.randint(0, 2, 100),
        "Title_Mr": np.random.randint(0, 2, 100),
        "Title_Mrs": np.random.randint(0, 2, 100),
        "Deck_C": np.random.randint(0, 2, 100),
        "Embarked_S": np.random.randint(0, 2, 100)
    })


@pytest.fixture
def sample_targets():
    """Sample target values for model testing."""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100), name="Survived")


@pytest.fixture
def sample_model_config():
    """Sample model configuration."""
    return {
        "name": "logistic",
        "params": {
            "C": 1.0,
            "random_state": 42,
            "max_iter": 1000
        }
    }


@pytest.fixture
def sample_feature_config():
    """Sample feature engineering configuration."""
    return {
        "numeric_features": ["Age", "Fare"],
        "categorical_features": ["Sex", "Pclass", "Embarked"],
        "use_family_features": True,
        "use_title_features": True,
        "use_deck_features": True,
        "use_ticket_features": False,
        "impute_age": True,
        "log_transform_fare": False,
        "create_age_bins": False
    }
