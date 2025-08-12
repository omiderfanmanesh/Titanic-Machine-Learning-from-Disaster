"""Test fixtures and utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_titanic_data():
    """Create sample Titanic-like data for testing."""
    np.random.seed(42)
    
    n_samples = 100
    
    data = {
        "PassengerId": range(1, n_samples + 1),
        "Survived": np.random.choice([0, 1], n_samples),
        "Pclass": np.random.choice([1, 2, 3], n_samples),
        "Name": [f"Person {i}, Mr." for i in range(n_samples)],
        "Sex": np.random.choice(["male", "female"], n_samples),
        "Age": np.random.normal(30, 15, n_samples).clip(0, 80),
        "SibSp": np.random.poisson(0.5, n_samples),
        "Parch": np.random.poisson(0.4, n_samples),
        "Ticket": [f"TICKET{i}" for i in range(n_samples)],
        "Fare": np.random.lognormal(3, 1, n_samples),
        "Cabin": np.random.choice(["A123", "B456", "C789", None], n_samples, p=[0.1, 0.1, 0.1, 0.7]),
        "Embarked": np.random.choice(["C", "Q", "S"], n_samples, p=[0.2, 0.1, 0.7])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data():
    """Create sample test data (without target)."""
    np.random.seed(43)
    
    n_samples = 50
    
    data = {
        "PassengerId": range(1001, 1001 + n_samples),
        "Pclass": np.random.choice([1, 2, 3], n_samples),
        "Name": [f"TestPerson {i}, Mrs." for i in range(n_samples)],
        "Sex": np.random.choice(["male", "female"], n_samples),
        "Age": np.random.normal(30, 15, n_samples).clip(0, 80),
        "SibSp": np.random.poisson(0.5, n_samples),
        "Parch": np.random.poisson(0.4, n_samples),
        "Ticket": [f"TEST_TICKET{i}" for i in range(n_samples)],
        "Fare": np.random.lognormal(3, 1, n_samples),
        "Cabin": np.random.choice(["A123", "B456", "C789", None], n_samples, p=[0.1, 0.1, 0.1, 0.7]),
        "Embarked": np.random.choice(["C", "Q", "S"], n_samples, p=[0.2, 0.1, 0.7])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "train_path": "data/train.csv",
        "test_path": "data/test.csv",
        "target_column": "Survived",
        "id_column": "PassengerId",
        "task_type": "binary",
        "model_name": "logistic",
        "model_params": {"random_state": 42},
        "cv_folds": 3,
        "cv_strategy": "stratified"
    }


@pytest.fixture
def feature_config():
    """Sample feature configuration."""
    return {
        "add_family_features": True,
        "add_title_features": True,
        "add_deck_features": True,
        "add_ticket_features": True,
        "transform_fare": True,
        "add_missing_indicators": True
    }
