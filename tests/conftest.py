"""
Test configuration and fixtures specifically for features module testing.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
from pathlib import Path


@pytest.fixture(name="features_sample_titanic_data")
def sample_titanic_data():
    """Standard Titanic dataset sample for testing in the features module."""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
        'Name': [
            'Braund, Mr. Owen Harris',
            'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
            'Heikkinen, Miss. Laina',
            'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
            'Allen, Mr. William Henry',
            'Moran, Master. James',
            'McCarthy, Mr. Timothy J',
            'Palsson, Master. Gosta Leonard',
            'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
            'Nasser, Mrs. Nicholas (Adele Achem)'
        ],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
        'Age': [22.0, 38.0, 26.0, 35.0, np.nan, 54.0, np.nan, 2.0, 27.0, 14.0],
        'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
        'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '330877', '17463', '349909', '347742', '237736'],
        'Fare': [7.2500, 71.2833, 7.9250, 53.1000, np.nan, 8.4583, 51.8625, 21.0750, 11.1333, 30.0708],
        'Cabin': ['', 'C85', '', 'C123', '', '', 'E46', '', '', ''],
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C']
    })


@pytest.fixture
def sample_target():
    """Standard target variable for testing."""
    return pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='Survived')


@pytest.fixture
def test_config_yaml():
    """Create a test YAML configuration that matches the real config structure."""
    config = {
        "train_path": "data/raw/train.csv",
        "test_path": "data/raw/test.csv",
        "target_column": "Survived",
        "id_column": "PassengerId",
        "task_type": "binary",

        "required_columns": [
            "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
            "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
        ],

        "numeric_columns": ["Age", "SibSp", "Parch", "Fare"],
        "categorical_columns": ["Sex", "Embarked", "Pclass", "Deck", "Title"],

        "handle_missing": True,
        "encode_categorical": True,
        "scale_features": True,

        "skip_encoding_columns": ["PassengerId"],

        "log_transform_fare": True,
        "age_bins": 5,
        "rare_title_threshold": 10,

        "encoding": {
            "default": {
                "method": "onehot",
                "handle_missing": "value",
                "handle_unknown": "ignore"
            },
            "per_column": {
                "Title": {"method": "catboost", "a": 1.0},
                "Deck": {"method": "onehot"},
                "Embarked": {"method": "onehot"}
            }
        },

        "imputation": {
            "order": ["Fare", "Age"],
            "exclude": ["PassengerId", "Name", "Ticket", "Title", "Deck"],
            "default": {
                "numeric": "median",
                "categorical": "constant",
                "fill_value": "Unknown",
                "add_missing_indicators": False,
                "missing_indicator_prefix": "__miss_",
                "debug": True
            },
            "per_column": {
                "Age": {
                    "method": "model",
                    "estimator": "random_forest",
                    "features": ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"],
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": 42,
                    "clip_min": 0,
                    "clip_max": 80
                },
                "Fare": {"method": "mean", "clip_min": 0},
                "Embarked": {"method": "constant", "fill_value": "S"},
                "Cabin": {"method": "most_frequent"}
            }
        },

        "feature_engineering": {
            "pre_impute": [
                "FamilySizeTransform",
                "DeckTransform",
                "TicketGroupTransform"
            ],
            "post_impute": [
                "FareTransform"
            ]
        }
    }

    return config


@pytest.fixture
def temp_yaml_config(test_config_yaml):
    """Create a temporary YAML file with test configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config_yaml, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def edge_case_data():
    """Data with various edge cases for robust testing."""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5],
        'Pclass': [1, 2, 3, 1, 2],
        'Name': [
            'Test, Mr. Normal',
            'Test, Mrs. Normal',
            'Test, Master. Child',
            'Test, Dr. Rare',  # Rare title
            'Test, Rev. VeryRare'  # Very rare title
        ],
        'Sex': ['male', 'female', 'male', 'male', 'male'],
        'Age': [25.0, np.nan, 5.0, -1.0, 150.0],  # Negative and extreme ages
        'SibSp': [0, 1, 8, 0, 0],  # Some extreme family sizes
        'Parch': [0, 2, 5, 0, 0],
        'Ticket': ['SAME', 'SAME', 'UNIQUE1', 'UNIQUE2', 'UNIQUE3'],  # Shared tickets
        'Fare': [0.0, np.nan, -5.0, 1000.0, np.nan],  # Zero, negative, extreme fares
        'Cabin': ['A1', '', 'B2 C3 D4', np.nan, 'Z99'],  # Various cabin formats
        'Embarked': ['S', np.nan, 'C', 'Q', 'X']  # Unknown port
    })


@pytest.fixture
def missing_heavy_data():
    """Data with heavy missing values to test robustness."""
    data = pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5],
        'Pclass': [1, 2, np.nan, 1, 3],
        'Name': ['A, Mr. Test', 'B, Mrs. Test', np.nan, 'D, Miss Test', 'E, Master Test'],
        'Sex': ['male', np.nan, 'female', 'male', np.nan],
        'Age': [np.nan, np.nan, np.nan, 25.0, np.nan],  # Mostly missing
        'SibSp': [0, 1, np.nan, 0, 2],
        'Parch': [0, np.nan, 1, 0, np.nan],
        'Ticket': ['T1', np.nan, 'T3', np.nan, 'T5'],
        'Fare': [np.nan, np.nan, 25.0, np.nan, np.nan],  # Mostly missing
        'Cabin': [np.nan, np.nan, np.nan, 'A1', np.nan],  # Mostly missing
        'Embarked': [np.nan, 'C', np.nan, 'S', np.nan]  # Mostly missing
    })
    return data


@pytest.fixture
def minimal_valid_data():
    """Minimal valid dataset that satisfies basic requirements."""
    return pd.DataFrame({
        'PassengerId': [1, 2],
        'Pclass': [1, 3],
        'Name': ['Test, Mr. A', 'Test, Mrs. B'],
        'Sex': ['male', 'female'],
        'Age': [30.0, 25.0],
        'SibSp': [0, 1],
        'Parch': [0, 0],
        'Ticket': ['T1', 'T2'],
        'Fare': [50.0, 20.0],
        'Cabin': ['A1', ''],
        'Embarked': ['S', 'C']
    })


@pytest.fixture
def config_variations():
    """Dictionary of configuration variations for testing different scenarios."""
    return {
        "minimal": {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True
        },

        "no_missing_handling": {
            "handle_missing": False,
            "encode_categorical": True,
            "scale_features": True
        },

        "no_encoding": {
            "handle_missing": True,
            "encode_categorical": False,
            "scale_features": True
        },

        "no_scaling": {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": False
        },

        "everything_disabled": {
            "handle_missing": False,
            "encode_categorical": False,
            "scale_features": False
        },

        "complex_feature_engineering": {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True,
            "log_transform_fare": True,
            "age_bins": 3,
            "feature_engineering": {
                "pre_impute": ["FamilySizeTransform", "DeckTransform", "TicketGroupTransform"],
                "post_impute": ["FareTransform", "AgeBinningTransform"]
            }
        },

        "advanced_imputation": {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True,
            "imputation": {
                "order": ["Fare", "Age"],
                "default": {"add_missing_indicators": True},
                "per_column": {
                    "Age": {
                        "method": "model",
                        "estimator": "random_forest",
                        "features": ["Pclass", "Sex", "SibSp", "Parch", "Fare"],
                        "random_state": 42
                    }
                }
            }
        },

        "mixed_encoding": {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True,
            "encoding": {
                "default": {"method": "onehot"},
                "per_column": {
                    "Sex": {"method": "target"},
                    "Embarked": {"method": "onehot"},
                    "Pclass": {"method": "catboost", "a": 1.0}
                }
            }
        }
    }


class PreprocessingTestCase:
    """Base class for preprocessing test cases with common utilities."""

    @staticmethod
    def assert_no_missing_values(df, exclude_columns=None):
        """Assert that dataframe has no missing values except in excluded columns."""
        exclude_columns = exclude_columns or []
        for col in df.columns:
            if col not in exclude_columns:
                assert not df[col].isnull().any(), f"Column {col} has missing values"

    @staticmethod
    def assert_proper_scaling(df, numeric_columns=None, tolerance=0.2):
        """Assert that numeric columns are properly scaled (mean~0, std~1)."""
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if df[col].std() > 1e-6:  # Avoid zero-variance columns
                assert abs(df[col].mean()) < tolerance, f"Column {col} mean not close to 0: {df[col].mean()}"
                assert abs(df[col].std() - 1.0) < tolerance, f"Column {col} std not close to 1: {df[col].std()}"

    @staticmethod
    def assert_binary_columns_valid(df, binary_columns):
        """Assert that binary columns contain only 0s and 1s."""
        for col in binary_columns:
            if col in df.columns:
                unique_vals = set(df[col].dropna().unique())
                assert unique_vals.issubset({0, 1}), f"Binary column {col} has invalid values: {unique_vals}"

    @staticmethod
    def assert_feature_engineering_applied(df, expected_features):
        """Assert that expected engineered features are present."""
        for feature in expected_features:
            assert feature in df.columns, f"Expected feature {feature} not found in columns"

    @staticmethod
    def assert_categorical_encoded(df, original_categorical_cols, skip_columns=None):
        """Assert that categorical columns were properly encoded."""
        skip_columns = skip_columns or []
        for col in original_categorical_cols:
            if col not in skip_columns:
                assert col not in df.columns, f"Original categorical column {col} still present after encoding"


@pytest.fixture
def preprocessing_test_case():
    """Provide the PreprocessingTestCase utilities."""
    return PreprocessingTestCase
