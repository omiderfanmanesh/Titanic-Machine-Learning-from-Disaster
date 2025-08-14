"""
Unit tests for encoding system - categorical variable encoding.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.features.encoding.orchestrator import EncodingOrchestrator
from src.features.encoding.factory import build_encoder


class TestEncodingOrchestrator:
    """Test the main encoding orchestrator."""

    @pytest.fixture
    def sample_config(self):
        """Sample encoding configuration."""
        return {
            "encode_categorical": True,
            "skip_encoding_columns": ["PassengerId"],
            "encoding": {
                "default": {
                    "method": "onehot",
                    "handle_missing": "value",
                    "handle_unknown": "ignore"
                },
                "per_column": {
                    "Title": {
                        "method": "catboost"
                    },
                    "Deck": {
                        "method": "onehot"
                    },
                    "Embarked": {
                        "method": "onehot"
                    }
                }
            }
        }

    @pytest.fixture
    def sample_data(self):
        """Sample data with categorical variables."""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Sex': ['male', 'female', 'female', 'male', 'female'],
            'Embarked': ['S', 'C', 'S', 'Q', 'S'],
            'Deck': ['A', 'B', 'C', 'A', 'U'],
            'Title': ['Mr', 'Mrs', 'Miss', 'Mr', 'Mrs'],
            'Age': [22.0, 38.0, 26.0, 35.0, 27.0],
            'Fare': [7.25, 71.28, 7.925, 53.1, 8.05]
        })

    @pytest.fixture
    def sample_target(self):
        """Sample target variable."""
        return pd.Series([0, 1, 1, 0, 1], name='Survived')

    def test_orchestrator_initialization(self, sample_config):
        """Test orchestrator initializes correctly."""
        orchestrator = EncodingOrchestrator(sample_config)

        assert orchestrator.config == sample_config
        assert orchestrator._encoders == {}

    def test_encoding_config_parsing(self, sample_config):
        """Test parsing of encoding configuration."""
        orchestrator = EncodingOrchestrator(sample_config)

        config = orchestrator._encoding_cfg()
        assert "default" in config
        assert "per_column" in config
        assert config["default"]["method"] == "onehot"
        assert config["per_column"]["Title"]["method"] == "catboost"

    def test_column_config_default(self, sample_config):
        """Test getting default column configuration."""
        orchestrator = EncodingOrchestrator(sample_config)

        config = orchestrator._col_cfg("Sex")  # Not in per_column
        assert config["method"] == "onehot"
        assert config["handle_missing"] == "value"

    def test_column_config_per_column(self, sample_config):
        """Test getting per-column configuration."""
        orchestrator = EncodingOrchestrator(sample_config)

        config = orchestrator._col_cfg("Title")
        assert config["method"] == "catboost"
        # Should inherit defaults for unspecified params
        assert config["handle_missing"] == "value"

    def test_fit_with_categorical_columns(self, sample_config, sample_data, sample_target):
        """Test fitting with specified categorical columns."""
        orchestrator = EncodingOrchestrator(sample_config)
        categorical_cols = ["Sex", "Embarked", "Deck", "Title"]

        orchestrator.fit(sample_data, sample_target, categorical_cols)

        # Should have encoders for non-skipped categorical columns
        expected_encoders = ["Sex", "Embarked", "Deck", "Title"]  # PassengerId skipped
        for col in expected_encoders:
            assert col in orchestrator._encoders

    def test_fit_auto_detect_categorical(self, sample_config, sample_data, sample_target):
        """Test auto-detection of categorical columns."""
        orchestrator = EncodingOrchestrator(sample_config)

        orchestrator.fit(sample_data, sample_target)

        # Should detect object/category columns automatically
        detected_categorical = list(orchestrator._encoders.keys())
        assert "Sex" in detected_categorical
        assert "Embarked" in detected_categorical
        # PassengerId should be skipped
        assert "PassengerId" not in detected_categorical

    def test_skip_encoding_columns(self, sample_config, sample_data, sample_target):
        """Test that skip_encoding_columns are excluded."""
        orchestrator = EncodingOrchestrator(sample_config)
        categorical_cols = ["PassengerId", "Sex", "Embarked"]

        orchestrator.fit(sample_data, sample_target, categorical_cols)

        assert "PassengerId" not in orchestrator._encoders
        assert "Sex" in orchestrator._encoders
        assert "Embarked" in orchestrator._encoders

    def test_encoding_disabled(self, sample_data, sample_target):
        """Test when encoding is disabled."""
        config = {"encode_categorical": False}
        orchestrator = EncodingOrchestrator(config)

        orchestrator.fit(sample_data, sample_target)

        assert orchestrator._encoders == {}

    def test_transform_with_encoders(self, sample_config, sample_data, sample_target):
        """Test transform when encoders are fitted."""
        orchestrator = EncodingOrchestrator(sample_config)
        orchestrator.fit(sample_data, sample_target)

        result = orchestrator.transform(sample_data)

        # Original categorical columns should be removed
        assert "Sex" not in result.columns
        assert "Embarked" not in result.columns

        # Should have encoded columns
        assert len(result.columns) > len(sample_data.columns) - 4  # Some columns were encoded

    def test_transform_without_encoders(self, sample_data):
        """Test transform when no encoders are fitted."""
        config = {"encode_categorical": False}
        orchestrator = EncodingOrchestrator(config)

        result = orchestrator.transform(sample_data)

        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, sample_data)

    def test_missing_column_in_transform(self, sample_config, sample_data, sample_target):
        """Test transform when expected column is missing."""
        orchestrator = EncodingOrchestrator(sample_config)
        orchestrator.fit(sample_data, sample_target)

        # Remove a column that was encoded
        test_data = sample_data.drop(columns=['Sex'])
        result = orchestrator.transform(test_data)

        # Should handle missing column gracefully
        assert isinstance(result, pd.DataFrame)

    def test_feature_names_generation(self, sample_config, sample_data, sample_target):
        """Test generation of feature names."""
        orchestrator = EncodingOrchestrator(sample_config)
        orchestrator.fit(sample_data, sample_target)

        feature_names = orchestrator.feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    @patch('src.features.encoding.orchestrator.build_encoder')
    def test_encoder_building(self, mock_build_encoder, sample_config, sample_data, sample_target):
        """Test that encoders are built correctly."""
        mock_encoder = MagicMock()
        mock_encoder.fit.return_value = mock_encoder
        mock_build_encoder.return_value = mock_encoder

        orchestrator = EncodingOrchestrator(sample_config)
        categorical_cols = ["Sex", "Title"]

        orchestrator.fit(sample_data, sample_target, categorical_cols)

        # Should build encoders for each categorical column
        assert mock_build_encoder.call_count == 2


class TestEncoderFactory:
    """Test the encoder factory function."""

    def test_build_onehot_encoder(self):
        """Test building one-hot encoder."""
        config = {
            "method": "onehot",
            "handle_missing": "value",
            "handle_unknown": "ignore"
        }

        encoder = build_encoder("test_col", config)

        # Should return an encoder instance
        assert encoder is not None
        assert hasattr(encoder, 'fit')
        assert hasattr(encoder, 'transform')

    def test_build_catboost_encoder(self):
        """Test building CatBoost encoder."""
        config = {
            "method": "catboost",
            "a": 1.0
        }

        encoder = build_encoder("test_col", config)

        assert encoder is not None
        assert hasattr(encoder, 'fit')
        assert hasattr(encoder, 'transform')

    def test_build_target_encoder(self):
        """Test building target encoder."""
        config = {
            "method": "target",
            "smoothing": 1.0
        }

        encoder = build_encoder("test_col", config)

        assert encoder is not None
        assert hasattr(encoder, 'fit')
        assert hasattr(encoder, 'transform')

    def test_unknown_encoder_method(self):
        """Test handling of unknown encoder method."""
        config = {
            "method": "unknown_method"
        }

        with pytest.raises(ValueError, match="Unknown encoding method"):
            build_encoder("test_col", config)


class TestOneHotEncoding:
    """Test one-hot encoding functionality."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'],
            'other_col': [1, 2, 3, 4, 5, 6]
        })

    def test_onehot_encoding_basic(self, sample_data):
        """Test basic one-hot encoding."""
        config = {
            "method": "onehot",
            "handle_missing": "value",
            "handle_unknown": "ignore"
        }

        encoder = build_encoder("category", config)
        encoder.fit(sample_data)
        result = encoder.transform(sample_data)

        # Should create binary columns for each category
        expected_columns = ['category_A', 'category_B', 'category_C']
        for col in expected_columns:
            assert col in result.columns

        # Values should be binary
        for col in expected_columns:
            assert set(result[col].unique()).issubset({0, 1})

    def test_onehot_with_missing_values(self):
        """Test one-hot encoding with missing values."""
        data = pd.DataFrame({
            'category': ['A', 'B', np.nan, 'A', np.nan],
            'other_col': [1, 2, 3, 4, 5]
        })

        config = {
            "method": "onehot",
            "handle_missing": "value"
        }

        encoder = build_encoder("category", config)
        encoder.fit(data)
        result = encoder.transform(data)

        # Should handle missing values (often creates missing indicator)
        assert not result.isnull().any().any()

    def test_onehot_with_unknown_categories(self, sample_data):
        """Test one-hot encoding with unknown categories."""
        config = {
            "method": "onehot",
            "handle_unknown": "ignore"
        }

        encoder = build_encoder("category", config)
        encoder.fit(sample_data)

        # Transform data with unknown category
        test_data = pd.DataFrame({
            'category': ['A', 'D', 'B'],  # 'D' is unknown
            'other_col': [1, 2, 3]
        })

        result = encoder.transform(test_data)

        # Should handle unknown category gracefully
        assert isinstance(result, pd.DataFrame)


class TestTargetEncoding:
    """Test target encoding functionality."""

    @pytest.fixture
    def sample_data_with_target(self):
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'other_col': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        target = pd.Series([1, 0, 1, 1, 0, 0, 1, 1])
        return data, target

    def test_target_encoding_basic(self, sample_data_with_target):
        """Test basic target encoding."""
        data, target = sample_data_with_target

        config = {
            "method": "catboost",
            "a": 1.0
        }

        encoder = build_encoder("category", config)
        encoder.fit(data, target)
        result = encoder.transform(data)

        # Should create a single numeric column
        encoded_col = f"category_catboost"
        assert encoded_col in result.columns
        assert result[encoded_col].dtype in ['float64', 'float32']

    def test_target_encoding_smoothing(self, sample_data_with_target):
        """Test target encoding with smoothing parameter."""
        data, target = sample_data_with_target

        config = {
            "method": "catboost",
            "a": 10.0  # High smoothing
        }

        encoder = build_encoder("category", config)
        encoder.fit(data, target)
        result = encoder.transform(data)

        # With high smoothing, values should be closer to global mean
        global_mean = target.mean()
        encoded_values = result[f"category_catboost"].unique()

        # All encoded values should be closer to global mean with high smoothing
        for value in encoded_values:
            assert abs(value - global_mean) < abs(1.0 - global_mean)  # Reasonable bound


class TestEncodingIntegration:
    """Test encoding system integration."""

    @pytest.fixture
    def complex_config(self):
        """Complex encoding configuration."""
        return {
            "encode_categorical": True,
            "skip_encoding_columns": ["PassengerId", "Name"],
            "encoding": {
                "default": {
                    "method": "onehot",
                    "handle_missing": "value",
                    "handle_unknown": "ignore"
                },
                "per_column": {
                    "Title": {
                        "method": "catboost",
                        "a": 1.0
                    },
                    "Sex": {
                        "method": "target",
                        "smoothing": 1.0
                    },
                    "Embarked": {
                        "method": "onehot"
                    },
                    "Deck": {
                        "method": "onehot"
                    }
                }
            }
        }

    @pytest.fixture
    def complex_data(self):
        """Complex dataset for integration testing."""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5, 6],
            'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson', 'David Lee'],
            'Sex': ['male', 'female', 'female', 'male', 'female', 'male'],
            'Age': [22, 38, 26, 35, 27, 45],
            'Embarked': ['S', 'C', 'S', 'Q', 'S', 'C'],
            'Deck': ['A', 'B', 'C', 'A', 'U', 'B'],
            'Title': ['Mr', 'Mrs', 'Miss', 'Mr', 'Mrs', 'Mr'],
            'Fare': [7.25, 71.28, 7.925, 53.1, 8.05, 26.55]
        })

    @pytest.fixture
    def complex_target(self):
        """Target variable for complex data."""
        return pd.Series([0, 1, 1, 0, 1, 0], name='Survived')

    def test_mixed_encoding_strategies(self, complex_config, complex_data, complex_target):
        """Test multiple encoding strategies working together."""
        orchestrator = EncodingOrchestrator(complex_config)

        orchestrator.fit(complex_data, complex_target)
        result = orchestrator.transform(complex_data)

        # Check that different encoding strategies were applied
        # One-hot encoded columns should exist
        assert any('Embarked_' in col for col in result.columns)
        assert any('Deck_' in col for col in result.columns)

        # Target encoded columns should exist
        assert any('Title_catboost' in col for col in result.columns)
        assert any('Sex_target' in col for col in result.columns)

        # Original categorical columns should be removed
        assert 'Sex' not in result.columns
        assert 'Embarked' not in result.columns
        assert 'Title' not in result.columns
        assert 'Deck' not in result.columns

        # Skipped columns should remain
        assert 'PassengerId' in result.columns
        assert 'Name' in result.columns

    def test_encoding_consistency(self, complex_config, complex_data, complex_target):
        """Test that encoding is consistent across multiple transforms."""
        orchestrator = EncodingOrchestrator(complex_config)

        orchestrator.fit(complex_data, complex_target)
        result1 = orchestrator.transform(complex_data)
        result2 = orchestrator.transform(complex_data)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_new_data_encoding(self, complex_config, complex_data, complex_target):
        """Test encoding of new data with same schema."""
        orchestrator = EncodingOrchestrator(complex_config)

        orchestrator.fit(complex_data, complex_target)

        # Create new data with same schema
        new_data = complex_data.copy()
        new_data.loc[0, 'Sex'] = 'female'  # Change some values
        new_data.loc[1, 'Embarked'] = 'Q'

        result = orchestrator.transform(new_data)

        # Should have same column structure as training
        original_result = orchestrator.transform(complex_data)
        assert result.columns.tolist() == original_result.columns.tolist()

    def test_unknown_categories_handling(self, complex_config, complex_data, complex_target):
        """Test handling of unknown categories in new data."""
        orchestrator = EncodingOrchestrator(complex_config)

        orchestrator.fit(complex_data, complex_target)

        # Create data with unknown categories
        new_data = complex_data.copy()
        new_data.loc[0, 'Embarked'] = 'X'  # Unknown port
        new_data.loc[1, 'Deck'] = 'Z'      # Unknown deck

        result = orchestrator.transform(new_data)

        # Should handle unknown categories gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(new_data)

    def test_feature_count_after_encoding(self, complex_config, complex_data, complex_target):
        """Test that feature count changes appropriately after encoding."""
        orchestrator = EncodingOrchestrator(complex_config)

        original_categorical_count = len(['Sex', 'Embarked', 'Deck', 'Title'])

        orchestrator.fit(complex_data, complex_target)
        result = orchestrator.transform(complex_data)

        # Should have more columns due to one-hot encoding
        # (some categorical columns become multiple binary columns)
        assert len(result.columns) >= len(complex_data.columns)

    def test_no_data_leakage_in_encoding(self, complex_config, complex_data, complex_target):
        """Test that target encoding doesn't cause data leakage."""
        orchestrator = EncodingOrchestrator(complex_config)

        # Split data into train/test
        train_data = complex_data.iloc[:4]
        test_data = complex_data.iloc[4:]
        train_target = complex_target.iloc[:4]

        # Fit on train data only
        orchestrator.fit(train_data, train_target)

        # Transform test data
        test_result = orchestrator.transform(test_data)

        # Should work without seeing test target
        assert isinstance(test_result, pd.DataFrame)
        assert len(test_result) == len(test_data)

        # Target encoded values should be based only on training data
        if 'Title_catboost' in test_result.columns:
            # Values should be reasonable (not extreme)
            assert test_result['Title_catboost'].between(0, 1).all()
