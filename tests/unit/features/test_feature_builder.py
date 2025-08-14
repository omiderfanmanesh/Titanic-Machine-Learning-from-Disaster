"""
Unit tests for TitanicFeatureBuilder - the main preprocessing orchestrator.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import yaml
from pathlib import Path

from src.features.builders.titanic import TitanicFeatureBuilder
from src.core import DataConfig


class TestTitanicFeatureBuilder:
    """Test the main feature builder orchestrator."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True,
            "log_transform_fare": True,
            "age_bins": 5,
            "rare_title_threshold": 10,
            "numeric_columns": ["Age", "SibSp", "Parch", "Fare"],
            "categorical_columns": ["Sex", "Embarked", "Pclass", "Deck", "Title"],
            "skip_encoding_columns": ["PassengerId"],
            "feature_engineering": {
                "pre_impute": ["FamilySizeTransform", "DeckTransform", "TicketGroupTransform"],
                "post_impute": ["FareTransform"]
            },
            "encoding": {
                "default": {"method": "onehot", "handle_missing": "value", "handle_unknown": "ignore"},
                "per_column": {
                    "Title": {"method": "catboost", "a": 1.0},
                    "Deck": {"method": "onehot"}
                }
            },
            "imputation": {
                "order": ["Fare", "Age"],
                "exclude": ["PassengerId", "Name", "Ticket", "Title", "Deck"],
                "default": {
                    "numeric": "median",
                    "categorical": "constant",
                    "fill_value": "Unknown"
                },
                "per_column": {
                    "Age": {
                        "method": "model",
                        "estimator": "random_forest",
                        "features": ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"],
                        "clip_min": 0,
                        "clip_max": 80
                    },
                    "Fare": {"method": "mean", "clip_min": 0},
                    "Embarked": {"method": "constant", "fill_value": "S"}
                }
            }
        }

    @pytest.fixture
    def sample_data(self):
        """Sample Titanic data for testing."""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath', 'Allen, Mr. William Henry'],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, np.nan],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
            'Fare': [7.2500, 71.2833, 7.9250, 53.1000, np.nan],
            'Cabin': ['', 'C85', '', 'C123', ''],
            'Embarked': ['S', 'C', 'S', 'S', 'S']
        })

    @pytest.fixture
    def sample_target(self):
        """Sample target variable."""
        return pd.Series([0, 1, 1, 1, 0], name='Survived')

    def test_init_with_config(self, sample_config):
        """Test initialization with configuration."""
        builder = TitanicFeatureBuilder(sample_config)

        assert builder.config == sample_config
        assert builder.handle_missing is True
        assert not builder._is_fitted
        assert builder._fitted_columns is None

    def test_init_without_config(self):
        """Test initialization without configuration."""
        builder = TitanicFeatureBuilder()

        assert builder.config == {}
        assert builder.handle_missing is True  # default
        assert not builder._is_fitted

    def test_fit_transform_workflow(self, sample_config, sample_data, sample_target):
        """Test the complete fit-transform workflow."""
        builder = TitanicFeatureBuilder(sample_config)

        # Test fit
        builder.fit(sample_data, sample_target)
        assert builder._is_fitted
        assert builder._fitted_columns is not None
        assert len(builder._fitted_columns) > 0

        # Test transform
        transformed = builder.transform(sample_data)
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)
        assert transformed.columns.tolist() == builder._fitted_columns

    def test_fit_transform_shortcut(self, sample_config, sample_data, sample_target):
        """Test fit_transform method."""
        builder = TitanicFeatureBuilder(sample_config)

        transformed = builder.fit_transform(sample_data, sample_target)
        assert isinstance(transformed, pd.DataFrame)
        assert builder._is_fitted
        assert len(transformed) == len(sample_data)

    def test_transform_before_fit_raises_error(self, sample_config, sample_data):
        """Test that transform raises error when called before fit."""
        builder = TitanicFeatureBuilder(sample_config)

        with pytest.raises(ValueError, match="Feature builder must be fitted before transform"):
            builder.transform(sample_data)

    def test_get_feature_names(self, sample_config, sample_data, sample_target):
        """Test getting feature names after fitting."""
        builder = TitanicFeatureBuilder(sample_config)

        # Before fitting
        feature_names = builder.get_feature_names()
        assert isinstance(feature_names, list)

        # After fitting
        builder.fit(sample_data, sample_target)
        feature_names = builder.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert feature_names == builder._fitted_columns

    def test_column_alignment_in_transform(self, sample_config, sample_data, sample_target):
        """Test that transform aligns columns correctly."""
        builder = TitanicFeatureBuilder(sample_config)
        builder.fit(sample_data, sample_target)

        # Create test data with missing columns
        test_data = sample_data.copy()
        original_transformed = builder.transform(test_data)

        # Should have same columns as fitted schema
        assert original_transformed.columns.tolist() == builder._fitted_columns

    def test_disabled_missing_handling(self, sample_data, sample_target):
        """Test builder with missing handling disabled."""
        config = {"handle_missing": False}
        builder = TitanicFeatureBuilder(config)

        assert builder.imputer is None

        # Should still work but without imputation
        transformed = builder.fit_transform(sample_data, sample_target)
        assert isinstance(transformed, pd.DataFrame)

    def test_disabled_encoding(self, sample_data, sample_target):
        """Test builder with encoding disabled."""
        config = {"encode_categorical": False}
        builder = TitanicFeatureBuilder(config)

        transformed = builder.fit_transform(sample_data, sample_target)
        assert isinstance(transformed, pd.DataFrame)

    def test_disabled_scaling(self, sample_data, sample_target):
        """Test builder with scaling disabled."""
        config = {"scale_features": False}
        builder = TitanicFeatureBuilder(config)

        transformed = builder.fit_transform(sample_data, sample_target)
        assert isinstance(transformed, pd.DataFrame)

    @patch('src.features.builders.titanic.build_pipeline_pre')
    @patch('src.features.builders.titanic.build_pipeline_post')
    def test_pipeline_building(self, mock_build_post, mock_build_pre, sample_config, sample_data, sample_target):
        """Test that pipelines are built correctly from config."""
        mock_pipeline = MagicMock()
        mock_pipeline.fit.return_value = mock_pipeline
        mock_pipeline.transform.return_value = sample_data

        mock_build_pre.return_value = mock_pipeline
        mock_build_post.return_value = mock_pipeline

        builder = TitanicFeatureBuilder(sample_config)
        builder.fit(sample_data, sample_target)

        # Verify pipelines were built with correct config
        mock_build_pre.assert_called_once_with(sample_config)
        mock_build_post.assert_called_once_with(sample_config)

    def test_logging_during_fit(self, sample_config, sample_data, sample_target, caplog):
        """Test that appropriate logging occurs during fit."""
        builder = TitanicFeatureBuilder(sample_config)

        with caplog.at_level("INFO"):
            builder.fit(sample_data, sample_target)

        assert "Fitting feature builder" in caplog.text
        assert "Feature builder fitted successfully" in caplog.text

    def test_logging_during_transform(self, sample_config, sample_data, sample_target, caplog):
        """Test that appropriate logging occurs during transform."""
        builder = TitanicFeatureBuilder(sample_config)
        builder.fit(sample_data, sample_target)

        with caplog.at_level("INFO"):
            builder.transform(sample_data)

        assert "Transforming data with" in caplog.text
        assert "Transformed data shape:" in caplog.text


class TestFeatureBuilderWithYAMLConfig:
    """Test feature builder with real YAML configuration."""

    @pytest.fixture
    def yaml_config_path(self, tmp_path):
        """Create a temporary YAML config file."""
        config_content = """
        # Data configuration
        handle_missing: true
        encode_categorical: true
        scale_features: true
        log_transform_fare: true
        age_bins: 5
        rare_title_threshold: 10
        
        numeric_columns: [Age, SibSp, Parch, Fare]
        categorical_columns: [Sex, Embarked, Pclass, Deck, Title]
        skip_encoding_columns: [PassengerId]
        
        feature_engineering:
          pre_impute:
            - FamilySizeTransform
            - DeckTransform
            - TicketGroupTransform
          post_impute:
            - FareTransform
        
        encoding:
          default:
            method: onehot
            handle_missing: value
            handle_unknown: ignore
          per_column:
            Title:
              method: catboost
              a: 1.0
            Deck:
              method: onehot
        
        imputation:
          order: [Fare, Age]
          exclude: [PassengerId, Name, Ticket, Title, Deck]
          default:
            numeric: median
            categorical: constant
            fill_value: "Unknown"
          per_column:
            Age:
              method: model
              estimator: random_forest
              features: [Pclass, Sex, SibSp, Parch, Fare, Embarked]
              clip_min: 0
              clip_max: 80
            Fare:
              method: mean
              clip_min: 0
            Embarked:
              method: constant
              fill_value: "S"
        """

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)
        return config_path

    @pytest.fixture
    def sample_data(self):
        """Sample Titanic data for testing."""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath', 'Allen, Mr. William Henry'],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, np.nan],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
            'Fare': [7.2500, 71.2833, 7.9250, 53.1000, np.nan],
            'Cabin': ['', 'C85', '', 'C123', ''],
            'Embarked': ['S', 'C', 'S', 'S', 'S']
        })

    def test_yaml_config_loading(self, yaml_config_path, sample_data):
        """Test loading configuration from YAML file."""
        # Load config from YAML
        with open(yaml_config_path, 'r') as f:
            config = yaml.safe_load(f)

        builder = TitanicFeatureBuilder(config)

        # Verify config was loaded correctly
        assert builder.config['handle_missing'] is True
        assert builder.config['log_transform_fare'] is True
        assert builder.config['age_bins'] == 5
        assert 'FamilySizeTransform' in builder.config['feature_engineering']['pre_impute']

    def test_end_to_end_with_yaml_config(self, yaml_config_path, sample_data):
        """Test complete preprocessing pipeline with YAML config."""
        # Load config from YAML
        with open(yaml_config_path, 'r') as f:
            config = yaml.safe_load(f)

        builder = TitanicFeatureBuilder(config)
        target = pd.Series([0, 1, 1, 1, 0], name='Survived')

        # Test complete workflow
        transformed = builder.fit_transform(sample_data, target)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)
        assert transformed.isnull().sum().sum() == 0  # No missing values after preprocessing

        # Test consistency across multiple transforms
        transformed2 = builder.transform(sample_data)
        pd.testing.assert_frame_equal(transformed, transformed2)
