"""
Integration tests for the complete features preprocessing pipeline.
Tests the full workflow with YAML configuration.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.features import create_feature_builder, TitanicFeatureBuilder
from src.core import DataConfig


class TestFullPreprocessingPipeline:
    """Test complete preprocessing pipeline end-to-end."""

    @pytest.fixture
    def real_yaml_config(self):
        """Load the actual data.yaml configuration."""
        config_path = Path("configs/data.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Fallback configuration for testing
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
                        "Deck": {"method": "onehot"},
                        "Embarked": {"method": "onehot"}
                    }
                },
                "imputation": {
                    "order": ["Fare", "Age"],
                    "exclude": ["PassengerId", "Name", "Ticket", "Title", "Deck"],
                    "default": {"numeric": "median", "categorical": "constant", "fill_value": "Unknown"},
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
    def titanic_sample_data(self):
        """Realistic Titanic dataset sample."""
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
    def titanic_target(self):
        """Sample target variable."""
        return pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='Survived')

    def test_pipeline_with_real_config(self, real_yaml_config, titanic_sample_data, titanic_target):
        """Test pipeline with actual YAML configuration."""
        builder = TitanicFeatureBuilder(real_yaml_config)

        # Test complete workflow
        transformed = builder.fit_transform(titanic_sample_data, titanic_target)

        # Basic validation
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(titanic_sample_data)
        assert transformed.isnull().sum().sum() == 0  # No missing values

        # Should have more columns due to feature engineering and encoding
        assert len(transformed.columns) >= len(titanic_sample_data.columns)

    def test_feature_engineering_stages(self, real_yaml_config, titanic_sample_data, titanic_target):
        """Test that feature engineering stages are applied correctly."""
        builder = TitanicFeatureBuilder(real_yaml_config)

        # Fit the builder to examine intermediate steps
        builder.fit(titanic_sample_data, titanic_target)

        # Test pre-imputation features
        pre_pipeline = builder.pipeline_pre
        pre_transformed = pre_pipeline.transform(titanic_sample_data)

        # Should have new features from pre-imputation transforms
        expected_pre_features = ['FamilySize', 'IsAlone', 'Deck', 'TicketGroupSize']
        for feature in expected_pre_features:
            assert feature in pre_transformed.columns

        # Test post-imputation features (after imputation)
        if builder.imputer:
            imputed = builder.imputer.transform(pre_transformed)
            post_pipeline = builder.pipeline_post
            post_transformed = post_pipeline.transform(imputed)

            # Should have fare processing
            if real_yaml_config.get('log_transform_fare'):
                assert 'Fare_log' in post_transformed.columns

    def test_imputation_workflow(self, real_yaml_config, titanic_sample_data, titanic_target):
        """Test that imputation follows configured order and strategies."""
        builder = TitanicFeatureBuilder(real_yaml_config)
        builder.fit(titanic_sample_data, titanic_target)

        transformed = builder.transform(titanic_sample_data)

        # Age and Fare should be imputed (originally had missing values)
        original_age_missing = titanic_sample_data['Age'].isnull().sum()
        original_fare_missing = titanic_sample_data['Fare'].isnull().sum()

        if original_age_missing > 0 or original_fare_missing > 0:
            # After full pipeline, no missing values should remain in final output
            assert transformed.isnull().sum().sum() == 0

    def test_encoding_strategies(self, real_yaml_config, titanic_sample_data, titanic_target):
        """Test that different encoding strategies are applied correctly."""
        builder = TitanicFeatureBuilder(real_yaml_config)
        transformed = builder.fit_transform(titanic_sample_data, titanic_target)

        # Original categorical columns should be removed
        categorical_cols = real_yaml_config.get('categorical_columns', [])
        skip_cols = real_yaml_config.get('skip_encoding_columns', [])

        for col in categorical_cols:
            if col not in skip_cols and col in titanic_sample_data.columns:
                assert col not in transformed.columns, f"Original categorical column {col} should be removed"

        # Should have encoded versions
        # One-hot encoded columns should exist
        assert any('Embarked_' in col for col in transformed.columns)
        assert any('Deck_' in col for col in transformed.columns)

        # Target encoded columns should exist
        if 'Title' in categorical_cols:
            assert any('Title_catboost' in col for col in transformed.columns)

    def test_scaling_applied(self, real_yaml_config, titanic_sample_data, titanic_target):
        """Test that scaling is applied to final features."""
        if not real_yaml_config.get('scale_features', True):
            pytest.skip("Scaling disabled in config")

        builder = TitanicFeatureBuilder(real_yaml_config)
        transformed = builder.fit_transform(titanic_sample_data, titanic_target)

        # Numeric features should be approximately standardized
        numeric_features = [col for col in transformed.columns if transformed[col].dtype in ['float64', 'float32']]

        for col in numeric_features:
            if transformed[col].std() > 1e-6:  # Avoid zero-variance features
                # Should be approximately standardized (mean ~0, std ~1)
                assert abs(transformed[col].mean()) < 1.0  # Reasonable bound
                assert 0.5 < transformed[col].std() < 2.0   # Reasonable bound

    def test_train_test_consistency(self, real_yaml_config, titanic_sample_data, titanic_target):
        """Test consistency between train and test preprocessing."""
        # Split data
        train_data = titanic_sample_data.iloc[:7]
        test_data = titanic_sample_data.iloc[7:]
        train_target = titanic_target.iloc[:7]

        builder = TitanicFeatureBuilder(real_yaml_config)

        # Fit on train data
        builder.fit(train_data, train_target)
        train_transformed = builder.transform(train_data)
        test_transformed = builder.transform(test_data)

        # Should have same column structure
        assert train_transformed.columns.tolist() == test_transformed.columns.tolist()

        # Both should have no missing values
        assert train_transformed.isnull().sum().sum() == 0
        assert test_transformed.isnull().sum().sum() == 0

    def test_feature_names_consistency(self, real_yaml_config, titanic_sample_data, titanic_target):
        """Test that feature names are consistent and retrievable."""
        builder = TitanicFeatureBuilder(real_yaml_config)
        transformed = builder.fit_transform(titanic_sample_data, titanic_target)

        feature_names = builder.get_feature_names()

        # Feature names should match transformed columns
        assert feature_names == transformed.columns.tolist()
        assert len(feature_names) > 0

    def test_pipeline_with_minimal_data(self, real_yaml_config):
        """Test pipeline with minimal data that might cause issues."""
        minimal_data = pd.DataFrame({
            'PassengerId': [1, 2],
            'Pclass': [1, 3],
            'Name': ['Test, Mr. A', 'Test, Mrs. B'],
            'Sex': ['male', 'female'],
            'Age': [30.0, np.nan],
            'SibSp': [0, 1],
            'Parch': [0, 0],
            'Ticket': ['T1', 'T2'],
            'Fare': [50.0, np.nan],
            'Cabin': ['A1', ''],
            'Embarked': ['S', 'C']
        })
        target = pd.Series([1, 0])

        builder = TitanicFeatureBuilder(real_yaml_config)

        # Should handle minimal data gracefully
        transformed = builder.fit_transform(minimal_data, target)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == 2
        assert transformed.isnull().sum().sum() == 0

    def test_new_data_with_unknown_categories(self, real_yaml_config, titanic_sample_data, titanic_target):
        """Test handling of new data with unknown categorical values."""
        builder = TitanicFeatureBuilder(real_yaml_config)
        builder.fit(titanic_sample_data, titanic_target)

        # Create test data with unknown categories
        new_data = titanic_sample_data.copy()
        new_data.loc[0, 'Embarked'] = 'X'  # Unknown port
        new_data.loc[1, 'Cabin'] = 'Z99'   # Unknown deck

        # Should handle unknown categories gracefully
        transformed = builder.transform(new_data)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(new_data)
        assert transformed.isnull().sum().sum() == 0


class TestDataConfigIntegration:
    """Test integration with DataConfig class."""

    @pytest.fixture
    def data_config(self):
        """Create a DataConfig instance."""
        config_dict = {
            "train_path": "data/raw/train.csv",
            "test_path": "data/raw/test.csv",
            "target_column": "Survived",
            "id_column": "PassengerId",
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True,
            "feature_engineering": {
                "pre_impute": ["FamilySizeTransform", "DeckTransform"],
                "post_impute": ["FareTransform"]
            }
        }
        return DataConfig(**config_dict)

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Pclass': [1, 2, 3],
            'Name': ['A, Mr. Test', 'B, Mrs. Test', 'C, Miss Test'],
            'Sex': ['male', 'female', 'female'],
            'Age': [30, 25, np.nan],
            'SibSp': [0, 1, 0],
            'Parch': [0, 0, 1],
            'Ticket': ['T1', 'T2', 'T3'],
            'Fare': [50, np.nan, 25],
            'Cabin': ['A1', '', 'B2'],
            'Embarked': ['S', 'C', 'S']
        })

    def test_create_feature_builder_function(self, data_config, sample_data):
        """Test the create_feature_builder factory function."""
        builder = create_feature_builder(data_config)

        assert isinstance(builder, TitanicFeatureBuilder)

        # Should work with the data
        target = pd.Series([1, 0, 1])
        transformed = builder.fit_transform(sample_data, target)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)

    def test_debug_mode_integration(self, data_config, sample_data):
        """Test debug mode functionality."""
        builder = create_feature_builder(data_config, debug=True)

        # Debug mode should be enabled in config
        assert builder.config.get("debug_mode") is True

        target = pd.Series([1, 0, 1])
        transformed = builder.fit_transform(sample_data, target)

        assert isinstance(transformed, pd.DataFrame)


class TestYAMLConfigValidation:
    """Test YAML configuration validation and error handling."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Sex': ['male', 'female', 'male'],
            'Age': [25, 30, 35],
            'Fare': [10, 20, 30]
        })

    def test_invalid_transform_name(self, sample_data):
        """Test handling of invalid transform names in config."""
        invalid_config = {
            "feature_engineering": {
                "pre_impute": ["InvalidTransform"]
            }
        }

        builder = TitanicFeatureBuilder(invalid_config)
        target = pd.Series([1, 0, 1])

        with pytest.raises(ValueError, match="Unknown transform"):
            builder.fit(sample_data, target)

    def test_missing_required_config_sections(self, sample_data):
        """Test handling of missing config sections."""
        minimal_config = {}

        builder = TitanicFeatureBuilder(minimal_config)
        target = pd.Series([1, 0, 1])

        # Should work with defaults
        transformed = builder.fit_transform(sample_data, target)
        assert isinstance(transformed, pd.DataFrame)

    def test_invalid_encoding_method(self, sample_data):
        """Test handling of invalid encoding methods."""
        invalid_config = {
            "encoding": {
                "default": {"method": "invalid_method"}
            }
        }

        builder = TitanicFeatureBuilder(invalid_config)
        target = pd.Series([1, 0, 1])

        with pytest.raises(ValueError, match="Unknown encoding method"):
            builder.fit(sample_data, target)

    def test_missing_imputation_features(self, sample_data):
        """Test handling when imputation features are missing from data."""
        config_with_missing_features = {
            "imputation": {
                "per_column": {
                    "Age": {
                        "method": "model",
                        "features": ["NonExistentColumn", "Sex"]
                    }
                }
            }
        }

        builder = TitanicFeatureBuilder(config_with_missing_features)
        target = pd.Series([1, 0, 1])

        # Should handle missing features gracefully
        transformed = builder.fit_transform(sample_data, target)
        assert isinstance(transformed, pd.DataFrame)


class TestPerformanceAndMemory:
    """Test performance and memory characteristics."""

    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create larger dataset
        n_samples = 1000
        large_data = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Pclass': np.random.choice([1, 2, 3], n_samples),
            'Name': [f'Person_{i}, Mr. Test' for i in range(n_samples)],
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.normal(30, 10, n_samples),
            'SibSp': np.random.poisson(0.5, n_samples),
            'Parch': np.random.poisson(0.2, n_samples),
            'Ticket': [f'T{i}' for i in range(n_samples)],
            'Fare': np.random.exponential(30, n_samples),
            'Cabin': [''] * n_samples,  # Mostly empty
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples)
        })

        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        large_data.loc[missing_indices, 'Age'] = np.nan

        target = pd.Series(np.random.choice([0, 1], n_samples))

        config = {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True,
            "feature_engineering": {
                "pre_impute": ["FamilySizeTransform", "DeckTransform"],
                "post_impute": ["FareTransform"]
            }
        }

        builder = TitanicFeatureBuilder(config)

        # Should handle large dataset
        transformed = builder.fit_transform(large_data, target)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == n_samples
        assert transformed.isnull().sum().sum() == 0

    def test_memory_efficiency(self):
        """Test that preprocessing doesn't cause excessive memory usage."""
        # This is a basic test - in practice you'd use memory profiling tools
        n_samples = 500
        data = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.normal(30, 10, n_samples),
            'Fare': np.random.exponential(30, n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples)
        })
        target = pd.Series(np.random.choice([0, 1], n_samples))

        config = {"handle_missing": True, "encode_categorical": True, "scale_features": True}
        builder = TitanicFeatureBuilder(config)

        # Multiple transforms shouldn't cause memory issues
        for _ in range(3):
            transformed = builder.fit_transform(data, target)
            assert isinstance(transformed, pd.DataFrame)

    def test_reproducibility(self):
        """Test that preprocessing is reproducible with same data and config."""
        np.random.seed(42)
        data = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Sex': ['male', 'female', 'male', 'female', 'male'],
            'Age': [25, np.nan, 35, 30, 28],
            'Fare': [10, 20, np.nan, 25, 15],
            'Embarked': ['S', 'C', 'S', 'Q', 'S']
        })
        target = pd.Series([1, 0, 1, 0, 1])

        config = {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True,
            "imputation": {
                "per_column": {
                    "Age": {"method": "model", "random_state": 42}
                }
            }
        }

        # Run twice with same config
        builder1 = TitanicFeatureBuilder(config)
        result1 = builder1.fit_transform(data, target)

        builder2 = TitanicFeatureBuilder(config)
        result2 = builder2.fit_transform(data, target)

        # Results should be identical (or very close for model-based imputation)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
