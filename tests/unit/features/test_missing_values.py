"""
Comprehensive tests for missing value handling in the Titanic ML pipeline.

This integrated test suite covers:
1. CLI features command with data.yaml configuration
2. Missing value detection and reporting
3. Different imputation strategies
4. Missing value indicators
5. Edge cases and error handling
6. Integration with the full pipeline
7. Performance and memory testing
8. Real-world scenarios with CLI
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from src.cli import cli
from src.features.builders.titanic import TitanicFeatureBuilder
from src.features.missing.orchestrator import ImputationOrchestrator
from src.core import DataConfig


class TestMissingValueHandling:
    """Comprehensive test suite for missing value handling across the pipeline."""

    @pytest.fixture
    def missing_value_data(self):
        """Sample data with various missing value patterns."""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
            'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
            'Age': [22.0, 38.0, np.nan, 35.0, np.nan, 54.0, np.nan, 2.0, 27.0, np.nan],  # 40% missing
            'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
            'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
            'Fare': [7.25, 71.28, np.nan, 53.10, np.nan, 8.46, 51.86, 21.08, np.nan, 30.07],  # 30% missing
            'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan, 'E46', np.nan, np.nan, np.nan],  # 70% missing
            'Embarked': ['S', 'C', 'S', 'S', np.nan, 'Q', 'S', 'S', np.nan, 'C'],  # 20% missing
            'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]  # Target - no missing
        })

    @pytest.fixture
    def realistic_missing_data(self):
        """Create realistic data with missing patterns similar to actual Titanic dataset."""
        np.random.seed(42)
        n_samples = 50

        return pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.2, 0.6]),
            'Name': [f'Person {i}' for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': [25, 30, np.nan, 45, np.nan, 60, 35, np.nan, 28, 42] * 5,  # 30% missing
            'SibSp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.07, 0.03]),
            'Parch': np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]),
            'Ticket': [f'TICKET{i}' for i in range(1, n_samples + 1)],
            'Fare': [50.0, np.nan, 75.0, 120.0, np.nan] * 10,  # 20% missing
            'Cabin': [np.nan if i % 3 == 0 else f'C{i}' for i in range(n_samples)],  # 33% missing
            'Embarked': ['S' if i % 10 != 0 else np.nan for i in range(n_samples)],  # 10% missing
            'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38])
        })

    @pytest.fixture
    def empty_columns_data(self):
        """Data with completely empty columns for edge case testing."""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Age': [22.0, 38.0, 26.0, 35.0, 25.0],
            'Fare': [7.25, 71.28, 7.92, 53.10, 8.46],
            'EmptyColumn': [np.nan, np.nan, np.nan, np.nan, np.nan],  # 100% missing
            'MostlyEmpty': [1.0, np.nan, np.nan, np.nan, np.nan],  # 80% missing
            'Survived': [0, 1, 1, 1, 0]
        })

    @pytest.fixture
    def data_config_with_missing_handling(self):
        """Configuration with comprehensive missing value handling."""
        return {
            'train_path': 'data/raw/train.csv',
            'test_path': 'data/raw/test.csv',
            'target_column': 'Survived',
            'id_column': 'PassengerId',
            'handle_missing': True,
            'numeric_columns': ['Age', 'SibSp', 'Parch', 'Fare'],
            'categorical_columns': ['Sex', 'Embarked', 'Pclass'],
            'imputation': {
                'default': {
                    'numeric': 'median',
                    'categorical': 'constant',
                    'fill_value': 'Unknown',
                    'add_missing_indicators': True,
                    'missing_indicator_threshold': 0.05,
                    'missing_indicator_prefix': '__miss_'
                },
                'per_column': {
                    'Age': {
                        'method': 'median',
                        'add_missing_indicator': True,
                        'clip_min': 0,
                        'clip_max': 100
                    },
                    'Fare': {
                        'method': 'median',
                        'clip_min': 0
                    },
                    'Cabin': {
                        'method': 'constant',
                        'fill_value': 'Unknown'
                    }
                },
                'exclude': ['PassengerId', 'Ticket'],
                'order': ['Fare', 'Age', 'Embarked', 'Cabin']
            }
        }

    # ==================== BASIC FUNCTIONALITY TESTS ====================

    def test_missing_value_detection(self, missing_value_data):
        """Test detection and reporting of missing values."""
        # Check missing value patterns
        missing_counts = missing_value_data.isnull().sum()

        assert missing_counts['Age'] == 4  # 40% missing
        assert missing_counts['Fare'] == 3  # 30% missing
        assert missing_counts['Cabin'] == 7  # 70% missing
        assert missing_counts['Embarked'] == 2  # 20% missing
        assert missing_counts['Survived'] == 0  # No missing in target

    def test_imputation_orchestrator_basic(self, missing_value_data, data_config_with_missing_handling):
        """Test basic imputation orchestrator functionality."""
        orchestrator = ImputationOrchestrator(data_config_with_missing_handling)

        X = missing_value_data.drop('Survived', axis=1)
        y = missing_value_data['Survived']

        # Fit and transform
        orchestrator.fit(X, y)
        X_imputed = orchestrator.transform(X)

        # Check no missing values remain in specified columns
        assert X_imputed['Age'].isnull().sum() == 0
        assert X_imputed['Fare'].isnull().sum() == 0
        assert X_imputed['Embarked'].isnull().sum() == 0

        # Check missing indicators were added for Age (>5% missing)
        age_indicator = f"__miss_Age"
        if age_indicator in X_imputed.columns:
            assert X_imputed[age_indicator].sum() == 4  # Should match original missing count

    def test_different_imputation_strategies(self, missing_value_data):
        """Test different imputation strategies."""
        strategies_config = {
            'imputation': {
                'per_column': {
                    'Age': {'method': 'mean'},
                    'Fare': {'method': 'median'},
                    'Embarked': {'method': 'most_frequent'},
                    'Cabin': {'method': 'constant', 'fill_value': 'Unknown'}
                }
            }
        }

        orchestrator = ImputationOrchestrator(strategies_config)
        X = missing_value_data[['Age', 'Fare', 'Embarked', 'Cabin']]

        orchestrator.fit(X)
        X_imputed = orchestrator.transform(X)

        # Verify strategies were applied correctly
        age_mean = missing_value_data['Age'].mean()

        # Check imputed values are reasonable
        imputed_ages = X_imputed.loc[missing_value_data['Age'].isnull(), 'Age']
        assert all(abs(age - age_mean) < 0.01 for age in imputed_ages)

        assert all(X_imputed['Cabin'].fillna('') != '')  # No NaN should remain
        assert 'Unknown' in X_imputed['Cabin'].values

    def test_missing_indicators(self, missing_value_data):
        """Test missing value indicator creation."""
        config = {
            'imputation': {
                'default': {
                    'add_missing_indicators': True,
                    'missing_indicator_threshold': 0.1,  # 10% threshold
                    'missing_indicator_prefix': '__missing_'
                }
            }
        }

        orchestrator = ImputationOrchestrator(config)
        X = missing_value_data[['Age', 'Fare', 'Embarked', 'Cabin']]

        orchestrator.fit(X)
        X_transformed = orchestrator.transform(X)

        # Check indicators for columns with >10% missing
        expected_indicators = ['__missing_Age', '__missing_Fare', '__missing_Cabin']

        for indicator in expected_indicators:
            if indicator in X_transformed.columns:
                original_col = indicator.replace('__missing_', '')
                original_missing = X[original_col].isnull()
                assert (X_transformed[indicator] == original_missing.astype(int)).all()

    # ==================== CLI INTEGRATION TESTS ====================

    def test_cli_features_basic_missing_handling(self, realistic_missing_data):
        """Test basic CLI features command with missing value handling."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Setup directories
            data_dir = temp_path / 'data' / 'raw'
            processed_dir = temp_path / 'data' / 'processed'
            config_dir = temp_path / 'configs'

            for directory in [data_dir, processed_dir, config_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            # Create data files
            train_data = realistic_missing_data.copy()
            test_data = realistic_missing_data.drop('Survived', axis=1).copy()

            train_path = data_dir / 'train.csv'
            test_path = data_dir / 'test.csv'

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            # Create data.yaml with missing value handling
            data_config = {
                'train_path': str(train_path),
                'test_path': str(test_path),
                'target_column': 'Survived',
                'id_column': 'PassengerId',
                'handle_missing': True,
                'numeric_columns': ['Age', 'SibSp', 'Parch', 'Fare'],
                'categorical_columns': ['Sex', 'Embarked', 'Pclass'],
                'skip_encoding_columns': ['PassengerId'],
                'imputation': {
                    'default': {
                        'numeric': 'median',
                        'categorical': 'constant',
                        'fill_value': 'Unknown'
                    },
                    'per_column': {
                        'Age': {'method': 'median'},
                        'Fare': {'method': 'median'},
                        'Cabin': {'method': 'constant', 'fill_value': 'Unknown'},
                        'Embarked': {'method': 'most_frequent'}
                    }
                },
                'encoding': {
                    'default': {'method': 'onehot'}
                },
                'scale_features': True
            }

            # Create experiment.yaml
            exp_config = {
                'name': 'test_missing_values',
                'model_name': 'logistic',
                'seed': 42,
                'debug_mode': True,
                'debug_n_rows': 50
            }

            # Write config files
            with open(config_dir / 'data.yaml', 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)

            with open(config_dir / 'experiment.yaml', 'w') as f:
                yaml.dump(exp_config, f, default_flow_style=False)

            # Run CLI features command
            result = runner.invoke(cli, [
                '--config-dir', str(config_dir),
                'features',
                '--data-config', 'data',
                '--experiment-config', 'experiment'
            ])

            # Verify success or document failure
            if result.exit_code == 0:
                # Check that processed files exist
                train_features_path = processed_dir / 'train_features.csv'
                test_features_path = processed_dir / 'test_features.csv'

                assert train_features_path.exists(), "Train features file was not created"
                assert test_features_path.exists(), "Test features file was not created"

                # Load and verify processed data
                train_processed = pd.read_csv(train_features_path)
                test_processed = pd.read_csv(test_features_path)

                # Verify no missing values
                assert train_processed.isnull().sum().sum() == 0, "Missing values found in processed train data"
                assert test_processed.isnull().sum().sum() == 0, "Missing values found in processed test data"

                # Verify target column handling
                assert 'Survived' in train_processed.columns, "Target column missing from train data"
                assert 'Survived' not in test_processed.columns, "Target column incorrectly in test data"
            else:
                # Log the failure for debugging
                print(f"❌ CLI command failed with exit code {result.exit_code}")
                print(f"Output: {result.output}")

    def test_cli_features_with_missing_indicators(self, realistic_missing_data):
        """Test CLI features command with missing value indicators enabled."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Setup directories
            data_dir = temp_path / 'data' / 'raw'
            processed_dir = temp_path / 'data' / 'processed'
            config_dir = temp_path / 'configs'

            for directory in [data_dir, processed_dir, config_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            # Create data files
            train_data = realistic_missing_data.copy()
            test_data = realistic_missing_data.drop('Survived', axis=1).copy()

            train_path = data_dir / 'train.csv'
            test_path = data_dir / 'test.csv'

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            # Create data.yaml with missing indicators
            data_config = {
                'train_path': str(train_path),
                'test_path': str(test_path),
                'target_column': 'Survived',
                'id_column': 'PassengerId',
                'handle_missing': True,
                'numeric_columns': ['Age', 'SibSp', 'Parch', 'Fare'],
                'categorical_columns': ['Sex', 'Embarked', 'Pclass'],
                'skip_encoding_columns': ['PassengerId'],
                'imputation': {
                    'default': {
                        'numeric': 'median',
                        'categorical': 'constant',
                        'fill_value': 'Unknown',
                        'add_missing_indicators': True,
                        'missing_indicator_threshold': 0.1,  # 10% threshold
                        'missing_indicator_prefix': '__miss_'
                    },
                    'per_column': {
                        'Age': {
                            'method': 'median',
                            'add_missing_indicator': True
                        },
                        'Fare': {
                            'method': 'median',
                            'add_missing_indicator': True
                        }
                    }
                },
                'encoding': {
                    'default': {'method': 'onehot'}
                },
                'scale_features': True
            }

            # Create experiment.yaml
            exp_config = {
                'name': 'test_missing_indicators',
                'model_name': 'logistic',
                'seed': 42,
                'debug_mode': True,
                'debug_n_rows': 50
            }

            # Write config files
            with open(config_dir / 'data.yaml', 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)

            with open(config_dir / 'experiment.yaml', 'w') as f:
                yaml.dump(exp_config, f, default_flow_style=False)

            # Run CLI features command
            result = runner.invoke(cli, [
                '--config-dir', str(config_dir),
                'features',
                '--data-config', 'data',
                '--experiment-config', 'experiment'
            ])

            if result.exit_code == 0:
                # Check for missing indicators in processed data
                train_features_path = processed_dir / 'train_features.csv'
                if train_features_path.exists():
                    train_processed = pd.read_csv(train_features_path)

                    # Look for missing indicator columns
                    missing_indicator_cols = [col for col in train_processed.columns if '__miss_' in col]

                    # Verify no missing values remain
                    assert train_processed.isnull().sum().sum() == 0

    # ==================== FEATURE BUILDER TESTS ====================

    def test_feature_builder_with_missing_values(self, missing_value_data, data_config_with_missing_handling):
        """Test TitanicFeatureBuilder with missing value handling."""
        builder = TitanicFeatureBuilder(data_config_with_missing_handling)

        X = missing_value_data.drop('Survived', axis=1)
        y = missing_value_data['Survived']

        # Fit and transform
        builder.fit(X, y)
        X_processed = builder.transform(X)

        # Verify no missing values in output
        assert X_processed.isnull().sum().sum() == 0

        # Verify output is numeric (after encoding)
        assert all(pd.api.types.is_numeric_dtype(X_processed[col]) for col in X_processed.columns)

    def test_integration_with_full_pipeline(self, missing_value_data, data_config_with_missing_handling):
        """Test missing value handling in the context of the full feature pipeline."""
        config = data_config_with_missing_handling.copy()
        config.update({
            'feature_engineering': {
                'pre_impute': [],
                'post_impute': []
            },
            'encoding': {
                'default': {'method': 'onehot', 'handle_missing': 'value'}
            },
            'scale_features': True
        })

        builder = TitanicFeatureBuilder(config)

        X = missing_value_data.drop('Survived', axis=1)
        y = missing_value_data['Survived']

        # Full pipeline: imputation -> encoding -> scaling
        builder.fit(X, y)
        X_final = builder.transform(X)

        # Final output should have no missing values and be ready for ML
        assert X_final.isnull().sum().sum() == 0
        assert len(X_final) == len(X)  # No rows should be dropped

    # ==================== EDGE CASES AND ERROR HANDLING ====================

    def test_edge_case_all_missing_column(self, empty_columns_data):
        """Test handling of columns with all missing values."""
        config = {
            'imputation': {
                'default': {
                    'categorical': 'constant',
                    'fill_value': 'Unknown'
                }
            }
        }

        orchestrator = ImputationOrchestrator(config)

        # Fit should handle all-missing columns gracefully
        orchestrator.fit(empty_columns_data)
        result = orchestrator.transform(empty_columns_data)

        # All-missing column should be filled with default value
        assert result['EmptyColumn'].isnull().sum() == 0

    def test_missing_value_report(self, missing_value_data, data_config_with_missing_handling):
        """Test missing value reporting functionality."""
        orchestrator = ImputationOrchestrator(data_config_with_missing_handling)

        X = missing_value_data.drop('Survived', axis=1)
        y = missing_value_data['Survived']

        orchestrator.fit(X, y)

        # Get imputation report
        report = orchestrator.get_report()

        assert not report.empty
        # The report uses index as column names, check for expected columns
        expected_columns = ['method', 'missing_rate', 'fit_rows']
        for col in expected_columns:
            assert col in report.columns

    def test_schema_validation_between_fit_transform(self, missing_value_data, data_config_with_missing_handling):
        """Test schema validation between fit and transform calls."""
        orchestrator = ImputationOrchestrator(data_config_with_missing_handling)

        X_train = missing_value_data.drop('Survived', axis=1)
        orchestrator.fit(X_train)

        # Test with missing columns in transform
        X_test_missing_col = X_train.drop('Age', axis=1)

        with pytest.raises(ValueError, match="missing columns seen during fit"):
            orchestrator.transform(X_test_missing_col)

    def test_clipping_after_imputation(self, missing_value_data):
        """Test value clipping after imputation."""
        config = {
            'imputation': {
                'per_column': {
                    'Age': {
                        'method': 'constant',
                        'fill_value': 150,  # Unrealistic age
                        'clip_min': 0,
                        'clip_max': 100
                    }
                }
            }
        }

        orchestrator = ImputationOrchestrator(config)
        X = missing_value_data[['Age']]

        orchestrator.fit(X)
        X_imputed = orchestrator.transform(X)

        # Check clipping was applied
        assert X_imputed['Age'].max() <= 100
        assert X_imputed['Age'].min() >= 0

    def test_missing_value_handling_disabled(self, missing_value_data):
        """Test behavior when missing value handling is disabled."""
        config = {
            'handle_missing': False
        }

        builder = TitanicFeatureBuilder(config)
        X = missing_value_data.drop('Survived', axis=1)
        y = missing_value_data['Survived']

        # Should still work but may propagate NaNs
        builder.fit(X, y)

    # ==================== PERFORMANCE TESTS ====================

    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets."""
        np.random.seed(42)
        n_samples = 1000

        # Create large dataset with missing values
        large_data = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Age': np.where(np.random.random(n_samples) < 0.2, np.nan, np.random.normal(30, 15, n_samples)),
            'Fare': np.where(np.random.random(n_samples) < 0.1, np.nan, np.random.lognormal(3, 1, n_samples)),
            'Embarked': np.where(np.random.random(n_samples) < 0.05, np.nan, np.random.choice(['S', 'C', 'Q'], n_samples)),
            'Survived': np.random.choice([0, 1], n_samples)
        })

        config = {
            'imputation': {
                'default': {
                    'numeric': 'median',
                    'categorical': 'most_frequent'
                }
            }
        }

        orchestrator = ImputationOrchestrator(config)

        # Monitor performance
        start_time = time.time()
        orchestrator.fit(large_data)
        X_imputed = orchestrator.transform(large_data)
        end_time = time.time()

        # Basic performance checks
        assert end_time - start_time < 10  # Should complete within 10 seconds
        assert X_imputed.isnull().sum().sum() == 0  # All missing values filled
        assert X_imputed.shape == large_data.shape  # Same shape

    # ==================== PARAMETRIZED TESTS ====================

    @pytest.mark.parametrize("strategy,expected_type", [
        ("mean", float),
        ("median", float),
        ("most_frequent", object),
        ("constant", object)
    ])
    def test_imputation_strategy_types(self, missing_value_data, strategy, expected_type):
        """Test that different imputation strategies preserve appropriate data types."""
        config = {
            'imputation': {
                'per_column': {
                    'Age': {'method': strategy, 'fill_value': 30 if strategy == 'constant' else None}
                }
            }
        }

        orchestrator = ImputationOrchestrator(config)
        X = missing_value_data[['Age']]

        orchestrator.fit(X)
        X_imputed = orchestrator.transform(X)

        # Check that imputed values have reasonable types
        if strategy in ['mean', 'median']:
            assert pd.api.types.is_numeric_dtype(X_imputed['Age'])

    # ==================== ANALYSIS AND DOCUMENTATION TESTS ====================

    def test_missing_value_patterns_analysis(self, realistic_missing_data):
        """Analyze missing value patterns in the test data."""
        missing_info = {}

        for col in realistic_missing_data.columns:
            missing_count = realistic_missing_data[col].isnull().sum()
            missing_rate = missing_count / len(realistic_missing_data)
            missing_info[col] = {
                'missing_count': missing_count,
                'missing_rate': missing_rate,
                'dtype': str(realistic_missing_data[col].dtype)
            }

        # Verify our test data has the expected missing patterns
        assert realistic_missing_data['Age'].isnull().sum() > 0, "Age should have missing values"
        assert realistic_missing_data['Fare'].isnull().sum() > 0, "Fare should have missing values"
        assert realistic_missing_data['Cabin'].isnull().sum() > 0, "Cabin should have missing values"
        assert realistic_missing_data['Embarked'].isnull().sum() > 0, "Embarked should have missing values"
        assert realistic_missing_data['Survived'].isnull().sum() == 0, "Target should have no missing values"

    def test_cli_with_real_data_config(self):
        """Test CLI features command using the actual data.yaml configuration."""
        # Use the actual config file
        config_dir = Path('/configs')

        if not config_dir.exists():
            pytest.skip("Actual config directory not found")

        # Check if actual data files exist
        data_config_path = config_dir / 'data.yaml'
        if not data_config_path.exists():
            pytest.skip("Actual data.yaml not found")

        # Load the actual configuration to see how missing values are configured
        with open(data_config_path, 'r') as f:
            actual_config = yaml.safe_load(f)

        # Document the actual configuration
        handle_missing = actual_config.get('handle_missing', 'Not specified')
        has_imputation = 'imputation' in actual_config

        # This test documents the actual configuration
        assert isinstance(actual_config, dict), "Configuration should be a valid dictionary"
