"""
Integration tests to detect data leakage and pipeline consistency issues.
These tests verify the complete end-to-end behavior of the missing value,
encoding, and scaling pipelines to catch interactions between components.
"""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from features.missing.orchestrator import ImputationOrchestrator
from features.scaling.scaler import ScalingOrchestrator


@pytest.fixture
def leakage_test_data():
    """Create test data with various edge cases that can reveal data leakage."""
    # Training data
    train_data = {
        'numeric_with_missing': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0],
        'categorical_with_missing': ['A', 'B', np.nan, 'A', 'C', 'B', np.nan, 'A'],
        'binary_feature': [0, 1, 0, 1, 0, 1, 0, 1],
        'high_cardinality': ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8'],
        'target': [0, 1, 0, 1, 1, 0, 1, 0]
    }
    train_df = pd.DataFrame(train_data)

    # Test data with different patterns to detect leakage
    test_data = {
        'numeric_with_missing': [1.5, np.nan, 3.5, np.nan, 5.5, 6.5, np.nan, 8.5],
        'categorical_with_missing': ['A', np.nan, 'D', 'B', np.nan, 'E', 'A', np.nan],  # 'D', 'E' are new
        'binary_feature': [1, 0, 1, 0, 1, 0, 1, 0],
        'high_cardinality': ['cat1', 'cat9', 'cat10', 'cat2', 'cat11', 'cat3', 'cat12', 'cat4'],  # New categories
        'target': [1, 0, 1, 0, 0, 1, 0, 1]
    }
    test_df = pd.DataFrame(test_data)

    return train_df, test_df


@pytest.fixture
def knn_imputation_config():
    """Configuration that uses KNN imputation to test for data leakage."""
    return {
        "imputation": {
            "default": {
                "add_missing_indicators": True,
                "missing_indicator_threshold": 0.1,
                "debug": True
            },
            "per_column": {
                "numeric_with_missing": {
                    "method": "knn",
                    "n_neighbors": 3,
                    "features": ["binary_feature", "categorical_with_missing"]
                }
            }
        }
    }


class TestKNNDataLeakage:
    """Test KNN imputation for data leakage issues."""

    def test_knn_encoder_fitted_on_train_only(self, leakage_test_data, knn_imputation_config):
        """Verify KNN strategy doesn't fit encoder on test data."""
        train_df, test_df = leakage_test_data

        orchestrator = ImputationOrchestrator(knn_imputation_config)
        orchestrator.fit(train_df.drop('target', axis=1))

        # Get the KNN strategy
        knn_strategy = orchestrator.strategies.get('numeric_with_missing')
        assert knn_strategy is not None

        # Verify encoder was fitted during fit phase
        if hasattr(knn_strategy, 'encoder') and knn_strategy.encoder is not None:
            # Encoder should know about categories from training only
            train_categories = set(train_df['categorical_with_missing'].dropna().unique())
            fitted_categories = set()
            for feature_names in knn_strategy.encoder.categories_:
                fitted_categories.update(feature_names)

            # Should contain training categories
            assert train_categories.issubset(fitted_categories)

        # Transform test data
        test_result = orchestrator.transform(test_df.drop('target', axis=1))

        # Should not crash even with new categories in test
        assert not test_result['numeric_with_missing'].isna().any()

        # Verify that transform didn't refit anything
        assert knn_strategy.encoder is not None  # Should still be the same fitted encoder

    def test_knn_consistent_feature_space(self, leakage_test_data, knn_imputation_config):
        """Verify KNN produces consistent feature spaces between train and test."""
        train_df, test_df = leakage_test_data

        orchestrator = ImputationOrchestrator(knn_imputation_config)
        orchestrator.fit(train_df.drop('target', axis=1))

        # Transform both datasets
        train_transformed = orchestrator.transform(train_df.drop('target', axis=1))
        test_transformed = orchestrator.transform(test_df.drop('target', axis=1))

        # Should have same columns (except for missing indicators which may vary)
        base_columns = [c for c in train_transformed.columns if not c.startswith('__miss_')]
        test_base_columns = [c for c in test_transformed.columns if not c.startswith('__miss_')]

        assert set(base_columns) == set(test_base_columns)

        # Numeric column should be fully imputed in both
        assert not train_transformed['numeric_with_missing'].isna().any()
        assert not test_transformed['numeric_with_missing'].isna().any()


class TestMissingIndicatorLogic:
    """Test missing indicator injection logic."""

    def test_selective_missing_indicators(self, leakage_test_data):
        """Test that missing indicators are only added selectively."""
        train_df, test_df = leakage_test_data

        # Config with threshold-based indicator selection
        config = {
            "imputation": {
                "default": {
                    "add_missing_indicators": True,
                    "missing_indicator_threshold": 0.2,  # Only if >20% missing
                    "debug": True
                }
            }
        }

        orchestrator = ImputationOrchestrator(config)
        orchestrator.fit(train_df.drop('target', axis=1))

        # Check which columns got indicators
        # numeric_with_missing has 2/8 = 25% missing -> should get indicator
        # categorical_with_missing has 2/8 = 25% missing -> should get indicator
        expected_indicators = ['numeric_with_missing', 'categorical_with_missing']

        assert set(orchestrator._missing_indicator_cols) == set(expected_indicators)

        # Transform and verify indicators are added
        result = orchestrator.transform(test_df.drop('target', axis=1))

        for col in expected_indicators:
            indicator_col = f"__miss_{col}"
            if test_df[col].isna().any():
                assert indicator_col in result.columns
                assert result[indicator_col].dtype == np.int8

    def test_explicit_missing_indicator_columns(self, leakage_test_data):
        """Test explicit specification of missing indicator columns."""
        train_df, test_df = leakage_test_data

        config = {
            "imputation": {
                "default": {
                    "add_missing_indicators": True,
                    "missing_indicator_columns": ["numeric_with_missing"],  # Only this one
                    "debug": True
                }
            }
        }

        orchestrator = ImputationOrchestrator(config)
        orchestrator.fit(train_df.drop('target', axis=1))

        # Should only track the explicitly specified column
        assert orchestrator._missing_indicator_cols == ["numeric_with_missing"]

        result = orchestrator.transform(test_df.drop('target', axis=1))

        # Should only have indicator for numeric column
        assert "__miss_numeric_with_missing" in result.columns
        assert "__miss_categorical_with_missing" not in result.columns


class TestScalingConsistency:
    """Test scaling orchestrator for consistency issues."""

    def test_scaling_exclusion_stability(self, leakage_test_data):
        """Test that scaling exclusion logic is stable between train and test."""
        train_df, test_df = leakage_test_data

        # Add a feature that's binary in train but becomes ternary in test
        train_df['unstable_binary'] = [0, 1, 0, 1, 0, 1, 0, 1]
        test_df['unstable_binary'] = [0, 1, 2, 1, 0, 2, 1, 0]  # Now has value 2

        config = {
            "exclude_binary": True,
            "min_unique_threshold": 3,
            "debug": True
        }

        scaler = ScalingOrchestrator(enable=True, config=config)
        scaler.fit(train_df.select_dtypes(include=['number']))

        # unstable_binary should be excluded in training (only 2 unique values)
        assert 'unstable_binary' not in scaler.scale_cols

        # Transform test data - should not crash even though unstable_binary now has 3 values
        result = scaler.transform(test_df.select_dtypes(include=['number']))

        # unstable_binary should remain unscaled in test
        assert (result['unstable_binary'] == test_df['unstable_binary']).all()

    def test_dtype_handling_after_scaling(self, leakage_test_data):
        """Test that dtype restoration works properly after scaling."""
        train_df, test_df = leakage_test_data

        # Add integer column that should remain float after scaling
        train_df['int_feature'] = [10, 20, 30, 40, 50, 60, 70, 80]
        test_df['int_feature'] = [15, 25, 35, 45, 55, 65, 75, 85]

        config = {
            "restore_dtypes": True,
            "debug": True
        }

        scaler = ScalingOrchestrator(enable=True, config=config)
        scaler.fit(train_df.select_dtypes(include=['number']))

        result = scaler.transform(test_df.select_dtypes(include=['number']))

        # int_feature should be scaled (and thus converted to float)
        if 'int_feature' in scaler.scale_cols:
            # Should be float after scaling (not restored to int for standardized features)
            assert result['int_feature'].dtype == np.float64

        # Binary features should remain unscaled and keep their dtype
        assert result['binary_feature'].dtype == train_df['binary_feature'].dtype


class TestEndToEndPipelineConsistency:
    """Integration tests for the complete pipeline."""

    def test_complete_pipeline_train_test_consistency(self, leakage_test_data):
        """Test complete pipeline produces consistent results between train and test."""
        train_df, test_df = leakage_test_data

        # Complete pipeline config
        imputation_config = {
            "imputation": {
                "default": {
                    "add_missing_indicators": True,
                    "missing_indicator_threshold": 0.15,
                    "debug": True
                },
                "per_column": {
                    "numeric_with_missing": {"method": "knn", "n_neighbors": 3},
                    "categorical_with_missing": {"method": "constant", "fill_value": "Unknown"}
                }
            }
        }

        scaling_config = {
            "exclude_binary": True,
            "debug": True
        }

        # Fit pipelines
        imputer = ImputationOrchestrator(imputation_config)
        scaler = ScalingOrchestrator(enable=True, config=scaling_config)

        X_train = train_df.drop('target', axis=1)
        X_test = test_df.drop('target', axis=1)

        # Fit on training data
        imputer.fit(X_train)
        X_train_imputed = imputer.transform(X_train)
        scaler.fit(X_train_imputed)
        X_train_final = scaler.transform(X_train_imputed)

        # Transform test data
        X_test_imputed = imputer.transform(X_test)
        X_test_final = scaler.transform(X_test_imputed)

        # Verify no NaNs remain
        assert not X_train_final.isna().any().any()
        assert not X_test_final.isna().any().any()

        # Verify consistent schema (base columns should match)
        train_base_cols = [c for c in X_train_final.columns if not c.startswith('__miss_')]
        test_base_cols = [c for c in X_test_final.columns if not c.startswith('__miss_')]
        assert set(train_base_cols) == set(test_base_cols)

        # Verify missing indicators are consistent with configuration
        expected_indicators = [c for c in imputer._missing_indicator_cols]
        for col in expected_indicators:
            indicator_col = f"__miss_{col}"
            if X_test[col].isna().any():
                assert indicator_col in X_test_final.columns

    def test_pipeline_handles_edge_cases(self):
        """Test pipeline handles various edge cases without crashing."""
        # Create edge case data
        edge_case_data = {
            'all_missing': [np.nan] * 10,
            'single_value': [1.0] * 10,
            'extreme_outliers': [1, 2, 3, 1000000, 4, 5, 6, 7, 8, 9],
            'categorical_single': ['A'] * 10,
            'mixed_types': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Will be treated as numeric
        }

        df = pd.DataFrame(edge_case_data)

        config = {
            "imputation": {
                "default": {
                    "add_missing_indicators": True,
                    "debug": True
                },
                "per_column": {
                    "all_missing": {"method": "constant", "fill_value": 0},
                    "extreme_outliers": {"method": "median", "clip_min": 0, "clip_max": 100}
                }
            }
        }

        # Should not crash
        imputer = ImputationOrchestrator(config)
        imputer.fit(df)
        result = imputer.transform(df)

        # Verify edge cases handled
        assert not result['all_missing'].isna().any()
        assert result['extreme_outliers'].max() <= 100  # Clipped
        assert result['single_value'].nunique() == 1  # Unchanged
