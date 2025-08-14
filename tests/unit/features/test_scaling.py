"""
Unit tests for scaling system - feature scaling and normalization.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.features.scaling.scaler import ScalingOrchestrator


class TestScalingOrchestrator:
    """Test the main scaling orchestrator."""

    @pytest.fixture
    def sample_data(self):
        """Sample numeric data for scaling."""
        return pd.DataFrame({
            'Age': [22.0, 38.0, 26.0, 35.0, 27.0],
            'Fare': [7.25, 71.28, 7.925, 53.1, 8.05],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'FamilySize': [2, 2, 1, 2, 1],
            'Sex_male': [1, 0, 0, 1, 0],
            'Sex_female': [0, 1, 1, 0, 1],
            'PassengerId': [1, 2, 3, 4, 5]  # Should be excluded from scaling
        })

    def test_scaler_initialization_enabled(self):
        """Test scaler initialization when enabled."""
        scaler = ScalingOrchestrator(enable=True)

        assert scaler.enable is True
        assert scaler.scaler is None
        assert not scaler._is_fitted

    def test_scaler_initialization_disabled(self):
        """Test scaler initialization when disabled."""
        scaler = ScalingOrchestrator(enable=False)

        assert scaler.enable is False
        assert scaler.scaler is None

    def test_fit_when_enabled(self, sample_data):
        """Test fitting when scaling is enabled."""
        scaler = ScalingOrchestrator(enable=True)

        scaler.fit(sample_data)

        assert scaler._is_fitted
        assert scaler.scaler is not None

    def test_fit_when_disabled(self, sample_data):
        """Test fitting when scaling is disabled."""
        scaler = ScalingOrchestrator(enable=False)

        scaler.fit(sample_data)

        assert not scaler._is_fitted
        assert scaler.scaler is None

    def test_transform_when_enabled(self, sample_data):
        """Test transform when scaling is enabled."""
        scaler = ScalingOrchestrator(enable=True)
        scaler.fit(sample_data)

        result = scaler.transform(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data.shape
        assert result.columns.tolist() == sample_data.columns.tolist()

        # Numeric columns should be scaled (different from original)
        numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        for col in numeric_cols:
            # Scaled values should have different mean/std (unless constant)
            if sample_data[col].std() > 0:
                assert not np.allclose(result[col], sample_data[col])

    def test_transform_when_disabled(self, sample_data):
        """Test transform when scaling is disabled."""
        scaler = ScalingOrchestrator(enable=False)
        scaler.fit(sample_data)

        result = scaler.transform(sample_data)

        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, sample_data)

    def test_transform_before_fit_raises_error(self, sample_data):
        """Test that transform raises error when called before fit."""
        scaler = ScalingOrchestrator(enable=True)

        with pytest.raises(ValueError, match="Scaler must be fitted before transform"):
            scaler.transform(sample_data)

    def test_fit_transform_shortcut(self, sample_data):
        """Test fit_transform method."""
        scaler = ScalingOrchestrator(enable=True)

        result = scaler.fit_transform(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert scaler._is_fitted

    def test_standardization_properties(self, sample_data):
        """Test that standardization produces expected statistical properties."""
        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(sample_data)

        # Numeric columns should be approximately standardized
        numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        for col in numeric_cols:
            if sample_data[col].std() > 1e-6:  # Avoid division by zero
                # Mean should be close to 0, std close to 1
                assert abs(result[col].mean()) < 0.1
                assert abs(result[col].std() - 1.0) < 0.1

    def test_binary_columns_handling(self, sample_data):
        """Test handling of binary columns (one-hot encoded)."""
        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(sample_data)

        # Binary columns might be scaled differently
        binary_cols = ['Sex_male', 'Sex_female']
        for col in binary_cols:
            assert col in result.columns
            # Values should still be numeric
            assert result[col].dtype in ['float64', 'float32', 'int64', 'int32']

    def test_constant_columns_handling(self):
        """Test handling of constant columns."""
        data = pd.DataFrame({
            'constant_col': [1.0] * 5,
            'variable_col': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(data)

        # Constant column should be handled gracefully (no division by zero)
        assert not result.isnull().any().any()
        # Constant column should remain constant (or become zero)
        assert result['constant_col'].nunique() <= 1

    def test_missing_values_handling(self):
        """Test handling of missing values in scaling."""
        data = pd.DataFrame({
            'col_with_missing': [1.0, 2.0, np.nan, 4.0, 5.0],
            'complete_col': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(data)

        # Missing values should remain missing
        assert result['col_with_missing'].isnull().sum() == 1
        # Complete column should be scaled
        assert not result['complete_col'].isnull().any()

    def test_scaling_consistency(self, sample_data):
        """Test that scaling is consistent across multiple transforms."""
        scaler = ScalingOrchestrator(enable=True)
        scaler.fit(sample_data)

        result1 = scaler.transform(sample_data)
        result2 = scaler.transform(sample_data)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_new_data_scaling(self, sample_data):
        """Test scaling of new data with same schema."""
        scaler = ScalingOrchestrator(enable=True)
        scaler.fit(sample_data)

        # Create new data with same schema but different values
        new_data = sample_data.copy()
        new_data['Age'] = [30.0, 40.0, 50.0, 60.0, 70.0]
        new_data['Fare'] = [10.0, 20.0, 30.0, 40.0, 50.0]

        result = scaler.transform(new_data)

        # Should have same column structure
        assert result.columns.tolist() == sample_data.columns.tolist()
        assert len(result) == len(new_data)

    @patch('src.features.scaling.scaler.StandardScaler')
    def test_scaler_type_used(self, mock_standard_scaler, sample_data):
        """Test that StandardScaler is used internally."""
        mock_scaler_instance = MagicMock()
        mock_standard_scaler.return_value = mock_scaler_instance

        scaler = ScalingOrchestrator(enable=True)
        scaler.fit(sample_data)

        # StandardScaler should be instantiated and fitted
        mock_standard_scaler.assert_called_once()
        mock_scaler_instance.fit.assert_called_once()


class TestScalingIntegration:
    """Test scaling integration with other preprocessing steps."""

    @pytest.fixture
    def preprocessed_data(self):
        """Data that has been through feature engineering and encoding."""
        return pd.DataFrame({
            'Age': [22.0, 38.0, 26.0, 35.0, 27.0],
            'Fare': [7.25, 71.28, 7.925, 53.1, 8.05],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'FamilySize': [2, 2, 1, 2, 1],
            'IsAlone': [0, 0, 1, 0, 1],
            'TicketGroupSize': [1, 1, 1, 1, 1],
            'Fare_log': [1.98, 4.27, 2.07, 3.97, 2.08],
            # One-hot encoded features
            'Sex_male': [1, 0, 0, 1, 0],
            'Sex_female': [0, 1, 1, 0, 1],
            'Embarked_S': [1, 0, 1, 0, 1],
            'Embarked_C': [0, 1, 0, 0, 0],
            'Embarked_Q': [0, 0, 0, 1, 0],
            'Deck_A': [0, 1, 0, 0, 0],
            'Deck_B': [1, 0, 0, 0, 0],
            'Deck_C': [0, 0, 1, 0, 0],
            'Deck_U': [0, 0, 0, 1, 1],
            # Target encoded features
            'Title_catboost': [0.4, 0.8, 0.9, 0.4, 0.8]
        })

    def test_scaling_after_preprocessing(self, preprocessed_data):
        """Test scaling of fully preprocessed data."""
        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(preprocessed_data)

        # All columns should be scaled
        assert result.shape == preprocessed_data.shape
        assert result.columns.tolist() == preprocessed_data.columns.tolist()

        # Continuous features should be standardized
        continuous_features = ['Age', 'Fare', 'Fare_log', 'Title_catboost']
        for col in continuous_features:
            if preprocessed_data[col].std() > 1e-6:
                assert abs(result[col].mean()) < 0.2
                assert abs(result[col].std() - 1.0) < 0.2

    def test_mixed_feature_types_scaling(self, preprocessed_data):
        """Test scaling with mixed feature types (continuous, binary, count)."""
        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(preprocessed_data)

        # Binary features (one-hot encoded)
        binary_features = ['Sex_male', 'Sex_female', 'Embarked_S', 'Embarked_C', 'Embarked_Q',
                          'Deck_A', 'Deck_B', 'Deck_C', 'Deck_U']

        # Count features
        count_features = ['SibSp', 'Parch', 'FamilySize', 'IsAlone', 'TicketGroupSize']

        # All feature types should be handled appropriately
        for col in binary_features + count_features:
            assert col in result.columns
            assert not result[col].isnull().any()

    def test_feature_importance_preservation(self, preprocessed_data):
        """Test that relative feature importance relationships are preserved."""
        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(preprocessed_data)

        # High-variance features should still have reasonable variance after scaling
        # (scaling shouldn't completely flatten important features)
        for col in result.columns:
            if preprocessed_data[col].std() > 0:
                assert result[col].std() > 0

    def test_outlier_impact_on_scaling(self):
        """Test impact of outliers on scaling."""
        # Data with outliers
        data_with_outliers = pd.DataFrame({
            'normal_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_with_outlier': [1.0, 2.0, 3.0, 4.0, 100.0]  # 100 is an outlier
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(data_with_outliers)

        # Outlier should be scaled but still detectable
        outlier_scaled = result['feature_with_outlier'].iloc[-1]
        normal_scaled = result['feature_with_outlier'].iloc[:-1]

        # Outlier should still be distinguishable from normal values
        assert abs(outlier_scaled) > abs(normal_scaled.max()) * 2

    def test_zero_variance_features_after_encoding(self):
        """Test handling of zero-variance features that might result from encoding."""
        # Data that creates zero-variance feature after one-hot encoding
        data = pd.DataFrame({
            'constant_category': ['A'] * 5,  # Will create single binary column
            'variable_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        # Simulate one-hot encoding result
        encoded_data = pd.DataFrame({
            'constant_category_A': [1] * 5,  # Zero variance
            'variable_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(encoded_data)

        # Should handle zero-variance feature gracefully
        assert not result.isnull().any().any()
        assert 'constant_category_A' in result.columns

    def test_scaling_with_different_data_types(self):
        """Test scaling with different numeric data types."""
        data = pd.DataFrame({
            'int_feature': [1, 2, 3, 4, 5],
            'float_feature': [1.1, 2.2, 3.3, 4.4, 5.5],
            'large_int_feature': [1000, 2000, 3000, 4000, 5000]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(data)

        # All features should be scaled regardless of original data type
        for col in data.columns:
            if data[col].std() > 1e-6:
                assert abs(result[col].mean()) < 0.1
                assert abs(result[col].std() - 1.0) < 0.1


class TestScalingEdgeCases:
    """Test edge cases in scaling functionality."""

    def test_empty_dataframe(self):
        """Test scaling of empty dataframe."""
        empty_data = pd.DataFrame()

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(empty_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row_data(self):
        """Test scaling with single row of data."""
        single_row = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(single_row)

        # Single row should be handled gracefully (will have zero variance)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_single_column_data(self):
        """Test scaling with single column."""
        single_col = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(single_col)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == single_col.shape
        assert 'feature1' in result.columns

    def test_all_missing_column(self):
        """Test scaling when a column is all missing values."""
        data_with_all_missing = pd.DataFrame({
            'all_missing': [np.nan] * 5,
            'complete_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(data_with_all_missing)

        # All-missing column should remain all missing
        assert result['all_missing'].isnull().all()
        # Complete column should be scaled
        assert not result['complete_feature'].isnull().any()

    def test_extremely_large_values(self):
        """Test scaling with extremely large values."""
        large_value_data = pd.DataFrame({
            'normal_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'large_feature': [1e6, 2e6, 3e6, 4e6, 5e6]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(large_value_data)

        # Should handle large values without overflow
        assert not result.isnull().any().any()
        assert np.isfinite(result).all().all()

    def test_extremely_small_values(self):
        """Test scaling with extremely small values."""
        small_value_data = pd.DataFrame({
            'normal_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'small_feature': [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
        })

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(small_value_data)

        # Should handle small values without underflow
        assert not result.isnull().any().any()
        assert np.isfinite(result).all().all()

    def test_scaling_preserves_dataframe_index(self):
        """Test that scaling preserves the original dataframe index."""
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [5.0, 4.0, 3.0, 2.0, 1.0]
        }, index=['a', 'b', 'c', 'd', 'e'])

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(data)

        # Index should be preserved
        pd.testing.assert_index_equal(result.index, data.index)

    def test_scaling_with_categorical_data_mixed_in(self):
        """Test that scaling handles mixed data types appropriately."""
        mixed_data = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'string_feature': ['a', 'b', 'c', 'd', 'e'],  # Should be ignored or cause error
            'binary_feature': [0, 1, 0, 1, 0]
        })

        # Remove string feature (preprocessing should have handled this)
        numeric_only_data = mixed_data.select_dtypes(include=[np.number])

        scaler = ScalingOrchestrator(enable=True)
        result = scaler.fit_transform(numeric_only_data)

        # Should work with numeric-only data
        assert isinstance(result, pd.DataFrame)
        assert result.shape == numeric_only_data.shape
