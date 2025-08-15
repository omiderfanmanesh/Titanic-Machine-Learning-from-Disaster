"""Regression tests for leak-safe feature processing in cross-validation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from core.interfaces import ITransformer
from modeling.trainers import TitanicTrainer


class MockLeakyTransformer(ITransformer):
    """A mock transformer that would cause data leakage if used incorrectly."""

    def __init__(self):
        self.fitted_data_size = None
        self.fit_call_count = 0
        self.transform_call_count = 0
        self.fitted_mean = None
        self.fitted_std = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the transformer and record what data it was fitted on."""
        self.fit_call_count += 1
        self.fitted_data_size = len(X)

        # Store statistics that would leak information if computed on validation data
        if 'feature_1' in X.columns:
            self.fitted_mean = X['feature_1'].mean()
            self.fitted_std = X['feature_1'].std()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters."""
        self.transform_call_count += 1

        if self.fitted_mean is None or self.fitted_std is None:
            raise ValueError("Transformer not fitted")

        X_transformed = X.copy()

        # Apply transformation using fitted statistics
        if 'feature_1' in X.columns:
            X_transformed['feature_1_normalized'] = (X['feature_1'] - self.fitted_mean) / self.fitted_std

        return X_transformed


class MockDeterministicTransformer(ITransformer):
    """A mock transformer that produces deterministic results."""

    def __init__(self):
        self.is_fitted = False
        self.fitted_columns = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the transformer."""
        self.is_fitted = True
        self.fitted_columns = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform deterministically."""
        if not self.is_fitted:
            raise ValueError("Transformer not fitted")

        X_transformed = X.copy()
        # Add a deterministic feature
        X_transformed['deterministic_feature'] = X_transformed.sum(axis=1)
        return X_transformed


class MockNonDeterministicTransformer(ITransformer):
    """A mock transformer that produces non-deterministic results."""

    def __init__(self):
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the transformer."""
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform non-deterministically."""
        if not self.is_fitted:
            raise ValueError("Transformer not fitted")

        X_transformed = X.copy()
        # Add a random feature that changes each call
        X_transformed['random_feature'] = np.random.random(len(X))
        return X_transformed


class TestLeakSafeTraining:
    """Test suite for leak-safe cross-validation training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
        })

        # Create a binary target with some correlation to features
        y = pd.Series((X['feature_1'] + X['feature_2'] + np.random.normal(0, 0.5, n_samples)) > 0).astype(int)

        return X, y

    @pytest.fixture
    def trainer_config(self):
        """Create trainer configuration."""
        return {
            "strategy": "stratified",
            "n_folds": 3,
            "shuffle": True,
            "random_state": 42,
            "model_name": "random_forest",
            "model_params": {"n_estimators": 10, "random_state": 42}
        }

    def test_feature_pipeline_fitted_only_on_training_fold(self, sample_data, trainer_config):
        """Test that feature pipeline is fitted only on training data in each fold."""
        X, y = sample_data

        # Create a mock transformer that tracks what data it was fitted on
        feature_pipeline = MockLeakyTransformer()

        # Create trainer
        trainer = TitanicTrainer(trainer_config)

        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Run cross-validation with feature pipeline
        results = trainer.cross_validate(model, X, y, trainer_config, feature_pipeline)

        # Verify that we have the expected number of fold pipelines
        assert len(trainer.fold_feature_pipelines) == 3

        # Verify that each fold's transformer was fitted on training data only
        total_samples = len(X)
        expected_train_size = int(total_samples * 2 / 3)  # 2/3 for training in 3-fold CV

        for i, fold_pipeline in enumerate(trainer.fold_feature_pipelines):
            # Each pipeline should be fitted exactly once
            assert fold_pipeline.fit_call_count == 1

            # Each pipeline should be fitted on roughly 2/3 of the data (training fold)
            assert abs(fold_pipeline.fitted_data_size - expected_train_size) <= 2  # Allow small variance

            # Each pipeline should have been used to transform data multiple times (train + val)
            assert fold_pipeline.transform_call_count >= 2

    def test_no_data_leakage_between_folds(self, sample_data, trainer_config):
        """Test that feature statistics don't leak between folds."""
        X, y = sample_data

        # Create trainer
        trainer = TitanicTrainer(trainer_config)

        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Run cross-validation with feature pipeline
        feature_pipeline = MockLeakyTransformer()
        results = trainer.cross_validate(model, X, y, trainer_config, feature_pipeline)

        # Get the fitted statistics from each fold
        fold_means = [fp.fitted_mean for fp in trainer.fold_feature_pipelines]
        fold_stds = [fp.fitted_std for fp in trainer.fold_feature_pipelines]

        # The statistics should be different across folds because they're fitted on different data
        assert len(set(np.round(fold_means, 4))) > 1, "Fold means should be different (no leakage)"
        assert len(set(np.round(fold_stds, 4))) > 1, "Fold stds should be different (no leakage)"

        # But they should all be reasonable values (not NaN or extreme)
        for mean, std in zip(fold_means, fold_stds):
            assert not np.isnan(mean), "Fitted mean should not be NaN"
            assert not np.isnan(std), "Fitted std should not be NaN"
            assert std > 0, "Fitted std should be positive"

    def test_validation_leakage_check_passes_for_deterministic_transform(self, sample_data, trainer_config):
        """Test that validation leakage check passes for deterministic transformers."""
        X, y = sample_data

        feature_pipeline = MockDeterministicTransformer()
        trainer = TitanicTrainer(trainer_config)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # This should complete without errors
        results = trainer.cross_validate(model, X, y, trainer_config, feature_pipeline)

        # Verify successful completion
        assert "cv_scores" in results
        assert len(trainer.fold_scores) == 3

    def test_validation_leakage_check_warns_for_nondeterministic_transform(self, sample_data, trainer_config):
        """Test that validation leakage check warns for non-deterministic transformers."""
        X, y = sample_data

        feature_pipeline = MockNonDeterministicTransformer()
        trainer = TitanicTrainer(trainer_config)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Mock the logger to capture warnings
        with patch.object(trainer.logger, 'warning') as mock_warning:
            results = trainer.cross_validate(model, X, y, trainer_config, feature_pipeline)

            # Should have warned about non-deterministic results
            warning_calls = mock_warning.call_args_list
            assert len(warning_calls) == 3  # One warning per fold

            for call in warning_calls:
                assert "non-deterministic results" in str(call)

    def test_feature_pipeline_artifacts_saved(self, sample_data, trainer_config, tmp_path):
        """Test that feature pipeline artifacts are saved for each fold."""
        X, y = sample_data

        # Mock the path manager to use tmp_path
        feature_pipeline = MockDeterministicTransformer()
        trainer = TitanicTrainer(trainer_config)

        with patch.object(trainer.path_manager, 'create_run_directory', return_value=tmp_path):
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            results = trainer.cross_validate(model, X, y, trainer_config, feature_pipeline)

            # Check that feature pipeline files were created
            for i in range(3):  # 3 folds
                pipeline_file = tmp_path / f"fold_{i}_feature_pipeline.joblib"
                assert pipeline_file.exists(), f"Feature pipeline file for fold {i} should exist"

    def test_cross_validation_without_feature_pipeline(self, sample_data, trainer_config):
        """Test that cross-validation works without a feature pipeline."""
        X, y = sample_data

        trainer = TitanicTrainer(trainer_config)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Run cross-validation without feature pipeline
        results = trainer.cross_validate(model, X, y, trainer_config, feature_builder=None)

        # Should complete successfully
        assert "cv_scores" in results
        assert len(trainer.fold_scores) == 3

        # All fold feature pipelines should be None
        assert all(fp is None for fp in trainer.fold_feature_pipelines)

    def test_feature_pipeline_deep_copy_isolation(self, sample_data, trainer_config):
        """Test that feature pipelines are properly isolated using deep copy."""
        X, y = sample_data

        # Create a transformer with mutable state
        original_pipeline = MockLeakyTransformer()

        trainer = TitanicTrainer(trainer_config)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        results = trainer.cross_validate(model, X, y, trainer_config, original_pipeline)

        # Original pipeline should not have been modified
        assert original_pipeline.fit_call_count == 0, "Original pipeline should not be fitted"
        assert original_pipeline.transform_call_count == 0, "Original pipeline should not be used for transform"

        # Each fold should have its own independent copy
        for i, fold_pipeline in enumerate(trainer.fold_feature_pipelines):
            assert fold_pipeline is not original_pipeline, f"Fold {i} pipeline should be a copy"
            assert fold_pipeline.fit_call_count == 1, f"Fold {i} pipeline should be fitted once"

    def test_regression_no_validation_data_in_fit(self, sample_data, trainer_config):
        """Regression test: Ensure validation data never reaches the fit method of feature pipeline."""
        X, y = sample_data

        # Create a spy transformer that records all data it sees during fit
        class SpyTransformer(ITransformer):
            def __init__(self):
                self.fitted_row_indices = set()
                self.is_fitted = False

            def fit(self, X: pd.DataFrame, y: pd.Series = None):
                # Record the indices of data used in fit
                self.fitted_row_indices.update(X.index.tolist())
                self.is_fitted = True
                return self

            def transform(self, X: pd.DataFrame) -> pd.DataFrame:
                if not self.is_fitted:
                    raise ValueError("Not fitted")
                return X.copy()

        spy_pipeline = SpyTransformer()
        trainer = TitanicTrainer(trainer_config)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        results = trainer.cross_validate(model, X, y, trainer_config, spy_pipeline)

        # Collect all validation indices across all folds
        from cv.folds import FoldSplitterFactory
        splitter = FoldSplitterFactory.create_splitter("stratified", n_splits=3, shuffle=True, random_state=42)
        all_validation_indices = set()

        for train_idx, val_idx in splitter.split(X, y):
            all_validation_indices.update(val_idx)

        # Check that no validation indices appeared in any fit call
        fitted_indices = set()
        for fold_pipeline in trainer.fold_feature_pipelines:
            fitted_indices.update(fold_pipeline.fitted_row_indices)

        validation_leak = fitted_indices.intersection(all_validation_indices)
        assert len(validation_leak) == 0, f"Validation indices {validation_leak} were used in feature pipeline fit()"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def small_data(self):
        """Create very small dataset for edge case testing."""
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [5, 4, 3, 2, 1],
        })
        y = pd.Series([0, 1, 0, 1, 0])
        return X, y

    def test_feature_pipeline_error_handling(self, small_data):
        """Test error handling when feature pipeline fails."""
        X, y = small_data

        class FailingTransformer(ITransformer):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                raise ValueError("Intentional failure")

        trainer_config = {
            "strategy": "stratified",
            "n_folds": 2,
            "shuffle": False,
            "random_state": 42
        }

        trainer = TitanicTrainer(trainer_config)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        failing_pipeline = FailingTransformer()

        # Should raise an error with context about which fold failed
        with pytest.raises(RuntimeError, match=r"Feature pipeline validation failed"):
            trainer.cross_validate(model, X, y, trainer_config, failing_pipeline)

    def test_feature_pipeline_validation_with_empty_validation_set(self):
        """Test validation when validation set becomes empty (edge case)."""
        # This is an edge case that shouldn't happen in practice but tests robustness
        X = pd.DataFrame({'feature_1': [1, 2]})
        y = pd.Series([0, 1])

        trainer_config = {
            "strategy": "stratified",
            "n_folds": 2,
            "shuffle": False,
            "random_state": 42
        }

        trainer = TitanicTrainer(trainer_config)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        pipeline = MockDeterministicTransformer()

        # Should handle gracefully
        results = trainer.cross_validate(model, X, y, trainer_config, pipeline)
        assert "cv_scores" in results
