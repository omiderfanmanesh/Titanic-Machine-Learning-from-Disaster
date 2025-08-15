#!/usr/bin/env python3
"""Simple test script to verify leak-safe training functionality."""

import sys
import os
sys.path.append('../src')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy

# Import our components
from modeling.trainers import TitanicTrainer
from core.interfaces import ITransformer


class TestFeatureTransformer(ITransformer):
    """A test transformer that tracks what data it was fitted on."""

    def __init__(self):
        self.fitted_data_size = None
        self.fit_call_count = 0
        self.fitted_mean = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit and record statistics."""
        self.fit_call_count += 1
        self.fitted_data_size = len(X)
        if 'feature_1' in X.columns:
            self.fitted_mean = X['feature_1'].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted statistics."""
        if self.fitted_mean is None:
            raise ValueError("Not fitted")

        X_transformed = X.copy()
        if 'feature_1' in X.columns:
            X_transformed['feature_1_scaled'] = X['feature_1'] - self.fitted_mean
        return X_transformed


def test_leak_safe_training():
    """Test that feature pipeline is fitted only on training folds."""
    print("🧪 Testing leak-safe cross-validation training...")

    # Create sample data
    np.random.seed(42)
    n_samples = 60
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
    })
    y = pd.Series((X['feature_1'] + np.random.normal(0, 0.5, n_samples)) > 0).astype(int)

    print(f"📊 Created dataset with {len(X)} samples")

    # Create trainer configuration
    trainer_config = {
        "strategy": "stratified",
        "n_folds": 3,
        "shuffle": True,
        "random_state": 42,
        "model_name": "random_forest",
        "model_params": {"n_estimators": 10, "random_state": 42}
    }

    # Create feature transformer
    feature_pipeline = TestFeatureTransformer()

    # Create trainer and model
    trainer = TitanicTrainer(trainer_config)
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    print("🚀 Running cross-validation with feature processing...")

    # Run cross-validation
    results = trainer.cross_validate(model, X, y, trainer_config, feature_pipeline)

    print("✅ Cross-validation completed successfully!")

    # Verify results
    print(f"📈 CV Results:")
    cv_scores = results['cv_scores']
    print(f"  - Mean CV Score: {cv_scores['mean_score']:.4f}")
    print(f"  - Std CV Score: {cv_scores['std_score']:.4f}")
    print(f"  - OOF Score: {cv_scores['oof_score']:.4f}")

    # Verify no data leakage
    print("\n🔍 Verifying no data leakage...")

    # Check that we have separate pipelines for each fold
    assert len(trainer.fold_feature_pipelines) == 3, "Should have 3 fold pipelines"
    print(f"✅ Created {len(trainer.fold_feature_pipelines)} separate feature pipelines")

    # Check that each pipeline was fitted independently
    total_samples = len(X)
    expected_train_size = int(total_samples * 2 / 3)  # ~2/3 for training in 3-fold CV

    fitted_means = []
    for i, fold_pipeline in enumerate(trainer.fold_feature_pipelines):
        print(f"  Fold {i+1}:")
        print(f"    - Fit calls: {fold_pipeline.fit_call_count}")
        print(f"    - Training size: {fold_pipeline.fitted_data_size}")
        print(f"    - Fitted mean: {fold_pipeline.fitted_mean:.4f}")

        assert fold_pipeline.fit_call_count == 1, f"Fold {i+1} should be fitted exactly once"
        assert abs(fold_pipeline.fitted_data_size - expected_train_size) <= 3, f"Fold {i+1} training size should be ~{expected_train_size}"

        fitted_means.append(fold_pipeline.fitted_mean)

    # Verify that different folds have different statistics (no leakage)
    unique_means = len(set(np.round(fitted_means, 4)))
    print(f"✅ Found {unique_means} unique fitted means across folds (indicates no leakage)")
    assert unique_means > 1, "Different folds should have different fitted statistics"

    # Verify original pipeline was not modified
    assert feature_pipeline.fit_call_count == 0, "Original pipeline should not be fitted"
    print("✅ Original feature pipeline was not modified (proper isolation)")

    print("\n🎉 All leak-safe training tests passed!")
    return True


def test_validation_leakage_regression():
    """Regression test to ensure validation data never reaches fit method."""
    print("\n🔬 Running regression test for validation data leakage...")

    # Create a spy transformer that records all indices it sees during fit
    class SpyTransformer(ITransformer):
        def __init__(self):
            self.fitted_indices = set()

        def fit(self, X: pd.DataFrame, y: pd.Series = None):
            self.fitted_indices.update(X.index.tolist())
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            return X.copy()

    # Create sample data with explicit indices
    np.random.seed(42)
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 30),
        'feature_2': np.random.normal(0, 1, 30),
    }, index=range(30))  # Explicit indices 0-29
    y = pd.Series(np.random.choice([0, 1], 30), index=range(30))

    # Set up trainer
    trainer_config = {
        "strategy": "stratified",
        "n_folds": 3,
        "shuffle": False,  # No shuffle for predictable splits
        "random_state": 42
    }

    spy_pipeline = SpyTransformer()
    trainer = TitanicTrainer(trainer_config)
    model = RandomForestClassifier(n_estimators=5, random_state=42)

    # Run cross-validation
    results = trainer.cross_validate(model, X, y, trainer_config, spy_pipeline)

    # Determine which indices should have been validation indices
    from cv.folds import FoldSplitterFactory
    splitter = FoldSplitterFactory.create_splitter("stratified", n_splits=3, shuffle=False, random_state=42)
    all_validation_indices = set()

    for train_idx, val_idx in splitter.split(X, y):
        all_validation_indices.update(val_idx)

    # Check that no validation indices were used in any fit call
    all_fitted_indices = set()
    for fold_pipeline in trainer.fold_feature_pipelines:
        all_fitted_indices.update(fold_pipeline.fitted_indices)

    validation_leak = all_fitted_indices.intersection(all_validation_indices)

    print(f"📊 Total validation indices across all folds: {len(all_validation_indices)}")
    print(f"📊 Total fitted indices across all folds: {len(all_fitted_indices)}")
    print(f"🔍 Validation indices that leaked into fit: {len(validation_leak)}")

    assert len(validation_leak) == 0, f"Validation indices {validation_leak} were leaked into fit()"
    print("✅ Regression test passed: No validation data leaked into feature pipeline fit()")

    return True


if __name__ == "__main__":
    print("🔧 Testing Refactored TitanicTrainer with Leak-Safe Feature Processing")
    print("=" * 70)

    try:
        test_leak_safe_training()
        test_validation_leakage_regression()
        print("\n🎯 All tests passed! The refactored TitanicTrainer prevents data leakage.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
