"""Tests for model components."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import Mock, patch

from modeling.model_registry import (
    ModelRegistry,
    BaseModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel
)
from modeling.trainers import TitanicTrainer
from cv.folds import StratifiedKFoldSplitter


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_get_available_models(self):
        """Test getting list of available models."""
        registry = ModelRegistry()
        models = registry.get_available_models()
        
        # Should have basic models
        assert "logistic" in models
        assert "random_forest" in models
        
    def test_create_logistic_model(self):
        """Test creating logistic regression model."""
        registry = ModelRegistry()
        model = registry.create_model("logistic", {"C": 0.5, "random_state": 42})
        
        assert isinstance(model, LogisticRegressionModel)
        assert model.model.C == 0.5
        assert model.model.random_state == 42
        
    def test_create_model_invalid_name(self):
        """Test error for invalid model name."""
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="Unknown model"):
            registry.create_model("invalid_model")
            
    def test_create_model_with_missing_dependency(self):
        """Test graceful handling of missing dependencies."""
        registry = ModelRegistry()
        
        # Mock missing dependency
        with patch('titanic_ml.modeling.model_registry.xgb', None):
            with pytest.raises(ValueError, match="XGBoost not available"):
                registry.create_model("xgboost")


class TestBaseModel:
    """Test base model functionality."""
    
    def test_base_model_interface(self):
        """Test base model abstract interface."""
        # Cannot instantiate abstract base class
        with pytest.raises(TypeError):
            BaseModel()


class TestLogisticRegressionModel:
    """Test logistic regression model wrapper."""
    
    def test_model_initialization(self):
        """Test model initialization with parameters."""
        model = LogisticRegressionModel(C=0.5, max_iter=200)
        
        assert isinstance(model.model, LogisticRegression)
        assert model.model.C == 0.5
        assert model.model.max_iter == 200
        
    def test_fit_predict_workflow(self, sample_features, sample_targets):
        """Test complete fit/predict workflow."""
        model = LogisticRegressionModel()
        
        # Fit model
        model.fit(sample_features, sample_targets)
        
        # Make predictions
        predictions = model.predict(sample_features)
        probabilities = model.predict_proba(sample_features)
        
        assert len(predictions) == len(sample_targets)
        assert len(probabilities) == len(sample_targets)
        assert probabilities.shape[1] == 2  # Binary classification
        
        # Predictions should be 0 or 1
        assert set(predictions).issubset({0, 1})
        
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(probabilities.sum(axis=1), 1.0)
        
    def test_get_feature_importance(self, sample_features, sample_targets):
        """Test feature importance extraction."""
        model = LogisticRegressionModel()
        model.fit(sample_features, sample_targets)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == sample_features.shape[1]
        assert isinstance(importance, np.ndarray)


class TestRandomForestModel:
    """Test random forest model wrapper."""
    
    def test_model_initialization(self):
        """Test model initialization with parameters."""
        model = RandomForestModel(n_estimators=50, max_depth=5)
        
        assert isinstance(model.model, RandomForestClassifier)
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 5
        
    def test_feature_importance_available(self, sample_features, sample_targets):
        """Test that feature importance is available for tree models."""
        model = RandomForestModel()
        model.fit(sample_features, sample_targets)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == sample_features.shape[1]
        assert all(imp >= 0 for imp in importance)  # Should be non-negative


class TestTitanicTrainer:
    """Test training functionality."""
    
    def test_trainer_initialization(self, sample_config, temp_dir):
        """Test trainer initialization."""
        trainer = TitanicTrainer(
            config=sample_config,
            output_dir=temp_dir
        )
        
        assert trainer.config == sample_config
        assert trainer.output_dir == temp_dir
        
    def test_single_fold_training(self, sample_features, sample_targets, sample_config, temp_dir):
        """Test training on a single fold."""
        trainer = TitanicTrainer(
            config=sample_config,
            output_dir=temp_dir
        )
        
        # Create simple train/val split
        split_idx = len(sample_features) // 2
        train_idx = list(range(split_idx))
        val_idx = list(range(split_idx, len(sample_features)))
        
        model = LogisticRegressionModel()
        
        fold_result = trainer._train_fold(
            model=model,
            X=sample_features,
            y=sample_targets,
            train_idx=train_idx,
            val_idx=val_idx,
            fold=0
        )
        
        # Check fold result structure
        assert "fold" in fold_result
        assert "train_score" in fold_result
        assert "val_score" in fold_result
        assert "val_predictions" in fold_result
        
        assert fold_result["fold"] == 0
        assert isinstance(fold_result["train_score"], float)
        assert isinstance(fold_result["val_score"], float)
        
    def test_full_cv_training(self, sample_features, sample_targets, sample_config, temp_dir):
        """Test complete cross-validation training."""
        trainer = TitanicTrainer(
            config=sample_config,
            output_dir=temp_dir
        )
        
        # Use simple fold splitter
        fold_splitter = StratifiedKFoldSplitter(n_splits=3, shuffle=True, random_state=42)
        model = LogisticRegressionModel()
        
        result = trainer.train(
            model=model,
            X=sample_features,
            y=sample_targets,
            fold_splitter=fold_splitter
        )
        
        # Check result structure
        assert "cv_scores" in result
        assert "oof_predictions" in result
        assert "fold_results" in result
        assert "model" in result
        
        # Should have 3 folds
        assert len(result["fold_results"]) == 3
        assert len(result["oof_predictions"]) == len(sample_targets)
        
        # Check artifacts are saved
        expected_files = [
            temp_dir / "model_logistic.joblib",
            temp_dir / "oof_logistic.npy", 
            temp_dir / "cv_scores_logistic.json"
        ]
        
        for file_path in expected_files:
            assert file_path.exists(), f"Expected file {file_path} not found"
            
    def test_training_with_validation_data(self, sample_features, sample_targets, sample_config, temp_dir):
        """Test training with separate validation data."""
        trainer = TitanicTrainer(
            config=sample_config,
            output_dir=temp_dir
        )
        
        # Split data
        split_idx = len(sample_features) // 2
        X_train = sample_features.iloc[:split_idx]
        y_train = sample_targets.iloc[:split_idx]
        X_val = sample_features.iloc[split_idx:]
        y_val = sample_targets.iloc[split_idx:]
        
        fold_splitter = StratifiedKFoldSplitter(n_splits=2, shuffle=True, random_state=42)
        model = LogisticRegressionModel()
        
        result = trainer.train(
            model=model,
            X=X_train,
            y=y_train,
            fold_splitter=fold_splitter,
            X_val=X_val,
            y_val=y_val
        )
        
        # Should include validation scores
        assert "validation_score" in result
        assert isinstance(result["validation_score"], float)
        
    def test_training_report_generation(self, sample_features, sample_targets, sample_config, temp_dir):
        """Test that training generates a report."""
        trainer = TitanicTrainer(
            config=sample_config,
            output_dir=temp_dir
        )
        
        fold_splitter = StratifiedKFoldSplitter(n_splits=2, shuffle=True, random_state=42)
        model = LogisticRegressionModel()
        
        trainer.train(
            model=model,
            X=sample_features,
            y=sample_targets,
            fold_splitter=fold_splitter,
            generate_report=True
        )
        
        # Should create report
        report_file = temp_dir / "report.md"
        assert report_file.exists()
        
        # Report should have some content
        report_content = report_file.read_text()
        assert "Training Report" in report_content
        assert "Cross-Validation Scores" in report_content
