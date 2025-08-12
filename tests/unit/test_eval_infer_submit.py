"""Tests for evaluation and prediction components."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from eval.evaluator import TitanicEvaluator
from infer.predictor import TitanicPredictor, ModelLoader, TTAPredictor
from submit.build_submission import TitanicSubmissionBuilder
from modeling.model_registry import LogisticRegressionModel


class TestTitanicEvaluator:
    """Test model evaluation functionality."""
    
    def test_basic_evaluation(self, sample_targets):
        """Test basic evaluation metrics."""
        evaluator = TitanicEvaluator()
        
        # Create some realistic predictions
        np.random.seed(42)
        oof_predictions = (sample_targets + np.random.normal(0, 0.3, len(sample_targets))).clip(0, 1)
        cv_scores = {"accuracy": [0.8, 0.82, 0.78], "mean": 0.8, "std": 0.016}
        
        result = evaluator.evaluate(
            y_true=sample_targets,
            oof_predictions=oof_predictions,
            cv_scores=cv_scores
        )
        
        # Check required metrics are present
        required_metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        for metric in required_metrics:
            assert metric in result, f"Missing metric: {metric}"
            assert isinstance(result[metric], (int, float))
            assert 0 <= result[metric] <= 1  # All metrics should be between 0 and 1
            
        # Check CV scores are included
        assert "cv_accuracy_mean" in result
        assert "cv_accuracy_std" in result
        assert result["cv_accuracy_mean"] == 0.8
        
    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        evaluator = TitanicEvaluator()
        
        y_true = pd.Series([0, 1, 0, 1, 0])
        oof_predictions = np.array([0.0, 1.0, 0.0, 1.0, 0.0])  # Perfect predictions
        cv_scores = {"accuracy": [1.0, 1.0, 1.0], "mean": 1.0, "std": 0.0}
        
        result = evaluator.evaluate(y_true, oof_predictions, cv_scores)
        
        # Most metrics should be 1.0 for perfect predictions
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        
    def test_random_predictions(self):
        """Test evaluation with random predictions."""
        evaluator = TitanicEvaluator()
        
        np.random.seed(42)
        y_true = pd.Series(np.random.randint(0, 2, 100))
        oof_predictions = np.random.rand(100)  # Random predictions
        cv_scores = {"accuracy": [0.5, 0.48, 0.52], "mean": 0.5, "std": 0.016}
        
        result = evaluator.evaluate(y_true, oof_predictions, cv_scores)
        
        # Metrics should be around 0.5 for random predictions
        assert 0.3 <= result["accuracy"] <= 0.7  # Allow some variation
        assert 0.3 <= result["auc"] <= 0.7
        
    def test_evaluation_report_generation(self, sample_targets, temp_dir):
        """Test evaluation report generation."""
        evaluator = TitanicEvaluator()
        
        np.random.seed(42)
        oof_predictions = (sample_targets + np.random.normal(0, 0.2, len(sample_targets))).clip(0, 1)
        cv_scores = {"accuracy": [0.8, 0.82, 0.78], "mean": 0.8, "std": 0.016}
        
        result = evaluator.evaluate(
            y_true=sample_targets,
            oof_predictions=oof_predictions,
            cv_scores=cv_scores,
            output_dir=temp_dir,
            generate_report=True
        )
        
        # Check report file was created
        report_file = temp_dir / "evaluation_report.md"
        assert report_file.exists()
        
        # Check report content
        report_content = report_file.read_text()
        assert "Model Evaluation Report" in report_content
        assert "Performance Metrics" in report_content
        assert "Cross-Validation Scores" in report_content


class TestTitanicPredictor:
    """Test prediction functionality."""
    
    def test_single_model_prediction(self, sample_features):
        """Test prediction with single model."""
        predictor = TitanicPredictor()
        
        # Create and train a simple model
        model = LogisticRegressionModel()
        y_dummy = np.random.randint(0, 2, len(sample_features))
        model.fit(sample_features, y_dummy)
        
        predictions = predictor.predict(model, sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [0, 1] for pred in predictions)
        
    def test_prediction_probabilities(self, sample_features):
        """Test probability prediction."""
        predictor = TitanicPredictor()
        
        # Create and train a model
        model = LogisticRegressionModel()
        y_dummy = np.random.randint(0, 2, len(sample_features))
        model.fit(sample_features, y_dummy)
        
        probabilities = predictor.predict_proba(model, sample_features)
        
        assert len(probabilities) == len(sample_features)
        assert probabilities.shape[1] == 2  # Binary classification
        
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(probabilities.sum(axis=1), 1.0)
        
        # All probabilities should be between 0 and 1
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()
        
    def test_ensemble_prediction(self, sample_features):
        """Test ensemble prediction with multiple models."""
        predictor = TitanicPredictor()
        
        # Create multiple models
        models = []
        for i in range(3):
            model = LogisticRegressionModel(random_state=i)
            y_dummy = np.random.randint(0, 2, len(sample_features))
            model.fit(sample_features, y_dummy)
            models.append(model)
        
        predictions = predictor.predict_ensemble(models, sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [0, 1] for pred in predictions)


class TestModelLoader:
    """Test model loading functionality."""
    
    def test_load_single_model(self, sample_features, temp_dir):
        """Test loading a single saved model."""
        import joblib
        
        # Create and save a model
        model = LogisticRegressionModel()
        y_dummy = np.random.randint(0, 2, len(sample_features))
        model.fit(sample_features, y_dummy)
        
        model_file = temp_dir / "test_model.joblib"
        joblib.dump(model, model_file)
        
        # Load model
        loader = ModelLoader()
        loaded_model = loader.load_model(model_file)
        
        # Test that loaded model works
        predictions = loaded_model.predict(sample_features)
        assert len(predictions) == len(sample_features)
        
    def test_load_multiple_models(self, sample_features, temp_dir):
        """Test loading multiple models."""
        import joblib
        
        # Create and save multiple models
        model_files = []
        for i in range(3):
            model = LogisticRegressionModel(random_state=i)
            y_dummy = np.random.randint(0, 2, len(sample_features))
            model.fit(sample_features, y_dummy)
            
            model_file = temp_dir / f"model_{i}.joblib"
            joblib.dump(model, model_file)
            model_files.append(model_file)
        
        # Load models
        loader = ModelLoader()
        models = loader.load_models(model_files)
        
        assert len(models) == 3
        
        # All models should work
        for model in models:
            predictions = model.predict(sample_features)
            assert len(predictions) == len(sample_features)


class TestTitanicSubmissionBuilder:
    """Test submission building functionality."""
    
    def test_basic_submission_building(self, temp_dir):
        """Test basic submission creation."""
        builder = TitanicSubmissionBuilder()
        
        passenger_ids = pd.Series([892, 893, 894, 895])
        predictions = np.array([0, 1, 0, 1])
        
        submission = builder.build_submission(
            passenger_ids=passenger_ids,
            predictions=predictions,
            output_file=temp_dir / "test_submission.csv"
        )
        
        # Check submission format
        assert len(submission) == len(passenger_ids)
        assert list(submission.columns) == ["PassengerId", "Survived"]
        
        # Check values
        pd.testing.assert_series_equal(submission["PassengerId"], passenger_ids)
        pd.testing.assert_series_equal(
            submission["Survived"], 
            pd.Series(predictions, name="Survived")
        )
        
        # Check file was saved
        submission_file = temp_dir / "test_submission.csv"
        assert submission_file.exists()
        
        # Check saved file content
        saved_submission = pd.read_csv(submission_file)
        pd.testing.assert_frame_equal(saved_submission, submission)
        
    def test_submission_validation(self, temp_dir):
        """Test submission validation."""
        builder = TitanicSubmissionBuilder()
        
        passenger_ids = pd.Series([892, 893, 894])
        
        # Test invalid predictions (not 0 or 1)
        invalid_predictions = np.array([0, 1, 2])  # 2 is invalid
        
        with pytest.raises(ValueError, match="must be 0 or 1"):
            builder.build_submission(passenger_ids, invalid_predictions)
            
        # Test mismatched lengths
        predictions = np.array([0, 1])  # One less than passenger_ids
        
        with pytest.raises(ValueError, match="Length mismatch"):
            builder.build_submission(passenger_ids, predictions)
            
    def test_ensemble_submission_building(self, temp_dir):
        """Test building submission from ensemble predictions."""
        builder = TitanicSubmissionBuilder()
        
        passenger_ids = pd.Series([892, 893, 894, 895])
        
        # Multiple model predictions (probabilities)
        ensemble_predictions = np.array([
            [0.3, 0.7],  # Prediction: 1
            [0.6, 0.4],  # Prediction: 0
            [0.2, 0.8],  # Prediction: 1
            [0.9, 0.1]   # Prediction: 0
        ])
        
        submission = builder.build_ensemble_submission(
            passenger_ids=passenger_ids,
            ensemble_predictions=ensemble_predictions,
            output_file=temp_dir / "ensemble_submission.csv"
        )
        
        # Check predictions match expected (argmax)
        expected_predictions = [1, 0, 1, 0]
        assert submission["Survived"].tolist() == expected_predictions
        
    def test_submission_metadata(self, temp_dir):
        """Test submission with metadata."""
        builder = TitanicSubmissionBuilder()
        
        passenger_ids = pd.Series([892, 893])
        predictions = np.array([0, 1])
        
        metadata = {
            "model_name": "test_model",
            "cv_score": 0.85,
            "timestamp": "2024-01-01T12:00:00"
        }
        
        submission = builder.build_submission(
            passenger_ids=passenger_ids,
            predictions=predictions,
            output_file=temp_dir / "submission_with_metadata.csv",
            metadata=metadata
        )
        
        # Metadata should not affect the submission dataframe
        assert list(submission.columns) == ["PassengerId", "Survived"]
        
        # But metadata file should be created
        metadata_file = temp_dir / "submission_with_metadata_metadata.json"
        assert metadata_file.exists()
        
        # Check metadata content
        import json
        with open(metadata_file) as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["model_name"] == "test_model"
        assert saved_metadata["cv_score"] == 0.85
