"""Inference and prediction components with ensembling support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from core.interfaces import IPredictor
from core.utils import LoggerFactory


class TitanicPredictor(IPredictor):
    """Main predictor for Titanic competition with ensemble support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)
        
    def predict(self, X: pd.DataFrame, models: List[BaseEstimator],
               config: Dict[str, Any]) -> pd.DataFrame:
        """Generate binary predictions."""
        predictions_proba = self.predict_proba(X, models, config)
        
        threshold = config.get("threshold", 0.5)
        predictions_binary = (predictions_proba["prediction"] >= threshold).astype(int)
        
        result = pd.DataFrame({
            "PassengerId": X.index,
            "prediction": predictions_binary,
            "prediction_proba": predictions_proba["prediction"]
        })
        
        return result
    
    def predict_proba(self, X: pd.DataFrame, models: List[BaseEstimator],
                     config: Dict[str, Any]) -> pd.DataFrame:
        """Generate probability predictions with ensembling."""
        
        if not models:
            raise ValueError("No models provided for prediction")
            
        self.logger.info(f"Making predictions with {len(models)} models")
        
        # Collect predictions from all models
        all_predictions = []
        
        for i, model in enumerate(models):
            try:
                pred = self._predict_single_model(model, X)
                all_predictions.append(pred)
                self.logger.debug(f"Model {i+1} prediction range: "
                                f"{pred.min():.3f} - {pred.max():.3f}")
            except Exception as e:
                self.logger.error(f"Model {i+1} prediction failed: {e}")
                continue
        
        if not all_predictions:
            raise RuntimeError("All model predictions failed")
            
        # Ensemble predictions
        ensemble_method = config.get("ensemble_method", "average")
        ensemble_weights = config.get("ensemble_weights")
        
        final_predictions = self._ensemble_predictions(
            all_predictions, ensemble_method, ensemble_weights
        )
        
        result = pd.DataFrame({
            "PassengerId": X.index,
            "prediction": final_predictions
        })
        
        self.logger.info(f"Generated predictions for {len(result)} samples")
        
        return result
    
    def _predict_single_model(self, model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from a single model."""
        try:
            # Try predict_proba first (for classifiers)
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
        except AttributeError:
            # Fall back to decision_function
            try:
                return model.decision_function(X)
            except AttributeError:
                # Fall back to predict
                return model.predict(X)
    
    def _ensemble_predictions(self, predictions: List[np.ndarray],
                            method: str, weights: Optional[List[float]] = None) -> np.ndarray:
        """Ensemble multiple predictions."""
        
        if method == "average":
            if weights is not None:
                if len(weights) != len(predictions):
                    self.logger.warning("Weights length mismatch, using equal weights")
                    weights = None
                else:
                    weights = np.array(weights) / np.sum(weights)  # Normalize
                    
            if weights is None:
                return np.mean(predictions, axis=0)
            else:
                return np.average(predictions, axis=0, weights=weights)
                
        elif method == "rank_average":
            # Convert to ranks and average
            ranked_predictions = []
            for pred in predictions:
                ranks = pd.Series(pred).rank(pct=True).values
                ranked_predictions.append(ranks)
            return np.mean(ranked_predictions, axis=0)
            
        elif method == "geometric_mean":
            # Geometric mean of probabilities
            predictions_array = np.array(predictions)
            # Clip to avoid log(0)
            predictions_array = np.clip(predictions_array, 1e-8, 1 - 1e-8)
            return np.exp(np.mean(np.log(predictions_array), axis=0))
            
        elif method == "median":
            return np.median(predictions, axis=0)
            
        elif method == "max":
            return np.max(predictions, axis=0)
            
        elif method == "min":
            return np.min(predictions, axis=0)
            
        else:
            self.logger.warning(f"Unknown ensemble method: {method}, using average")
            return np.mean(predictions, axis=0)


class ModelLoader:
    """Utility for loading trained models from various sources."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
    
    def load_models_from_paths(self, model_paths: List[Union[str, Path]]) -> List[BaseEstimator]:
        """Load models from file paths."""
        models = []
        
        for path in model_paths:
            try:
                model = joblib.load(path)
                models.append(model)
                self.logger.info(f"Loaded model from {path}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {path}: {e}")
                
        if not models:
            raise RuntimeError("No models successfully loaded")
            
        return models
    
    def load_fold_models(self, run_dir: Union[str, Path]) -> List[BaseEstimator]:
        """Load all fold models from a training run directory."""
        run_dir = Path(run_dir)
        
        fold_model_paths = sorted(run_dir.glob("fold_*_model.joblib"))
        
        if not fold_model_paths:
            raise FileNotFoundError(f"No fold models found in {run_dir}")
            
        model_paths = [str(p) for p in fold_model_paths]
        return self.load_models_from_paths(model_paths)
    
    def load_best_model(self, run_dir: Union[str, Path]) -> BaseEstimator:
        """Load the best model from a training run."""
        run_dir = Path(run_dir)
        
        # Try to find CV scores to identify best model
        scores_path = run_dir / "cv_scores.json"
        
        if scores_path.exists():
            import json
            with open(scores_path, "r") as f:
                scores_data = json.load(f)
                
            fold_scores = scores_data.get("fold_scores", [])
            if fold_scores:
                best_fold = np.argmax(fold_scores)
                best_model_path = run_dir / f"fold_{best_fold}_model.joblib"
                
                if best_model_path.exists():
                    return joblib.load(best_model_path)
        
        # Fallback: load first fold model
        fold_models = sorted(run_dir.glob("fold_*_model.joblib"))
        if fold_models:
            return joblib.load(fold_models[0])
            
        raise FileNotFoundError(f"No models found in {run_dir}")


class TTAPredictor(TitanicPredictor):
    """Test-Time Augmentation predictor (for future extension)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def predict_proba(self, X: pd.DataFrame, models: List[BaseEstimator],
                     config: Dict[str, Any]) -> pd.DataFrame:
        """Predict with test-time augmentation."""
        
        tta_rounds = config.get("tta_rounds", 5)
        
        if tta_rounds <= 1 or not config.get("use_tta", False):
            # No TTA, use parent method
            return super().predict_proba(X, models, config)
        
        self.logger.info(f"Using TTA with {tta_rounds} rounds")
        
        all_tta_predictions = []
        
        for round_idx in range(tta_rounds):
            # For tabular data, TTA might involve:
            # - Adding small noise to numerical features
            # - Randomly dropping some features
            # - etc.
            
            X_augmented = self._augment_data(X, round_idx, config)
            round_predictions = super().predict_proba(X_augmented, models, config)
            all_tta_predictions.append(round_predictions["prediction"].values)
            
        # Average TTA predictions
        final_predictions = np.mean(all_tta_predictions, axis=0)
        
        result = pd.DataFrame({
            "PassengerId": X.index,
            "prediction": final_predictions
        })
        
        return result
    
    def _augment_data(self, X: pd.DataFrame, round_idx: int, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply test-time augmentation to data."""
        # For now, just return original data
        # In future, could implement:
        # - Gaussian noise addition to numerical features
        # - Random feature dropout
        # - etc.
        
        np.random.seed(round_idx)  # Ensure reproducible augmentation
        
        X_aug = X.copy()
        
        # Example: add small gaussian noise to numerical columns
        noise_scale = config.get("tta_noise_scale", 0.01)
        numeric_cols = X_aug.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != "PassengerId":  # Don't augment ID column
                noise = np.random.normal(0, noise_scale * X_aug[col].std(), len(X_aug))
                X_aug[col] = X_aug[col] + noise
        
        return X_aug


def create_predictor(config: Dict[str, Any]) -> IPredictor:
    """Factory function to create predictor."""
    
    if config.get("use_tta", False):
        return TTAPredictor(config)
    else:
        return TitanicPredictor(config)
