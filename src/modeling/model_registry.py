"""Model registry for managing different ML models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from core.interfaces import IModel
from core.utils import LoggerFactory


class BaseModel(IModel):
    """Base model wrapper implementing common functionality."""
    
    def __init__(self, **params):
        self.params = params
        self.model: Optional[BaseEstimator] = None
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.is_fitted = False
    
    @abstractmethod
    def _create_model(self, **params) -> BaseEstimator:
        """Create the actual sklearn model instance."""
        pass
    
    def build(self, config: Dict[str, Any]) -> BaseEstimator:
        """Build model from configuration."""
        model_params = config.get("model_params", {})
        model_params.update(self.params)
        
        self.model = self._create_model(**model_params)
        self.logger.info(f"Built {self.__class__.__name__} with params: {model_params}")
        
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model to training data."""
        if self.model is None:
            self.model = self._create_model(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
        self.logger.info(f"Fitted {self.__class__.__name__} on {len(X)} samples")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self.model.predict(X)
            # Convert to pseudo-probabilities
            proba = np.column_stack([(1 - predictions), predictions])
            return proba
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if not self.is_fitted or self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use coefficient magnitudes
            return np.abs(self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_)
        else:
            return None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump(self, path)
        self.logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """Load model from disk."""
        model_instance = joblib.load(path)
        return model_instance


class LogisticRegressionModel(BaseModel):
    """Logistic regression model wrapper."""
    
    def _create_model(self, **params) -> BaseEstimator:
        """Create logistic regression model."""
        default_params = {
            "random_state": 42,
            "max_iter": 1000,
            "C": 1.0
        }
        default_params.update(params)
        return LogisticRegression(**default_params)


class RandomForestModel(BaseModel):
    """Random forest model wrapper."""
    
    def _create_model(self, **params) -> BaseEstimator:
        """Create random forest model."""
        default_params = {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
        default_params.update(params)
        return RandomForestClassifier(**default_params)


class GradientBoostingModel(BaseModel):
    """Gradient boosting model wrapper."""
    
    def _create_model(self, **params) -> BaseEstimator:
        """Create gradient boosting model."""
        default_params = {
            "n_estimators": 100,
            "random_state": 42,
            "learning_rate": 0.1,
            "max_depth": 3
        }
        default_params.update(params)
        return GradientBoostingClassifier(**default_params)


class SVMModel(BaseModel):
    """Support Vector Machine model wrapper."""
    
    def _create_model(self, **params) -> BaseEstimator:
        """Create SVM model."""
        default_params = {
            "random_state": 42,
            "C": 1.0,
            "probability": True  # Enable probability predictions
        }
        default_params.update(params)
        return SVC(**default_params)


# Optional models that require additional dependencies
try:
    import xgboost as xgb
    
    class XGBoostModel(BaseModel):
        """XGBoost model wrapper."""
        
        def _create_model(self, **params) -> BaseEstimator:
            """Create XGBoost model."""
            default_params = {
                "random_state": 42,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6
            }
            default_params.update(params)
            return xgb.XGBClassifier(**default_params)
    
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


try:
    import catboost as cb
    
    class CatBoostModel(BaseModel):
        """CatBoost model wrapper."""
        
        def _create_model(self, **params) -> BaseEstimator:
            """Create CatBoost model."""
            default_params = {
                "random_state": 42,
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "verbose": False
            }
            default_params.update(params)
            return cb.CatBoostClassifier(**default_params)
    
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


try:
    import lightgbm as lgb
    
    class LightGBMModel(BaseModel):
        """LightGBM model wrapper."""
        
        def _create_model(self, **params) -> BaseEstimator:
            """Create LightGBM model."""
            default_params = {
                "random_state": 42,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "verbose": -1
            }
            default_params.update(params)
            return lgb.LGBMClassifier(**default_params)
    
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelRegistry:
    """Registry for managing available models."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self._models = self._register_models()
    
    def _register_models(self) -> Dict[str, BaseModel]:
        """Register all available models."""
        models = {
            "logistic": LogisticRegressionModel,
            "random_forest": RandomForestModel,
            "gradient_boosting": GradientBoostingModel,
            "svm": SVMModel
        }
        
        # Add optional models if available
        if XGBOOST_AVAILABLE:
            models["xgboost"] = XGBoostModel
        
        if CATBOOST_AVAILABLE:
            models["catboost"] = CatBoostModel
        
        if LIGHTGBM_AVAILABLE:
            models["lightgbm"] = LightGBMModel
        
        return models
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self._models.keys())
    
    def create_model(self, model_name: str, params: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Create model instance by name."""
        if model_name not in self._models:
            available = ", ".join(self.get_available_models())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
        
        model_class = self._models[model_name]
        model_instance = model_class(**(params or {}))
        
        self.logger.info(f"Created {model_name} model")
        return model_instance
    
    def create_model_from_config(self, config: Dict[str, Any]) -> BaseModel:
        """Create model from configuration dictionary."""
        model_name = config.get("model_name", config.get("name"))
        if not model_name:
            raise ValueError("Model name not specified in config")
        
        model_params = config.get("model_params", config.get("params", {}))
        return self.create_model(model_name, model_params)
    
    def register_custom_model(self, name: str, model_class: type) -> None:
        """Register a custom model class."""
        if not issubclass(model_class, BaseModel):
            raise ValueError("Custom model must inherit from BaseModel")
        
        self._models[name] = model_class
        self.logger.info(f"Registered custom model: {name}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = self._models[model_name]
        
        # Try to create a dummy instance to get default parameters
        try:
            dummy_model = model_class()
            dummy_sklearn_model = dummy_model._create_model()
            default_params = dummy_sklearn_model.get_params()
        except:
            default_params = {}
        
        return {
            "name": model_name,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "default_params": default_params
        }


# Global registry instance
_registry = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
