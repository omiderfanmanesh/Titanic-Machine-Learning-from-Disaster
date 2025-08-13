"""Core interfaces for the Titanic ML pipeline following SOLID principles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator


class IDataLoader(ABC):
    """Interface for data loading components."""
    
    @abstractmethod
    def load(self, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load data from source."""
        pass
    
    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate that data conforms to expected schema."""
        pass


class ITransformer(ABC):
    """Interface for data transformation components."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ITransformer":
        """Fit transformer to training data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data."""
        pass
    
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        pass


class IModel(ABC):
    """Interface for ML models."""
    
    @abstractmethod
    def build(self, config: Dict[str, Any]) -> BaseEstimator:
        """Build model from configuration."""
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "IModel":
        """Load model from disk."""
        pass


class ITrainer(ABC):
    """Interface for model training components."""
    
    @abstractmethod
    def fit(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, 
            config: Dict[str, Any]) -> Dict[str, Any]:
        """Train model and return training artifacts."""
        pass
    
    @abstractmethod
    def cross_validate(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                      cv_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-validation."""
        pass
    
    @abstractmethod
    def best_checkpoint(self) -> str:
        """Return path to best model checkpoint."""
        pass


class IEvaluator(ABC):
    """Interface for model evaluation components."""
    
    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, 
                config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate predictions and return metrics."""
        pass
    
    @abstractmethod
    def evaluate_cv(self, oof_predictions: pd.Series, y_true: pd.Series,
                   fold_scores: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate cross-validation results."""
        pass


class IPredictor(ABC):
    """Interface for inference components."""
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, models: List[BaseEstimator],
               config: Dict[str, Any]) -> pd.DataFrame:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, models: List[BaseEstimator],
                     config: Dict[str, Any]) -> pd.DataFrame:
        """Generate probability predictions."""
        pass

    def _resolve_threshold(self, inference_cfg):
        pass


class IFoldSplitter(ABC):
    """Interface for cross-validation fold splitting."""
    
    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series, 
             groups: Optional[pd.Series] = None) -> List[Tuple[List[int], List[int]]]:
        """Generate train/validation splits."""
        pass
    
    @abstractmethod
    def get_n_splits(self) -> int:
        """Return number of splits."""
        pass


class ISubmissionBuilder(ABC):
    """Interface for building Kaggle submissions."""
    
    @abstractmethod
    def build_submission(self, predictions: pd.DataFrame, 
                        config: Dict[str, Any]) -> pd.DataFrame:
        """Build submission file from predictions."""
        pass
    
    @abstractmethod
    def validate_submission(self, submission: pd.DataFrame) -> bool:
        """Validate submission format."""
        pass


class IDataValidator(ABC):
    """Interface for data validation components."""
    
    @abstractmethod
    def validate_train(self, df: pd.DataFrame) -> bool:
        """Validate training data."""
        pass
    
    @abstractmethod
    def validate_test(self, df: pd.DataFrame) -> bool:
        """Validate test data."""
        pass
    
    @abstractmethod
    def validate_consistency(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate consistency between train and test data."""
        pass


class ICache(ABC):
    """Interface for caching components."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass


class IEncoderStrategy:
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "IEncoderStrategy":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def output_columns(self) -> List[str]:
        raise NotImplementedError
