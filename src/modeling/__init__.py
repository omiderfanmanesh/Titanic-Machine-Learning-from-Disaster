"""Model components including registry and trainers."""

from .model_registry import ModelRegistry, BaseModel
from .trainers import TitanicTrainer

# Import specific model classes
try:
    from .model_registry import (
        LogisticRegressionModel,
        RandomForestModel,
        GradientBoostingModel,
        SVMModel
    )
except ImportError:
    pass

__all__ = [
    "ModelRegistry",
    "BaseModel", 
    "TitanicTrainer",
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "SVMModel"
]
