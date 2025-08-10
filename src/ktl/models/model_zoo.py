from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

from ktl.models.metrics import TaskType
from ktl.utils.exceptions import TrainingError


class ModelFactory:
    """Factory for instantiating models by name.

    Supported core models: 'logistic' (binary), 'ridge' (regression).
    Optional: 'lgbm', 'xgb', 'catboost' (require extras installed).
    """

    @staticmethod
    def make(name: str, task: TaskType, params: Dict[str, Any]) -> BaseEstimator:
        key = name.lower()
        if key == "logistic":
            return LogisticRegression(**{**{"max_iter": 1000, "n_jobs": -1}, **params})
        if key == "ridge":
            return Ridge(**params)
        if key in {"rf", "random_forest", "randomforest"}:
            if task != "binary":
                return RandomForestClassifier(**{**{"n_estimators": 300, "random_state": 42, "n_jobs": -1}, **params})
            return RandomForestClassifier(**{**{"n_estimators": 300, "random_state": 42, "n_jobs": -1}, **params})
        if key in {"lgbm", "xgb", "catboost"}:
            raise TrainingError(
                f"Model '{name}' requires optional dependency. Install extras: `pip install -e .[extras]`"
            )
        raise TrainingError(f"Unknown model name: {name}")
