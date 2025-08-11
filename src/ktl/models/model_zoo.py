from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
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
        if key in {"gbc", "gradient_boosting"}:
            # GradientBoostingClassifier handles interactions, deterministic by default
            default = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}
            return GradientBoostingClassifier(**{**default, **params})
        if key in {"hgb", "histgb", "hist_gradient_boosting"}:
            # Fast tree boosting; handles missing values natively
            default = {"learning_rate": 0.06, "max_depth": 3, "l2_regularization": 0.0, "random_state": 42}
            return HistGradientBoostingClassifier(**{**default, **params})
        if key in {"lgbm", "xgb", "catboost"}:
            # Optional libraries: provide graceful imports with helpful errors
            if key == "xgb":
                try:
                    from xgboost import XGBClassifier  # type: ignore
                except Exception as e:  # pragma: no cover
                    raise TrainingError(
                        "XGBoost not installed. Install with `pip install xgboost`"
                    ) from e
                default = dict(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    tree_method="hist",
                    eval_metric="logloss",
                    n_jobs=-1,
                    use_label_encoder=False,
                    verbosity=0,  # Suppress warnings
                )
                return XGBClassifier(**{**default, **params})
            if key == "catboost":
                try:
                    from catboost import CatBoostClassifier  # type: ignore
                except Exception as e:  # pragma: no cover
                    raise TrainingError(
                        "CatBoost not installed. Install with `pip install catboost`"
                    ) from e
                default = dict(
                    iterations=800,
                    depth=4,
                    learning_rate=0.05,
                    l2_leaf_reg=3.0,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=42,
                    verbose=False,
                )
                return CatBoostClassifier(**{**default, **params})
            # LightGBM can be added similarly if desired
            raise TrainingError(f"Optional model '{name}' not supported in this build.")
        raise TrainingError(f"Unknown model name: {name}")
