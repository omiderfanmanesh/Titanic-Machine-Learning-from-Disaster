from __future__ import annotations
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from core.interfaces import EstimatorFactory

class DefaultEstimatorFactory(EstimatorFactory):
    """DIP: Orchestrator depends on this abstraction, not concrete estimators."""
    def make(self, plan: Dict[str, Any]):
        name = str(plan.get("estimator", "random_forest")).lower()
        if name == "random_forest":
            return RandomForestRegressor(
                n_estimators=int(plan.get("n_estimators", 300)),
                max_depth=plan.get("max_depth", None),
                random_state=plan.get("random_state", 42),
                n_jobs=-1,
            )
        if name == "linear":
            return Ridge(alpha=float(plan.get("alpha", 1.0)), random_state=plan.get("random_state", 42))

        if name == "xgboost":
            from xgboost import XGBRegressor  # optional dependency
            return XGBRegressor(
                n_estimators=int(plan.get("n_estimators", 400)),
                max_depth=plan.get("max_depth", 6),
                learning_rate=float(plan.get("learning_rate", 0.05)),
                subsample=float(plan.get("subsample", 0.8)),
                colsample_bytree=float(plan.get("colsample_bytree", 0.8)),
                random_state=plan.get("random_state", 42),
                n_jobs=-1,
            )
        if name == "lightgbm":
            from lightgbm import LGBMRegressor  # optional dependency
            return LGBMRegressor(
                n_estimators=int(plan.get("n_estimators", 400)),
                max_depth=plan.get("max_depth", -1),
                learning_rate=float(plan.get("learning_rate", 0.05)),
                subsample=float(plan.get("subsample", 0.8)),
                colsample_bytree=float(plan.get("colsample_bytree", 0.8)),
                random_state=plan.get("random_state", 42),
                n_jobs=-1,
            )
        raise ValueError(f"Unsupported estimator '{name}'")
