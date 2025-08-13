from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

from core.interfaces import ImputationStrategy, EstimatorFactory, FeaturePreprocessor

class ModelImputerStrategy(ImputationStrategy):
    """Model-based imputer for a single numeric column."""
    def __init__(
        self,
        column: str,
        plan: Dict[str, Any],
        estimator_factory: EstimatorFactory,
        preprocessor: FeaturePreprocessor,
        debug: bool = False,
    ):
        super().__init__(column, plan)
        self.est_factory = estimator_factory
        self.preprocessor = preprocessor
        self.debug = debug

        self.features: List[str] = plan.get("features", [])
        self.pipe: Optional[Pipeline] = None
        self._feats_present: List[str] = []
        self.report: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        mask = X[self.column].notna()
        if mask.sum() == 0:
            return

        pre, feats_present = self.preprocessor.build(X, self.features or [c for c in X.columns if c != self.column])
        self._feats_present = feats_present
        est = self.est_factory.make(self.plan)
        self.pipe = Pipeline([("pre", pre), ("model", est)])
        self.pipe.fit(X.loc[mask, feats_present], X.loc[mask, self.column])

        if self.debug:
            kf = KFold(n_splits=min(5, max(2, int(mask.sum() // 20))), shuffle=True, random_state=42)
            r2 = cross_val_score(self.pipe, X.loc[mask, feats_present], X.loc[mask, self.column], cv=kf, scoring="r2")
            mae = -cross_val_score(self.pipe, X.loc[mask, feats_present], X.loc[mask, self.column], cv=kf, scoring="neg_mean_absolute_error")
            self.report.update({
                "cv_r2_mean": float(np.mean(r2)),
                "cv_r2_std": float(np.std(r2)),
                "cv_mae_mean": float(np.mean(mae)),
                "cv_mae_std": float(np.std(mae)),
            })

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipe is None:
            return X
        X = X.copy()
        mask = X[self.column].isna()
        if not mask.any():
            return X
        feats = [f for f in self._feats_present if f in X.columns]
        if not feats:
            return X
        X.loc[mask, self.column] = self.pipe.predict(X.loc[mask, feats])
        return X
