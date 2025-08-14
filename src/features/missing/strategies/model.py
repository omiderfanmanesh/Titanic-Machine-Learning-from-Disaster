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

        # Control preprocessing to avoid conflicts with orchestrator
        self.skip_imputation_in_preprocessor: bool = plan.get("skip_imputation_in_preprocessor", True)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        mask = X[self.column].notna()
        if mask.sum() == 0:
            return

        # Use only complete cases for fitting to avoid circular dependencies
        X_complete = X.loc[mask].copy()

        # Get feature list
        available_features = self.features or [c for c in X.columns if c != self.column]
        self._feats_present = [f for f in available_features if f in X.columns]

        if not self._feats_present:
            return

        # Build preprocessor with controls to avoid double-processing
        preprocessor_config = {
            "skip_missing_imputation": self.skip_imputation_in_preprocessor,
            "minimal_preprocessing": True  # Use minimal preprocessing to avoid conflicts
        }

        # Create a lightweight preprocessor or use the existing one carefully
        if hasattr(self.preprocessor, 'build_minimal'):
            pre, feats_final = self.preprocessor.build_minimal(X_complete, self._feats_present, preprocessor_config)
        else:
            # Fallback: use existing build but on complete data only
            pre, feats_final = self.preprocessor.build(X_complete, self._feats_present)

        self._feats_present = feats_final

        est = self.est_factory.make(self.plan)
        self.pipe = Pipeline([("pre", pre), ("model", est)])

        # Fit on complete cases only
        y_target = X_complete[self.column]
        self.pipe.fit(X_complete[self._feats_present], y_target)

        if self.debug:
            try:
                n_samples = len(y_target)
                n_splits = min(5, max(2, n_samples // 20))
                if n_splits >= 2 and n_samples >= n_splits:
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    r2 = cross_val_score(self.pipe, X_complete[self._feats_present], y_target, cv=kf, scoring="r2")
                    mae = -cross_val_score(self.pipe, X_complete[self._feats_present], y_target, cv=kf, scoring="neg_mean_absolute_error")
                    self.report.update({
                        "cv_r2_mean": float(np.mean(r2)),
                        "cv_r2_std": float(np.std(r2)),
                        "cv_mae_mean": float(np.mean(mae)),
                        "cv_mae_std": float(np.std(mae)),
                        "n_train_samples": n_samples
                    })
                else:
                    self.report.update({
                        "cv_r2_mean": None,
                        "cv_r2_std": None,
                        "cv_mae_mean": None,
                        "cv_mae_std": None,
                        "n_train_samples": n_samples,
                        "note": "Insufficient samples for cross-validation"
                    })
            except Exception as e:
                self.report.update({
                    "cv_error": str(e),
                    "n_train_samples": len(y_target)
                })

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipe is None:
            return X
        X = X.copy()
        mask = X[self.column].isna()
        if not mask.any():
            return X

        # Ensure all required features are present
        feats = [f for f in self._feats_present if f in X.columns]
        if not feats:
            return X

        # Only predict for rows with missing values in target column
        X_missing = X.loc[mask, feats]
        if len(X_missing) > 0:
            try:
                predictions = self.pipe.predict(X_missing)
                X.loc[mask, self.column] = predictions
            except Exception as e:
                # Log the error but don't fail the entire pipeline
                print(f"Warning: Model imputation failed for column {self.column}: {e}")

        return X
