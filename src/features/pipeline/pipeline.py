from __future__ import annotations
from typing import List, Optional, Dict, Any
import time
import pandas as pd

from features.transforms.base import BaseTransform  # already inherits BaseEstimator, TransformerMixin


class FeaturePipeline(BaseTransform):
    """
    Chains multiple feature transforms.

    Extras:
      - sklearn-compatible via BaseTransform
      - per-step timing & error context
      - optional schema & dtype validation between fit/transform
      - frozen output-column list after fit for sanity checks
    """

    def __init__(
        self,
        transforms: List[BaseTransform],
        strict_schema: bool = False,
        validate_output_dtypes: bool = False,
    ):
        super().__init__(name="FeaturePipeline")
        self.transforms = list(transforms)
        self.strict_schema = strict_schema
        self.validate_output_dtypes = validate_output_dtypes

        self._fitted_columns: Optional[List[str]] = None
        self._fitted_dtypes: Optional[Dict[str, Any]] = None

    # ---- sklearn plumbing (optional overrides) ----
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "transforms": self.transforms,
            "strict_schema": self.strict_schema,
            "validate_output_dtypes": self.validate_output_dtypes,
        }
        if deep:
            for i, t in enumerate(self.transforms):
                params[f"transform__{i}"] = t
        return params

    def set_params(self, **params):
        if "transforms" in params:
            self.transforms = list(params.pop("transforms"))
        # allow setting individual steps: transform__0=..., transform__1=...
        for k in list(params.keys()):
            if k.startswith("transform__"):
                idx = int(k.split("__", 1)[1])
                self.transforms[idx] = params.pop(k)
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- core API ----
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeaturePipeline":
        self._validate_X(X)
        Xc = X.copy()

        for i, t in enumerate(self.transforms):
            name = type(t).__name__
            t0 = time.time()
            try:
                self.logger.debug(f"[{i+1}/{len(self.transforms)}] Fitting {name}")
                t.fit(Xc, y)
                Xc = t.transform(Xc)
            except Exception as e:
                raise RuntimeError(f"FeaturePipeline.fit failed at step {i} ({name}): {e}") from e
            finally:
                self.logger.debug(f"[{i+1}/{len(self.transforms)}] {name} fit+transform in {time.time()-t0:.3f}s")

            if self.strict_schema and not isinstance(Xc, pd.DataFrame):
                raise TypeError(f"{name} did not return a pandas DataFrame.")

        # freeze schema
        self._fitted_columns = list(Xc.columns)
        if self.validate_output_dtypes:
            self._fitted_dtypes = {c: Xc[c].dtype for c in Xc.columns}

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_X(X)
        Xc = X.copy()

        for i, t in enumerate(self.transforms):
            name = type(t).__name__
            t0 = time.time()
            try:
                Xc = t.transform(Xc)
            except Exception as e:
                raise RuntimeError(f"FeaturePipeline.transform failed at step {i} ({name}): {e}") from e
            finally:
                self.logger.debug(f"[{i+1}/{len(self.transforms)}] {name} transform in {time.time()-t0:.3f}s")

            if self.strict_schema and not isinstance(Xc, pd.DataFrame):
                raise TypeError(f"{name} did not return a pandas DataFrame.")

        # optional checks vs. fit schema
        if self.strict_schema and self._fitted_columns is not None:
            missing = [c for c in self._fitted_columns if c not in Xc.columns]
            if missing:
                raise ValueError(f"FeaturePipeline.transform: columns missing compared to fit(): {missing}")

        if self.validate_output_dtypes and self._fitted_dtypes is not None:
            for c, dt in self._fitted_dtypes.items():
                if c in Xc.columns and Xc[c].dtype != dt:
                    self.logger.debug(f"Column '{c}': dtype changed {dt} -> {Xc[c].dtype}")

        return Xc

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        # mirror fit() to keep single-pass behavior (some steps add columns for later ones)
        self._validate_X(X)
        Xc = X.copy()

        for i, t in enumerate(self.transforms):
            name = type(t).__name__
            t0 = time.time()
            try:
                self.logger.debug(f"[{i+1}/{len(self.transforms)}] Fitting {name}")
                t.fit(Xc, y)
                Xc = t.transform(Xc)
            except Exception as e:
                raise RuntimeError(f"FeaturePipeline.fit_transform failed at step {i} ({name}): {e}") from e
            finally:
                self.logger.debug(f"[{i+1}/{len(self.transforms)}] {name} fit+transform in {time.time()-t0:.3f}s")

        self._fitted_columns = list(Xc.columns)
        if self.validate_output_dtypes:
            self._fitted_dtypes = {c: Xc[c].dtype for c in Xc.columns}
        self.is_fitted = True
        return Xc
