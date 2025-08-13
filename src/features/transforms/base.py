from __future__ import annotations
from typing import Optional, List, Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.interfaces import ITransformer
from core.utils import LoggerFactory


class BaseTransform(BaseEstimator, TransformerMixin, ITransformer):
    """
    Small, SOLID-friendly base for all feature transforms.

    What you get:
      - logger
      - fitted-state handling & guard
      - consistent feature-name reporting via `_new_cols`
      - light input validation
      - sklearn compatibility (get/set params through BaseEstimator)
    """

    def __init__(self, *, name: Optional[str] = None, validate_input: bool = True):
        self.logger = LoggerFactory.get_logger(name or self.__class__.__name__)
        self.is_fitted: bool = False
        self._new_cols: List[str] = []
        self._validate_input: bool = validate_input

    # ---- public API expected by sklearn/ITransformer ----
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseTransform":
        self._validate_X(X)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_X(X)
        # default: passthrough (subclasses override)
        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        # keep it explicit so subclasses that only override transform still get the guard
        return self.fit(X, y).transform(X)

    # ---- convenience ----
    def get_feature_names(self) -> List[str]:
        """Names of columns this transform creates (if any)."""
        return list(self._new_cols)

    def reset(self) -> None:
        """Clear fitted state and any learned attributes."""
        self.is_fitted = False
        # subclasses can override and also clear their learned attrs

    # ---- helpers for subclasses ----
    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform")

    def _validate_X(self, X: Any) -> None:
        if not self._validate_input:
            return
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__} expects a pandas DataFrame, got {type(X).__name__}"
            )

    def _set_new_cols(self, cols: List[str]) -> None:
        """Call in fit() to declare which new columns will be added."""
        self._new_cols = list(cols)
