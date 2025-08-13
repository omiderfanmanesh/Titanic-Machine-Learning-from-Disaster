from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class MissingValueIndicatorTransform(BaseTransform):
    """Creates binary indicators for missing values."""

    def __init__(self, columns: Optional[List[str]] = None,
                 missing_threshold: float = 0.01):
        super().__init__()
        self.columns = columns
        self.missing_threshold = missing_threshold
        self.indicator_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MissingValueIndicatorTransform":
        """Identify columns with significant missing values."""
        if self.columns is None:
            missing_pct = X.isnull().mean()
            self.indicator_columns = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        else:
            self.indicator_columns = [col for col in self.columns if col in X.columns]

        self.logger.info(f"Creating missing indicators for: {self.indicator_columns}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create missing value indicators."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        X = X.copy()
        for col in self.indicator_columns:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isnull().astype(int)
        return X

