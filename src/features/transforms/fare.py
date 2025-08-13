from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class FareTransform(BaseTransform):
    """Transforms fare values with log transformation and missing value handling."""

    def __init__(self, fare_col: str = "Fare", log_transform: bool = False):
        super().__init__()
        self.fare_col = fare_col
        self.log_transform = log_transform
        self.median_fare: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FareTransform":
        if self.fare_col not in X.columns:
            raise ValueError(f"Column {self.fare_col} not found")

        self.median_fare = X[self.fare_col].median()
        self.logger.info(f"Learned median fare: {self.median_fare:.2f}")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        X = X.copy()
        X[self.fare_col] = X[self.fare_col].fillna(self.median_fare)
        if self.log_transform:
            X[f"{self.fare_col}_log"] = np.log1p(X[self.fare_col])
        return X

