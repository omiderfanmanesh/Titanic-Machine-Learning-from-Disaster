from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from features.transforms.base import BaseTransform


class FareTransform(BaseTransform):
    """
    - Replaces Fare == 0 with the mean Fare per Pclass (learned on fit).
    - Mean can be computed excluding zeros (default) or including them.
    - Optionally applies log1p transform.
    """

    def __init__(
        self,
        fare_col: str = "Fare",
        class_col: str = "Pclass",
        log_transform: bool = False,
        output_col: str = "Fare_Transformed",
        exclude_zero_in_mean: bool = True,
    ):
        super().__init__()
        self.fare_col = fare_col
        self.class_col = class_col
        self.log_transform = log_transform
        self.exclude_zero_in_mean = exclude_zero_in_mean
        self._means: Optional[pd.Series] = None
        self.output_col = output_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FareTransform":
        if self.fare_col not in X.columns:
            raise ValueError(f"Column {self.fare_col} not found")
        if self.class_col not in X.columns:
            raise ValueError(f"Column {self.class_col} not found")

        fares = pd.to_numeric(X[self.fare_col], errors="coerce")
        grp = X[self.class_col]

        if self.exclude_zero_in_mean:
            mask = fares > 0
            self._means = fares[mask].groupby(grp[mask]).mean()
        else:
            self._means = fares.groupby(grp).mean()

        self.is_fitted = True
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        X = X.copy()
        if self._means is None:
            return X

        fares = pd.to_numeric(X[self.fare_col], errors="coerce")
        classes = X[self.class_col]

        # Replace Fare == 0 with class mean (vectorized)
        mask = (fares == 0) & (classes.isin(self._means.index))
        fares = fares.copy()
        fares[mask] = classes[mask].map(self._means)

        # Optional log1p
        if self.log_transform:
            # Guard against negatives (clip at 0 for stability)
            fares = fares.clip(lower=0)
            fares = np.log1p(fares)

        X[self.output_col] = fares
        return X