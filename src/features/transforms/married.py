from __future__ import annotations

from typing import Optional
import pandas as pd

from features.transforms.base import BaseTransform


class MarriedTransform(BaseTransform):
    """
    Create a binary feature Is_Married based on the normalized Title column.

    Assumes TitleTransform has already run and produced a normalized 'Title' column.
    Is_Married = 1 if Title == 'Mrs' else 0.
    """

    def __init__(self, title_col: str = "Title", output_col: str = "IsMarried"):
        super().__init__(name="MarriedTransform")
        self.title_col = title_col
        self.output_col = output_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MarriedTransform":
        self._validate_X(X)
        if self.title_col not in X.columns:
            raise ValueError(f"Column '{self.title_col}' not found — ensure TitleTransform runs before MarriedTransform")
        self._set_new_cols([self.output_col])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_X(X)
        X = X.copy()
        if self.title_col not in X.columns:
            # Title missing (unexpected if pipeline order correct); default to 0
            X[self.output_col] = 0
            return X
        titles = X[self.title_col].astype("string").str.strip().str.lower()
        X[self.output_col] = (titles == "mrs").astype(int)
        try:
            cnt = int(X[self.output_col].sum())
            self.logger.info(f"MarriedTransform: Is_Married positives={cnt}/{len(X)}")
        except Exception:
            pass
        return X

