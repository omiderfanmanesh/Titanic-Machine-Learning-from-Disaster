from __future__ import annotations

import re
import string
from typing import Optional

import pandas as pd

from features.transforms.base import BaseTransform


class FamilyTransform(BaseTransform):
    """
    Extract a cleaned family (surname) string from the raw 'Name' column.

    Logic (vectorized):
      - Remove text in parentheses (maiden names), if any
      - Take substring before the comma as the family name
      - Remove punctuation characters and trim whitespace

    Output column: 'Family'
    """

    def __init__(self, name_col: str = "Name", output_col: str = "Family"):
        super().__init__(name="FamilyTransform")
        self.name_col = name_col
        self.output_col = output_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FamilyTransform":
        self._validate_X(X)
        if self.name_col not in X.columns:
            raise ValueError(f"Column '{self.name_col}' not found")
        self._set_new_cols([self.output_col])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_X(X)
        X = X.copy()
        if self.name_col not in X.columns:
            X[self.output_col] = pd.NA
            return X

        names = X[self.name_col].astype("string")
        # Remove content inside parentheses
        no_paren = names.str.split("(", n=1, expand=False).str[0]
        # Take part before comma as family
        family = no_paren.str.split(",", n=1, expand=False).str[0]
        # Remove punctuation
        punct_re = f"[{re.escape(string.punctuation)}]"
        family = family.str.replace(punct_re, "", regex=True).str.strip()

        X[self.output_col] = family

        try:
            vc = X[self.output_col].value_counts(dropna=True)
            self.logger.info(
                f"FamilyTransform: unique={vc.size}, top5={vc.head(5).to_dict()}"
            )
        except Exception:
            pass

        return X

