from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from features.transforms.base import BaseTransform


class DeckTransform(BaseTransform):
    """Extracts deck information from cabin numbers."""

    def __init__(self, cabin_col: str = "Cabin"):
        super().__init__(name="DeckTransform")
        self.cabin_col = cabin_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DeckTransform":
        if self.cabin_col not in X.columns:
            raise ValueError(f"Column '{self.cabin_col}' not found")
        # Declare the new column this transform creates
        self._set_new_cols(["Deck"])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_X(X)
        X = X.copy()

        # Get first letter of cabin; keep missing as 'U'
        raw = X[self.cabin_col].astype("string") if self.cabin_col in X.columns else pd.Series(pd.NA, index=X.index, dtype="string")
        letter = raw.str.slice(0, 1)
        # Normalize: map 'T' (rare) to 'A' (as commonly done in Titanic)
        letter = letter.fillna("U").replace({"T": "A"})

        # Map into 3 grouped categories {ABC, DE, FG}; anything else -> 'U'
        mapping = {
            "A": "ABC", "B": "ABC", "C": "ABC",
            "D": "DE",  "E": "DE",
            "F": "FG",  "G": "FG",
        }
        deck_group = letter.map(mapping).fillna("U")
        X["Deck"] = deck_group

        try:
            counts = X["Deck"].value_counts(dropna=False).to_dict()
            self.logger.info(f"DeckTransform: distribution={counts}")
        except Exception:
            pass

        return X
