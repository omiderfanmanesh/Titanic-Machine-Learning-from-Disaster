from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class DeckTransform(BaseTransform):
    """Extracts deck information from cabin numbers."""

    def __init__(self, cabin_col: str = "Cabin"):
        super().__init__()
        self.cabin_col = cabin_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DeckTransform":
        if self.cabin_col not in X.columns:
            raise ValueError(f"Column {self.cabin_col} not found")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")
        X = X.copy()
        deck = X[self.cabin_col].astype(str).str[0]
        valid_decks = list('ABCDEFGT')
        X["Deck"] = np.where(deck.isin(valid_decks), deck, 'U')
        self.logger.debug(f"Deck distribution: {X['Deck'].value_counts().to_dict()}")
        return X

