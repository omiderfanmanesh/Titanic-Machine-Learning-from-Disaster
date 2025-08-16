from __future__ import annotations

from typing import Optional

import pandas as pd

from features.transforms.base import BaseTransform


class TicketParseTransform(BaseTransform):
    """
    Extracts ticket prefix and numeric part.

    - Ticket_prefix: non-digit, non-dot prefix ('' -> 'NUMBER')
    - Ticket_number: trailing digits as float (NaN if absent)
    """

    def __init__(self, ticket_col: str = "Ticket"):
        super().__init__()
        self.ticket_col = ticket_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TicketParseTransform":
        if self.ticket_col not in X.columns:
            raise ValueError(f"Column {self.ticket_col} not found")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        X = X.copy()

        # Prefix: remove digits and dots, strip spaces
        prefix = (
            X[self.ticket_col]
            .astype(str)
            .str.replace(r"\d+", "", regex=True)
            .str.replace(".", "", regex=False)
            .str.strip()
        )
        X["Ticket_prefix"] = prefix.replace("", "NUMBER")

        # Numeric (last group of digits)
        num = X[self.ticket_col].astype(str).str.extract(r"(\d+)$")[0]
        X["Ticket_number"] = pd.to_numeric(num, errors="coerce")

        return X

