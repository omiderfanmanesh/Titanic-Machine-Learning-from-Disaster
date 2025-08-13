from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class TicketGroupTransform(BaseTransform):
    """Creates ticket group size feature."""

    def __init__(self, ticket_col: str = "Ticket"):
        super().__init__()
        self.ticket_col = ticket_col
        self.ticket_counts: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TicketGroupTransform":
        """Learn ticket counts from training data."""
        if self.ticket_col not in X.columns:
            raise ValueError(f"Column {self.ticket_col} not found")

        self.ticket_counts = X[self.ticket_col].value_counts()
        self.logger.info(f"Learned ticket counts for {len(self.ticket_counts)} unique tickets")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by adding ticket group size."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        X = X.copy()
        X["TicketGroupSize"] = X[self.ticket_col].map(self.ticket_counts).fillna(1)
        self.logger.debug(f"Ticket group size distribution: {X['TicketGroupSize'].value_counts().sort_index().to_dict()}")
        return X

