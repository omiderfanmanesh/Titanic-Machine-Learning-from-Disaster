from __future__ import annotations

from typing import Optional
import pandas as pd

from features.transforms.base import BaseTransform


class TicketFrequencyTransform(BaseTransform):
    """
    Adds a numeric column with the frequency of each ticket string:
      Ticket_Frequency = count of rows sharing the same Ticket value

    Leak-safe: frequencies are learned on fit() and applied on transform().
    Unseen/missing tickets during transform default to 1.
    """

    def __init__(self, ticket_col: str = "Ticket", output_col: str = "Ticket_Frequency"):
        super().__init__(name="TicketFrequencyTransform")
        self.ticket_col = ticket_col
        self.output_col = output_col
        self._freq: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TicketFrequencyTransform":
        self._validate_X(X)
        if self.ticket_col not in X.columns:
            raise ValueError(f"Column '{self.ticket_col}' not found")

        tickets = X[self.ticket_col].astype("string").fillna("__MISSING__")
        vc = tickets.value_counts(dropna=False)
        self._freq = vc
        self._set_new_cols([self.output_col])
        self.is_fitted = True

        try:
            self.logger.info(
                f"Learned ticket frequencies: unique={len(vc)}, top5={vc.head(5).to_dict()}"
            )
        except Exception:
            pass
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_X(X)
        X = X.copy()

        if self.ticket_col not in X.columns:
            # If ticket column missing, default to 1
            X[self.output_col] = 1
            return X

        tickets = X[self.ticket_col].astype("string").fillna("__MISSING__")
        if self._freq is not None:
            freq = tickets.map(self._freq).fillna(1).astype(int)
        else:
            freq = pd.Series(1, index=X.index, dtype=int)
        X[self.output_col] = freq
        return X

