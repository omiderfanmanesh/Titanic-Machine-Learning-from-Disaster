from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class AgeBinningTransform(BaseTransform):
    """Creates age bins from continuous age values."""

    def __init__(self, age_col: str = "Age", n_bins: int = 5,
                 bin_labels: Optional[List[str]] = None):
        super().__init__()
        self.age_col = age_col
        self.n_bins = n_bins
        self.bin_labels = bin_labels or [f"Age_Bin_{i}" for i in range(n_bins)]
        self.bin_edges: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AgeBinningTransform":
        """Learn age bin edges from training data."""
        if self.age_col not in X.columns:
            raise ValueError(f"Column {self.age_col} not found")

        age_values = X[self.age_col].dropna()
        if len(age_values) == 0:
            raise ValueError("No valid age values found")

        # Create quantile-based bins
        self.bin_edges = np.quantile(age_values, np.linspace(0, 1, self.n_bins + 1))

        # Ensure unique edges
        self.bin_edges = np.unique(self.bin_edges)

        self.logger.info(f"Learned age bins: {self.bin_edges}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform age into bins."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        X = X.copy()

        # Create age bins
        X["AgeBin"] = pd.cut(
            X[self.age_col],
            bins=self.bin_edges,
            labels=self.bin_labels[:len(self.bin_edges)-1],
            include_lowest=True
        )

        return X

