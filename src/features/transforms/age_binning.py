from __future__ import annotations

from typing import Optional, Literal, List
import numpy as np
import pandas as pd

from features.transforms.base import BaseTransform


class AgeBinningTransform(BaseTransform):
    """
    Discretize Age into bins and output numeric bin indices.

    - strategy: "quantile" (default) or "uniform"
    - output is an integer column (int16) with -1 for missing/unassigned
    """

    def __init__(
        self,
        age_col: str = "Age",
        n_bins: int = 5,
        strategy: Literal["quantile", "uniform"] = "quantile",
        output_col: str = "AgeBin",
    ):
        super().__init__(name="AgeBinningTransform")
        self.age_col = age_col
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.output_col = output_col

        self.bin_edges: Optional[np.ndarray] = None
        self._bins_effective: Optional[int] = None  # actual number of bins after dedup

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AgeBinningTransform":
        self._validate_X(X)
        if self.age_col not in X.columns:
            raise ValueError(f"Column '{self.age_col}' not found")

        s = pd.to_numeric(X[self.age_col], errors="coerce").dropna()
        if s.empty:
            # No ages available to learn from: mark as unfitted bins
            self.logger.warning("No valid age values found; AgeBin will be set to -1 at transform()")
            self.bin_edges = None
            self._bins_effective = 0
            self._set_new_cols([self.output_col])
            self.is_fitted = True
            return self

        if self.strategy == "quantile":
            qs = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.quantile(s.to_numpy(), qs)
        else:  # uniform
            lo, hi = float(s.min()), float(s.max())
            edges = np.linspace(lo, hi, self.n_bins + 1)

        # Ensure edges are strictly monotonic; drop duplicates if necessary
        edges = np.unique(edges)
        if edges.size < 2:
            # Degenerate: all ages identical
            self.logger.warning(
                "Degenerate age distribution (all identical); AgeBin will be 0 for non-missing ages."
            )
            # Create a minimal 2-edge bin so pd.cut works; values will fall into single bin
            v = float(s.iloc[0])
            edges = np.array([v - 1e-6, v + 1e-6])

        # Remember actual number of bins
        self._bins_effective = int(edges.size - 1)
        self.bin_edges = edges.astype(float)

        self.logger.info(f"Learned age bins (edges): {np.round(self.bin_edges, 4).tolist()}  "
                         f"(bins={self._bins_effective})")

        self._set_new_cols([self.output_col])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X).copy()

        # If we couldn't learn edges, output sentinel -1
        if self.bin_edges is None or self._bins_effective is None or self.age_col not in X.columns:
            X[self.output_col] = -1
            X[self.output_col] = X[self.output_col].astype("int16")
            return X

        # Bin to integer codes
        s = pd.to_numeric(X[self.age_col], errors="coerce")
        codes = pd.cut(
            s,
            bins=self.bin_edges,
            labels=False,           # -> integer codes
            include_lowest=True,
            right=True,
            duplicates="drop",      # safety if edges still collapse
        )

        # Missing/unassigned -> -1, cast to small int
        X[self.output_col] = codes.fillna(-1).astype("int16")
        return X
