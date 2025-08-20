from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd

from features.transforms.base import BaseTransform


class FareBinningTransform(BaseTransform):
    """
    Discretize Fare into bins and output integer bin codes.

    - Learns bin edges on training data using either quantiles or uniform width.
    - Handles zeros and missing values robustly.

    Parameters
    ----------
    fare_col : str, default "Fare"
        Column with numeric fares.
    n_bins : int, default 13
        Number of bins to create.
    strategy : {"quantile", "uniform"}, default "quantile"
        Binning strategy. Quantile yields balanced bin counts.
    output_col : str, default "FareBin"
        Name of the integer-coded bin column.
    include_zero : bool, default True
        Whether to include zero fares in edge computation (kept as numeric).
    """

    def __init__(
        self,
        fare_col: str = "Fare",
        n_bins: int = 13,
        strategy: Literal["quantile", "uniform"] = "quantile",
        output_col: str = "FareBin",
        include_zero: bool = True,
    ):
        super().__init__(name="FareBinningTransform")
        self.fare_col = fare_col
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.output_col = output_col
        self.include_zero = include_zero

        self.bin_edges: Optional[np.ndarray] = None
        self._bins_effective: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FareBinningTransform":
        self._validate_X(X)
        if self.fare_col not in X.columns:
            raise ValueError(f"Column '{self.fare_col}' not found")

        fares = pd.to_numeric(X[self.fare_col], errors="coerce")
        mask = fares.notna()
        if not self.include_zero:
            mask &= fares > 0
        fares_fit = fares[mask]

        if fares_fit.empty:
            self.logger.warning("No valid fare values found; FareBin will be set to -1 at transform().")
            self.bin_edges = None
            self._bins_effective = 0
            self._set_new_cols([self.output_col])
            self.is_fitted = True
            return self

        if self.strategy == "quantile":
            qs = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.quantile(fares_fit.to_numpy(), qs)
        else:
            lo, hi = float(fares_fit.min()), float(fares_fit.max())
            edges = np.linspace(lo, hi, self.n_bins + 1)

        edges = np.unique(edges)
        if edges.size < 2:
            v = float(fares_fit.iloc[0])
            self.logger.warning(
                "Degenerate fare distribution (all identical); FareBin will be 0 for non-missing fares."
            )
            edges = np.array([v - 1e-6, v + 1e-6])

        self._bins_effective = int(edges.size - 1)
        self.bin_edges = edges.astype(float)
        self._set_new_cols([self.output_col])
        self.is_fitted = True

        try:
            self.logger.info(
                f"Learned fare bins (edges): {np.round(self.bin_edges, 4).tolist()} (bins={self._bins_effective})"
            )
        except Exception:
            pass
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_X(X)
        X = X.copy()

        if self.fare_col not in X.columns:
            X[self.output_col] = -1
            X[self.output_col] = X[self.output_col].astype("int16")
            return X

        fares = pd.to_numeric(X[self.fare_col], errors="coerce")
        if self.bin_edges is None or not self._bins_effective:
            X[self.output_col] = -1
            X[self.output_col] = X[self.output_col].astype("int16")
            return X

        codes = pd.cut(
            fares,
            bins=self.bin_edges,
            labels=False,
            include_lowest=True,
            right=True,
            duplicates="drop",
        )
        # Missing fares -> -1
        codes = codes.astype("float").fillna(-1).astype("int16")
        X[self.output_col] = codes
        return X

