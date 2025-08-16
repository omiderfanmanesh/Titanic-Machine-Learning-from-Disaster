from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd

from features.transforms.base import BaseTransform


class AgeBinningTransform(BaseTransform):
    """
    Pipeline:
      1) Impute Age: fill NaNs using mean Age per Title learned on fit data.
         - Unseen/missing titles at transform-time fall back to the global mean Age (learned on fit data).
      2) (Optional) Round Age to nearest integer.
      3) Bin Age: discretize the (filled) Age into bins and output integer bin codes.

    Parameters
    ----------
    age_col : str, default "Age"
        Column with numeric ages.
    title_col : str, default "Title"
        Column with titles (e.g., "Mr", "Mrs", "Miss", "Master").
    n_bins : int, default 5
        Number of bins.
    strategy : {"quantile", "uniform"}, default "quantile"
        Binning strategy.
    output_col : str, default "AgeBin"
        Name of the integer-coded bin column.
    output_filled_age_col : Optional[str], default None
        If provided, also writes the filled ages to this column (float).
    round_age : bool, default False
        If True, filled ages are rounded to the nearest integer before binning.
    """

    def __init__(
        self,
        age_col: str = "Age",
        title_col: str = "Title",
        n_bins: int = 5,
        strategy: Literal["quantile", "uniform"] = "quantile",
        output_col: str = "AgeBin",
        output_filled_age_col: Optional[str] = None,
        round_age: bool = True,
    ):
        super().__init__(name="AgeBinningTransform")
        self.age_col = age_col
        self.title_col = title_col
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.output_col = output_col
        self.output_filled_age_col = output_filled_age_col
        self.round_age = round_age

        self.bin_edges: Optional[np.ndarray] = None
        self._bins_effective: Optional[int] = None
        self.title_age_means_: Optional[pd.Series] = None
        self.global_age_mean_: Optional[float] = None

    # ---- helpers ----
    def _fill_ages(self, X: pd.DataFrame) -> pd.Series:
        """Return ages with NaNs filled using learned title means then global mean."""
        s = pd.to_numeric(X[self.age_col], errors="coerce")

        # If we weren't able to learn means, just return s
        if self.title_age_means_ is None and self.global_age_mean_ is None:
            return s

        filled = s.copy()
        mask_na = filled.isna()
        if mask_na.any() and self.title_col in X.columns and self.title_age_means_ is not None:
            titles = X[self.title_col].astype("string")
            mapped = titles.map(self.title_age_means_)  # unseen titles -> NaN
            filled.loc[mask_na] = mapped.loc[mask_na]

        if filled.isna().any() and self.global_age_mean_ is not None:
            filled = filled.fillna(self.global_age_mean_)

        # Optionally round to nearest integer
        if self.round_age:
            filled = filled.round()

        return filled

    # ---- core API ----
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AgeBinningTransform":
        self._validate_X(X)
        if self.age_col not in X.columns:
            raise ValueError(f"Column '{self.age_col}' not found")

        s_raw = pd.to_numeric(X[self.age_col], errors="coerce")
        s_valid = s_raw.dropna()

        # Learn title means/global mean
        if s_valid.empty:
            self.logger.warning("No valid age values found; AgeBin will be set to -1 at transform().")
            self.title_age_means_ = None
            self.global_age_mean_ = None
            self.bin_edges = None
            self._bins_effective = 0
            new_cols = [self.output_col]
            if self.output_filled_age_col:
                new_cols.append(self.output_filled_age_col)
            self._set_new_cols(new_cols)
            self.is_fitted = True
            return self

        self.global_age_mean_ = float(s_valid.mean())

        if self.title_col not in X.columns:
            self.logger.warning(f"'{self.title_col}' not found; title-based imputation unavailable.")
            self.title_age_means_ = None
        else:
            df_fit = pd.DataFrame({"age": s_raw, "title": X[self.title_col].astype("string")})
            df_fit = df_fit.dropna(subset=["age"])
            self.title_age_means_ = None if df_fit.empty else df_fit.groupby("title")["age"].mean()

        # Use filled ages (rounded if required) to learn bin edges
        ages_for_binning = self._fill_ages(X).dropna().to_numpy()
        if ages_for_binning.size == 0:
            self.logger.warning("No valid ages to compute bins; AgeBin will be -1 at transform().")
            self.bin_edges = None
            self._bins_effective = 0
        else:
            if self.strategy == "quantile":
                qs = np.linspace(0.0, 1.0, self.n_bins + 1)
                edges = np.quantile(ages_for_binning, qs)
            else:  # uniform
                lo, hi = float(np.min(ages_for_binning)), float(np.max(ages_for_binning))
                edges = np.linspace(lo, hi, self.n_bins + 1)

            edges = np.unique(edges)
            if edges.size < 2:
                v = float(ages_for_binning[0])
                self.logger.warning(
                    "Degenerate age distribution (all identical); AgeBin will be 0 for non-missing ages."
                )
                edges = np.array([v - 1e-6, v + 1e-6])

            self._bins_effective = int(edges.size - 1)
            self.bin_edges = edges.astype(float)

            self.logger.info(
                f"Learned age bins (edges): {np.round(self.bin_edges, 4).tolist()} "
                f"(bins={self._bins_effective})"
            )

        new_cols = [self.output_col]
        if self.output_filled_age_col:
            new_cols.append(self.output_filled_age_col)
        self._set_new_cols(new_cols)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X).copy()

        if self.age_col not in X.columns:
            # No age column at all
            if self.output_filled_age_col:
                X[self.output_filled_age_col] = np.nan
            X[self.output_col] = -1
            X[self.output_col] = X[self.output_col].astype("int16")
            return X

        # 1) Fill ages by title mean -> global mean, then optionally round
        s_filled = self._fill_ages(X)

        # (optional) keep the filled age column
        if self.output_filled_age_col:
            X[self.output_filled_age_col] = s_filled.astype(float)

        # 2) Bin the filled ages
        if self.bin_edges is None or self._bins_effective is None or self._bins_effective == 0:
            X[self.output_col] = -1
            X[self.output_col] = X[self.output_col].astype("int16")
            return X

        codes = pd.cut(
            s_filled,
            bins=self.bin_edges,
            labels=False,
            include_lowest=True,
            right=True,
            duplicates="drop",
        )

        X[self.output_col] = codes.fillna(-1).astype("int16")
        return X
