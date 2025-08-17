from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from features.transforms.base import BaseTransform


class AgeImputeByTitleTransform(BaseTransform):
    """
    Impute Age by:
      1) Filling NaNs with the mean Age per Title learned on fit data.
      2) Falling back to the global mean Age (learned on fit data) for unseen/missing titles.
      3) (Optional) Rounding the filled ages.

    Parameters
    ----------
    age_col : str, default "Age"
        Numeric age column to impute.
    title_col : str, default "Title"
        Categorical title column (e.g., "Mr", "Mrs", "Miss", "Master").
    output_filled_age_col : Optional[str], default None
        If provided, also writes the fully filled ages to this column (float).
        Regardless, the transform fills NaNs in `age_col` in-place.
    round_age : bool, default True
        If True, filled ages are rounded to the nearest integer.
    """

    def __init__(
        self,
        age_col: str = "Age",
        title_col: str = "Title",
        output_filled_age_col: Optional[str] = None,
        round_age: bool = True,
    ):
        super().__init__(name="AgeImputeByTitleTransform")
        self.age_col = age_col
        self.title_col = title_col
        self.output_filled_age_col = output_filled_age_col
        self.round_age = round_age

        self.title_age_means_: Optional[pd.Series] = None
        self.global_age_mean_: Optional[float] = None

    # ---- helpers ----
    def _fill_ages(self, X: pd.DataFrame) -> pd.Series:
        """Return age series with NaNs filled using learned means."""
        s = pd.to_numeric(X[self.age_col], errors="coerce")

        # If we didn't learn anything, just return as-is
        if self.title_age_means_ is None and self.global_age_mean_ is None:
            return s

        filled = s.copy()
        mask_na = filled.isna()

        # Try title-based fill
        if mask_na.any() and self.title_col in X.columns and self.title_age_means_ is not None:
            titles = X[self.title_col].astype("string")
            mapped = titles.map(self.title_age_means_)  # unseen titles -> NaN
            filled.loc[mask_na] = mapped.loc[mask_na]

        # Fallback to global mean
        if filled.isna().any() and self.global_age_mean_ is not None:
            filled = filled.fillna(self.global_age_mean_)

        if self.round_age:
            filled = filled.round()

        return filled.astype(float)

    # ---- core API ----
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AgeImputeByTitleTransform":
        self._validate_X(X)
        if self.age_col not in X.columns:
            raise ValueError(f"Column '{self.age_col}' not found")

        s_raw = pd.to_numeric(X[self.age_col], errors="coerce")
        s_valid = s_raw.dropna()

        if s_valid.empty:
            self.logger.warning("No valid age values found; nothing will be imputed at transform().")
            self.title_age_means_ = None
            self.global_age_mean_ = None
        else:
            self.global_age_mean_ = float(s_valid.mean())

            if self.title_col not in X.columns:
                self.logger.warning(f"'{self.title_col}' not found; title-based imputation unavailable.")
                self.title_age_means_ = None
            else:
                df_fit = pd.DataFrame({"age": s_raw, "title": X[self.title_col].astype("string")})
                df_fit = df_fit.dropna(subset=["age"])
                self.title_age_means_ = None if df_fit.empty else df_fit.groupby("title")["age"].mean()

        # We only add the optional filled-age column
        new_cols = []
        if self.output_filled_age_col:
            new_cols.append(self.output_filled_age_col)
        self._set_new_cols(new_cols)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X).copy()

        if self.age_col not in X.columns:
            # Nothing to impute
            if self.output_filled_age_col:
                X[self.output_filled_age_col] = np.nan
            return X

        s_filled = self._fill_ages(X)

        # Write auxiliary filled column if requested
        if self.output_filled_age_col:
            X[self.output_filled_age_col] = s_filled

        # Fill the original age column in place (only where missing)
        # Preserve original non-missing values
        age_orig = pd.to_numeric(X[self.age_col], errors="coerce")
        X[self.age_col] = age_orig.fillna(s_filled).astype(float)

        return X
