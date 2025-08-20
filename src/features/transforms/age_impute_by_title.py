from __future__ import annotations

from typing import Optional, List
import numpy as np
import pandas as pd

from features.transforms.base import BaseTransform


class AgeImputeByTitleTransform(BaseTransform):
    """
    General Age imputation with configurable grouping.

    Default behavior: fill missing Age using the median Age per (Sex, Pclass)
    group learned on fit data, then fall back to the global median.
    Backward-compatible: if no group columns are provided, falls back to
    Title-based means (legacy behavior), then global mean.

    Parameters
    ----------
    age_col : str, default "Age"
        Numeric age column to impute.
    group_cols : Optional[List[str]], default ["Sex", "Pclass"]
        List of columns to group by for median imputation. Each must exist in X at
        fit/transform time. If None, uses legacy title-based strategy.
    title_col : str, default "Title"
        Legacy fallback: categorical title column (e.g., "Mr", "Mrs", ...).
    output_filled_age_col : Optional[str], default None
        If provided, also writes the fully filled ages to this column (float).
        Regardless, the transform fills NaNs in `age_col` in-place.
    round_age : bool, default True
        If True, filled ages are rounded to the nearest integer.
    """

    def __init__(
        self,
        age_col: str = "Age",
        group_cols: Optional[List[str]] = None,
        title_col: str = "Title",
        output_filled_age_col: Optional[str] = None,
        round_age: bool = True,
    ):
        super().__init__(name="AgeImputeByTitleTransform")
        self.age_col = age_col
        self.group_cols = group_cols if group_cols is not None else ["Sex", "Pclass"]
        self.title_col = title_col
        self.output_filled_age_col = output_filled_age_col
        self.round_age = round_age

        self.group_medians_: Optional[pd.Series] = None
        self.title_age_means_: Optional[pd.Series] = None
        self.global_age_median_: Optional[float] = None

    # ---- helpers ----
    def _fill_ages(self, X: pd.DataFrame) -> pd.Series:
        """Return age series with NaNs filled using learned means."""
        s = pd.to_numeric(X[self.age_col], errors="coerce")

        # If we didn't learn anything, just return as-is
        if self.group_medians_ is None and self.title_age_means_ is None and self.global_age_median_ is None:
            return s

        filled = s.copy()
        mask_na = filled.isna()

        # 1) Group-based median fill (preferred)
        if mask_na.any() and self.group_medians_ is not None and self.group_cols:
            # Build a key DataFrame for mapping
            present = [c for c in self.group_cols if c in X.columns]
            if len(present) == len(self.group_cols):
                key_df = X[present].copy()
                # Normalize dtypes for stable lookup
                for c in present:
                    # Treat all as string for mapping consistency
                    key_df[c] = key_df[c].astype("string")
                # Build a MultiIndex to align with learned medians
                try:
                    med = self.group_medians_
                    # Create series of medians aligned to rows
                    mapped = med.reindex(pd.MultiIndex.from_frame(key_df)).values
                    mapped = pd.Series(mapped, index=X.index)
                    filled.loc[mask_na] = mapped.loc[mask_na]
                except Exception:
                    # If mapping fails (unlikely), skip silently to next fallback
                    pass

        # 2) Legacy title-based mean fill (fallback if configured on fit)
        if filled.isna().any() and self.title_age_means_ is not None and self.title_col in X.columns:
            titles = X[self.title_col].astype("string")
            mapped = titles.map(self.title_age_means_)  # unseen titles -> NaN
            filled.loc[filled.isna()] = mapped.loc[filled.isna()]

        # 3) Global median fallback
        if filled.isna().any() and self.global_age_median_ is not None:
            filled = filled.fillna(self.global_age_median_)

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
            self.group_medians_ = None
            self.title_age_means_ = None
            self.global_age_median_ = None
        else:
            self.global_age_median_ = float(s_valid.median())
            n_missing = int(s_raw.isna().sum())
            self.logger.info(
                f"Age fit stats: n={len(s_raw)}, missing={n_missing}, global_median={self.global_age_median_:.2f}"
            )

            # Learn group medians if all group columns exist
            present = [c for c in (self.group_cols or []) if c in X.columns]
            if self.group_cols and len(present) == len(self.group_cols):
                df_fit = pd.DataFrame({"age": s_raw})
                for c in self.group_cols:
                    df_fit[c] = X[c].astype("string")
                df_fit = df_fit.dropna(subset=["age"])  # only rows with age present
                if not df_fit.empty:
                    med = df_fit.groupby(self.group_cols)["age"].median()
                    self.group_medians_ = med
                    try:
                        sample = med.head(5)
                    except Exception:
                        sample = med
                    self.logger.info(
                        f"Learned Age medians by groups {self.group_cols}: {len(med)} combos; sample:\n{sample.to_string()}"
                    )
                else:
                    self.group_medians_ = None
            else:
                self.group_medians_ = None

            # Legacy title means (only if group medians not available)
            if self.group_medians_ is None:
                if self.title_col not in X.columns:
                    self.title_age_means_ = None
                else:
                    df_fit_t = pd.DataFrame({"age": s_raw, "title": X[self.title_col].astype("string")})
                    df_fit_t = df_fit_t.dropna(subset=["age"])
                    self.title_age_means_ = None if df_fit_t.empty else df_fit_t.groupby("title")["age"].mean()

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
        try:
            n_missing_before = int(pd.to_numeric(X[self.age_col], errors="coerce").isna().sum())
            n_missing_after = int(pd.to_numeric(s_filled, errors="coerce").isna().sum())
            self.logger.info(
                f"Age transform: missing before={n_missing_before}, remaining after fill={n_missing_after}"
            )
        except Exception:
            pass

        # Write auxiliary filled column if requested
        if self.output_filled_age_col:
            X[self.output_filled_age_col] = s_filled

        # Fill the original age column in place (only where missing)
        # Preserve original non-missing values
        age_orig = pd.to_numeric(X[self.age_col], errors="coerce")
        X[self.age_col] = age_orig.fillna(s_filled).round().astype("int64")

        return X
