from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from .factory import StrategyFactory
from .report import ImputationReport

_NUMERIC_KINDS = ("i", "u", "f")
def _is_numeric(series: pd.Series) -> bool:
    return series.dtype.kind in _NUMERIC_KINDS


class ImputationOrchestrator:
    """
    Coordinates per-column strategies. Orchestrator contains no imputation logic.
    Responsibilities:
      - Build per-column plans from config
      - Add optional missing indicators
      - Apply per-column clipping
      - Validate schema between fit/transform
      - Aggregate a simple report
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.cfg = (self.config.get("imputation") or {})
        self.default_cfg = self.cfg.get("default", {})
        self.per_column = self.cfg.get("per_column", {})

        # More selective missing indicator controls
        self.add_missing_indicators: bool = bool(self.default_cfg.get("add_missing_indicators", False))  # Default to False
        self.missing_indicator_columns: Optional[List[str]] = self.default_cfg.get("missing_indicator_columns")  # Explicit column list
        self.missing_indicator_threshold: float = float(self.default_cfg.get("missing_indicator_threshold", 0.05))  # Only add if >5% missing
        self.missing_indicator_prefix: str = str(self.default_cfg.get("missing_indicator_prefix", "__miss_"))
        self.debug: bool = bool(self.default_cfg.get("debug", False))

        # Optional controls
        self.include: Optional[List[str]] = self.cfg.get("include")  # only these columns (if provided)
        self.exclude: List[str] = list(self.cfg.get("exclude", []))  # columns to skip entirely
        self.order: Optional[List[str]] = self.cfg.get("order")      # explicit fit/transform order

        self.factory = StrategyFactory(debug=self.debug)
        self.strategies: Dict[str, Any] = {}
        self.clipping: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self.report = ImputationReport()

        # Schema freeze
        self._fitted_columns: List[str] = []
        self._col_dtypes: Dict[str, Any] = {}
        self._missing_indicator_cols: List[str] = []  # Track which indicators were actually added

    # ---- lifecycle ----
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ImputationOrchestrator":
        X = X.copy()

        cols = self._select_columns(X)
        cols = self._apply_order(cols)
        
        # Determine which columns should get missing indicators during fit
        self._determine_missing_indicators(X, cols)

        for col in cols:
            plan = self._plan_for(col, X[col])
            self.clipping[col] = (plan.get("clip_min", None), plan.get("clip_max", None))
            self.report.init_row(col, plan["method"], float(X[col].isna().mean()), int(X[col].notna().sum()))

            strat = self.factory.make(col, plan)
            strat.fit(X, y)
            if hasattr(strat, "report"):
                self.report.update(col, getattr(strat, "report"))
            self.strategies[col] = strat

            # remember dtype for restoration after clipping
            self._col_dtypes[col] = X[col].dtype

        self._fitted_columns = list(cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 0) schema check: ensure all fitted columns exist in incoming frame
        missing_cols = [c for c in self._fitted_columns if c not in X.columns]
        if missing_cols:
            raise ValueError(
                "ImputationOrchestrator.transform: input is missing columns seen during fit: "
                f"{missing_cols}"
            )

        # 1) Add missing indicators only for selected columns (before filling)
        for col in self._missing_indicator_cols:
            if col in X.columns and X[col].isna().any():
                indicator_name = f"{self.missing_indicator_prefix}{col}"
                X[indicator_name] = X[col].isna().astype(np.int8)

        # 2) apply strategies in the same order frozen at fit
        for col in self._fitted_columns:
            strat = self.strategies.get(col)
            if strat is None:
                # If there are NaNs and no strategy, be explicit
                if X[col].isna().any():
                    raise ValueError(
                        f"No fitted imputation strategy found for column '{col}', "
                        "but NaNs detected in transform()."
                    )
                continue
            X = strat.transform(X)

        # 3) clipping (after all imputations) with better dtype handling
        for col, (lo, hi) in self.clipping.items():
            if (lo is not None or hi is not None) and col in X.columns and _is_numeric(X[col]):
                original_dtype = self._col_dtypes.get(col, X[col].dtype)

                # Only clip if there are no NaN values after imputation
                if not X[col].isna().any():
                    X[col] = X[col].astype(float)

                    # Apply clipping
                    if lo is not None:
                        X[col] = X[col].clip(lower=lo)
                    if hi is not None:
                        X[col] = X[col].clip(upper=hi)

                    # Attempt dtype restoration
                    self._restore_dtype(X, col, original_dtype)

        # 4) sanity: any column in fitted set still has NaNs?
        still_nan = [c for c in self._fitted_columns if c in X.columns and X[c].isna().any()]
        if still_nan:
            raise ValueError(
                "ImputationOrchestrator.transform: some fitted columns still contain NaNs after imputation: "
                f"{still_nan}"
            )

        return X

    def get_report(self) -> pd.DataFrame:
        return self.report.to_df()

    # ---- helpers ----
    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        cols = list(X.columns)
        if self.include is not None:
            cols = [c for c in cols if c in self.include]
        if self.exclude:
            cols = [c for c in cols if c not in self.exclude]
        return cols

    def _apply_order(self, cols: List[str]) -> List[str]:
        if not self.order:
            return cols
        in_order = [c for c in self.order if c in cols]
        the_rest = [c for c in cols if c not in in_order]
        return in_order + the_rest

    def _plan_for(self, col: str, s: pd.Series) -> Dict[str, Any]:
        if col in self.per_column:
            plan = dict(self.per_column[col])
            plan["method"] = str(plan.get("method", "")).lower()
            return plan
        # defaults by dtype
        if _is_numeric(s):
            return {"method": str(self.default_cfg.get("numeric", "median")).lower()}
        else:
            return {
                "method": str(self.default_cfg.get("categorical", "constant")).lower(),
                "fill_value": self.default_cfg.get("fill_value", "Unknown"),
            }

    def _determine_missing_indicators(self, X: pd.DataFrame, cols: List[str]) -> None:
        """Determine which columns should get missing indicators based on configuration."""
        if not self.add_missing_indicators:
            return

        if self.missing_indicator_columns is not None:
            # Use explicitly specified columns
            self._missing_indicator_cols = [
                col for col in self.missing_indicator_columns
                if col in cols and X[col].isna().any()
            ]
        else:
            # Use threshold-based selection
            self._missing_indicator_cols = [
                col for col in cols
                if X[col].isna().any() and X[col].isna().mean() >= self.missing_indicator_threshold
            ]

    def _restore_dtype(self, X: pd.DataFrame, col: str, original_dtype) -> None:
        """Safely restore dtype after clipping operations."""
        try:
            if np.issubdtype(original_dtype, np.integer):
                # Check if all values are actually integers
                if np.all(np.equal(np.mod(X[col].values, 1), 0)):
                    X[col] = X[col].astype(original_dtype)
                else:
                    # Keep as float if clipping introduced non-integer values
                    if self.debug:
                        print(f"Warning: Column '{col}' kept as float after clipping introduced non-integer values")
            else:
                # For float dtypes, try to restore but don't fail if it doesn't work
                X[col] = X[col].astype(original_dtype)
        except (ValueError, OverflowError) as e:
            if self.debug:
                print(f"Warning: Could not restore dtype for column '{col}': {e}")
            # Keep as float64 if restoration fails
