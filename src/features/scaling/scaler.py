from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from core.utils import LoggerFactory

class ScalingOrchestrator:
    def __init__(self, enable: bool = True, config: Optional[Dict[str, Any]] = None):
        self.enable = enable
        self.config = config or {}

        # Scaling configuration
        self.scaler: Optional[StandardScaler] = None
        self.scale_cols: List[str] = []
        self.original_dtypes: Dict[str, Any] = {}

        # More robust exclusion logic
        self.exclude_patterns: List[str] = self.config.get("exclude_patterns", ["PassengerId", "IsAlone"])
        self.exclude_binary: bool = self.config.get("exclude_binary", True)
        self.min_unique_threshold: int = self.config.get("min_unique_threshold", 3)  # Exclude if < 3 unique values
        self.restore_dtypes: bool = self.config.get("restore_dtypes", True)
        self.debug: bool = self.config.get("debug", False)
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

    def fit(self, X: pd.DataFrame) -> "ScalingOrchestrator":
        if not self.enable:
            return self

        # Get numeric columns
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

        # Determine columns to exclude from scaling
        exclude_cols = set()

        # Add explicit exclusions
        exclude_cols.update(self.exclude_patterns)
        # Exclude known categorical columns from config if provided
        cat_cols = set(self.config.get("categorical_columns", []) or [])
        exclude_cols.update(cat_cols)

        # Add low-cardinality columns if configured
        if self.exclude_binary:
            # Compute unique counts vectorized for stability
            present_numeric = [c for c in numeric_cols if c in X.columns]
            if present_numeric:
                uniq_series = X[present_numeric].nunique()
                for col, unique_count in uniq_series.items():
                    if int(unique_count) < int(self.min_unique_threshold):
                        exclude_cols.add(col)
                        if self.debug:
                            print(f"Excluding '{col}' from scaling (only {int(unique_count)} unique values)")

        # Final list of columns to scale
        self.scale_cols = [c for c in numeric_cols if c not in exclude_cols and c in X.columns]

        if self.debug:
            print(f"Scaling columns: {self.scale_cols}")
            print(f"Excluded columns: {list(exclude_cols & set(numeric_cols))}")
        self.logger.info(
            f"Scaling: numeric={len(numeric_cols)}, scale_cols={len(self.scale_cols)}, excluded={len(set(numeric_cols)-set(self.scale_cols))}"
        )

        # Store original dtypes for restoration
        if self.restore_dtypes:
            self.original_dtypes = {col: X.dtypes[col] for col in self.scale_cols if col in X.columns}

        # Fit scaler
        if self.scale_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.scale_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enable or not self.scale_cols or self.scaler is None:
            return X

        X = X.copy()

        # Check for column availability (some may be missing in test set)
        available_cols = [c for c in self.scale_cols if c in X.columns]
        missing_cols = [c for c in self.scale_cols if c not in X.columns]

        if missing_cols and self.debug:
            print(f"Warning: Scaling columns missing in transform: {missing_cols}")

        if available_cols:
            # Apply scaling
            X[available_cols] = self.scaler.transform(X[available_cols])
            self.logger.info(f"Scaling applied to {len(available_cols)} columns")

            # Restore dtypes if requested and safe to do
            if self.restore_dtypes:
                self._restore_dtypes_safely(X, available_cols)

        return X

    def _restore_dtypes_safely(self, X: pd.DataFrame, cols: List[str]) -> None:
        """Attempt to restore original dtypes after scaling."""
        for col in cols:
            if col not in self.original_dtypes:
                continue

            original_dtype = self.original_dtypes[col]
            try:
                # Only attempt restoration for integer types if values are close to integers
                if np.issubdtype(original_dtype, np.integer):
                    values = X[col].values
                    # Check if scaled values could reasonably be cast back to integers
                    # This is generally not recommended after standardization, so skip
                    if self.debug:
                        print(f"Skipping dtype restoration for '{col}' (scaled integers should remain float)")
                    continue
                else:
                    # For float types, try to restore original precision
                    X[col] = X[col].astype(original_dtype)

            except (ValueError, OverflowError) as e:
                if self.debug:
                    print(f"Could not restore dtype for '{col}': {e}")
                # Keep as float64 if restoration fails

    def get_feature_names_out(self) -> List[str]:
        """Return the names of features that were scaled."""
        return self.scale_cols.copy()

    def get_scaling_info(self) -> Dict[str, Any]:
        """Return information about the scaling operation."""
        info = {
            "enabled": self.enable,
            "scaled_columns": self.scale_cols.copy(),
            "n_scaled_features": len(self.scale_cols),
            "scaler_fitted": self.scaler is not None
        }

        if self.scaler is not None:
            info.update({
                "feature_means": dict(zip(self.scale_cols, self.scaler.mean_)),
                "feature_scales": dict(zip(self.scale_cols, self.scaler.scale_))
            })

        return info
