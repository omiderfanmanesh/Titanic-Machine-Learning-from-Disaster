from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from core.interfaces import ImputationStrategy

_NUMERIC_KINDS = ("i", "u", "f")

def _is_numeric(series: pd.Series) -> bool:
    return series.dtype.kind in _NUMERIC_KINDS

class KNNImputerStrategy(ImputationStrategy):
    """KNN imputation for a single target column using configured feature set."""
    def __init__(self, column: str, plan: Dict[str, Any]):
        super().__init__(column, plan)
        self.n_neighbors = int(plan.get("n_neighbors", 5))
        self.features: List[str] = plan.get("features", [])

        # Store fitted components to prevent data leakage
        self.encoder: Optional[OneHotEncoder] = None
        self.knn_imputer: Optional[KNNImputer] = None
        self._fitted_features: List[str] = []
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._encoded_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit the OneHotEncoder and KNNImputer on training data only."""
        if not X[self.column].isna().any():
            return  # No missing values to impute

        # Determine features to use
        self._fitted_features = self.features or [c for c in X.columns if c != self.column]
        self._fitted_features = [f for f in self._fitted_features if f in X.columns]

        if not self._fitted_features:
            return

        cols = self._fitted_features + [self.column]

        # Separate numeric and categorical columns
        self._num_cols = [c for c in cols if _is_numeric(X[c])]
        self._cat_cols = [c for c in cols if c not in self._num_cols and c != self.column]

        # Fit OneHotEncoder on categorical columns if any exist
        if self._cat_cols:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.encoder.fit(X[self._cat_cols])
            self._encoded_feature_names = list(self.encoder.get_feature_names_out(self._cat_cols))

        # Prepare training matrix for KNN
        mat = self._prepare_matrix(X)

        # Fit KNN imputer
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.knn_imputer.fit(mat)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted components only."""
        if self.knn_imputer is None or not X[self.column].isna().any():
            return X

        X = X.copy()

        # Check that all required features are present
        missing_features = [f for f in self._fitted_features if f not in X.columns]
        if missing_features:
            raise ValueError(f"Missing required features for KNN imputation: {missing_features}")

        # Prepare matrix using fitted encoder
        mat = self._prepare_matrix(X)

        # Apply KNN imputation
        mat_imp = pd.DataFrame(
            self.knn_imputer.transform(mat),
            index=mat.index,
            columns=mat.columns
        )

        # Update only the missing values in the target column
        mask = X[self.column].isna()
        X.loc[mask, self.column] = mat_imp.loc[mask, self.column]

        return X

    def _prepare_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for KNN imputation."""
        components = []

        # Add numeric columns
        if self._num_cols:
            num_subset = [c for c in self._num_cols if c in X.columns]
            if num_subset:
                components.append(X[num_subset])

        # Add encoded categorical columns
        if self._cat_cols and self.encoder is not None:
            cat_subset = [c for c in self._cat_cols if c in X.columns]
            if cat_subset:
                encoded = pd.DataFrame(
                    self.encoder.transform(X[cat_subset]),
                    index=X.index,
                    columns=self._encoded_feature_names
                )
                components.append(encoded)

        # Add target column
        components.append(X[[self.column]])

        if not components:
            raise ValueError("No valid features available for KNN imputation")

        return pd.concat(components, axis=1)
