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
        # stateless (no fit artifacts needed)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        feats = self.features or [c for c in X.columns if c != self.column]
        feats_present = [f for f in feats if f in X.columns]
        if not feats_present or not X[self.column].isna().any():
            return X

        cols = feats_present + [self.column]
        mat = X[cols].copy()
        num_cols = [c for c in cols if _is_numeric(X[c])]
        cat_cols = [c for c in cols if c not in num_cols]

        if cat_cols:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
            enc_df = pd.DataFrame(
                enc.fit_transform(X[cat_cols]),
                index=X.index,
                columns=enc.get_feature_names_out(cat_cols),
            )
            mat = pd.concat([X[num_cols], enc_df, X[[self.column]]], axis=1)

        knn = KNNImputer(n_neighbors=self.n_neighbors)
        mat_imp = pd.DataFrame(knn.fit_transform(mat), index=mat.index, columns=mat.columns)
        mask = X[self.column].isna()
        X.loc[mask, self.column] = mat_imp.loc[mask, self.column]
        return X
