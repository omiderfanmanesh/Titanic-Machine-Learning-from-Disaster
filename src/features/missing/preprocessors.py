from __future__ import annotations
from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from core.interfaces import FeaturePreprocessor

_NUMERIC_KINDS = ("i", "u", "f")

def _is_numeric(series: pd.Series) -> bool:
    return series.dtype.kind in _NUMERIC_KINDS

class DefaultFeaturePreprocessor(FeaturePreprocessor):
    """ISP: a tiny class only responsible for building the preprocessor."""
    def build(self, X: pd.DataFrame, features: List[str]):
        feats_present = [f for f in features if f in X.columns]
        num_cols = [c for c in feats_present if _is_numeric(X[c])]
        cat_cols = [c for c in feats_present if c not in num_cols]

        transformers = []
        if num_cols:
            transformers.append(("num", "passthrough", num_cols))
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop"), feats_present
