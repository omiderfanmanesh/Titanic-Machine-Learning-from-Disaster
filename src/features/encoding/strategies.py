from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
try:
    import category_encoders as ce  # optional dependency
except Exception:  # pragma: no cover - runtime optional
    ce = None  # type: ignore
from sklearn.preprocessing import LabelEncoder

from core import IEncoderStrategy

_MISSING = "__MISSING__"

def _as_string_frame(X: pd.DataFrame, col: str) -> pd.DataFrame:
    return X[[col]].astype("string").fillna(_MISSING)

class OneHotStrategy(IEncoderStrategy):
    def __init__(self, col: str, **kwargs: Any):
        self.col = col
        self._cols_out: List[str] = []
        self._use_ce = ce is not None
        if self._use_ce:
            self.enc = ce.OneHotEncoder(cols=[col], return_df=True, **kwargs)
        else:
            self._categories: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        Xc = _as_string_frame(X, self.col)
        if self._use_ce:
            self.enc.fit(Xc, y)
            self._cols_out = self.enc.transform(Xc.iloc[:1]).columns.tolist()
        else:
            cats = pd.Index(Xc[self.col].astype("string").fillna(_MISSING).unique()).tolist()
            self._categories = [str(c) for c in cats]
            self._cols_out = [f"{self.col}_{c}" for c in self._categories]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = _as_string_frame(X, self.col)
        if self._use_ce:
            df = self.enc.transform(Xc)
            # Robustness: category_encoders may emit NaN for some missing/unknown cases
            # Ensure a dense 0/1 frame without NaNs
            return df.fillna(0).astype(int)
        # pandas get_dummies fallback with fixed columns
        dummies = pd.get_dummies(Xc[self.col].astype("string").fillna(_MISSING), prefix=self.col)
        # Ensure all expected columns exist
        for c in self._cols_out:
            if c not in dummies.columns:
                dummies[c] = 0
        return dummies[self._cols_out].astype(int)

    def output_columns(self) -> List[str]:
        return self._cols_out

class OrdinalStrategy(IEncoderStrategy):
    def __init__(self, col: str, mapping=None, **_):
        self.col = col
        self._use_ce = ce is not None
        self._cols_out = [col]
        self._mapping = mapping
        if self._use_ce:
            self.enc = ce.OrdinalEncoder(
                cols=[col],
                mapping=mapping,
                handle_missing="return_nan",
                handle_unknown="value",
                return_df=True
            )
        else:
            self._internal_map: Dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        Xc = _as_string_frame(X, self.col)
        if self._use_ce:
            self.enc.fit(Xc, y)
        else:
            if self._mapping is not None:
                # convert ce-style mapping into dict
                mp = self._mapping[0]["mapping"] if isinstance(self._mapping, list) else self._mapping
                self._internal_map = {str(k): int(v) for k, v in mp.items()}
            else:
                cats = pd.Index(Xc[self.col].astype("string").fillna(_MISSING).unique()).tolist()
                self._internal_map = {str(v): i for i, v in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = _as_string_frame(X, self.col)
        if self._use_ce:
            return self.enc.transform(Xc)
        s = Xc[self.col].astype("string").fillna(_MISSING).map(lambda v: self._internal_map.get(str(v), -1))
        return pd.DataFrame({self.col: s.astype(int)}, index=X.index)

    def output_columns(self) -> List[str]:
        return self._cols_out

class TargetStrategy(IEncoderStrategy):
    def __init__(self, col: str, **kwargs: Any):
        if ce is None:
            raise ImportError("category_encoders is required for TargetStrategy. Install with: pip install category-encoders")
        self.col = col
        self.enc = ce.TargetEncoder(cols=[col], return_df=True, **kwargs)
        self._cols_out = [col]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            raise ValueError("TargetStrategy requires y")
        self.enc.fit(_as_string_frame(X, self.col), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.enc.transform(_as_string_frame(X, self.col))

    def output_columns(self) -> List[str]:
        return self._cols_out

class LabelStrategy(IEncoderStrategy):
    def __init__(self, col: str):
        self.col = col
        self.le = LabelEncoder()
        self._classes: np.ndarray = np.array([])
        self._cols_out = [col]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        s = _as_string_frame(X, self.col)[self.col]
        self.le.fit(s.astype(str))
        self._classes = self.le.classes_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        s = _as_string_frame(X, self.col)[self.col]
        known = set(self._classes.tolist())
        safe = s.map(lambda v: v if v in known else self._classes[0])
        out = pd.DataFrame({self.col: self.le.transform(safe.astype(str))}, index=X.index)
        return out

    def output_columns(self) -> List[str]:
        return self._cols_out


# -------------------------
# NEW: CatBoost / LOO / WOE
# -------------------------

class CatBoostStrategy(IEncoderStrategy):
    """
    Ordered target encoding that reduces leakage by using an internal permutation.
    Useful when you want target-based signal but less leakage than plain mean encoding.
    """
    def __init__(self, col: str, **kwargs: Any):
        if ce is None:
            raise ImportError("category_encoders is required for CatBoostStrategy. Install with: pip install category-encoders")
        self.col = col
        # 'a' is a smoothing parameter in ce.CatBoostEncoder
        self.enc = ce.CatBoostEncoder(cols=[col], return_df=True, **kwargs)
        self._cols_out = [col]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            raise ValueError("CatBoostStrategy requires y")
        self.enc.fit(_as_string_frame(X, self.col), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.enc.transform(_as_string_frame(X, self.col))

    def output_columns(self) -> List[str]:
        return self._cols_out


class LeaveOneOutStrategy(IEncoderStrategy):
    """
    Leave-one-out target encoding: for each row, uses the mean target
    of the category computed on all *other* rows. Reduces leakage vs plain mean.
    """
    def __init__(self, col: str, **kwargs: Any):
        if ce is None:
            raise ImportError("category_encoders is required for LeaveOneOutStrategy. Install with: pip install category-encoders")
        self.col = col
        # 'sigma' controls Gaussian noise added during training (regularization)
        self.enc = ce.LeaveOneOutEncoder(cols=[col], return_df=True, **kwargs)
        self._cols_out = [col]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            raise ValueError("LeaveOneOutStrategy requires y")
        self.enc.fit(_as_string_frame(X, self.col), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.enc.transform(_as_string_frame(X, self.col))

    def output_columns(self) -> List[str]:
        return self._cols_out


class WOEStrategy(IEncoderStrategy):
    """
    Weight of Evidence encoding (typically for binary classification).
    Encodes each category with log((pos_rate)/(neg_rate)).
    """
    def __init__(self, col: str, **kwargs: Any):
        if ce is None:
            raise ImportError("category_encoders is required for WOEStrategy. Install with: pip install category-encoders")
        self.col = col
        self.enc = ce.WOEEncoder(cols=[col], return_df=True, **kwargs)
        self._cols_out = [col]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            raise ValueError("WOEStrategy requires y (binary target recommended)")
        self.enc.fit(_as_string_frame(X, self.col), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.enc.transform(_as_string_frame(X, self.col))

    def output_columns(self) -> List[str]:
        return self._cols_out
