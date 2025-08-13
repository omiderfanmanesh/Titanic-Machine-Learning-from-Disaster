from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

from src.core import IEncoderStrategy

_MISSING = "__MISSING__"

def _as_string_frame(X: pd.DataFrame, col: str) -> pd.DataFrame:
    return X[[col]].astype("string").fillna(_MISSING)

class OneHotStrategy(IEncoderStrategy):
    def __init__(self, col: str, **kwargs: Any):
        self.col = col
        self.enc = ce.OneHotEncoder(cols=[col], return_df=True, **kwargs)
        self._cols_out: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        Xc = _as_string_frame(X, self.col)
        self.enc.fit(Xc, y)
        self._cols_out = self.enc.transform(Xc.iloc[:1]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.enc.transform(_as_string_frame(X, self.col))

    def output_columns(self) -> List[str]:
        return self._cols_out

class OrdinalStrategy(IEncoderStrategy):
    def __init__(self, col: str, mapping=None, **_):
        self.col = col
        self.enc = ce.OrdinalEncoder(
            cols=[col],
            mapping=mapping,
            handle_missing="return_nan",
            handle_unknown="value",
            return_df=True
        )
        self._cols_out = [col]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.enc.fit(_as_string_frame(X, self.col), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.enc.transform(_as_string_frame(X, self.col))

    def output_columns(self) -> List[str]:
        return self._cols_out

class TargetStrategy(IEncoderStrategy):
    def __init__(self, col: str, **kwargs: Any):
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
