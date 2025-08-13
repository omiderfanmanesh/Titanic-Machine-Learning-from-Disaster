from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
from core.interfaces import ImputationStrategy

class ForwardFillStrategy(ImputationStrategy):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.column] = X[self.column].ffill()
        return X

class BackwardFillStrategy(ImputationStrategy):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.column] = X[self.column].bfill()
        return X
