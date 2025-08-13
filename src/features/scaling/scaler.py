from typing import List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ScalingOrchestrator:
    def __init__(self, enable: bool = True):
        self.enable = enable
        self.scaler: Optional[StandardScaler] = None
        self.scale_cols: List[str] = []

    def fit(self, X: pd.DataFrame) -> "ScalingOrchestrator":
        if not self.enable:
            return self
        numeric_cols = X.select_dtypes(include=["number"]).columns
        exclude = ["PassengerId", "IsAlone"] + [c for c in numeric_cols if X[c].nunique() == 2]
        self.scale_cols = [c for c in numeric_cols if c not in exclude]
        if self.scale_cols:
            self.scaler = StandardScaler().fit(X[self.scale_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enable or not self.scale_cols or self.scaler is None:
            return X
        X = X.copy()
        cols = [c for c in self.scale_cols if c in X.columns]
        if cols:
            X[cols] = self.scaler.transform(X[cols])
        return X
