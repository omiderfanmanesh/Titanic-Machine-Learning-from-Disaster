from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.impute import SimpleImputer
from core.interfaces import ImputationStrategy

class SimpleImputerStrategy(ImputationStrategy):
    def __init__(self, column: str, plan: Dict[str, Any]):
        super().__init__(column, plan)
        self.imp: Optional[SimpleImputer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        strat = self.plan["method"]
        if strat == "constant":
            fill_value = self.plan.get("fill_value", "Unknown")
            self.imp = SimpleImputer(strategy="constant", fill_value=fill_value)
        else:
            self.imp = SimpleImputer(strategy=strat)
        self.imp.fit(X[[self.column]])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[[self.column]] = self.imp.transform(X[[self.column]])
        return X
