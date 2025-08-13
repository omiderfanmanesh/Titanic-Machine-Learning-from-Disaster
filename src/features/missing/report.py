from __future__ import annotations
from typing import Dict, Any
import pandas as pd

class ImputationReport:
    """SRP: holds metrics/summary only."""
    def __init__(self):
        self.rows: Dict[str, Dict[str, Any]] = {}

    def init_row(self, column: str, method: str, missing_rate: float, fit_rows: int):
        self.rows[column] = {"method": method, "missing_rate": missing_rate, "fit_rows": fit_rows}

    def update(self, column: str, extra: Dict[str, Any]):
        if column in self.rows:
            self.rows[column].update(extra)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).T if self.rows else pd.DataFrame()
