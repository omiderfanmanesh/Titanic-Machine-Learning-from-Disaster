from __future__ import annotations
from typing import Optional
import pandas as pd

from features.transforms.base import BaseTransform


class FamilyRoleTransform(BaseTransform):
    """
    Adds two common Titanic features:
      - IsChild: 1 if Age < 16, else 0
      - IsMother: 1 if female adult with Parch>0 and Title indicates not 'Miss'
    Requires columns: Age (imputed), Sex, Parch, Title
    """

    def __init__(self):
        super().__init__(name="FamilyRoleTransform")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FamilyRoleTransform":
        self._set_new_cols(["IsChild", "IsMother"])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        Xo = X.copy()
        # IsChild: Age strictly less than 16
        age = pd.to_numeric(Xo.get("Age"), errors="coerce")
        Xo["IsChild"] = (age < 16).astype("int8")

        # IsMother: female adult (>=18), has Parch>0, and title not Miss
        sex = Xo.get("Sex").astype("string")
        parch = pd.to_numeric(Xo.get("Parch"), errors="coerce").fillna(0)
        title = Xo.get("Title").astype("string")
        is_adult = (age >= 18)
        not_miss = ~(title.str.lower() == "miss")
        is_female = (sex.str.lower() == "female")
        Xo["IsMother"] = (is_female & is_adult.fillna(False) & (parch > 0) & not_miss.fillna(False)).astype("int8")

        return Xo

