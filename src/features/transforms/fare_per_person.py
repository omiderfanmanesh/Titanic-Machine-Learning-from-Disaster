from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from features.transforms.base import BaseTransform


class FarePerPersonTransform(BaseTransform):
    """
    Adds FarePerPerson = Fare / FamilySize (with safe division and clipping).
    Requires: Fare (imputed), FamilySize (from FamilySizeTransform).
    """

    def __init__(self, clip_min: float = 0.0, clip_max: Optional[float] = None):
        super().__init__(name="FarePerPersonTransform")
        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FarePerPersonTransform":
        self._set_new_cols(["FarePerPerson"])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        Xo = X.copy()
        fare = pd.to_numeric(Xo.get("Fare"), errors="coerce")
        fam = pd.to_numeric(Xo.get("FamilySize"), errors="coerce")
        fam = fam.replace(0, np.nan)  # avoid division by zero
        fpp = fare / fam
        # handle NaNs from zero family or missing fare
        fpp = fpp.replace([np.inf, -np.inf], np.nan).fillna(fare)  # fallback to Fare when size invalid
        # optional clipping
        if self.clip_min is not None:
            fpp = fpp.clip(lower=self.clip_min)
        if self.clip_max is not None:
            fpp = fpp.clip(upper=self.clip_max)
        Xo["FarePerPerson"] = fpp.astype(float)
        return Xo

