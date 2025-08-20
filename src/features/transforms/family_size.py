import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.interfaces import ITransformer
from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class FamilySizeTransform(BaseTransform):
    """Creates FamilySize and a grouped categorical FamilySizeGrouped.

    FamilySize = SibSp + Parch + 1
    FamilySizeGrouped (string):
      - Alone (size == 1)
      - Small (2–4)
      - Medium (5–6)
      - Large (>=7)

    This grouping generalizes the example mapping (1:Alone; 2–4:Small; 5–6:Medium; 7,8,11:Large).
    """

    def __init__(self, sibsp_col: str = "SibSp", parch_col: str = "Parch"):
        super().__init__(name="FamilySizeTransform")
        self.sibsp_col = sibsp_col
        self.parch_col = parch_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FamilySizeTransform":
        """Fit transform - no parameters to learn."""
        required_cols = [self.sibsp_col, self.parch_col]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Declare new columns (IsAlone disabled by request)
        self._set_new_cols(["FamilySize", "FamilySizeGrouped"])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by adding family size features."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        X = X.copy()
        X["FamilySize"] = pd.to_numeric(X[self.sibsp_col], errors="coerce").fillna(0) + \
                           pd.to_numeric(X[self.parch_col], errors="coerce").fillna(0) + 1
        X["FamilySize"] = X["FamilySize"].astype(int)
        # IsAlone disabled (kept out of outputs)

        # Grouping into categories
        def _group(sz: int) -> str:
            if sz <= 1:
                return "Alone"
            if 2 <= sz <= 4:
                return "Small"
            if 5 <= sz <= 6:
                return "Medium"
            return "Large"

        X["FamilySizeGrouped"] = X["FamilySize"].apply(_group).astype("string")

        try:
            counts = X["FamilySizeGrouped"].value_counts().to_dict()
            self.logger.info(
                f"FamilySize: range={int(X['FamilySize'].min())}-{int(X['FamilySize'].max())}, "
                f"grouped_dist={counts}"
            )
        except Exception:
            pass

        return X
