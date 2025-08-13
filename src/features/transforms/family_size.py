import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.interfaces import ITransformer
from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class FamilySizeTransform(BaseTransform):
    """Creates family size and is_alone features."""

    def __init__(self, sibsp_col: str = "SibSp", parch_col: str = "Parch"):
        super().__init__()
        self.sibsp_col = sibsp_col
        self.parch_col = parch_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FamilySizeTransform":
        """Fit transform - no parameters to learn."""
        required_cols = [self.sibsp_col, self.parch_col]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by adding family size features."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        X = X.copy()
        X["FamilySize"] = X[self.sibsp_col] + X[self.parch_col] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        self.logger.debug(f"Created FamilySize (range: {X['FamilySize'].min()}-{X['FamilySize'].max()}) "
                          f"and IsAlone ({X['IsAlone'].sum()}/{len(X)} alone)")

        return X

