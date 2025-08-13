from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.interfaces import ITransformer
from core.utils import LoggerFactory



class BaseTransform(BaseEstimator, TransformerMixin, ITransformer):
    """Base class for all feature transformations."""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.is_fitted = False

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
