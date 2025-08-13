"""Atomic feature transformations following SOLID principles."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.interfaces import ITransformer
from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class FeaturePipeline(BaseTransform):
    """Pipeline for chaining multiple feature transforms."""

    def __init__(self, transforms: List[BaseTransform]):
        super().__init__()
        self.transforms = transforms

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeaturePipeline":
        """Fit all transforms sequentially."""
        X_current = X.copy()

        for i, transform in enumerate(self.transforms):
            self.logger.debug(f"Fitting transform {i + 1}/{len(self.transforms)}: {type(transform).__name__}")
            transform.fit(X_current, y)
            X_current = transform.transform(X_current)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transforms sequentially."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before calling transform")

        X_current = X.copy()

        for transform in self.transforms:
            X_current = transform.transform(X_current)

        return X_current