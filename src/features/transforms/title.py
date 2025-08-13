from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd
from core.utils import LoggerFactory
from features.transforms.base import BaseTransform


class TitleTransform(BaseTransform):
    """Extracts and encodes titles from passenger names."""

    def __init__(self, name_col: str = "Name", rare_threshold: int = 10):
        super().__init__()
        self.name_col = name_col
        self.rare_threshold = rare_threshold
        self.title_mapping: Optional[Dict[str, str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TitleTransform":
        """Learn title mappings from training data."""
        if self.name_col not in X.columns:
            raise ValueError(f"Column {self.name_col} not found")

        # Extract titles
        titles = X[self.name_col].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Count title frequencies
        title_counts = titles.value_counts()

        # Map rare titles to 'Rare'
        self.title_mapping = {}
        for title, count in title_counts.items():
            if count >= self.rare_threshold:
                self.title_mapping[title] = title
            else:
                self.title_mapping[title] = "Rare"

        self.logger.info(f"Learned {len(set(self.title_mapping.values()))} title categories: "
                        f"{sorted(set(self.title_mapping.values()))}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by extracting and mapping titles."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        X = X.copy()

        # Extract titles
        titles = X[self.name_col].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Map using learned mapping, default to 'Rare' for unseen titles
        X["Title"] = titles.map(self.title_mapping).fillna("Rare")

        return X

