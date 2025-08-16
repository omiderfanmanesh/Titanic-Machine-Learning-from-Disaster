from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd
from features.transforms.base import BaseTransform


class TitleTransform(BaseTransform):
    """
    Extract and normalize titles and related name features.

    Behavior:
      - rare_threshold: int  -> collapse infrequent titles to 'Rare' based on training freq.
      - rare_threshold: None -> no frequency collapsing; only apply DEFAULT_TITLE_MAP (and any user overrides).
    """

    DEFAULT_TITLE_MAP: Dict[str, str] = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Royal", "Countess": "Royal", "Dona": "Royal", "Sir": "Royal", "Don": "Royal",
        "Jonkheer": "Rare", "Capt": "Rare", "Col": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare","Master": "Rare",
    }

    def __init__(
        self,
        name_col: str = "Name",
        rare_threshold: Optional[int] = 10,
        # Backward-compat alias for configs using 'rare_title_threshold'
        rare_title_threshold: Optional[int] = None,
        # Allow callers to extend/override the mapping
        title_map_override: Optional[Dict[str, str]] = None,
        unknown_token: str = "Unknown",
        keep_unseen_if_collapsing: bool = False,
    ):
        super().__init__()
        # Resolve alias
        if rare_title_threshold is not None and rare_threshold is not None and rare_title_threshold != rare_threshold:
            raise ValueError(
                f"Inconsistent thresholds: rare_threshold={rare_threshold} "
                f"vs rare_title_threshold={rare_title_threshold}"
            )
        self.name_col = name_col
        self.rare_threshold: Optional[int] = rare_title_threshold if rare_title_threshold is not None else rare_threshold
        self.title_mapping: Optional[Dict[str, str]] = None
        self.title_map_override = dict(title_map_override) if title_map_override else {}
        self.unknown_token = unknown_token
        self.keep_unseen_if_collapsing = keep_unseen_if_collapsing

    @staticmethod
    def extract_title(full_name: str) -> Optional[str]:
        """Extract raw title (no dot), e.g. 'Mr.' -> 'Mr'. Returns None if not found."""
        if pd.isna(full_name):
            return None
        m = re.search(r" ([A-Za-z]+)\.", str(full_name))
        return m.group(1) if m else None

    def _split_after_comma(self, s: str) -> Optional[str]:
        """Return the substring after the first comma, stripped; None if no comma."""
        if pd.isna(s):
            return None
        parts = str(s).split(",", 1)
        return parts[1].strip() if len(parts) > 1 else None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TitleTransform":
        if self.name_col not in X.columns:
            raise ValueError(f"Column {self.name_col} not found")

        titles = X[self.name_col].map(self.extract_title)

        mapping: Dict[str, str] = {}
        # Frequency-based collapsing
        if self.rare_threshold is not None:
            counts = titles.value_counts(dropna=True)
            for title, cnt in counts.items():
                mapping[title] = title if cnt >= self.rare_threshold else "Rare"

        # Domain defaults then user overrides (overrides take precedence)
        mapping.update(self.DEFAULT_TITLE_MAP)
        mapping.update(self.title_map_override)

        self.title_mapping = mapping

        # Logging (guard if BaseTransform lacks logger)
        try:
            cats = sorted(set(self.title_mapping.values()))
            self.logger.info(
                "Configured title mapping with %d categories: %s (rare_threshold=%s, keep_unseen_if_collapsing=%s)",
                len(cats), cats, "None" if self.rare_threshold is None else self.rare_threshold,
                self.keep_unseen_if_collapsing,
            )
        except Exception:
            pass

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")

        X = X.copy()

        # Raw parsed bits (robust to missing comma/title)
        X["Surname"] = X[self.name_col].astype("string").str.split(",").str[0].str.strip()
        title_first_middle = X[self.name_col].map(self._split_after_comma)
        X["Title_First_Middle"] = title_first_middle.astype("string")

        X["Title_Raw"] = X[self.name_col].map(self.extract_title).astype("string")
        X["Title_Raw"] = X["Title_Raw"].str.replace(".", "", regex=False).str.strip()

        # First + middle names (after the raw title token)
        X["First_Middle"] = (
            X["Title_First_Middle"]
            .str.split(" ")
            .str[1:]
            .str.join(" ")
            .replace("", pd.NA)
            .fillna(self.unknown_token)
            .str.strip()
        )

        # Maiden name (inside parentheses)
        X["MaidenName"] = X[self.name_col].astype("string").str.extract(r"\((.*?)\)")

        # Final normalized Title
        if self.rare_threshold is None:
            # Only apply explicit mappings; otherwise keep raw; fill missing with unknown
            X["Title"] = (
                X["Title_Raw"]
                .map(self.title_mapping)
                .fillna(X["Title_Raw"])
                .fillna(self.unknown_token)
            )
        else:
            mapped = X["Title_Raw"].map(self.title_mapping)
            if self.keep_unseen_if_collapsing:
                X["Title"] = mapped.fillna(X["Title_Raw"]).fillna(self.unknown_token)
            else:
                X["Title"] = mapped.fillna("Rare")

        return X
