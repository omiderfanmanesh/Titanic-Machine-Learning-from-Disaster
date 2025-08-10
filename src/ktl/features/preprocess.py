from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeaturesConfig:
    """Configuration for feature preprocessing.

    Attributes
    ----------
    scale_numeric : bool
        Whether to scale numeric features.
    encode_categorical : bool
        Whether to one-hot encode categoricals.
    impute_numeric : str
        Strategy for numeric imputation.
    impute_categorical : str
        Strategy for categorical imputation.
    """

    scale_numeric: bool = True
    encode_categorical: bool = True
    impute_numeric: str = "median"
    impute_categorical: str = "most_frequent"
    add_family: bool = False
    add_is_alone: bool = False
    add_title: bool = False
    add_deck: bool = False
    add_ticket_group_size: bool = False
    log_fare: bool = False
    bin_age: bool = False
    rare_title_threshold: int = 10


class _FeatureEngineer(BaseEstimator, TransformerMixin):
    """Fold-safe feature engineering transformer for Titanic-style data.

    Adds engineered columns to a copy of the input DataFrame. Fit stores only
    per-fold artifacts like `ticket_counts_` and `title_mapping_` to avoid leakage.
    """

    def __init__(
        self,
        add_family: bool = False,
        add_is_alone: bool = False,
        add_title: bool = False,
        add_deck: bool = False,
        add_ticket_group_size: bool = False,
        log_fare: bool = False,
        bin_age: bool = False,
        rare_title_threshold: int = 10,
    ) -> None:
        self.add_family = add_family
        self.add_is_alone = add_is_alone
        self.add_title = add_title
        self.add_deck = add_deck
        self.add_ticket_group_size = add_ticket_group_size
        self.log_fare = log_fare
        self.bin_age = bin_age
        self.rare_title_threshold = rare_title_threshold
        # fitted artifacts
        self.ticket_counts_: Dict[str, int] = {}
        self.title_replace_map_: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):  # type: ignore[override]
        if self.add_ticket_group_size and 'Ticket' in X.columns:
            vc = X['Ticket'].astype(str).value_counts()
            self.ticket_counts_ = vc.to_dict()
        else:
            self.ticket_counts_ = {}

        if self.add_title and 'Name' in X.columns:
            def extract_title(name: str) -> str:
                m = re.search(r',\s*([^\.]+)\.', str(name))
                return m.group(1).strip() if m else 'Unknown'
            titles = X['Name'].map(extract_title)
            # normalize common variants
            titles = titles.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
            counts = titles.value_counts()
            rare = counts[counts < int(self.rare_title_threshold)].index
            rep = {t: ('Rare' if t in rare else t) for t in counts.index}
            self.title_replace_map_ = rep
        else:
            self.title_replace_map_ = {}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        df = X.copy()
        if self.add_family and {'SibSp', 'Parch'}.issubset(df.columns):
            df['FamilySize'] = df['SibSp'].astype(float) + df['Parch'].astype(float) + 1.0
        if self.add_is_alone:
            if 'FamilySize' in df.columns:
                df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            elif {'SibSp', 'Parch'}.issubset(df.columns):
                tmp = df['SibSp'].astype(float) + df['Parch'].astype(float) + 1.0
                df['IsAlone'] = (tmp == 1).astype(int)
        if self.add_title and 'Name' in df.columns:
            def extract_title(name: str) -> str:
                m = re.search(r',\s*([^\.]+)\.', str(name))
                return m.group(1).strip() if m else 'Unknown'
            titles = df['Name'].map(extract_title).replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
            if self.title_replace_map_:
                titles = titles.map(lambda t: self.title_replace_map_.get(t, t))
            df['Title'] = titles
        if self.add_deck and 'Cabin' in df.columns:
            deck = df['Cabin'].astype(str).str[0]
            df['Deck'] = np.where(deck.isin(list('ABCDEFGT')), deck, 'U')
        if self.add_ticket_group_size and 'Ticket' in df.columns:
            df['TicketGroupSize'] = df['Ticket'].astype(str).map(self.ticket_counts_).fillna(1).astype(int)
        if self.log_fare and 'Fare' in df.columns:
            df['LogFare'] = np.log1p(df['Fare'])
        if self.bin_age and 'Age' in df.columns:
            bins = pd.cut(df['Age'], [0, 5, 12, 18, 30, 45, 60, 80])
            df['AgeBin'] = bins.astype(str)
        return df


class PreprocessorBuilder:
    """Builds a preprocessing pipeline (feateng + ColumnTransformer).

    Returns a transformer compliant with sklearn pipeline APIs.
    """

    @staticmethod
    def build(df: pd.DataFrame, target: str, cfg: Optional[FeaturesConfig] = None) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
        cfg = cfg or FeaturesConfig()
        feats = df.drop(columns=[target]) if target in df.columns else df.copy()

        # Anticipate engineered columns for selection
        eng_num: List[str] = []
        eng_cat: List[str] = []
        if cfg.add_family:
            eng_num.append('FamilySize')
        if cfg.add_is_alone:
            eng_num.append('IsAlone')
        if cfg.add_ticket_group_size:
            eng_num.append('TicketGroupSize')
        if cfg.log_fare:
            eng_num.append('LogFare')
        if cfg.add_title:
            eng_cat.append('Title')
        if cfg.add_deck:
            eng_cat.append('Deck')
        if cfg.bin_age:
            eng_cat.append('AgeBin')

        # Base column detection
        base_num = feats.select_dtypes(include=[np.number]).columns.tolist()
        base_cat = feats.select_dtypes(exclude=[np.number]).columns.tolist()

        # Combined lists after feateng
        num_cols = sorted(list(dict.fromkeys(base_num + eng_num)))
        cat_cols = sorted(list(dict.fromkeys(base_cat + eng_cat)))
        dt_cols: List[str] = []  # reserved for future

        num_steps = []
        if num_cols:
            num_steps.append(("imputer", SimpleImputer(strategy=cfg.impute_numeric)))
            if cfg.scale_numeric:
                num_steps.append(("scaler", StandardScaler(with_mean=True)))

        cat_steps = []
        if cat_cols and cfg.encode_categorical:
            cat_steps.append(("imputer", SimpleImputer(strategy=cfg.impute_categorical)))
            cat_steps.append(("ohe", OneHotEncoder(handle_unknown="ignore")))

        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline(num_steps), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline(cat_steps), cat_cols))

        ct = ColumnTransformer(transformers=transformers, remainder="drop")

        # Prepend feature engineering so engineered columns exist before CT selection
        preproc = Pipeline([
            ("feateng", _FeatureEngineer(
                add_family=cfg.add_family,
                add_is_alone=cfg.add_is_alone,
                add_title=cfg.add_title,
                add_deck=cfg.add_deck,
                add_ticket_group_size=cfg.add_ticket_group_size,
                log_fare=cfg.log_fare,
                bin_age=cfg.bin_age,
                rare_title_threshold=int(cfg.rare_title_threshold),
            )),
            ("ct", ct),
        ])

        # Return the pipeline but keep signature
        return preproc, num_cols, cat_cols, dt_cols
