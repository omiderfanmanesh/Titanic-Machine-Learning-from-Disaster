from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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


class PreprocessorBuilder:
    """Builds a ColumnTransformer given a dataframe and config."""

    @staticmethod
    def build(df: pd.DataFrame, target: str, cfg: Optional[FeaturesConfig] = None) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
        cfg = cfg or FeaturesConfig()
        feats = df.drop(columns=[target]) if target in df.columns else df.copy()
        num_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = feats.select_dtypes(exclude=[np.number]).columns.tolist()
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
            from sklearn.pipeline import Pipeline

            transformers.append(("num", Pipeline(num_steps), num_cols))
        if cat_cols:
            from sklearn.pipeline import Pipeline

            transformers.append(("cat", Pipeline(cat_steps), cat_cols))

        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        return ct, num_cols, cat_cols, dt_cols
