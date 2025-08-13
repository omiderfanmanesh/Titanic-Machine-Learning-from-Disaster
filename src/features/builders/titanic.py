from typing import Dict, List, Optional, Any
import pandas as pd

from core.interfaces import ITransformer
from core.utils import LoggerFactory, DataConfig

from features.pipeline import build_pipeline
from features.encoding.orchestrator import EncodingOrchestrator
from features.missing.imputer import MissingValueHandler
from features.scaling.scaler import ScalingOrchestrator


class TitanicFeatureBuilder(ITransformer):
    """High-level orchestrator that applies FE pipeline, missing-value handling, encoding, and scaling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = LoggerFactory.get_logger(__name__)
        self.pipeline = None
        self.encoder = EncodingOrchestrator(self.config)
        self.imputer = MissingValueHandler()
        self.scaler = ScalingOrchestrator(enable=self.config.get("scale_features", True))
        self._is_fitted = False

        # NEW: freeze the exact post-encoding columns used for scaling
        self._fitted_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TitanicFeatureBuilder":
        self.logger.info("Fitting feature builder")

        # 1) Build & fit FE pipeline
        self.pipeline = build_pipeline(self.config)
        self.pipeline.fit(X, y)

        # 2) Transform once to get derived columns
        Xt = self.pipeline.transform(X)

        # 3) Impute BEFORE encoding
        Xt = self.imputer.handle(Xt)

        # 4) Fit encoders on imputed features
        cat_cols = self.config.get("categorical_columns", [])
        self.encoder.fit(Xt, y, categorical_cols=cat_cols)

        # 5) Encode to get the exact columns that the scaler will see
        Xt_enc = self.encoder.transform(Xt)

        # 6) Fit scaler on the encoded frame and freeze the column order
        self.scaler.fit(Xt_enc)
        self._fitted_columns = Xt_enc.columns.tolist()

        self._is_fitted = True
        self.logger.info("✅ Feature builder fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Feature builder must be fitted before transform")

        self.logger.info(f"Transforming data with {len(X)} samples")

        # Same order as in fit()
        Xt = self.pipeline.transform(X)
        Xt = self.imputer.handle(Xt)
        Xt = self.encoder.transform(Xt)

        # ✅ Ensure scaler sees the *same* columns in the *same* order
        if self._fitted_columns is not None:
            # Add any missing columns (e.g., OHE levels unseen in this batch)
            for col in self._fitted_columns:
                if col not in Xt.columns:
                    Xt[col] = 0
            # Drop unexpected extras (e.g., due to drift or config mismatch)
            extra_cols = [c for c in Xt.columns if c not in self._fitted_columns]
            if extra_cols:
                Xt = Xt.drop(columns=extra_cols)
            # Reorder to the frozen order
            Xt = Xt[self._fitted_columns]

        # Scale numeric features
        Xt = self.scaler.transform(Xt)

        self.logger.info(f"✅ Transformed data shape: {Xt.shape}")
        return Xt

    # Keep ITransformer contract happy
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        # Encoded feature names (post-encoding, pre-scaling) in frozen order
        return list(self._fitted_columns or self.encoder.feature_names())



