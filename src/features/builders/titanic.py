from typing import Dict, List, Optional, Any
import pandas as pd

from core.interfaces import ITransformer
from core.utils import LoggerFactory

# these builders should read config["feature_engineering"] internally
from features.pipeline import build_pipeline_pre, build_pipeline_post

from features.encoding.orchestrator import EncodingOrchestrator
from features.missing.orchestrator import ImputationOrchestrator
from features.scaling.scaler import ScalingOrchestrator


class TitanicFeatureBuilder(ITransformer):
    """
    1) FE (pre-impute)   — from config.feature_engineering.pre_impute
    2) Imputation        — per-column strategies (config.imputation)
    3) FE (post-impute)  — from config.feature_engineering.post_impute
    4) Encoding          — config.encoding
    5) Scaling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = LoggerFactory.get_logger(__name__)

        # Pipelines built from config-driven stage lists
        self.pipeline_pre = None
        self.pipeline_post = None

        self.encoder = EncodingOrchestrator(self.config)

        # Global on/off; orchestrator reads `imputation:` details
        self.handle_missing = bool(self.config.get("handle_missing", True))
        self.imputer = ImputationOrchestrator(self.config) if self.handle_missing else None

        self.scaler = ScalingOrchestrator(enable=self.config.get("scale_features", True))

        self._is_fitted = False
        self._fitted_columns: Optional[List[str]] = None  # frozen post-encoding schema

    # -------- fit --------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TitanicFeatureBuilder":
        self.logger.info("Fitting feature builder")

        # 1) FE (pre-impute)
        self.pipeline_pre = build_pipeline_pre(self.config)
        self.pipeline_pre.fit(X, y)
        Xt = self.pipeline_pre.transform(X)

        # 2) Impute BEFORE encoding
        if self.imputer is not None:
            self.imputer.fit(Xt, y)
            Xt = self.imputer.transform(Xt)
            # optional: brief report
            try:
                rep = self.imputer.get_report()
                if not rep.empty:
                    self.logger.info("Imputation report (head):\n%s", rep.head().to_string())
            except Exception:
                pass

        # 3) FE (post-impute)
        self.pipeline_post = build_pipeline_post(self.config)
        self.pipeline_post.fit(Xt, y)
        Xt = self.pipeline_post.transform(Xt)

        # 4) Encoding
        cat_cols = self.config.get("categorical_columns", [])
        self.encoder.fit(Xt, y, categorical_cols=cat_cols)
        Xt_enc = self.encoder.transform(Xt)

        # 5) Scaling (+ freeze schema)
        self.scaler.fit(Xt_enc)
        self._fitted_columns = Xt_enc.columns.tolist()

        self._is_fitted = True
        self.logger.info("✅ Feature builder fitted successfully")
        return self

    # -------- transform --------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Feature builder must be fitted before transform")

        self.logger.info(f"Transforming data with {len(X)} samples")

        # Same order as fit()
        Xt = self.pipeline_pre.transform(X)
        if self.imputer is not None:
            Xt = self.imputer.transform(Xt)
        Xt = self.pipeline_post.transform(Xt)
        Xt = self.encoder.transform(Xt)

        # Align to frozen encoded schema
        if self._fitted_columns is not None:
            for col in self._fitted_columns:
                if col not in Xt.columns:
                    Xt[col] = 0
            extra_cols = [c for c in Xt.columns if c not in self._fitted_columns]
            if extra_cols:
                Xt = Xt.drop(columns=extra_cols)
            Xt = Xt[self._fitted_columns]

        Xt = self.scaler.transform(Xt)
        self.logger.info(f"✅ Transformed data shape: {Xt.shape}")
        return Xt

    # -------- helpers --------
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        return list(self._fitted_columns or self.encoder.feature_names())
