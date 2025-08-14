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

        # 4.5) Remove original columns that have been transformed/encoded
        final_columns = self._get_final_columns(Xt_enc)
        Xt_enc = Xt_enc[final_columns]

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

    def _get_final_columns(self, Xt: pd.DataFrame) -> List[str]:
        """
        Determine which columns should be kept in the final dataset and order them properly.
        Orders: PassengerId first, then features, then Survived last.
        """
        # Get ID and target columns from config
        id_col = self.config.get("id_column", "PassengerId")
        target_col = self.config.get("target_column", "Survived")

        # Start with all current columns
        all_cols = set(Xt.columns)

        # Define original columns that should be removed (they've been transformed)
        original_cols_to_remove = {
            "Name", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin",
            "Sex", "Embarked", "Pclass"
        }

        # Remove original columns from the set
        final_cols = all_cols - original_cols_to_remove

        # Convert to list and create ordered column list
        ordered_cols = []

        # 1. Add ID column first (if present)
        if id_col in final_cols:
            ordered_cols.append(id_col)
            final_cols.remove(id_col)

        # 2. Add target column to separate list (will be added last)
        target_to_add = None
        if target_col in final_cols:
            target_to_add = target_col
            final_cols.remove(target_col)

        # 3. Add all feature columns (sorted for consistency)
        feature_cols = sorted(list(final_cols))
        ordered_cols.extend(feature_cols)

        # 4. Add target column last (if present)
        if target_to_add:
            ordered_cols.append(target_to_add)

        self.logger.info(f"Column order: {id_col} first, {len(feature_cols)} features, {target_col} last")
        self.logger.info(f"Removed original columns: {sorted(original_cols_to_remove & all_cols)}")

        return ordered_cols
