from typing import Dict, Any, Optional, List
import pandas as pd
from .factory import build_encoder
from core import IEncoderStrategy

class EncodingOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self._encoders: Dict[str, IEncoderStrategy] = {}
        from core.utils import LoggerFactory
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

    def _encoding_cfg(self) -> Dict[str, Any]:
        enc = (self.config.get("encoding") or {})
        default = enc.get("default", {"method": "onehot", "handle_missing": "value", "handle_unknown": "ignore"})
        return {"default": default, "per_column": enc.get("per_column", {})}

    def _col_cfg(self, col: str) -> Dict[str, Any]:
        enc = self._encoding_cfg()
        return {**enc["default"], **enc["per_column"].get(col, {})}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, categorical_cols: Optional[List[str]] = None):
        if not self.config.get("encode_categorical", True):
            return self

        if not categorical_cols:
            categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.logger.info(f"Encoding: candidate categorical columns={len(categorical_cols)}")

        skip = set(self.config.get("skip_encoding_columns", []) or [])
        built = 0
        for col in categorical_cols:
            if col in skip or col not in X.columns:
                continue
            cfg = self._col_cfg(col)
            enc = build_encoder(col, cfg).fit(X, y)
            self._encoders[col] = enc
            built += 1
        self.logger.info(f"Encoding: built encoders for {built} columns (skipped {len(skip)})")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._encoders:
            return X
        Xo = X.copy()
        keep_original = bool(self.config.get("add_original_columns", self.config.get("add_original_column", False)))
        for col, enc in self._encoders.items():
            if col not in Xo.columns:
                continue
            enc_df = enc.transform(Xo)
            if not keep_original:
                Xo.drop(columns=[col], inplace=True)
            Xo = pd.concat([Xo, enc_df], axis=1)
        return Xo

    def feature_names(self) -> List[str]:
        names: List[str] = []
        for enc in self._encoders.values():
            names.extend(enc.output_columns())
        return names
