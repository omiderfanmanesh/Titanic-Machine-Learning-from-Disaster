from typing import Dict, Any
from .strategies import (
    OneHotStrategy,
    OrdinalStrategy,
    TargetStrategy,
    LabelStrategy,
    CatBoostStrategy,
    LeaveOneOutStrategy,
    WOEStrategy,
)
from src.core import IEncoderStrategy

def build_encoder(col: str, cfg: Dict[str, Any]) -> IEncoderStrategy:
    method = (cfg.get("method") or "onehot").lower()

    if method == "onehot":
        return OneHotStrategy(
            col,
            handle_missing=cfg.get("handle_missing", "value"),
            handle_unknown=cfg.get("handle_unknown", "ignore"),
            drop_invariant=cfg.get("drop_invariant", False),
            use_cat_names=True,  # 👈 add this
        )

    if method == "ordinal":
        mapping = None
        if "categories" in cfg and cfg["categories"]:
            cats = cfg["categories"][0]
            mapping = [{"col": col, "mapping": {v: i for i, v in enumerate(cats)}, "data_type": "string"}]
        return OrdinalStrategy(col, mapping=mapping)

    if method == "target":
        return TargetStrategy(col, smoothing=cfg.get("smoothing", 10.0))

    if method == "label":
        return LabelStrategy(col)

    # NEW
    if method == "catboost":
        return CatBoostStrategy(col, a=cfg.get("a", 1.0))

    if method in ("leave_one_out", "loo"):
        return LeaveOneOutStrategy(col, sigma=cfg.get("sigma", 0.0))

    if method == "woe":
        return WOEStrategy(col)

    raise ValueError(f"Unknown encoding method '{method}' for '{col}'")
