"""Feature engineering and transformation components."""
from core import DataConfig
from features.builders.titanic import TitanicFeatureBuilder
from features.builders.advanced import AdvancedFeatureBuilder

__all__ = [
    "TitanicFeatureBuilder",
    "create_feature_builder", 
    "AdvancedFeatureBuilder"
]

def create_feature_builder(config: DataConfig, debug: bool = False):
    # Import here to avoid circular dependency

    cfg = config.to_dict(include_none=True)
    if debug:
        cfg["debug_mode"] = True
    return TitanicFeatureBuilder(cfg)