from features.pipeline.pipeline import FeaturePipeline
from features.transforms import (
    FamilySizeTransform,
    TitleTransform,
    DeckTransform,
    TicketGroupTransform,
    FareTransform,
    AgeBinningTransform,
)

TRANSFORM_MAP = {
    "FamilySizeTransform": FamilySizeTransform,
    "TitleTransform": TitleTransform,
    "DeckTransform": DeckTransform,
    "TicketGroupTransform": TicketGroupTransform,
    "FareTransform": FareTransform,
    "AgeBinningTransform": AgeBinningTransform,
}

def build_pipeline_from_config(stage_list: list, config: dict):
    transforms = []
    for name in stage_list:
        if name not in TRANSFORM_MAP:
            raise ValueError(f"Unknown transform: {name}")
        # Pass config to transforms if they take parameters
        if name == "FareTransform":
            transforms.append(FareTransform(log_transform=config.get("log_transform_fare", False)))
        elif name == "AgeBinningTransform":
            transforms.append(AgeBinningTransform(n_bins=config.get("age_bins", 5)))
        else:
            transforms.append(TRANSFORM_MAP[name]())
    return FeaturePipeline(transforms)

def build_pipeline_pre(config):
    stage_list = config.get("feature_engineering", {}).get("pre_impute", [])
    return build_pipeline_from_config(stage_list, config)

def build_pipeline_post(config):
    stage_list = config.get("feature_engineering", {}).get("post_impute", [])
    return build_pipeline_from_config(stage_list, config)
