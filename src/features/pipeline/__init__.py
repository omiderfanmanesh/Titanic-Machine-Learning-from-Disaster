from features.pipeline.pipeline import FeaturePipeline
from features.transforms import (
    FamilySizeTransform,
    TitleTransform,
    FamilyTransform,
    DeckTransform,
    MarriedTransform,
    TicketGroupTransform,
    TicketFrequencyTransform,
    FareTransform,
    FareBinningTransform,
    AgeBinningTransform, AgeImputeByTitleTransform,
)
from features.transforms.ticket_parse import TicketParseTransform
from features.transforms.child_mother import FamilyRoleTransform
from features.transforms.fare_per_person import FarePerPersonTransform

TRANSFORM_MAP = {
    "FamilySizeTransform": FamilySizeTransform,
    "TitleTransform": TitleTransform,
    "FamilyTransform": FamilyTransform,
    "MarriedTransform": MarriedTransform,
    "DeckTransform": DeckTransform,
    "TicketGroupTransform": TicketGroupTransform,
    "TicketFrequencyTransform": TicketFrequencyTransform,
    "FareTransform": FareTransform,
    "FareBinningTransform": FareBinningTransform,
    "AgeBinningTransform": AgeBinningTransform,
    "TicketParseTransform": TicketParseTransform,
    "AgeImputeByTitleTransform": AgeImputeByTitleTransform,
    "FamilyRoleTransform": FamilyRoleTransform,
    "FarePerPersonTransform": FarePerPersonTransform,
}

def _is_enabled(name: str, config: dict) -> bool:
    toggles = (config.get("feature_toggles") or {})
    # Default to True when unspecified
    return bool(toggles.get(name, True))


def build_pipeline_from_config(stage_list: list, config: dict):
    transforms = []
    for name in stage_list:
        # Toggle support per transform
        if not _is_enabled(name, config):
            continue
        if name not in TRANSFORM_MAP:
            raise ValueError(f"Unknown transform: {name}")
        # Pass config to transforms if they take parameters
        if name == "FareTransform":
            transforms.append(FareTransform(log_transform=config.get("log_transform_fare", False)))
        elif name == "AgeBinningTransform":
            transforms.append(AgeBinningTransform(n_bins=config.get("age_bins", 5)))
        elif name == "AgeImputeByTitleTransform":
            transforms.append(AgeImputeByTitleTransform())
        elif name == "TitleTransform":
            transforms.append(
                TitleTransform(
                    rare_title_threshold=config.get("rare_title_threshold", None),
                    title_map_override=config.get("title_map_override")
                )
            )
        else:
            transforms.append(TRANSFORM_MAP[name]())
    return FeaturePipeline(transforms)

def build_pipeline_pre(config):
    stage_list = config.get("feature_engineering", {}).get("pre_impute", [])
    return build_pipeline_from_config(stage_list, config)

def build_pipeline_post(config):
    stage_list = config.get("feature_engineering", {}).get("post_impute", [])
    return build_pipeline_from_config(stage_list, config)
