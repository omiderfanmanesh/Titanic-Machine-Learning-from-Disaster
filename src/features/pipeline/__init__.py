"""Atomic feature transformations following SOLID principles."""

from __future__ import annotations

from .pipeline import FeaturePipeline


__all__ = [

    "FeaturePipeline",

]

from typing import List, Dict, Any
from features.transforms import (
    FamilySizeTransform,
    TitleTransform,
    DeckTransform,
    TicketGroupTransform,
    FareTransform,
    AgeBinningTransform,
    MissingValueIndicatorTransform,
)
from core.interfaces import ITransformer


def build_pipeline(config: Dict[str, Any]) -> FeaturePipeline:
    transforms: List[ITransformer] = []
    if config.get("add_family_features", True):
        transforms.append(FamilySizeTransform())
    if config.get("add_title_features", True):
        transforms.append(TitleTransform())
    if config.get("add_deck_features", True):
        transforms.append(DeckTransform())
    if config.get("add_ticket_features", False):
        transforms.append(TicketGroupTransform())
    if config.get("transform_fare", True):
        transforms.append(FareTransform(log_transform=config.get("log_transform_fare", False)))
    if config.get("add_age_bins", False):
        transforms.append(AgeBinningTransform(n_bins=config.get("age_bins", 5)))
    if config.get("add_missing_indicators", True):
        transforms.append(MissingValueIndicatorTransform())
    return FeaturePipeline(transforms)
