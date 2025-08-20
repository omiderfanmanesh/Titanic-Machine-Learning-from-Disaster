"""Atomic feature transformations following SOLID principles."""

from __future__ import annotations

# Import all transformer classes from individual modules
from .title import TitleTransform
from .family import FamilyTransform
from .deck import DeckTransform
from .ticket_group import TicketGroupTransform
from .ticket_frequency import TicketFrequencyTransform
from .fare import FareTransform
from .fare_binning import FareBinningTransform
from .age_binning import AgeBinningTransform
from .family_size import FamilySizeTransform
from .age_impute_by_title import AgeImputeByTitleTransform
from .married import MarriedTransform

__all__ = [
    "TitleTransform",
    "FamilyTransform",
    "DeckTransform",
    "TicketGroupTransform",
    "TicketFrequencyTransform",
    "FareTransform",
    "FareBinningTransform",
    "AgeBinningTransform",
    "FamilySizeTransform",
    "AgeImputeByTitleTransform",
    "MarriedTransform",
]
