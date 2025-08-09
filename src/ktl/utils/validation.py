from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


SplitScheme = Literal["kfold", "stratified", "group", "timeseries"]


@dataclass
class CVConfig:
    """Cross-validation configuration.

    Attributes
    ----------
    scheme : SplitScheme
        CV strategy.
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle (where applicable).
    random_state : int
        Seed for reproducibility.
    group_column : Optional[str]
        Group column name for group CV.
    time_column : Optional[str]
        Time column name for time series CV.
    """

    scheme: SplitScheme = "kfold"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    group_column: Optional[str] = None
    time_column: Optional[str] = None


class SplitterFactory:
    """Factory for building sklearn splitters from CVConfig."""

    @staticmethod
    def build(cfg: CVConfig):  # type: ignore[override]
        if cfg.scheme == "kfold":
            return KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state)
        if cfg.scheme == "stratified":
            return StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state)
        if cfg.scheme == "group":
            return GroupKFold(n_splits=cfg.n_splits)
        if cfg.scheme == "timeseries":
            return TimeSeriesSplit(n_splits=cfg.n_splits)
        raise ValueError(f"Unknown CV scheme: {cfg.scheme}")
