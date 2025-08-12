"""Cross-validation splitting strategies with leakage prevention."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from core.interfaces import IFoldSplitter
from core.utils import LoggerFactory


class StratifiedKFoldSplitter(IFoldSplitter):
    """Stratified K-Fold cross-validation splitter."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        self.logger = LoggerFactory.get_logger(__name__)
    
    def split(self, X: pd.DataFrame, y: pd.Series, 
              groups: Optional[pd.Series] = None) -> List[Tuple[List[int], List[int]]]:
        """Generate stratified splits."""
        if groups is not None:
            self.logger.warning("Groups parameter ignored in StratifiedKFold")
            
        splits = []
        for train_idx, val_idx in self.splitter.split(X, y):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            
        self.logger.info(f"Generated {len(splits)} stratified folds")
        self._log_split_info(splits, y)
        
        return splits
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits
    
    def _log_split_info(self, splits: List[Tuple[List[int], List[int]]], y: pd.Series) -> None:
        """Log information about the splits."""
        for i, (train_idx, val_idx) in enumerate(splits):
            train_dist = y.iloc[train_idx].value_counts(normalize=True).sort_index()
            val_dist = y.iloc[val_idx].value_counts(normalize=True).sort_index()
            
            self.logger.debug(f"Fold {i+1} - Train: {train_dist.to_dict()}, "
                             f"Val: {val_dist.to_dict()}")


class KFoldSplitter(IFoldSplitter):
    """Standard K-Fold cross-validation splitter."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        self.logger = LoggerFactory.get_logger(__name__)
    
    def split(self, X: pd.DataFrame, y: pd.Series,
              groups: Optional[pd.Series] = None) -> List[Tuple[List[int], List[int]]]:
        """Generate K-Fold splits."""
        if groups is not None:
            self.logger.warning("Groups parameter ignored in KFold")
            
        splits = []
        for train_idx, val_idx in self.splitter.split(X, y):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            
        self.logger.info(f"Generated {len(splits)} K-fold splits")
        
        return splits
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class GroupKFoldSplitter(IFoldSplitter):
    """Group K-Fold splitter to prevent data leakage within groups."""
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.splitter = GroupKFold(n_splits=n_splits)
        self.logger = LoggerFactory.get_logger(__name__)
    
    def split(self, X: pd.DataFrame, y: pd.Series,
              groups: Optional[pd.Series] = None) -> List[Tuple[List[int], List[int]]]:
        """Generate group-aware splits."""
        if groups is None:
            raise ValueError("Groups must be provided for GroupKFold")
            
        splits = []
        for train_idx, val_idx in self.splitter.split(X, y, groups):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            
        self.logger.info(f"Generated {len(splits)} group K-fold splits")
        self._log_group_info(splits, groups)
        
        return splits
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits
    
    def _log_group_info(self, splits: List[Tuple[List[int], List[int]]], 
                       groups: pd.Series) -> None:
        """Log group distribution information."""
        for i, (train_idx, val_idx) in enumerate(splits):
            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])
            
            # Check for group leakage
            overlap = train_groups.intersection(val_groups)
            if overlap:
                self.logger.error(f"Fold {i+1}: Group leakage detected! "
                                f"Overlapping groups: {overlap}")
            else:
                self.logger.debug(f"Fold {i+1}: No group leakage. "
                                f"Train groups: {len(train_groups)}, "
                                f"Val groups: {len(val_groups)}")


class TimeSeriesSplitter(IFoldSplitter):
    """Time series cross-validation splitter."""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        self.logger = LoggerFactory.get_logger(__name__)
    
    def split(self, X: pd.DataFrame, y: pd.Series,
              groups: Optional[pd.Series] = None) -> List[Tuple[List[int], List[int]]]:
        """Generate time series splits."""
        if groups is not None:
            self.logger.warning("Groups parameter ignored in TimeSeriesSplit")
            
        splits = []
        for train_idx, val_idx in self.splitter.split(X, y):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            
        self.logger.info(f"Generated {len(splits)} time series splits")
        self._validate_temporal_order(splits)
        
        return splits
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits
    
    def _validate_temporal_order(self, splits: List[Tuple[List[int], List[int]]]) -> None:
        """Validate that validation sets come after training sets."""
        for i, (train_idx, val_idx) in enumerate(splits):
            if max(train_idx) >= min(val_idx):
                self.logger.error(f"Fold {i+1}: Temporal leakage detected! "
                                f"Max train index ({max(train_idx)}) >= "
                                f"Min val index ({min(val_idx)})")


class CustomStratifiedSplitter(IFoldSplitter):
    """Custom stratified splitter with additional controls."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42,
                 min_samples_per_class: int = 1):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_samples_per_class = min_samples_per_class
        self.logger = LoggerFactory.get_logger(__name__)
    
    def split(self, X: pd.DataFrame, y: pd.Series,
              groups: Optional[pd.Series] = None) -> List[Tuple[List[int], List[int]]]:
        """Generate custom stratified splits with validation."""
        # Check class distribution
        class_counts = y.value_counts()
        min_count = class_counts.min()
        
        if min_count < self.n_splits * self.min_samples_per_class:
            self.logger.warning(f"Insufficient samples for stratification. "
                              f"Min class count: {min_count}, "
                              f"Required: {self.n_splits * self.min_samples_per_class}")
            
        # Use regular StratifiedKFold
        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        splits = []
        for train_idx, val_idx in splitter.split(X, y):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            
        # Validate splits
        self._validate_splits(splits, y)
        
        return splits
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits
    
    def _validate_splits(self, splits: List[Tuple[List[int], List[int]]], y: pd.Series) -> None:
        """Validate split quality."""
        overall_dist = y.value_counts(normalize=True).sort_index()
        
        for i, (train_idx, val_idx) in enumerate(splits):
            train_dist = y.iloc[train_idx].value_counts(normalize=True).sort_index()
            val_dist = y.iloc[val_idx].value_counts(normalize=True).sort_index()
            
            # Check for significant distribution differences
            for class_label in overall_dist.index:
                overall_pct = overall_dist[class_label]
                train_pct = train_dist.get(class_label, 0)
                val_pct = val_dist.get(class_label, 0)
                
                train_diff = abs(train_pct - overall_pct)
                val_diff = abs(val_pct - overall_pct)
                
                if train_diff > 0.1 or val_diff > 0.1:  # 10% threshold
                    self.logger.warning(f"Fold {i+1}: Large distribution difference "
                                      f"for class {class_label}")


class FoldSplitterFactory:
    """Factory for creating fold splitters."""
    
    @staticmethod
    def create_splitter(strategy: str, **kwargs) -> IFoldSplitter:
        """Create fold splitter by strategy name."""
        strategy = strategy.lower()
        
        if strategy == "kfold":
            return KFoldSplitter(**kwargs)
        elif strategy == "stratified":
            return StratifiedKFoldSplitter(**kwargs)
        elif strategy == "group":
            return GroupKFoldSplitter(**kwargs)
        elif strategy == "timeseries":
            return TimeSeriesSplitter(**kwargs)
        elif strategy == "custom_stratified":
            return CustomStratifiedSplitter(**kwargs)
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available splitting strategies."""
        return ["kfold", "stratified", "group", "timeseries", "custom_stratified"]


def create_splits_with_validation(X: pd.DataFrame, y: pd.Series, config: Dict[str, any],
                                 groups: Optional[pd.Series] = None) -> Tuple[List[Tuple[List[int], List[int]]], Dict[str, any]]:
    """Create CV splits with comprehensive validation."""
    logger = LoggerFactory.get_logger(__name__)
    
    # Create splitter
    splitter = FoldSplitterFactory.create_splitter(
        config.get("cv_strategy", "stratified"),
        n_splits=config.get("cv_folds", 5),
        shuffle=config.get("cv_shuffle", True),
        random_state=config.get("cv_random_state", 42)
    )
    
    # Generate splits
    splits = splitter.split(X, y, groups)
    
    # Validate splits
    validation_info = {
        "n_splits": len(splits),
        "strategy": config.get("cv_strategy", "stratified"),
        "train_sizes": [],
        "val_sizes": [],
        "class_distributions": []
    }
    
    for i, (train_idx, val_idx) in enumerate(splits):
        validation_info["train_sizes"].append(len(train_idx))
        validation_info["val_sizes"].append(len(val_idx))
        
        if hasattr(y, 'value_counts'):
            train_dist = y.iloc[train_idx].value_counts(normalize=True).to_dict()
            val_dist = y.iloc[val_idx].value_counts(normalize=True).to_dict()
            validation_info["class_distributions"].append({
                "fold": i,
                "train": train_dist,
                "validation": val_dist
            })
    
    logger.info(f"Created {len(splits)} CV splits using {config.get('cv_strategy', 'stratified')} strategy")
    
    return splits, validation_info
