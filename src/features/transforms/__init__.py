"""Atomic feature transformations following SOLID principles."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.interfaces import ITransformer
from core.utils import LoggerFactory


class BaseTransform(BaseEstimator, TransformerMixin, ITransformer):
    """Base class for all feature transformations."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class FamilySizeTransform(BaseTransform):
    """Creates family size and is_alone features."""
    
    def __init__(self, sibsp_col: str = "SibSp", parch_col: str = "Parch"):
        super().__init__()
        self.sibsp_col = sibsp_col
        self.parch_col = parch_col
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FamilySizeTransform":
        """Fit transform - no parameters to learn."""
        required_cols = [self.sibsp_col, self.parch_col]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by adding family size features."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")
            
        X = X.copy()
        X["FamilySize"] = X[self.sibsp_col] + X[self.parch_col] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        
        self.logger.debug(f"Created FamilySize (range: {X['FamilySize'].min()}-{X['FamilySize'].max()}) "
                         f"and IsAlone ({X['IsAlone'].sum()}/{len(X)} alone)")
        
        return X


class TitleTransform(BaseTransform):
    """Extracts and encodes titles from passenger names."""
    
    def __init__(self, name_col: str = "Name", rare_threshold: int = 10):
        super().__init__()
        self.name_col = name_col
        self.rare_threshold = rare_threshold
        self.title_mapping: Optional[Dict[str, str]] = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TitleTransform":
        """Learn title mappings from training data."""
        if self.name_col not in X.columns:
            raise ValueError(f"Column {self.name_col} not found")
            
        # Extract titles
        titles = X[self.name_col].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Count title frequencies
        title_counts = titles.value_counts()
        
        # Map rare titles to 'Rare'
        self.title_mapping = {}
        for title, count in title_counts.items():
            if count >= self.rare_threshold:
                self.title_mapping[title] = title
            else:
                self.title_mapping[title] = "Rare"
                
        self.logger.info(f"Learned {len(set(self.title_mapping.values()))} title categories: "
                        f"{sorted(set(self.title_mapping.values()))}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by extracting and mapping titles."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")
            
        X = X.copy()
        
        # Extract titles
        titles = X[self.name_col].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Map using learned mapping, default to 'Rare' for unseen titles
        X["Title"] = titles.map(self.title_mapping).fillna("Rare")
        
        return X


class DeckTransform(BaseTransform):
    """Extracts deck information from cabin numbers."""
    
    def __init__(self, cabin_col: str = "Cabin"):
        super().__init__()
        self.cabin_col = cabin_col
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DeckTransform":
        """Fit transform - no parameters to learn."""
        if self.cabin_col not in X.columns:
            raise ValueError(f"Column {self.cabin_col} not found")
            
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by extracting deck from cabin."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")
            
        X = X.copy()
        
        # Extract first character of cabin (deck)
        deck = X[self.cabin_col].astype(str).str[0]
        
        # Map valid deck letters, unknown/missing to 'U'
        valid_decks = list('ABCDEFGT')
        X["Deck"] = np.where(deck.isin(valid_decks), deck, 'U')
        
        deck_counts = X["Deck"].value_counts()
        self.logger.debug(f"Deck distribution: {deck_counts.to_dict()}")
        
        return X


class TicketGroupTransform(BaseTransform):
    """Creates ticket group size feature."""
    
    def __init__(self, ticket_col: str = "Ticket"):
        super().__init__()
        self.ticket_col = ticket_col
        self.ticket_counts: Optional[pd.Series] = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TicketGroupTransform":
        """Learn ticket counts from training data."""
        if self.ticket_col not in X.columns:
            raise ValueError(f"Column {self.ticket_col} not found")
            
        self.ticket_counts = X[self.ticket_col].value_counts()
        
        self.logger.info(f"Learned ticket counts for {len(self.ticket_counts)} unique tickets")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by adding ticket group size."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")
            
        X = X.copy()
        
        # Map ticket to group size, default to 1 for unseen tickets
        X["TicketGroupSize"] = X[self.ticket_col].map(self.ticket_counts).fillna(1)
        
        group_dist = X["TicketGroupSize"].value_counts().sort_index()
        self.logger.debug(f"Ticket group size distribution: {group_dist.to_dict()}")
        
        return X


class FareTransform(BaseTransform):
    """Transforms fare values with log transformation and missing value handling."""
    
    def __init__(self, fare_col: str = "Fare", log_transform: bool = False):
        super().__init__()
        self.fare_col = fare_col
        self.log_transform = log_transform
        self.median_fare: Optional[float] = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FareTransform":
        """Learn median fare for imputation."""
        if self.fare_col not in X.columns:
            raise ValueError(f"Column {self.fare_col} not found")
            
        self.median_fare = X[self.fare_col].median()
        
        self.logger.info(f"Learned median fare: {self.median_fare:.2f}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform fare column."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")
            
        X = X.copy()
        
        # Impute missing values with median
        X[self.fare_col] = X[self.fare_col].fillna(self.median_fare)
        
        # Apply log transformation if requested
        if self.log_transform:
            # Add small constant to avoid log(0)
            X[f"{self.fare_col}_log"] = np.log1p(X[self.fare_col])
            
        return X


class AgeBinningTransform(BaseTransform):
    """Creates age bins from continuous age values."""
    
    def __init__(self, age_col: str = "Age", n_bins: int = 5, 
                 bin_labels: Optional[List[str]] = None):
        super().__init__()
        self.age_col = age_col
        self.n_bins = n_bins
        self.bin_labels = bin_labels or [f"Age_Bin_{i}" for i in range(n_bins)]
        self.bin_edges: Optional[np.ndarray] = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AgeBinningTransform":
        """Learn age bin edges from training data."""
        if self.age_col not in X.columns:
            raise ValueError(f"Column {self.age_col} not found")
            
        age_values = X[self.age_col].dropna()
        if len(age_values) == 0:
            raise ValueError("No valid age values found")
            
        # Create quantile-based bins
        self.bin_edges = np.quantile(age_values, np.linspace(0, 1, self.n_bins + 1))
        
        # Ensure unique edges
        self.bin_edges = np.unique(self.bin_edges)
        
        self.logger.info(f"Learned age bins: {self.bin_edges}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform age into bins."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")
            
        X = X.copy()
        
        # Create age bins
        X["AgeBin"] = pd.cut(X[self.age_col], bins=self.bin_edges, 
                           labels=self.bin_labels[:len(self.bin_edges)-1],
                           include_lowest=True)
        
        return X


class MissingValueIndicatorTransform(BaseTransform):
    """Creates binary indicators for missing values."""
    
    def __init__(self, columns: Optional[List[str]] = None, 
                 missing_threshold: float = 0.01):
        super().__init__()
        self.columns = columns
        self.missing_threshold = missing_threshold
        self.indicator_columns: Optional[List[str]] = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MissingValueIndicatorTransform":
        """Identify columns with significant missing values."""
        if self.columns is None:
            # Auto-detect columns with missing values above threshold
            missing_pct = X.isnull().mean()
            self.indicator_columns = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        else:
            self.indicator_columns = [col for col in self.columns if col in X.columns]
            
        self.logger.info(f"Creating missing indicators for: {self.indicator_columns}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create missing value indicators."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform")
            
        X = X.copy()
        
        for col in self.indicator_columns:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isnull().astype(int)
                
        return X


class FeaturePipeline(BaseTransform):
    """Pipeline for chaining multiple feature transforms."""
    
    def __init__(self, transforms: List[BaseTransform]):
        super().__init__()
        self.transforms = transforms
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeaturePipeline":
        """Fit all transforms sequentially."""
        X_current = X.copy()
        
        for i, transform in enumerate(self.transforms):
            self.logger.debug(f"Fitting transform {i+1}/{len(self.transforms)}: {type(transform).__name__}")
            transform.fit(X_current, y)
            X_current = transform.transform(X_current)
            
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transforms sequentially."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before calling transform")
            
        X_current = X.copy()
        
        for transform in self.transforms:
            X_current = transform.transform(X_current)
            
        return X_current
