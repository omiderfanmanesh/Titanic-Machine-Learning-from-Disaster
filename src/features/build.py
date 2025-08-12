"""Feature building and engineering pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from core.interfaces import ITransformer
from core.utils import LoggerFactory
from features.transforms import (
    FamilySizeTransform,
    TitleTransform,
    DeckTransform,
    TicketGroupTransform,
    FareTransform,
    AgeBinningTransform,
    MissingValueIndicatorTransform,
    FeaturePipeline
)


class TitanicFeatureBuilder(ITransformer):
    """Main feature builder for Titanic competition."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = LoggerFactory.get_logger(__name__)
        self.pipeline: Optional[FeaturePipeline] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self._scale_columns: Optional[List[str]] = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TitanicFeatureBuilder":
        """Fit the feature builder on training data."""
        self.logger.info("Fitting feature builder")
        
        # Create transformation pipeline
        transforms = self._build_transforms()
        self.pipeline = FeaturePipeline(transforms)
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        
        # Apply initial transformations to get feature set
        X_transformed = self.pipeline.transform(X)
        
        # Fit encoders and scalers
        self._fit_encoders(X_transformed)
        self._fit_scaler(X_transformed)
        
        self.is_fitted = True
        self.logger.info("✅ Feature builder fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data using fitted transformers."""
        if not self.is_fitted:
            raise ValueError("Feature builder must be fitted before transform")
        
        self.logger.info(f"Transforming data with {len(X)} samples")
        
        # Apply feature engineering pipeline
        X_transformed = self.pipeline.transform(X)
        
        # Handle missing values
        X_transformed = self._handle_missing_values(X_transformed)
        
        # Apply encodings
        X_transformed = self._apply_encodings(X_transformed)
        
        # Apply scaling only to fitted columns
        if self.scaler is not None and self._scale_columns:
            available_cols = [c for c in self._scale_columns if c in X_transformed.columns]
            if available_cols:
                X_transformed[available_cols] = self.scaler.transform(X_transformed[available_cols])
        
        self.logger.info(f"✅ Transformed data shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _build_transforms(self) -> List[ITransformer]:
        """Build list of transformations based on config."""
        transforms = []
        
        if self.config.get("add_family_features", True):
            transforms.append(FamilySizeTransform())
        
        if self.config.get("add_title_features", True):
            transforms.append(TitleTransform())
        
        if self.config.get("add_deck_features", True):
            transforms.append(DeckTransform())
        
        if self.config.get("add_ticket_features", False):
            transforms.append(TicketGroupTransform())
        
        if self.config.get("transform_fare", True):
            transforms.append(FareTransform(
                log_transform=self.config.get("log_transform_fare", False)
            ))
        
        if self.config.get("add_age_bins", False):
            transforms.append(AgeBinningTransform(
                n_bins=self.config.get("age_bins", 5)
            ))
        
        if self.config.get("add_missing_indicators", True):
            transforms.append(MissingValueIndicatorTransform())
        
        return transforms
    
    def _fit_encoders(self, X: pd.DataFrame) -> None:
        """Fit label encoders for categorical features."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in ['PassengerId', 'Name', 'Ticket']:  # Skip ID-like columns
                encoder = LabelEncoder()
                # Handle missing values by treating them as a separate category
                values_to_fit = X[col].fillna('__MISSING__').astype(str)
                encoder.fit(values_to_fit)
                self.encoders[col] = encoder
                self.logger.debug(f"Fitted encoder for '{col}' with {len(encoder.classes_)} categories")
    
    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit scaler for numeric features."""
        if not self.config.get("scale_features", True):
            return
        
        numeric_cols = X.select_dtypes(include=['number']).columns
        # Exclude ID columns and binary features
        exclude_cols = ['PassengerId', 'IsAlone'] + [col for col in numeric_cols if X[col].nunique() == 2]
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if scale_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(X[scale_cols])
            self._scale_columns = list(scale_cols)
            self.logger.debug(f"Fitted scaler for {len(scale_cols)} numeric features: {self._scale_columns}")
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the transformed data."""
        X = X.copy()
        
        # Fill missing values with appropriate defaults
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['object', 'category']:
                    X[col] = X[col].fillna('Unknown')
                else:
                    X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def _apply_encodings(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders to categorical features."""
        X = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X.columns:
                # Handle missing values and unseen categories
                values = X[col].fillna('__MISSING__').astype(str)
                
                # Handle unseen categories
                seen_values = set(encoder.classes_)
                values = values.map(lambda x: x if x in seen_values else '__MISSING__')
                
                X[col] = encoder.transform(values)
        
        return X
    
    def _default_config(self) -> Dict[str, Any]:
        """Default feature engineering configuration."""
        return {
            "add_family_features": True,
            "add_title_features": True,
            "add_deck_features": True,
            "add_ticket_features": False,
            "transform_fare": True,
            "log_transform_fare": False,
            "add_age_bins": False,
            "age_bins": 5,
            "add_missing_indicators": True,
            "scale_features": True
        }
    
    def get_feature_names(self) -> List[str]:
        """Get names of output features."""
        if not self.is_fitted:
            raise ValueError("Feature builder must be fitted first")
        
        # This would need to be implemented based on the actual pipeline
        # For now, return empty list
        return []


def create_feature_builder(config: Optional[Dict[str, Any]] = None, debug: bool = False) -> TitanicFeatureBuilder:
    """Factory function to create feature builder."""
    if debug:
        # Add debug-specific configuration
        config = config or {}
        config["debug_mode"] = True
    
    return TitanicFeatureBuilder(config)


class AdvancedFeatureBuilder(TitanicFeatureBuilder):
    """Advanced feature builder with additional engineering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
    
    def _build_transforms(self) -> List[ITransformer]:
        """Build advanced transformation list."""
        transforms = super()._build_transforms()
        
        # Add advanced transforms
        if self.config.get("add_interaction_features", False):
            # Could add interaction feature transforms here
            pass
        
        if self.config.get("add_polynomial_features", False):
            # Could add polynomial feature transforms here
            pass
        
        return transforms
