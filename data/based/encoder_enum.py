"""Encoder constants and utilities.

This module defines supported encoding methods for categorical variables
and provides utilities for encoder selection and configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union


class EncoderType(Enum):
    """Supported encoding methods for categorical variables."""
    
    # Basic encoders
    ONE_HOT = "one_hot"
    LABEL = "label"
    ORDINAL = "ordinal"
    
    # Advanced encoders
    TARGET = "target"
    FREQUENCY = "frequency"
    BINARY = "binary"
    HELMERT = "helmert"
    SUM_CODING = "sum_coding"
    POLYNOMIAL = "polynomial"
    BACKWARD_DIFFERENCE = "backward_difference"
    
    # Hash-based encoders
    HASH = "hash"
    FEATURE_HASH = "feature_hash"
    
    # Custom encoders
    CUSTOM = "custom"
    
    @property
    def sklearn_class(self) -> Optional[str]:
        """Get the corresponding sklearn encoder class name.
        
        Returns:
            Name of sklearn encoder class, if available
        """
        mapping = {
            self.ONE_HOT: "OneHotEncoder",
            self.LABEL: "LabelEncoder", 
            self.ORDINAL: "OrdinalEncoder",
        }
        return mapping.get(self)
    
    @property
    def category_encoders_class(self) -> Optional[str]:
        """Get the corresponding category_encoders class name.
        
        Returns:
            Name of category_encoders class, if available
        """
        mapping = {
            self.ONE_HOT: "OneHotEncoder",
            self.LABEL: "OrdinalEncoder",  # Similar functionality
            self.ORDINAL: "OrdinalEncoder",
            self.TARGET: "TargetEncoder",
            self.FREQUENCY: "CountEncoder",
            self.BINARY: "BinaryEncoder",
            self.HELMERT: "HelmertEncoder",
            self.SUM_CODING: "SumEncoder",
            self.POLYNOMIAL: "PolynomialEncoder",
            self.BACKWARD_DIFFERENCE: "BackwardDifferenceEncoder",
            self.HASH: "HashingEncoder",
        }
        return mapping.get(self)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this encoder type.
        
        Returns:
            Dictionary of default parameters
        """
        defaults = {
            self.ONE_HOT: {
                "handle_unknown": "ignore",
                "sparse": False,
                "drop": "first",
            },
            self.LABEL: {},
            self.ORDINAL: {
                "handle_unknown": "use_encoded_value",
                "unknown_value": -1,
            },
            self.TARGET: {
                "smoothing": 1.0,
                "min_samples_leaf": 1,
                "handle_unknown": "value",
                "handle_missing": "value",
            },
            self.FREQUENCY: {
                "normalize": False,
                "handle_unknown": "value",
                "handle_missing": "value",
            },
            self.BINARY: {
                "handle_unknown": "value",
                "handle_missing": "value",
            },
            self.HASH: {
                "n_components": 8,
                "hash_method": "md5",
            },
        }
        return defaults.get(self, {})
    
    @property
    def requires_target(self) -> bool:
        """Check if encoder requires target variable during fitting.
        
        Returns:
            True if target is required
        """
        return self in {self.TARGET}
    
    @property
    def handles_missing(self) -> bool:
        """Check if encoder can handle missing values natively.
        
        Returns:
            True if missing values are handled
        """
        return self in {
            self.TARGET,
            self.FREQUENCY,
            self.BINARY,
            self.HASH,
        }
    
    @property
    def creates_new_features(self) -> bool:
        """Check if encoder creates new features (vs replacing existing).
        
        Returns:
            True if new features are created
        """
        return self in {
            self.ONE_HOT,
            self.BINARY,
            self.HELMERT,
            self.SUM_CODING,
            self.POLYNOMIAL,
            self.BACKWARD_DIFFERENCE,
            self.HASH,
        }


class NumericalEncoderType(Enum):
    """Encoding methods for numerical variables."""
    
    # Scaling
    STANDARD = "standard"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    NORMALIZER = "normalizer"
    
    # Transformation
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"
    POWER_TRANSFORMER = "power_transformer"
    
    # Binning
    K_BINS_UNIFORM = "k_bins_uniform"
    K_BINS_QUANTILE = "k_bins_quantile"
    K_BINS_KMEANS = "k_bins_kmeans"
    
    @property
    def sklearn_class(self) -> str:
        """Get the corresponding sklearn transformer class name.
        
        Returns:
            Name of sklearn transformer class
        """
        mapping = {
            self.STANDARD: "StandardScaler",
            self.MIN_MAX: "MinMaxScaler",
            self.ROBUST: "RobustScaler",
            self.NORMALIZER: "Normalizer",
            self.QUANTILE_UNIFORM: "QuantileTransformer",
            self.QUANTILE_NORMAL: "QuantileTransformer",
            self.POWER_TRANSFORMER: "PowerTransformer",
            self.K_BINS_UNIFORM: "KBinsDiscretizer",
            self.K_BINS_QUANTILE: "KBinsDiscretizer",
            self.K_BINS_KMEANS: "KBinsDiscretizer",
        }
        return mapping[self]
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this encoder type.
        
        Returns:
            Dictionary of default parameters
        """
        defaults = {
            self.STANDARD: {"with_mean": True, "with_std": True},
            self.MIN_MAX: {"feature_range": (0, 1)},
            self.ROBUST: {"quantile_range": (25.0, 75.0)},
            self.NORMALIZER: {"norm": "l2"},
            self.QUANTILE_UNIFORM: {
                "output_distribution": "uniform",
                "n_quantiles": 1000,
            },
            self.QUANTILE_NORMAL: {
                "output_distribution": "normal", 
                "n_quantiles": 1000,
            },
            self.POWER_TRANSFORMER: {"method": "yeo-johnson"},
            self.K_BINS_UNIFORM: {
                "n_bins": 5,
                "encode": "onehot-dense",
                "strategy": "uniform",
            },
            self.K_BINS_QUANTILE: {
                "n_bins": 5,
                "encode": "onehot-dense", 
                "strategy": "quantile",
            },
            self.K_BINS_KMEANS: {
                "n_bins": 5,
                "encode": "onehot-dense",
                "strategy": "kmeans",
            },
        }
        return defaults.get(self, {})


def get_recommended_encoder(
    column_type: str,
    cardinality: int,
    has_missing: bool = False,
    has_target: bool = False,
) -> EncoderType:
    """Get recommended encoder based on data characteristics.
    
    Args:
        column_type: Type of column ('categorical' or 'numerical')
        cardinality: Number of unique values
        has_missing: Whether column has missing values
        has_target: Whether target variable is available
        
    Returns:
        Recommended EncoderType
    """
    if column_type == "numerical":
        # For numerical data, return appropriate encoder based on use case
        # This would typically be handled by NumericalEncoderType
        return EncoderType.ORDINAL  # Placeholder
    
    # Categorical encoding recommendations
    if cardinality <= 2:
        return EncoderType.LABEL
    elif cardinality <= 10:
        return EncoderType.ONE_HOT
    elif cardinality <= 50:
        if has_target:
            return EncoderType.TARGET
        else:
            return EncoderType.FREQUENCY
    else:  # High cardinality
        if has_target:
            return EncoderType.TARGET
        else:
            return EncoderType.HASH


def create_encoder_config(
    encoder_type: EncoderType,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create encoder configuration with defaults and custom parameters.
    
    Args:
        encoder_type: Type of encoder to configure
        custom_params: Custom parameters to override defaults
        
    Returns:
        Complete encoder configuration
    """
    config = {
        "type": encoder_type.value,
        "params": encoder_type.get_default_params(),
    }
    
    if custom_params:
        config["params"].update(custom_params)
    
    return config


def validate_encoder_params(
    encoder_type: EncoderType,
    params: Dict[str, Any],
) -> List[str]:
    """Validate encoder parameters.
    
    Args:
        encoder_type: Type of encoder
        params: Parameters to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Basic validation rules
    if encoder_type == EncoderType.ONE_HOT:
        if "handle_unknown" in params:
            valid_values = ["error", "ignore", "infrequent_if_exist"]
            if params["handle_unknown"] not in valid_values:
                errors.append(f"handle_unknown must be one of {valid_values}")
    
    elif encoder_type == EncoderType.HASH:
        if "n_components" in params:
            if not isinstance(params["n_components"], int) or params["n_components"] <= 0:
                errors.append("n_components must be a positive integer")
    
    elif encoder_type == EncoderType.TARGET:
        if "smoothing" in params:
            if not isinstance(params["smoothing"], (int, float)) or params["smoothing"] < 0:
                errors.append("smoothing must be a non-negative number")
    
    return errors
