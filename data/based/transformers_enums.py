"""Data transformer constants and utilities.

This module defines supported data transformation methods and provides
utilities for transformer selection and configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union


class TransformerType(Enum):
    """Supported data transformation methods."""
    
    # Mathematical transformations
    LOG = "log"
    LOG1P = "log1p"
    SQRT = "sqrt"
    SQUARE = "square"
    RECIPROCAL = "reciprocal"
    
    # Power transformations
    POWER = "power"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"
    
    # Trigonometric transformations
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ARCSIN = "arcsin"
    ARCCOS = "arccos"
    ARCTAN = "arctan"
    
    # Statistical transformations
    Z_SCORE = "z_score"
    RANK = "rank"
    PERCENTILE = "percentile"
    
    # Binning transformations
    EQUAL_WIDTH = "equal_width"
    EQUAL_FREQUENCY = "equal_frequency"
    KMEANS_BINNING = "kmeans_binning"
    
    # Custom transformations
    CUSTOM = "custom"
    
    @property
    def requires_positive(self) -> bool:
        """Check if transformer requires positive values.
        
        Returns:
            True if positive values are required
        """
        return self in {
            self.LOG,
            self.SQRT,
            self.RECIPROCAL,
            self.BOX_COX,
        }
    
    @property
    def requires_finite(self) -> bool:
        """Check if transformer requires finite values.
        
        Returns:
            True if finite values are required
        """
        return self in {
            self.LOG,
            self.LOG1P,
            self.SQRT,
            self.RECIPROCAL,
            self.ARCSIN,
            self.ARCCOS,
        }
    
    @property
    def is_invertible(self) -> bool:
        """Check if transformation is invertible.
        
        Returns:
            True if transformation can be inverted
        """
        return self in {
            self.LOG,
            self.LOG1P,
            self.SQRT,
            self.SQUARE,
            self.RECIPROCAL,
            self.POWER,
            self.BOX_COX,
            self.YEO_JOHNSON,
            self.SIN,
            self.COS,
            self.TAN,
            self.ARCSIN,
            self.ARCCOS,
            self.ARCTAN,
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this transformer.
        
        Returns:
            Dictionary of default parameters
        """
        defaults = {
            self.LOG: {"base": "e"},
            self.LOG1P: {},
            self.SQRT: {},
            self.SQUARE: {},
            self.RECIPROCAL: {"epsilon": 1e-8},
            
            self.POWER: {"power": 2.0},
            self.BOX_COX: {"lambda": None},  # Auto-optimize
            self.YEO_JOHNSON: {"lambda": None},  # Auto-optimize
            
            self.SIN: {},
            self.COS: {},
            self.TAN: {},
            self.ARCSIN: {},
            self.ARCCOS: {},
            self.ARCTAN: {},
            
            self.Z_SCORE: {"center": True, "scale": True},
            self.RANK: {"method": "average"},
            self.PERCENTILE: {"method": "linear"},
            
            self.EQUAL_WIDTH: {"n_bins": 5, "strategy": "uniform"},
            self.EQUAL_FREQUENCY: {"n_bins": 5, "strategy": "quantile"},
            self.KMEANS_BINNING: {"n_bins": 5, "strategy": "kmeans"},
        }
        return defaults.get(self, {})
    
    def get_inverse_transformer(self) -> Optional[TransformerType]:
        """Get the inverse transformer if available.
        
        Returns:
            Inverse TransformerType or None
        """
        inverse_mapping = {
            self.LOG: self.POWER,  # exp
            self.LOG1P: None,      # expm1 (not in enum)
            self.SQRT: self.SQUARE,
            self.SQUARE: self.SQRT,
            self.SIN: self.ARCSIN,
            self.COS: self.ARCCOS,
            self.TAN: self.ARCTAN,
            self.ARCSIN: self.SIN,
            self.ARCCOS: self.COS,
            self.ARCTAN: self.TAN,
        }
        return inverse_mapping.get(self)


class FeatureTransformationType(Enum):
    """Types of feature transformations."""
    
    # Numerical transformations
    NUMERICAL = "numerical"
    
    # Categorical transformations
    CATEGORICAL = "categorical"
    
    # Text transformations
    TEXT = "text"
    
    # Date/time transformations
    DATETIME = "datetime"
    
    # Composite transformations
    INTERACTION = "interaction"
    POLYNOMIAL = "polynomial"
    
    def get_applicable_transformers(self) -> List[TransformerType]:
        """Get applicable transformers for this feature type.
        
        Returns:
            List of applicable TransformerType values
        """
        mapping = {
            self.NUMERICAL: [
                TransformerType.LOG,
                TransformerType.LOG1P,
                TransformerType.SQRT,
                TransformerType.SQUARE,
                TransformerType.POWER,
                TransformerType.BOX_COX,
                TransformerType.YEO_JOHNSON,
                TransformerType.Z_SCORE,
                TransformerType.RANK,
                TransformerType.PERCENTILE,
                TransformerType.EQUAL_WIDTH,
                TransformerType.EQUAL_FREQUENCY,
                TransformerType.KMEANS_BINNING,
            ],
            self.CATEGORICAL: [
                # Categorical transformations would be handled by encoder_enum
            ],
            self.TEXT: [
                # Text transformations would be specialized
            ],
            self.DATETIME: [
                # DateTime transformations would be specialized
            ],
            self.INTERACTION: [
                TransformerType.POWER,
                TransformerType.LOG,
                TransformerType.CUSTOM,
            ],
            self.POLYNOMIAL: [
                TransformerType.POWER,
                TransformerType.SQUARE,
                TransformerType.CUSTOM,
            ],
        }
        return mapping.get(self, [])


def get_recommended_transformer(
    data_characteristics: Dict[str, Any],
    transformation_goal: str = "normalize",
) -> TransformerType:
    """Get recommended transformer based on data characteristics.
    
    Args:
        data_characteristics: Dictionary with data info like:
            - distribution: str (normal, skewed, uniform)
            - skewness: float
            - has_negative: bool
            - has_zero: bool
            - min_value: float
            - max_value: float
        transformation_goal: Goal of transformation:
            - "normalize": Make distribution more normal
            - "stabilize": Stabilize variance
            - "compress": Compress dynamic range
            - "expand": Expand dynamic range
        
    Returns:
        Recommended TransformerType
    """
    distribution = data_characteristics.get("distribution", "unknown")
    skewness = data_characteristics.get("skewness", 0.0)
    has_negative = data_characteristics.get("has_negative", True)
    has_zero = data_characteristics.get("has_zero", False)
    
    if transformation_goal == "normalize":
        # For normalizing skewed distributions
        if abs(skewness) > 2:  # Highly skewed
            if has_negative:
                return TransformerType.YEO_JOHNSON
            elif has_zero:
                return TransformerType.LOG1P
            else:
                return TransformerType.BOX_COX
        elif abs(skewness) > 1:  # Moderately skewed
            if has_negative:
                return TransformerType.YEO_JOHNSON
            else:
                return TransformerType.LOG1P
    
    elif transformation_goal == "stabilize":
        # For stabilizing variance
        if not has_negative and not has_zero:
            return TransformerType.LOG
        else:
            return TransformerType.LOG1P
    
    elif transformation_goal == "compress":
        # For compressing large values
        if has_negative:
            return TransformerType.YEO_JOHNSON
        else:
            return TransformerType.LOG1P
    
    elif transformation_goal == "expand":
        # For expanding small values
        return TransformerType.SQUARE
    
    # Default: no transformation
    return TransformerType.LOG1P  # Safe default


def create_transformer_config(
    transformer_type: TransformerType,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create transformer configuration with defaults and custom parameters.
    
    Args:
        transformer_type: Type of transformer to configure
        custom_params: Custom parameters to override defaults
        
    Returns:
        Complete transformer configuration
    """
    config = {
        "type": transformer_type.value,
        "params": transformer_type.get_default_params(),
    }
    
    if custom_params:
        config["params"].update(custom_params)
    
    return config


def validate_transformer_params(
    transformer_type: TransformerType,
    params: Dict[str, Any],
    data_info: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate transformer parameters against data characteristics.
    
    Args:
        transformer_type: Type of transformer
        params: Parameters to validate
        data_info: Optional data characteristics for validation
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check if positive values are required
    if transformer_type.requires_positive and data_info:
        min_value = data_info.get("min_value")
        if min_value is not None and min_value <= 0:
            errors.append(
                f"{transformer_type.value} requires positive values, "
                f"but data has min={min_value}"
            )
    
    # Check if finite values are required
    if transformer_type.requires_finite and data_info:
        has_inf = data_info.get("has_inf", False)
        has_nan = data_info.get("has_nan", False)
        if has_inf or has_nan:
            errors.append(
                f"{transformer_type.value} requires finite values, "
                f"but data has inf={has_inf}, nan={has_nan}"
            )
    
    # Validate specific parameters
    if transformer_type == TransformerType.POWER:
        if "power" in params:
            power = params["power"]
            if not isinstance(power, (int, float)):
                errors.append("power must be a number")
    
    elif transformer_type == TransformerType.RECIPROCAL:
        if "epsilon" in params:
            epsilon = params["epsilon"]
            if not isinstance(epsilon, (int, float)) or epsilon < 0:
                errors.append("epsilon must be a non-negative number")
    
    elif transformer_type in {
        TransformerType.EQUAL_WIDTH,
        TransformerType.EQUAL_FREQUENCY,
        TransformerType.KMEANS_BINNING,
    }:
        if "n_bins" in params:
            n_bins = params["n_bins"]
            if not isinstance(n_bins, int) or n_bins <= 0:
                errors.append("n_bins must be a positive integer")
            if data_info and "n_samples" in data_info:
                n_samples = data_info["n_samples"]
                if n_bins > n_samples:
                    errors.append(f"n_bins ({n_bins}) cannot exceed n_samples ({n_samples})")
    
    return errors


def get_transformation_pipeline(
    transformations: List[Dict[str, Any]],
    validate: bool = True,
) -> List[Dict[str, Any]]:
    """Create a transformation pipeline from a list of transformations.
    
    Args:
        transformations: List of transformation configs
        validate: Whether to validate the pipeline
        
    Returns:
        Validated transformation pipeline
        
    Raises:
        ValueError: If pipeline validation fails
    """
    if not transformations:
        return []
    
    pipeline = []
    
    for i, transform_config in enumerate(transformations):
        # Ensure each transformation has required fields
        if "type" not in transform_config:
            raise ValueError(f"Transformation {i} missing 'type' field")
        
        try:
            transformer_type = TransformerType(transform_config["type"])
        except ValueError:
            raise ValueError(f"Unknown transformer type: {transform_config['type']}")
        
        # Add default parameters if not provided
        config = create_transformer_config(
            transformer_type,
            transform_config.get("params", {}),
        )
        
        pipeline.append(config)
    
    if validate:
        # Basic pipeline validation
        for i, transform_config in enumerate(pipeline[:-1]):
            current_type = TransformerType(transform_config["type"])
            next_type = TransformerType(pipeline[i + 1]["type"])
            
            # Check for incompatible sequences
            if (current_type.requires_positive and 
                next_type in {TransformerType.LOG, TransformerType.SQRT}):
                # This is actually compatible
                continue
    
    return pipeline
