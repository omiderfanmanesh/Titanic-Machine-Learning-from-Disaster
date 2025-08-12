"""Scaling type constants and utilities.

This module defines supported scaling methods for numerical features
and provides utilities for scaler selection and configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ScaleType(Enum):
    """Supported scaling methods for numerical features."""
    
    # No scaling
    NONE = "none"
    
    # Basic scaling
    STANDARD = "standard"
    MIN_MAX = "min_max"
    MAX_ABS = "max_abs"
    ROBUST = "robust"
    
    # Normalization
    L1_NORMALIZE = "l1_normalize"
    L2_NORMALIZE = "l2_normalize"
    MAX_NORMALIZE = "max_normalize"
    
    # Quantile-based
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"
    
    # Power transformations
    POWER_TRANSFORMER = "power_transformer"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"
    
    # Custom scaling
    UNIT_VECTOR = "unit_vector"
    DECIMAL_SCALING = "decimal_scaling"
    
    @property
    def sklearn_class(self) -> Optional[str]:
        """Get the corresponding sklearn scaler class name.
        
        Returns:
            Name of sklearn scaler class, if available
        """
        mapping = {
            self.STANDARD: "StandardScaler",
            self.MIN_MAX: "MinMaxScaler",
            self.MAX_ABS: "MaxAbsScaler",
            self.ROBUST: "RobustScaler",
            self.L1_NORMALIZE: "Normalizer",
            self.L2_NORMALIZE: "Normalizer",
            self.MAX_NORMALIZE: "Normalizer",
            self.QUANTILE_UNIFORM: "QuantileTransformer",
            self.QUANTILE_NORMAL: "QuantileTransformer",
            self.POWER_TRANSFORMER: "PowerTransformer",
        }
        return mapping.get(self)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this scaling method.
        
        Returns:
            Dictionary of default parameters
        """
        defaults = {
            self.STANDARD: {
                "with_mean": True,
                "with_std": True,
                "copy": True,
            },
            self.MIN_MAX: {
                "feature_range": (0, 1),
                "copy": True,
                "clip": False,
            },
            self.MAX_ABS: {
                "copy": True,
            },
            self.ROBUST: {
                "quantile_range": (25.0, 75.0),
                "with_centering": True,
                "with_scaling": True,
                "copy": True,
            },
            self.L1_NORMALIZE: {
                "norm": "l1",
                "copy": True,
            },
            self.L2_NORMALIZE: {
                "norm": "l2",
                "copy": True,
            },
            self.MAX_NORMALIZE: {
                "norm": "max",
                "copy": True,
            },
            self.QUANTILE_UNIFORM: {
                "n_quantiles": 1000,
                "output_distribution": "uniform",
                "subsample": 100000,
                "random_state": 42,
                "copy": True,
            },
            self.QUANTILE_NORMAL: {
                "n_quantiles": 1000,
                "output_distribution": "normal",
                "subsample": 100000,
                "random_state": 42,
                "copy": True,
            },
            self.POWER_TRANSFORMER: {
                "method": "yeo-johnson",
                "standardize": True,
                "copy": True,
            },
            self.BOX_COX: {
                "method": "box-cox",
                "standardize": True,
                "copy": True,
            },
            self.YEO_JOHNSON: {
                "method": "yeo-johnson",
                "standardize": True,
                "copy": True,
            },
        }
        return defaults.get(self, {})
    
    @property
    def requires_positive_values(self) -> bool:
        """Check if scaling method requires positive values.
        
        Returns:
            True if positive values are required
        """
        return self in {self.BOX_COX}
    
    @property
    def preserves_sparsity(self) -> bool:
        """Check if scaling method preserves sparsity.
        
        Returns:
            True if sparsity is preserved
        """
        return self in {self.MAX_ABS, self.L1_NORMALIZE, self.L2_NORMALIZE, self.MAX_NORMALIZE}
    
    @property
    def handles_outliers(self) -> bool:
        """Check if scaling method is robust to outliers.
        
        Returns:
            True if method handles outliers well
        """
        return self in {self.ROBUST, self.QUANTILE_UNIFORM, self.QUANTILE_NORMAL}
    
    @property
    def is_distribution_aware(self) -> bool:
        """Check if scaling method considers data distribution.
        
        Returns:
            True if method is distribution-aware
        """
        return self in {
            self.QUANTILE_UNIFORM,
            self.QUANTILE_NORMAL,
            self.POWER_TRANSFORMER,
            self.BOX_COX,
            self.YEO_JOHNSON,
        }
    
    def get_output_range(self) -> Optional[Tuple[float, float]]:
        """Get the expected output range for this scaling method.
        
        Returns:
            Tuple of (min, max) values, or None if unbounded
        """
        ranges = {
            self.MIN_MAX: (0.0, 1.0),
            self.QUANTILE_UNIFORM: (0.0, 1.0),
            self.MAX_ABS: (-1.0, 1.0),
        }
        return ranges.get(self)


class ScalingContext(Enum):
    """Context in which scaling is applied."""
    
    GENERAL = "general"
    NEURAL_NETWORK = "neural_network"
    TREE_BASED = "tree_based"
    LINEAR_MODEL = "linear_model"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    
    def get_recommended_scalers(self) -> List[ScaleType]:
        """Get recommended scaling methods for this context.
        
        Returns:
            List of recommended ScaleType values
        """
        recommendations = {
            self.GENERAL: [ScaleType.STANDARD, ScaleType.MIN_MAX, ScaleType.ROBUST],
            self.NEURAL_NETWORK: [ScaleType.STANDARD, ScaleType.MIN_MAX],
            self.TREE_BASED: [ScaleType.NONE, ScaleType.ROBUST],
            self.LINEAR_MODEL: [ScaleType.STANDARD, ScaleType.ROBUST],
            self.CLUSTERING: [ScaleType.STANDARD, ScaleType.MIN_MAX],
            self.DIMENSIONALITY_REDUCTION: [ScaleType.STANDARD],
        }
        return recommendations.get(self, [ScaleType.STANDARD])


def get_recommended_scaler(
    data_characteristics: Dict[str, Any],
    context: ScalingContext = ScalingContext.GENERAL,
) -> ScaleType:
    """Get recommended scaler based on data characteristics.
    
    Args:
        data_characteristics: Dictionary with data info like:
            - has_outliers: bool
            - has_negative: bool
            - is_sparse: bool
            - distribution: str (normal, skewed, uniform)
            - min_value: float
            - max_value: float
        context: Context in which scaling will be used
        
    Returns:
        Recommended ScaleType
    """
    has_outliers = data_characteristics.get("has_outliers", False)
    has_negative = data_characteristics.get("has_negative", True)
    is_sparse = data_characteristics.get("is_sparse", False)
    distribution = data_characteristics.get("distribution", "unknown")
    
    # If data is sparse, prefer sparsity-preserving scalers
    if is_sparse:
        if has_outliers:
            return ScaleType.MAX_ABS
        else:
            return ScaleType.L2_NORMALIZE
    
    # If data has outliers, use robust methods
    if has_outliers:
        if context == ScalingContext.NEURAL_NETWORK:
            return ScaleType.QUANTILE_UNIFORM
        else:
            return ScaleType.ROBUST
    
    # For highly skewed data, consider power transformations
    if distribution == "skewed":
        if has_negative:
            return ScaleType.YEO_JOHNSON
        else:
            return ScaleType.BOX_COX
    
    # Default recommendations by context
    recommended = context.get_recommended_scalers()
    return recommended[0] if recommended else ScaleType.STANDARD


def create_scaler_config(
    scale_type: ScaleType,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create scaler configuration with defaults and custom parameters.
    
    Args:
        scale_type: Type of scaler to configure
        custom_params: Custom parameters to override defaults
        
    Returns:
        Complete scaler configuration
    """
    config = {
        "type": scale_type.value,
        "params": scale_type.get_default_params(),
    }
    
    if custom_params:
        config["params"].update(custom_params)
    
    return config


def validate_scaler_params(
    scale_type: ScaleType,
    params: Dict[str, Any],
    data_info: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate scaler parameters against data characteristics.
    
    Args:
        scale_type: Type of scaler
        params: Parameters to validate
        data_info: Optional data characteristics for validation
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check if positive values are required
    if scale_type.requires_positive_values and data_info:
        min_value = data_info.get("min_value")
        if min_value is not None and min_value <= 0:
            errors.append(f"{scale_type.value} requires positive values, but data has min={min_value}")
    
    # Validate specific parameters
    if scale_type == ScaleType.MIN_MAX:
        if "feature_range" in params:
            feature_range = params["feature_range"]
            if not isinstance(feature_range, (tuple, list)) or len(feature_range) != 2:
                errors.append("feature_range must be a tuple/list of length 2")
            elif feature_range[0] >= feature_range[1]:
                errors.append("feature_range min must be less than max")
    
    elif scale_type == ScaleType.ROBUST:
        if "quantile_range" in params:
            quantile_range = params["quantile_range"]
            if not isinstance(quantile_range, (tuple, list)) or len(quantile_range) != 2:
                errors.append("quantile_range must be a tuple/list of length 2")
            elif not (0 <= quantile_range[0] < quantile_range[1] <= 100):
                errors.append("quantile_range must be between 0 and 100 with min < max")
    
    elif scale_type in {ScaleType.QUANTILE_UNIFORM, ScaleType.QUANTILE_NORMAL}:
        if "n_quantiles" in params:
            n_quantiles = params["n_quantiles"]
            if not isinstance(n_quantiles, int) or n_quantiles <= 0:
                errors.append("n_quantiles must be a positive integer")
            if data_info and "n_samples" in data_info:
                n_samples = data_info["n_samples"]
                if n_quantiles > n_samples:
                    errors.append(f"n_quantiles ({n_quantiles}) cannot exceed n_samples ({n_samples})")
    
    return errors


def analyze_data_for_scaling(data: np.ndarray) -> Dict[str, Any]:
    """Analyze data characteristics relevant for scaling decisions.
    
    Args:
        data: Numerical data array
        
    Returns:
        Dictionary with data characteristics
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    characteristics = {}
    
    # Basic statistics
    characteristics["n_samples"] = data.shape[0]
    characteristics["n_features"] = data.shape[1]
    characteristics["min_value"] = float(np.min(data))
    characteristics["max_value"] = float(np.max(data))
    characteristics["mean"] = float(np.mean(data))
    characteristics["std"] = float(np.std(data))
    
    # Check for special properties
    characteristics["has_negative"] = np.any(data < 0)
    characteristics["has_zero"] = np.any(data == 0)
    characteristics["is_sparse"] = np.count_nonzero(data) / data.size < 0.1
    
    # Outlier detection using IQR method
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    outliers = (data < outlier_bounds[0]) | (data > outlier_bounds[1])
    characteristics["has_outliers"] = np.any(outliers)
    characteristics["outlier_ratio"] = np.sum(outliers) / data.size
    
    # Distribution analysis
    from scipy import stats
    try:
        # Test for normality
        _, p_value = stats.normaltest(data.flatten())
        characteristics["is_normal"] = p_value > 0.05
        
        # Skewness
        skewness = stats.skew(data.flatten())
        characteristics["skewness"] = float(skewness)
        
        if abs(skewness) > 1:
            characteristics["distribution"] = "skewed"
        elif characteristics["is_normal"]:
            characteristics["distribution"] = "normal"
        else:
            characteristics["distribution"] = "unknown"
            
    except Exception:
        characteristics["is_normal"] = False
        characteristics["skewness"] = 0.0
        characteristics["distribution"] = "unknown"
    
    return characteristics
