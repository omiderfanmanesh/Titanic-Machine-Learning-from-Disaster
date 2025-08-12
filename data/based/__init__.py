"""Base data handling components.

This module provides base classes, enums, and utilities for data handling,
preprocessing, and transformation.
"""

from .file_types import (
    FileType,
    CompressionType,
    detect_file_type,
    get_supported_extensions,
    validate_file_path,
)

from .encoder_enum import (
    EncoderType,
    NumericalEncoderType,
    get_recommended_encoder,
    create_encoder_config,
    validate_encoder_params,
)

from .sampling_types import (
    SamplingType,
    SamplingStrategy,
    get_recommended_sampling,
    create_sampling_config,
    validate_sampling_params,
)

from .scale_types import (
    ScaleType,
    ScalingContext,
    get_recommended_scaler,
    create_scaler_config,
    validate_scaler_params,
    analyze_data_for_scaling,
)

from .transformers_enums import (
    TransformerType,
    FeatureTransformationType,
    get_recommended_transformer,
    create_transformer_config,
    validate_transformer_params,
    get_transformation_pipeline,
)

__all__ = [
    # File types
    "FileType",
    "CompressionType", 
    "detect_file_type",
    "get_supported_extensions",
    "validate_file_path",
    
    # Encoders
    "EncoderType",
    "NumericalEncoderType",
    "get_recommended_encoder",
    "create_encoder_config",
    "validate_encoder_params",
    
    # Sampling
    "SamplingType",
    "SamplingStrategy",
    "get_recommended_sampling",
    "create_sampling_config",
    "validate_sampling_params",
    
    # Scaling
    "ScaleType",
    "ScalingContext",
    "get_recommended_scaler",
    "create_scaler_config",
    "validate_scaler_params",
    "analyze_data_for_scaling",
    
    # Transformers
    "TransformerType",
    "FeatureTransformationType",
    "get_recommended_transformer",
    "create_transformer_config",
    "validate_transformer_params",
    "get_transformation_pipeline",
]
