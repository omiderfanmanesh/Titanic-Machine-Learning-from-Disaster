"""Data module for dataset handling and preprocessing.

This module provides classes and utilities for loading, preprocessing,
and managing datasets with a focus on machine learning workflows.
"""

from .based import (
    # File types
    FileType,
    CompressionType,
    detect_file_type,
    get_supported_extensions,
    validate_file_path,
    
    # Encoders
    EncoderType,
    NumericalEncoderType,
    get_recommended_encoder,
    create_encoder_config,
    validate_encoder_params,
    
    # Sampling
    SamplingType,
    SamplingStrategy,
    get_recommended_sampling,
    create_sampling_config,
    validate_sampling_params,
    
    # Scaling
    ScaleType,
    ScalingContext,
    get_recommended_scaler,
    create_scaler_config,
    validate_scaler_params,
    analyze_data_for_scaling,
    
    # Transformers
    TransformerType,
    FeatureTransformationType,
    get_recommended_transformer,
    create_transformer_config,
    validate_transformer_params,
    get_transformation_pipeline,
    
    # Base dataset
    BasedDataset,
)

from .titanic import TitanicDataset

from .loader import (
    DatasetRegistry,
    DatasetLoader,
    dataset_loader,
    load_dataset,
    load_titanic,
    register_dataset,
)

__all__ = [
    # Base classes and enums
    "BasedDataset",
    "FileType",
    "CompressionType",
    "EncoderType",
    "NumericalEncoderType",
    "SamplingType",
    "SamplingStrategy",
    "ScaleType",
    "ScalingContext",
    "TransformerType",
    "FeatureTransformationType",
    
    # Utilities
    "detect_file_type",
    "get_supported_extensions",
    "validate_file_path",
    "get_recommended_encoder",
    "create_encoder_config",
    "validate_encoder_params",
    "get_recommended_sampling",
    "create_sampling_config",
    "validate_sampling_params",
    "get_recommended_scaler",
    "create_scaler_config",
    "validate_scaler_params",
    "analyze_data_for_scaling",
    "get_recommended_transformer",
    "create_transformer_config",
    "validate_transformer_params",
    "get_transformation_pipeline",
    
    # Dataset classes
    "TitanicDataset",
    
    # Loader classes and functions
    "DatasetRegistry",
    "DatasetLoader",
    "dataset_loader",
    "load_dataset",
    "load_titanic",
    "register_dataset",
]
