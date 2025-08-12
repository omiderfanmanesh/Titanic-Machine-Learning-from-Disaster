"""Sampling type constants and utilities.

This module defines supported sampling methods for handling imbalanced datasets
and provides utilities for sampler selection and configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union


class SamplingType(Enum):
    """Supported sampling methods for imbalanced datasets."""
    
    # No sampling
    NONE = "none"
    
    # Under-sampling methods
    RANDOM_UNDER = "random_under"
    CLUSTER_CENTROIDS = "cluster_centroids"
    CONDENSED_NEAREST_NEIGHBOUR = "condensed_nearest_neighbour"
    EDITED_NEAREST_NEIGHBOUR = "edited_nearest_neighbour"
    REPEATED_EDITED_NEAREST_NEIGHBOUR = "repeated_edited_nearest_neighbour"
    ALL_KNN = "all_knn"
    INSTANCE_HARDNESS_THRESHOLD = "instance_hardness_threshold"
    NEIGHBOURHOOD_CLEANING_RULE = "neighbourhood_cleaning_rule"
    NEAR_MISS = "near_miss"
    TOMEK_LINKS = "tomek_links"
    
    # Over-sampling methods
    RANDOM_OVER = "random_over"
    SMOTE = "smote"
    SMOTE_BORDERLINE = "smote_borderline"
    SMOTE_SVM = "smote_svm"
    ADASYN = "adasyn"
    KMeans_SMOTE = "kmeans_smote"
    SMOTE_NC = "smote_nc"
    SMOTE_ENN = "smote_enn"
    SMOTE_TOMEK = "smote_tomek"
    
    # Combined methods
    SMOTE_ENN_COMBINED = "smote_enn_combined"
    SMOTE_TOMEK_COMBINED = "smote_tomek_combined"
    
    @property
    def imblearn_class(self) -> Optional[str]:
        """Get the corresponding imbalanced-learn class name.
        
        Returns:
            Name of imblearn class, if available
        """
        mapping = {
            # Under-sampling
            self.RANDOM_UNDER: "RandomUnderSampler",
            self.CLUSTER_CENTROIDS: "ClusterCentroids",
            self.CONDENSED_NEAREST_NEIGHBOUR: "CondensedNearestNeighbour",
            self.EDITED_NEAREST_NEIGHBOUR: "EditedNearestNeighbours",
            self.REPEATED_EDITED_NEAREST_NEIGHBOUR: "RepeatedEditedNearestNeighbours",
            self.ALL_KNN: "AllKNN",
            self.INSTANCE_HARDNESS_THRESHOLD: "InstanceHardnessThreshold",
            self.NEIGHBOURHOOD_CLEANING_RULE: "NeighbourhoodCleaningRule",
            self.NEAR_MISS: "NearMiss",
            self.TOMEK_LINKS: "TomekLinks",
            
            # Over-sampling
            self.RANDOM_OVER: "RandomOverSampler",
            self.SMOTE: "SMOTE",
            self.SMOTE_BORDERLINE: "BorderlineSMOTE",
            self.SMOTE_SVM: "SVMSMOTE",
            self.ADASYN: "ADASYN",
            self.KMeans_SMOTE: "KMeansSMOTE",
            self.SMOTE_NC: "SMOTENC",
            
            # Combined
            self.SMOTE_ENN: "SMOTEENN",
            self.SMOTE_TOMEK: "SMOTETomek",
        }
        return mapping.get(self)
    
    @property
    def sampling_strategy_type(self) -> str:
        """Get the sampling strategy type (under, over, combined).
        
        Returns:
            Type of sampling strategy
        """
        under_sampling = {
            self.RANDOM_UNDER, self.CLUSTER_CENTROIDS, self.CONDENSED_NEAREST_NEIGHBOUR,
            self.EDITED_NEAREST_NEIGHBOUR, self.REPEATED_EDITED_NEAREST_NEIGHBOUR,
            self.ALL_KNN, self.INSTANCE_HARDNESS_THRESHOLD, self.NEIGHBOURHOOD_CLEANING_RULE,
            self.NEAR_MISS, self.TOMEK_LINKS,
        }
        
        over_sampling = {
            self.RANDOM_OVER, self.SMOTE, self.SMOTE_BORDERLINE, self.SMOTE_SVM,
            self.ADASYN, self.KMeans_SMOTE, self.SMOTE_NC,
        }
        
        combined = {
            self.SMOTE_ENN, self.SMOTE_TOMEK, self.SMOTE_ENN_COMBINED, self.SMOTE_TOMEK_COMBINED,
        }
        
        if self in under_sampling:
            return "under"
        elif self in over_sampling:
            return "over"
        elif self in combined:
            return "combined"
        else:
            return "none"
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this sampling method.
        
        Returns:
            Dictionary of default parameters
        """
        defaults = {
            # Under-sampling
            self.RANDOM_UNDER: {
                "sampling_strategy": "auto",
                "random_state": 42,
                "replacement": False,
            },
            self.CLUSTER_CENTROIDS: {
                "sampling_strategy": "auto",
                "random_state": 42,
                "n_jobs": None,
            },
            self.NEAR_MISS: {
                "sampling_strategy": "auto",
                "version": 1,
                "n_neighbors": 3,
                "n_jobs": None,
            },
            self.TOMEK_LINKS: {
                "sampling_strategy": "auto",
                "n_jobs": None,
            },
            
            # Over-sampling
            self.RANDOM_OVER: {
                "sampling_strategy": "auto",
                "random_state": 42,
                "shrinkage": None,
            },
            self.SMOTE: {
                "sampling_strategy": "auto",
                "random_state": 42,
                "k_neighbors": 5,
                "n_jobs": None,
            },
            self.SMOTE_BORDERLINE: {
                "sampling_strategy": "auto",
                "random_state": 42,
                "k_neighbors": 5,
                "m_neighbors": 10,
                "kind": "borderline-1",
                "n_jobs": None,
            },
            self.ADASYN: {
                "sampling_strategy": "auto",
                "random_state": 42,
                "n_neighbors": 5,
                "n_jobs": None,
            },
            
            # Combined
            self.SMOTE_ENN: {
                "sampling_strategy": "auto",
                "random_state": 42,
                "smote": None,  # Will use default SMOTE
                "enn": None,    # Will use default ENN
            },
            self.SMOTE_TOMEK: {
                "sampling_strategy": "auto",
                "random_state": 42,
                "smote": None,  # Will use default SMOTE
                "tomek": None,  # Will use default Tomek
            },
        }
        return defaults.get(self, {})
    
    @property
    def requires_neighbors(self) -> bool:
        """Check if sampling method requires neighbor calculation.
        
        Returns:
            True if neighbors are required
        """
        neighbor_methods = {
            self.SMOTE, self.SMOTE_BORDERLINE, self.SMOTE_SVM, self.ADASYN,
            self.KMeans_SMOTE, self.SMOTE_NC, self.CONDENSED_NEAREST_NEIGHBOUR,
            self.EDITED_NEAREST_NEIGHBOUR, self.REPEATED_EDITED_NEAREST_NEIGHBOUR,
            self.ALL_KNN, self.NEIGHBOURHOOD_CLEANING_RULE, self.NEAR_MISS,
        }
        return self in neighbor_methods
    
    @property
    def supports_categorical(self) -> bool:
        """Check if sampling method supports categorical features.
        
        Returns:
            True if categorical features are supported
        """
        categorical_support = {
            self.RANDOM_UNDER, self.RANDOM_OVER, self.SMOTE_NC,
        }
        return self in categorical_support
    
    def get_min_samples_required(self, n_classes: int = 2) -> int:
        """Get minimum samples required for this sampling method.
        
        Args:
            n_classes: Number of classes in the dataset
            
        Returns:
            Minimum number of samples required per class
        """
        if self.requires_neighbors:
            # Most neighbor-based methods need at least k+1 samples
            k_neighbors = self.get_default_params().get("k_neighbors", 5)
            return k_neighbors + 1
        else:
            return 1


class SamplingStrategy(Enum):
    """Sampling strategy options."""
    
    AUTO = "auto"
    MAJORITY = "majority"
    NOT_MINORITY = "not minority"
    NOT_MAJORITY = "not majority"
    ALL = "all"
    MINORITY = "minority"
    
    @classmethod
    def get_strategy_for_method(cls, sampling_type: SamplingType) -> SamplingStrategy:
        """Get recommended strategy for a sampling method.
        
        Args:
            sampling_type: The sampling method
            
        Returns:
            Recommended sampling strategy
        """
        strategy_type = sampling_type.sampling_strategy_type
        
        if strategy_type == "under":
            return cls.AUTO
        elif strategy_type == "over":
            return cls.AUTO
        elif strategy_type == "combined":
            return cls.AUTO
        else:
            return cls.AUTO


def get_recommended_sampling(
    imbalance_ratio: float,
    n_samples: int,
    n_features: int,
    has_categorical: bool = False,
) -> SamplingType:
    """Get recommended sampling method based on data characteristics.
    
    Args:
        imbalance_ratio: Ratio of majority to minority class
        n_samples: Total number of samples
        n_features: Number of features
        has_categorical: Whether dataset has categorical features
        
    Returns:
        Recommended SamplingType
    """
    # No sampling if data is relatively balanced
    if imbalance_ratio < 3.0:
        return SamplingType.NONE
    
    # For small datasets, prefer simple methods
    if n_samples < 1000:
        return SamplingType.RANDOM_OVER
    
    # For datasets with categorical features
    if has_categorical:
        if imbalance_ratio < 10.0:
            return SamplingType.RANDOM_OVER
        else:
            return SamplingType.SMOTE_NC
    
    # For moderate imbalance, use SMOTE
    if imbalance_ratio < 10.0:
        return SamplingType.SMOTE
    
    # For severe imbalance, use combined methods
    if imbalance_ratio < 20.0:
        return SamplingType.SMOTE_ENN
    else:
        return SamplingType.SMOTE_TOMEK


def create_sampling_config(
    sampling_type: SamplingType,
    custom_params: Optional[Dict[str, Any]] = None,
    custom_strategy: Optional[Union[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Create sampling configuration with defaults and custom parameters.
    
    Args:
        sampling_type: Type of sampling to configure
        custom_params: Custom parameters to override defaults
        custom_strategy: Custom sampling strategy
        
    Returns:
        Complete sampling configuration
    """
    config = {
        "type": sampling_type.value,
        "params": sampling_type.get_default_params(),
    }
    
    if custom_strategy:
        config["params"]["sampling_strategy"] = custom_strategy
    
    if custom_params:
        config["params"].update(custom_params)
    
    return config


def validate_sampling_params(
    sampling_type: SamplingType,
    params: Dict[str, Any],
    n_samples_per_class: Dict[int, int],
) -> List[str]:
    """Validate sampling parameters against data characteristics.
    
    Args:
        sampling_type: Type of sampling
        params: Parameters to validate
        n_samples_per_class: Number of samples per class
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check minimum samples requirement
    min_samples = min(n_samples_per_class.values())
    required_samples = sampling_type.get_min_samples_required(len(n_samples_per_class))
    
    if min_samples < required_samples:
        errors.append(
            f"Sampling method {sampling_type.value} requires at least "
            f"{required_samples} samples per class, but minimum class has {min_samples}"
        )
    
    # Validate specific parameters
    if "k_neighbors" in params:
        k_neighbors = params["k_neighbors"]
        if k_neighbors >= min_samples:
            errors.append(
                f"k_neighbors ({k_neighbors}) must be less than minimum class size ({min_samples})"
            )
    
    if "random_state" in params:
        random_state = params["random_state"]
        if random_state is not None and not isinstance(random_state, int):
            errors.append("random_state must be an integer or None")
    
    return errors
