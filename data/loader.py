"""Data loader utility for managing different datasets.

This module provides a centralized loader that can instantiate and manage
different dataset types based on configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union


from core import LoggerFactory
from .based.based_dataset import BasedDataset
from .titanic import TitanicDataset



class DatasetRegistry:
    """Registry for available dataset implementations."""
    
    _datasets: Dict[str, Type[BasedDataset]] = {
        "titanic": TitanicDataset,
    }
    
    @classmethod
    def register(cls, name: str, dataset_class: Type[BasedDataset]) -> None:
        """Register a new dataset class.
        
        Args:
            name: Name to register the dataset under
            dataset_class: Dataset class to register
        """
        cls._datasets[name.lower()] = dataset_class
    
    @classmethod
    def get(cls, name: str) -> Type[BasedDataset]:
        """Get a registered dataset class.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dataset class
            
        Raises:
            ValueError: If dataset is not registered
        """
        name_lower = name.lower()
        if name_lower not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(f"Dataset '{name}' not registered. Available: {available}")
        
        return cls._datasets[name_lower]
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List all available dataset names.
        
        Returns:
            List of available dataset names
        """
        return list(cls._datasets.keys())


class DatasetLoader:
    """Main dataset loader class.
    
    This class provides a unified interface for loading different types of datasets
    with consistent configuration and preprocessing options.
    """
    
    def __init__(self, logger_name: str = "data.loader") -> None:
        """Initialize the dataset loader.
        
        Args:
            logger_name: Name for the logger
        """
        self.logger = LoggerFactory.get_logger(logger_name)
        self._loaded_datasets: Dict[str, BasedDataset] = {}
    
    def load_dataset(
        self,
        name: str,
        data_dir: Union[str, Path] = "data",
        auto_load: bool = True,
        **kwargs: Any,
    ) -> BasedDataset:
        """Load a dataset by name.
        
        Args:
            name: Name of the dataset to load
            data_dir: Directory containing the dataset files
            auto_load: Whether to automatically load the data files
            **kwargs: Additional dataset-specific parameters
            
        Returns:
            Loaded dataset instance
            
        Raises:
            ValueError: If dataset name is not recognized
        """
        self.logger.info(f"Loading dataset: {name}")
        
        # Get the dataset class
        dataset_class = DatasetRegistry.get(name)
        
        # Create dataset instance
        dataset = dataset_class(data_dir=data_dir, **kwargs)
        
        # Automatically load data if requested
        if auto_load:
            dataset.load_data()
        
        # Cache the loaded dataset
        cache_key = f"{name}_{hash(str(data_dir))}"
        self._loaded_datasets[cache_key] = dataset
        
        self.logger.info(f"Dataset {name} loaded successfully")
        return dataset
    
    def load_titanic(
        self,
        data_dir: Union[str, Path] = "data",
        auto_preprocess: bool = False,
        **kwargs: Any,
    ) -> TitanicDataset:
        """Load the Titanic dataset with specific configurations.
        
        Args:
            data_dir: Directory containing the dataset files
            auto_preprocess: Whether to automatically preprocess the data
            **kwargs: Additional Titanic-specific parameters
            
        Returns:
            Loaded Titanic dataset
        """
        self.logger.info("Loading Titanic dataset")
        
        # Default configurations for Titanic
        default_config = {
            "add_family_size": True,
            "add_is_alone": True,
            "add_title": True,
            "add_deck": True,
            "add_ticket_group_size": False,
            "log_fare": True,
            "bin_age": False,
            "rare_title_threshold": 10,
        }
        
        # Merge with user-provided config
        config = {**default_config, **kwargs}
        
        # Load the dataset
        dataset = self.load_dataset("titanic", data_dir=data_dir, **config)
        
        # Automatically preprocess if requested
        if auto_preprocess:
            self.logger.info("Auto-preprocessing Titanic dataset")
            dataset.engineer_features()
            dataset.handle_missing_values(strategy="auto")
        
        return dataset
    
    def get_cached_dataset(self, name: str, data_dir: Union[str, Path] = "data") -> Optional[BasedDataset]:
        """Get a cached dataset if available.
        
        Args:
            name: Name of the dataset
            data_dir: Directory containing the dataset files
            
        Returns:
            Cached dataset or None if not found
        """
        cache_key = f"{name}_{hash(str(data_dir))}"
        return self._loaded_datasets.get(cache_key)
    
    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self.logger.info("Clearing dataset cache")
        self._loaded_datasets.clear()
    
    def create_dataset_config(
        self,
        dataset_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a configuration dictionary for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            **kwargs: Configuration parameters
            
        Returns:
            Configuration dictionary
        """
        config = {
            "name": dataset_name,
            "type": dataset_name,
            "parameters": kwargs,
        }
        
        return config
    
    def load_from_config(
        self,
        config: Dict[str, Any],
        data_dir: Union[str, Path] = "data",
    ) -> BasedDataset:
        """Load a dataset from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            data_dir: Directory containing the dataset files
            
        Returns:
            Loaded dataset
        """
        dataset_name = config.get("name") or config.get("type")
        if not dataset_name:
            raise ValueError("Configuration must include 'name' or 'type'")
        
        parameters = config.get("parameters", {})
        
        return self.load_dataset(
            dataset_name,
            data_dir=data_dir,
            **parameters,
        )
    
    def validate_dataset_config(self, config: Dict[str, Any]) -> list[str]:
        """Validate a dataset configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if "name" not in config and "type" not in config:
            errors.append("Configuration must include 'name' or 'type'")
        
        # Check if dataset is available
        dataset_name = config.get("name") or config.get("type")
        if dataset_name:
            try:
                DatasetRegistry.get(dataset_name)
            except ValueError as e:
                errors.append(str(e))
        
        # Validate parameters
        parameters = config.get("parameters", {})
        if not isinstance(parameters, dict):
            errors.append("'parameters' must be a dictionary")
        
        return errors
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a dataset type.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Information dictionary
        """
        try:
            dataset_class = DatasetRegistry.get(name)
            
            info = {
                "name": name,
                "class": dataset_class.__name__,
                "module": dataset_class.__module__,
                "docstring": dataset_class.__doc__,
            }
            
            # Try to get more information from class attributes
            if hasattr(dataset_class, "expected_features"):
                info["expected_features"] = getattr(dataset_class, "expected_features", [])
            
            return info
            
        except ValueError:
            return {"name": name, "available": False}
    
    def list_datasets(self) -> list[Dict[str, Any]]:
        """List all available datasets with their information.
        
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        for name in DatasetRegistry.list_available():
            info = self.get_dataset_info(name)
            datasets.append(info)
        
        return datasets


# Global dataset loader instance
dataset_loader = DatasetLoader()


def load_dataset(
    name: str,
    data_dir: Union[str, Path] = "data",
    **kwargs: Any,
) -> BasedDataset:
    """Convenience function to load a dataset.
    
    Args:
        name: Name of the dataset to load
        data_dir: Directory containing the dataset files
        **kwargs: Additional dataset-specific parameters
        
    Returns:
        Loaded dataset instance
    """
    return dataset_loader.load_dataset(name, data_dir=data_dir, **kwargs)


def load_titanic(
    data_dir: Union[str, Path] = "data",
    **kwargs: Any,
) -> TitanicDataset:
    """Convenience function to load the Titanic dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        **kwargs: Additional Titanic-specific parameters
        
    Returns:
        Loaded Titanic dataset
    """
    return dataset_loader.load_titanic(data_dir=data_dir, **kwargs)


def register_dataset(name: str, dataset_class: Type[BasedDataset]) -> None:
    """Convenience function to register a new dataset class.
    
    Args:
        name: Name to register the dataset under
        dataset_class: Dataset class to register
    """
    DatasetRegistry.register(name, dataset_class)
