"""Base dataset class for data handling and manipulation.

This module provides the abstract base class for all dataset implementations,
defining the interface for data loading, preprocessing, and manipulation.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from core import LoggerFactory
from .file_types import FileType, detect_file_type
from .encoder_enum import EncoderType, get_recommended_encoder
from .sampling_types import SamplingType, get_recommended_sampling
from .scale_types import ScaleType, get_recommended_scaler, analyze_data_for_scaling
from .transformers_enums import TransformerType, get_recommended_transformer


class BasedDataset(ABC):
    """Abstract base class for all dataset implementations.
    
    This class provides the interface and common functionality for data loading,
    preprocessing, and manipulation. Subclasses should implement dataset-specific
    logic while inheriting the common functionality.
    
    Attributes:
        name: Name of the dataset
        data_dir: Directory containing the dataset files
        target_column: Name of the target column
        id_column: Name of the ID column
        df_train: Training dataset
        df_test: Test dataset
        df_validation: Validation dataset (if created)
        metadata: Dataset metadata
    """
    
    def __init__(
        self,
        name: str,
        data_dir: Union[str, Path] = "data",
        target_column: Optional[str] = None,
        id_column: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            name: Name of the dataset
            data_dir: Directory containing the dataset files
            target_column: Name of the target column
            id_column: Name of the ID column
            **kwargs: Additional dataset-specific parameters
        """
        self.name = name
        self.data_dir = Path(data_dir)
        self.target_column = target_column
        self.id_column = id_column
        
        # Data containers
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.df_validation: Optional[pd.DataFrame] = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {
            "name": name,
            "target_column": self.target_column,
            "id_column": self.id_column,
            "loaded": False,
            "preprocessed": False,
        }
        
        # Configuration
        self.config = kwargs
        
        # Initialize logger
        self.logger = LoggerFactory.get_logger(f"dataset.{name}")

    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset from files.
        
        This method should be implemented by subclasses to load the specific
        dataset. It should populate self.df_train and self.df_test.
        """
        pass
    
    @abstractmethod
    def get_feature_types(self) -> Dict[str, str]:
        """Get the types of features in the dataset.
        
        Returns:
            Dictionary mapping column names to types ('numerical', 'categorical', etc.)
        """
        pass
    
    def load_file(
        self,
        file_path: Union[str, Path],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a single file into a DataFrame.
        
        Args:
            file_path: Path to the file to load
            **kwargs: Additional parameters for the file reader
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            file_type, compression = detect_file_type(file_path)
        except ValueError as e:
            raise ValueError(f"Unsupported file type: {e}")
        
        # Get default read parameters
        read_kwargs = file_type.get_read_kwargs()
        if compression.value:
            read_kwargs["compression"] = compression.value
        
        # Override with custom parameters
        read_kwargs.update(kwargs)
        
        # Load the data
        reader_name = file_type.pandas_reader
        reader_func = getattr(pd, reader_name)
        
        self.logger.info(f"Loading file: {file_path} (type: {file_type.value})")
        df = reader_func(file_path, **read_kwargs)
        
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def save_file(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs: Any,
    ) -> None:
        """Save a DataFrame to a file.
        
        Args:
            df: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional parameters for the file writer
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            file_type, compression = detect_file_type(file_path)
        except ValueError as e:
            raise ValueError(f"Unsupported file type: {e}")
        
        # Get default write parameters
        write_kwargs = file_type.get_write_kwargs()
        if compression.value:
            write_kwargs["compression"] = compression.value
        
        # Override with custom parameters
        write_kwargs.update(kwargs)
        
        # Save the data
        writer_name = file_type.pandas_writer
        writer_func = getattr(df, writer_name)
        
        self.logger.info(f"Saving to: {file_path} (type: {file_type.value})")
        writer_func(file_path, **write_kwargs)
        
        self.logger.info(f"Saved {len(df)} rows, {len(df.columns)} columns")
    
    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset.
        
        Returns:
            Dictionary with basic dataset information
        """
        if not self.is_loaded():
            self.logger.warning("Dataset not loaded yet")
            return {}
        
        info = {
            "name": self.name,
            "train_shape": self.df_train.shape if self.df_train is not None else None,
            "test_shape": self.df_test.shape if self.df_test is not None else None,
            "validation_shape": self.df_validation.shape if self.df_validation is not None else None,
            "target_column": self.target_column,
            "id_column": self.id_column,
        }
        
        if self.df_train is not None:
            info.update({
                "feature_count": len(self.df_train.columns),
                "numerical_features": len(self.df_train.select_dtypes(include=[np.number]).columns),
                "categorical_features": len(self.df_train.select_dtypes(exclude=[np.number]).columns),
                "missing_values": self.df_train.isnull().sum().sum(),
                "duplicate_rows": self.df_train.duplicated().sum(),
            })
            
            if self.target_column and self.target_column in self.df_train.columns:
                target_info = self._analyze_target()
                info.update(target_info)
        
        return info
    
    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze the target variable.
        
        Returns:
            Dictionary with target analysis results
        """
        if not self.df_train or self.target_column not in self.df_train.columns:
            return {}
        
        target = self.df_train[self.target_column]
        
        analysis = {
            "target_type": "numerical" if target.dtype in [np.number] else "categorical",
            "target_unique_values": int(target.nunique()),
            "target_missing": int(target.isnull().sum()),
        }
        
        if analysis["target_type"] == "categorical" or analysis["target_unique_values"] <= 20:
            # Categorical target analysis
            value_counts = target.value_counts()
            analysis.update({
                "target_distribution": value_counts.to_dict(),
                "target_balance_ratio": float(value_counts.max() / value_counts.min()) if len(value_counts) > 1 else 1.0,
                "is_balanced": analysis.get("target_balance_ratio", 1.0) <= 3.0,
            })
        else:
            # Numerical target analysis
            analysis.update({
                "target_mean": float(target.mean()),
                "target_std": float(target.std()),
                "target_min": float(target.min()),
                "target_max": float(target.max()),
                "target_skewness": float(target.skew()),
            })
        
        return analysis
    
    def create_validation_split(
        self,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42,
    ) -> None:
        """Create a validation split from the training data.
        
        Args:
            test_size: Proportion of data to use for validation
            stratify: Whether to stratify the split
            random_state: Random state for reproducibility
        """
        if not self.is_loaded():
            raise ValueError("Dataset must be loaded before creating validation split")
        
        if self.df_train is None:
            raise ValueError("Training data not available")
        
        stratify_column = None
        if stratify and self.target_column and self.target_column in self.df_train.columns:
            target_nunique = self.df_train[self.target_column].nunique()
            if target_nunique <= 20:  # Only stratify for categorical targets
                stratify_column = self.df_train[self.target_column]
        
        train_data, val_data = train_test_split(
            self.df_train,
            test_size=test_size,
            stratify=stratify_column,
            random_state=random_state,
        )
        
        self.df_train = train_data.reset_index(drop=True)
        self.df_validation = val_data.reset_index(drop=True)
        
        self.logger.info(
            f"Created validation split: train={len(self.df_train)}, "
            f"validation={len(self.df_validation)}"
        )
    
    def handle_missing_values(
        self,
        strategy: str = "auto",
        columns: Optional[List[str]] = None,
    ) -> None:
        """Handle missing values in the dataset.
        
        Args:
            strategy: Strategy for handling missing values
                - "auto": Automatic strategy based on data type
                - "drop_rows": Drop rows with missing values
                - "drop_columns": Drop columns with missing values
                - "fill_mean": Fill with mean (numerical) / mode (categorical)
                - "fill_median": Fill with median (numerical) / mode (categorical)
                - "fill_mode": Fill with mode
                - "forward_fill": Forward fill
                - "backward_fill": Backward fill
            columns: Specific columns to process (None for all)
        """
        if not self.is_loaded():
            raise ValueError("Dataset must be loaded before handling missing values")
        
        datasets = [("train", self.df_train)]
        if self.df_test is not None:
            datasets.append(("test", self.df_test))
        if self.df_validation is not None:
            datasets.append(("validation", self.df_validation))
        
        for name, df in datasets:
            if df is None:
                continue
                
            self.logger.info(f"Handling missing values in {name} set")
            original_missing = df.isnull().sum().sum()
            
            target_columns = columns or df.columns.tolist()
            if self.target_column and self.target_column in target_columns:
                target_columns.remove(self.target_column)  # Don't process target
            
            for col in target_columns:
                if col not in df.columns:
                    continue
                
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue
                
                col_strategy = strategy
                if strategy == "auto":
                    # Automatic strategy based on data type and missing percentage
                    missing_pct = missing_count / len(df)
                    if missing_pct > 0.5:  # Arbitrary threshold for dropping columns
                        col_strategy = "drop_columns"
                    elif df[col].dtype in [np.number]:
                        col_strategy = "fill_median"
                    else:
                        col_strategy = "fill_mode"
                
                # Apply the strategy
                if col_strategy == "drop_columns":
                    df.drop(columns=[col], inplace=True)
                    self.logger.info(f"Dropped column {col} ({missing_pct:.1%} missing)")
                elif col_strategy in ["fill_mean", "fill_median", "fill_mode"]:
                    if df[col].dtype in [np.number]:
                        fill_value = df[col].mean() if col_strategy == "fill_mean" else df[col].median()
                    else:
                        fill_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown"
                    
                    df[col].fillna(fill_value, inplace=True)
                    self.logger.info(f"Filled {col} with {fill_value}")
                elif col_strategy == "forward_fill":
                    df[col].fillna(method="ffill", inplace=True)
                elif col_strategy == "backward_fill":
                    df[col].fillna(method="bfill", inplace=True)
                elif col_strategy == "drop_rows":
                    df.dropna(subset=[col], inplace=True)
            
            final_missing = df.isnull().sum().sum()
            self.logger.info(
                f"Missing values in {name}: {original_missing} -> {final_missing}"
            )
    
    def detect_outliers(
        self,
        method: str = "iqr",
        threshold: float = 3.0,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Detect outliers in numerical columns.
        
        Args:
            method: Method for outlier detection
                - "iqr": Interquartile range method
                - "zscore": Z-score method
                - "isolation_forest": Isolation forest method
            threshold: Threshold for outlier detection
            columns: Specific columns to check (None for all numerical)
            
        Returns:
            Dictionary mapping column names to boolean arrays of outlier indices
        """
        if not self.is_loaded() or self.df_train is None:
            raise ValueError("Dataset must be loaded before detecting outliers")
        
        numerical_columns = self.df_train.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column and self.target_column in numerical_columns:
            numerical_columns.remove(self.target_column)
        
        target_columns = columns or numerical_columns
        outliers = {}
        
        for col in target_columns:
            if col not in self.df_train.columns:
                continue
            
            data = self.df_train[col].dropna()
            if len(data) == 0:
                continue
            
            if method == "iqr":
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_mask = (self.df_train[col] < lower_bound) | (self.df_train[col] > upper_bound)
                
            elif method == "zscore":
                from scipy import stats
                z_scores = np.abs(stats.zscore(data))
                outlier_indices = data.index[z_scores > threshold]
                outlier_mask = self.df_train.index.isin(outlier_indices)
                
            elif method == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
                outlier_mask = pd.Series(outlier_labels == -1, index=data.index).reindex(self.df_train.index, fill_value=False)
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outliers[col] = outlier_mask
            outlier_count = outlier_mask.sum()
            outlier_pct = outlier_count / len(self.df_train) * 100
            
            self.logger.info(f"Column {col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
        
        return outliers
    
    def is_loaded(self) -> bool:
        """Check if the dataset is loaded.
        
        Returns:
            True if dataset is loaded
        """
        return self.metadata.get("loaded", False)
    
    def is_preprocessed(self) -> bool:
        """Check if the dataset is preprocessed.
        
        Returns:
            True if dataset is preprocessed
        """
        return self.metadata.get("preprocessed", False)
    
    def get_X_y(
        self,
        dataset: str = "train",
        include_id: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Get features and target from a dataset split.
        
        Args:
            dataset: Which dataset to use ("train", "test", "validation")
            include_id: Whether to include ID column in features
            
        Returns:
            Tuple of (features, target)
        """
        if dataset == "train":
            df = self.df_train
        elif dataset == "test":
            df = self.df_test
        elif dataset == "validation":
            df = self.df_validation
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        if df is None:
            raise ValueError(f"Dataset {dataset} is not available")
        
        # Get features
        exclude_columns = []
        if self.target_column and self.target_column in df.columns:
            exclude_columns.append(self.target_column)
        if not include_id and self.id_column and self.id_column in df.columns:
            exclude_columns.append(self.id_column)
        
        X = df.drop(columns=exclude_columns, errors="ignore")
        
        # Get target
        y = None
        if self.target_column and self.target_column in df.columns:
            y = df[self.target_column]
        
        return X, y
    
    def update_metadata(self, **kwargs: Any) -> None:
        """Update dataset metadata.
        
        Args:
            **kwargs: Key-value pairs to update in metadata
        """
        self.metadata.update(kwargs)
    
    def summary(self) -> str:
        """Get a summary string of the dataset.
        
        Returns:
            Summary string
        """
        info = self.get_basic_info()
        
        summary_lines = [
            f"Dataset: {self.name}",
            f"Status: {'Loaded' if self.is_loaded() else 'Not loaded'}",
        ]
        
        if self.is_loaded():
            summary_lines.extend([
                f"Train shape: {info.get('train_shape', 'N/A')}",
                f"Test shape: {info.get('test_shape', 'N/A')}",
                f"Validation shape: {info.get('validation_shape', 'N/A')}",
                f"Features: {info.get('feature_count', 0)} total",
                f"  - Numerical: {info.get('numerical_features', 0)}",
                f"  - Categorical: {info.get('categorical_features', 0)}",
                f"Missing values: {info.get('missing_values', 0)}",
                f"Target column: {self.target_column}",
            ])
            
            if "target_distribution" in info:
                summary_lines.append(f"Target distribution: {info['target_distribution']}")
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"BasedDataset(name='{self.name}', loaded={self.is_loaded()})"
