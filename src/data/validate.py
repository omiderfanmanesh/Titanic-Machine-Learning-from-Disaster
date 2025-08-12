"""Data validation components with leakage detection and quality checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from core.interfaces import IDataValidator
from core.utils import LoggerFactory


class TitanicDataValidator(IDataValidator):
    """Comprehensive data validator for Titanic competition."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        
        # Expected schema for Titanic data
        self.required_train_columns = {
            "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", 
            "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
        }
        
        self.required_test_columns = self.required_train_columns - {"Survived"}
        
        self.categorical_columns = {"Sex", "Embarked", "Pclass"}
        self.numeric_columns = {"Age", "SibSp", "Parch", "Fare"}
        
    def validate_train(self, df: pd.DataFrame) -> bool:
        """Validate training data."""
        self.logger.info("Validating training data")
        
        self._check_required_columns(df, self.required_train_columns, "train")
        self._check_data_types(df)
        self._check_target_values(df)
        self._check_duplicate_ids(df)
        self._check_data_quality(df)
        
        self.logger.info("✅ Training data validation passed")
        return True
    
    def validate_test(self, df: pd.DataFrame) -> bool:
        """Validate test data."""
        self.logger.info("Validating test data")
        
        self._check_required_columns(df, self.required_test_columns, "test")
        self._check_data_types(df)
        self._check_duplicate_ids(df)
        self._check_data_quality(df)
        
        # Test data should not have target column
        if "Survived" in df.columns:
            raise ValueError("Test data should not contain 'Survived' column")
            
        self.logger.info("✅ Test data validation passed")
        return True
    
    def validate_consistency(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate consistency between train and test data."""
        self.logger.info("Validating train/test consistency")
        
        # Check for overlapping PassengerIds
        train_ids = set(train_df["PassengerId"])
        test_ids = set(test_df["PassengerId"])
        
        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise ValueError(f"Found overlapping PassengerId values: {len(overlap)} IDs overlap")
        
        # Check feature distributions
        self._check_feature_distributions(train_df, test_df)
        
        self.logger.info("✅ Train/test consistency validation passed")
        return True
    
    def _check_required_columns(self, df: pd.DataFrame, required: Set[str], dataset_type: str) -> None:
        """Check that all required columns are present."""
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {dataset_type} data: {missing}")
    
    def _check_data_types(self, df: pd.DataFrame) -> None:
        """Check data types are reasonable."""
        # PassengerId should be numeric
        if not pd.api.types.is_numeric_dtype(df["PassengerId"]):
            raise ValueError("PassengerId should be numeric")
        
        # Pclass should be 1, 2, or 3
        if "Pclass" in df.columns:
            valid_classes = {1, 2, 3}
            invalid_classes = set(df["Pclass"].dropna().unique()) - valid_classes
            if invalid_classes:
                raise ValueError(f"Invalid Pclass values: {invalid_classes}")
        
        # Sex should be male/female
        if "Sex" in df.columns:
            valid_sexes = {"male", "female"}
            invalid_sexes = set(df["Sex"].dropna().unique()) - valid_sexes
            if invalid_sexes:
                raise ValueError(f"Invalid Sex values: {invalid_sexes}")
        
        # Age should be positive if present
        if "Age" in df.columns:
            invalid_ages = df["Age"].dropna()
            if (invalid_ages < 0).any() or (invalid_ages > 150).any():
                raise ValueError("Age values should be between 0 and 150")
    
    def _check_target_values(self, df: pd.DataFrame) -> None:
        """Check target values are valid."""
        if "Survived" not in df.columns:
            return
            
        valid_values = {0, 1}
        invalid_values = set(df["Survived"].dropna().unique()) - valid_values
        if invalid_values:
            raise ValueError(f"Invalid Survived values: {invalid_values}. Must be 0 or 1")
    
    def _check_duplicate_ids(self, df: pd.DataFrame) -> None:
        """Check for duplicate PassengerIds."""
        duplicates = df["PassengerId"].duplicated()
        if duplicates.any():
            duplicate_ids = df.loc[duplicates, "PassengerId"].tolist()
            raise ValueError(f"Found duplicate PassengerId values: {duplicate_ids}")
    
    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """Check general data quality issues."""
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1)
        if empty_rows.any():
            empty_count = empty_rows.sum()
            self.logger.warning(f"Found {empty_count} completely empty rows")
        
        # Check missing value patterns
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.5]
        if not high_missing.empty:
            self.logger.warning(f"Columns with >50% missing values: {high_missing.to_dict()}")
        
        # Check for constant columns
        for col in df.select_dtypes(include=[np.number, object]).columns:
            if df[col].nunique(dropna=True) <= 1:
                self.logger.warning(f"Column '{col}' has only one unique value")
    
    def _check_feature_distributions(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Check that feature distributions are similar between train and test."""
        warnings = []
        
        # Check numeric features
        for col in self.numeric_columns:
            if col in train_df.columns and col in test_df.columns:
                train_mean = train_df[col].mean()
                test_mean = test_df[col].mean()
                
                if abs(train_mean - test_mean) / abs(train_mean) > 0.5:  # 50% difference
                    warnings.append(f"Large mean difference in '{col}': train={train_mean:.2f}, test={test_mean:.2f}")
        
        # Check categorical features
        for col in self.categorical_columns:
            if col in train_df.columns and col in test_df.columns:
                train_cats = set(train_df[col].dropna().unique())
                test_cats = set(test_df[col].dropna().unique())
                
                # Categories in test but not in train
                new_cats = test_cats - train_cats
                if new_cats:
                    warnings.append(f"New categories in test '{col}': {new_cats}")
                
                # Categories in train but not in test (less concerning)
                missing_cats = train_cats - test_cats
                if missing_cats:
                    self.logger.info(f"Categories missing from test '{col}': {missing_cats}")
        
        for warning in warnings:
            self.logger.warning(warning)


class DataLeakageDetector:
    """Detector for various types of data leakage."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
    
    def check_temporal_leakage(self, df: pd.DataFrame, time_col: str, 
                             train_end_time: Any = None) -> List[str]:
        """Check for temporal data leakage."""
        issues = []
        
        if time_col not in df.columns:
            return issues
        
        if train_end_time is not None:
            future_data = df[df[time_col] > train_end_time]
            if not future_data.empty:
                issues.append(f"Found {len(future_data)} rows with future timestamps")
        
        return issues
    
    def check_target_leakage(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Check for features that might be leaking the target."""
        issues = []
        
        if target_col not in df.columns:
            return issues
        
        # Check for perfect correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(target_col, errors='ignore')
        
        for col in numeric_cols:
            correlation = df[col].corr(df[target_col])
            if abs(correlation) > 0.99:
                issues.append(f"Column '{col}' has suspiciously high correlation with target: {correlation:.4f}")
        
        return issues
    
    def check_duplicate_leakage(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                              exclude_cols: Optional[List[str]] = None) -> List[str]:
        """Check for duplicated rows between train and test."""
        issues = []
        
        if exclude_cols is None:
            exclude_cols = ["PassengerId", "Survived"]
        
        # Get columns to compare (exclude ID and target)
        compare_cols = [col for col in train_df.columns 
                       if col in test_df.columns and col not in exclude_cols]
        
        if not compare_cols:
            return issues
        
        # Check for exact duplicates
        train_subset = train_df[compare_cols].drop_duplicates()
        test_subset = test_df[compare_cols].drop_duplicates()
        
        # Find intersection
        merged = train_subset.merge(test_subset, on=compare_cols, how='inner')
        
        if not merged.empty:
            issues.append(f"Found {len(merged)} duplicated feature combinations between train and test")
        
        return issues
