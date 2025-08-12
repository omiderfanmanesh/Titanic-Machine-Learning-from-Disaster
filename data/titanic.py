"""Titanic dataset implementation.

This module provides the Titanic-specific dataset class with all configurations
and methods needed to load and preprocess the Titanic dataset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .based.based_dataset import BasedDataset
from configs import DATASET_DEFAULTS


class TitanicDataset(BasedDataset):
    """Titanic dataset implementation with specific preprocessing logic.
    
    This class handles loading and preprocessing of the Titanic dataset,
    including feature engineering specific to the problem domain.
    """
    
    def __init__(
        self,
        data_dir: str | Path = "data",
        **kwargs: Any,
    ) -> None:
        """Initialize the Titanic dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="titanic",
            data_dir=data_dir,
            target_column="Survived",
            id_column="PassengerId",
            **kwargs,
        )
        
        # Titanic-specific configurations
        self.expected_features = [
            "PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", 
            "Parch", "Ticket", "Fare", "Cabin", "Embarked"
        ]
        
        # Feature engineering flags
        self.feature_config = {
            "add_family_size": kwargs.get("add_family_size", True),
            "add_is_alone": kwargs.get("add_is_alone", True),
            "add_title": kwargs.get("add_title", True),
            "add_deck": kwargs.get("add_deck", True),
            "add_ticket_group_size": kwargs.get("add_ticket_group_size", False),
            "log_fare": kwargs.get("log_fare", True),
            "bin_age": kwargs.get("bin_age", False),
            "rare_title_threshold": kwargs.get("rare_title_threshold", 10),
        }
    
    def load_data(self) -> None:
        """Load the Titanic dataset from CSV files."""
        self.logger.info("Loading Titanic dataset...")
        
        # Load training data
        train_path = self.data_dir / "train.csv"
        if train_path.exists():
            self.df_train = self.load_file(train_path)
            self.logger.info(f"Loaded training data: {self.df_train.shape}")
        else:
            raise FileNotFoundError(f"Training file not found: {train_path}")
        
        # Load test data
        test_path = self.data_dir / "test.csv"
        if test_path.exists():
            self.df_test = self.load_file(test_path)
            self.logger.info(f"Loaded test data: {self.df_test.shape}")
        else:
            self.logger.warning(f"Test file not found: {test_path}")
        
        # Validate the loaded data
        self._validate_data()
        
        # Update metadata
        self.update_metadata(loaded=True)
        self.logger.info("Titanic dataset loaded successfully")
    
    def _validate_data(self) -> None:
        """Validate the loaded Titanic data."""
        if self.df_train is None:
            raise ValueError("Training data not loaded")
        
        # Check expected columns in training data
        missing_cols = set(self.expected_features) - set(self.df_train.columns)
        if self.target_column not in self.df_train.columns:
            missing_cols.add(self.target_column)
        
        if missing_cols:
            self.logger.warning(f"Missing expected columns in training data: {missing_cols}")
        
        # Check for duplicate PassengerIds
        if self.id_column in self.df_train.columns:
            duplicates = self.df_train[self.id_column].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate PassengerIds in training data")
        
        # Validate test data if available
        if self.df_test is not None:
            test_missing = set(self.expected_features) - {self.target_column} - set(self.df_test.columns)
            if test_missing:
                self.logger.warning(f"Missing expected columns in test data: {test_missing}")
    
    def get_feature_types(self) -> Dict[str, str]:
        """Get the types of features in the Titanic dataset.
        
        Returns:
            Dictionary mapping column names to feature types
        """
        feature_types = {
            "PassengerId": "id",
            "Survived": "target",
            "Pclass": "categorical",
            "Name": "text",
            "Sex": "categorical",
            "Age": "numerical",
            "SibSp": "numerical",
            "Parch": "numerical",
            "Ticket": "categorical",
            "Fare": "numerical",
            "Cabin": "categorical",
            "Embarked": "categorical",
        }
        
        # Add engineered features
        if self.feature_config.get("add_family_size"):
            feature_types["FamilySize"] = "numerical"
        if self.feature_config.get("add_is_alone"):
            feature_types["IsAlone"] = "categorical"
        if self.feature_config.get("add_title"):
            feature_types["Title"] = "categorical"
        if self.feature_config.get("add_deck"):
            feature_types["Deck"] = "categorical"
        if self.feature_config.get("add_ticket_group_size"):
            feature_types["TicketGroupSize"] = "numerical"
        if self.feature_config.get("log_fare"):
            feature_types["LogFare"] = "numerical"
        if self.feature_config.get("bin_age"):
            feature_types["AgeBin"] = "categorical"
        
        return feature_types
    
    def engineer_features(self) -> None:
        """Perform Titanic-specific feature engineering."""
        if not self.is_loaded():
            raise ValueError("Dataset must be loaded before feature engineering")
        
        self.logger.info("Starting feature engineering...")
        
        # Process all available datasets
        datasets = [("train", self.df_train)]
        if self.df_test is not None:
            datasets.append(("test", self.df_test))
        if self.df_validation is not None:
            datasets.append(("validation", self.df_validation))
        
        for name, df in datasets:
            if df is None:
                continue
            
            self.logger.info(f"Engineering features for {name} set")
            self._engineer_features_single(df)
        
        self.update_metadata(preprocessed=True)
        self.logger.info("Feature engineering completed")
    
    def _engineer_features_single(self, df: pd.DataFrame) -> None:
        """Engineer features for a single dataset.
        
        Args:
            df: DataFrame to process
        """
        # Family size feature
        if self.feature_config.get("add_family_size"):
            if {"SibSp", "Parch"}.issubset(df.columns):
                df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
                self.logger.debug("Added FamilySize feature")
        
        # Is alone feature
        if self.feature_config.get("add_is_alone"):
            if "FamilySize" in df.columns:
                df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
            elif {"SibSp", "Parch"}.issubset(df.columns):
                df["IsAlone"] = ((df["SibSp"] + df["Parch"]) == 0).astype(int)
            self.logger.debug("Added IsAlone feature")
        
        # Title extraction
        if self.feature_config.get("add_title") and "Name" in df.columns:
            df["Title"] = self._extract_titles(df["Name"])
            self.logger.debug("Added Title feature")
        
        # Deck extraction
        if self.feature_config.get("add_deck") and "Cabin" in df.columns:
            df["Deck"] = self._extract_deck(df["Cabin"])
            self.logger.debug("Added Deck feature")
        
        # Ticket group size
        if self.feature_config.get("add_ticket_group_size") and "Ticket" in df.columns:
            df["TicketGroupSize"] = self._get_ticket_group_sizes(df["Ticket"])
            self.logger.debug("Added TicketGroupSize feature")
        
        # Log fare
        if self.feature_config.get("log_fare") and "Fare" in df.columns:
            df["LogFare"] = np.log1p(df["Fare"].fillna(0))
            self.logger.debug("Added LogFare feature")
        
        # Age binning
        if self.feature_config.get("bin_age") and "Age" in df.columns:
            df["AgeBin"] = self._bin_age(df["Age"])
            self.logger.debug("Added AgeBin feature")
    
    def _extract_titles(self, names: pd.Series) -> pd.Series:
        """Extract titles from passenger names.
        
        Args:
            names: Series of passenger names
            
        Returns:
            Series of extracted titles
        """
        def extract_title(name: str) -> str:
            if pd.isna(name):
                return "Unknown"
            
            match = re.search(r',\s*([^\.]+)\.', str(name))
            if match:
                title = match.group(1).strip()
                
                # Normalize common titles
                if title in ["Mlle", "Ms"]:
                    return "Miss"
                elif title == "Mme":
                    return "Mrs"
                elif title in ["Capt", "Col", "Major", "Dr", "Rev"]:
                    return "Officer"
                elif title in ["Don", "Sir", "the Countess", "Dona", "Lady", "Jonkheer"]:
                    return "Royalty"
                else:
                    return title
            
            return "Unknown"
        
        titles = names.apply(extract_title)
        
        # Group rare titles
        threshold = self.feature_config.get("rare_title_threshold", 10)
        if hasattr(self, "df_train") and self.df_train is not None:
            # Use training data to determine rare titles
            train_titles = self.df_train["Name"].apply(extract_title) if "Name" in self.df_train.columns else titles
            title_counts = train_titles.value_counts()
            rare_titles = title_counts[title_counts < threshold].index
            titles = titles.apply(lambda x: "Rare" if x in rare_titles else x)
        
        return titles
    
    def _extract_deck(self, cabins: pd.Series) -> pd.Series:
        """Extract deck information from cabin numbers.
        
        Args:
            cabins: Series of cabin numbers
            
        Returns:
            Series of deck letters
        """
        def get_deck(cabin: str) -> str:
            if pd.isna(cabin) or cabin == "":
                return "Unknown"
            
            deck = str(cabin)[0].upper()
            if deck in "ABCDEFGT":
                return deck
            else:
                return "Unknown"
        
        return cabins.apply(get_deck)
    
    def _get_ticket_group_sizes(self, tickets: pd.Series) -> pd.Series:
        """Get the group sizes for tickets.
        
        Args:
            tickets: Series of ticket numbers
            
        Returns:
            Series of group sizes
        """
        # Use training data ticket counts if available
        if hasattr(self, "df_train") and self.df_train is not None and "Ticket" in self.df_train.columns:
            ticket_counts = self.df_train["Ticket"].value_counts().to_dict()
        else:
            ticket_counts = tickets.value_counts().to_dict()
        
        return tickets.map(ticket_counts).fillna(1).astype(int)
    
    def _bin_age(self, ages: pd.Series) -> pd.Series:
        """Bin ages into categories.
        
        Args:
            ages: Series of ages
            
        Returns:
            Series of age bins
        """
        # Define age bins
        bins = [0, 5, 12, 18, 30, 45, 60, 80]
        labels = ["Baby", "Child", "Teen", "Young Adult", "Adult", "Middle Age", "Senior"]
        
        age_bins = pd.cut(ages, bins=bins, labels=labels, include_lowest=True)
        return age_bins.astype(str).fillna("Unknown")
    
    def get_survival_analysis(self) -> Dict[str, Any]:
        """Perform survival analysis on the training data.
        
        Returns:
            Dictionary with survival analysis results
        """
        if not self.is_loaded() or self.df_train is None:
            raise ValueError("Training data must be loaded for survival analysis")
        
        if self.target_column not in self.df_train.columns:
            raise ValueError("Target column not found in training data")
        
        analysis = {}
        
        # Overall survival rate
        survival_rate = self.df_train[self.target_column].mean()
        analysis["overall_survival_rate"] = float(survival_rate)
        
        # Survival by categorical features
        categorical_features = ["Pclass", "Sex", "Embarked"]
        if self.feature_config.get("add_title"):
            categorical_features.append("Title")
        if self.feature_config.get("add_deck"):
            categorical_features.append("Deck")
        if self.feature_config.get("add_is_alone"):
            categorical_features.append("IsAlone")
        
        survival_by_category = {}
        for feature in categorical_features:
            if feature in self.df_train.columns:
                survival_rates = self.df_train.groupby(feature)[self.target_column].agg([
                    "mean", "count"
                ]).round(3)
                survival_by_category[feature] = survival_rates.to_dict("index")
        
        analysis["survival_by_category"] = survival_by_category
        
        # Survival by numerical features (binned)
        numerical_features = ["Age", "Fare"]
        if self.feature_config.get("add_family_size"):
            numerical_features.append("FamilySize")
        
        survival_by_numerical = {}
        for feature in numerical_features:
            if feature in self.df_train.columns:
                # Create quartile bins
                quartiles = self.df_train[feature].quantile([0.25, 0.5, 0.75]).values
                bins = [-np.inf] + quartiles.tolist() + [np.inf]
                labels = ["Q1", "Q2", "Q3", "Q4"]
                
                binned = pd.cut(self.df_train[feature], bins=bins, labels=labels)
                survival_rates = self.df_train.groupby(binned)[self.target_column].agg([
                    "mean", "count"
                ]).round(3)
                survival_by_numerical[feature] = survival_rates.to_dict("index")
        
        analysis["survival_by_numerical"] = survival_by_numerical
        
        return analysis
    
    def get_missing_value_analysis(self) -> Dict[str, Any]:
        """Analyze missing values in the dataset.
        
        Returns:
            Dictionary with missing value analysis
        """
        if not self.is_loaded():
            raise ValueError("Dataset must be loaded for missing value analysis")
        
        analysis = {}
        
        datasets = [("train", self.df_train)]
        if self.df_test is not None:
            datasets.append(("test", self.df_test))
        if self.df_validation is not None:
            datasets.append(("validation", self.df_validation))
        
        for name, df in datasets:
            if df is None:
                continue
            
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df) * 100).round(2)
            
            missing_info = {}
            for col in df.columns:
                if missing_counts[col] > 0:
                    missing_info[col] = {
                        "count": int(missing_counts[col]),
                        "percentage": float(missing_percentages[col])
                    }
            
            analysis[name] = missing_info
        
        return analysis
    
    def get_feature_correlations(self) -> Dict[str, float]:
        """Get correlations between features and target variable.
        
        Returns:
            Dictionary mapping feature names to correlation coefficients
        """
        if not self.is_loaded() or self.df_train is None:
            raise ValueError("Training data must be loaded for correlation analysis")
        
        if self.target_column not in self.df_train.columns:
            raise ValueError("Target column not found in training data")
        
        # Get numerical features
        numerical_cols = self.df_train.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)
        
        correlations = {}
        for col in numerical_cols:
            if col in self.df_train.columns:
                corr = self.df_train[col].corr(self.df_train[self.target_column])
                if not pd.isna(corr):
                    correlations[col] = float(corr)
        
        return correlations
    
    def preprocess_for_training(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess the data for machine learning training.
        
        Returns:
            Tuple of (training features, test features)
        """
        if not self.is_loaded():
            raise ValueError("Dataset must be loaded before preprocessing")
        
        # Engineer features if not done already
        if not self.is_preprocessed():
            self.engineer_features()
        
        # Handle missing values
        self.handle_missing_values(strategy="auto")
        
        # Get features for training
        X_train, _ = self.get_X_y("train", include_id=False)
        
        X_test = None
        if self.df_test is not None:
            X_test, _ = self.get_X_y("test", include_id=False)
        
        return X_train, X_test
    
    def create_submission_template(self, predictions: np.ndarray) -> pd.DataFrame:
        """Create a submission DataFrame from predictions.
        
        Args:
            predictions: Array of predictions
            
        Returns:
            Submission DataFrame
        """
        if self.df_test is None:
            raise ValueError("Test data not available for submission")
        
        if self.id_column not in self.df_test.columns:
            raise ValueError("ID column not found in test data")
        
        if len(predictions) != len(self.df_test):
            raise ValueError(
                f"Predictions length ({len(predictions)}) doesn't match "
                f"test data length ({len(self.df_test)})"
            )
        
        submission = pd.DataFrame({
            self.id_column: self.df_test[self.id_column],
            self.target_column: predictions
        })
        
        return submission
