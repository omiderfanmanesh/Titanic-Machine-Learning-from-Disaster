"""Kaggle submission builder with validation and formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core.interfaces import ISubmissionBuilder
from core.utils import LoggerFactory


class TitanicSubmissionBuilder(ISubmissionBuilder):
    """Submission builder for Titanic competition."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        
        # Titanic competition requirements
        self.required_columns = ["PassengerId", "Survived"]
        self.id_column = "PassengerId"
        self.target_column = "Survived"
        
    def build_submission(self, predictions: pd.DataFrame,
                        config: Dict[str, Any]) -> pd.DataFrame:
        """Build Kaggle submission from predictions."""
        
        self.logger.info("Building Kaggle submission file")
        
        # Ensure we have the right columns
        if "PassengerId" not in predictions.columns:
            if predictions.index.name == "PassengerId" or "PassengerId" in str(predictions.index):
                predictions = predictions.reset_index()
            else:
                raise ValueError("PassengerId not found in predictions")
                
        if "prediction" in predictions.columns:
            pred_col = "prediction"
        elif "prediction_proba" in predictions.columns:
            # Convert probabilities to binary predictions
            threshold = config.get("threshold", 0.5)
            pred_col = "prediction_proba"
            predictions["prediction"] = (predictions[pred_col] >= threshold).astype(int)
            pred_col = "prediction"
        else:
            available_cols = list(predictions.columns)
            raise ValueError(f"No prediction column found. Available columns: {available_cols}")
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            self.id_column: predictions["PassengerId"],
            self.target_column: predictions[pred_col]
        })
        
        # Apply post-processing if configured
        submission = self._apply_postprocessing(submission, config)
        
        # Sort by PassengerId to ensure consistent ordering
        submission = submission.sort_values(self.id_column).reset_index(drop=True)
        
        self.logger.info(f"Created submission with {len(submission)} predictions")
        
        return submission
    
    def validate_submission(self, submission: pd.DataFrame) -> bool:
        """Validate submission format."""
        
        validation_errors = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(submission.columns)
        if missing_cols:
            validation_errors.append(f"Missing required columns: {missing_cols}")
        
        if validation_errors:
            for error in validation_errors:
                self.logger.error(error)
            return False
        
        # Check data types and values
        try:
            # PassengerId should be integer
            if not pd.api.types.is_integer_dtype(submission[self.id_column]):
                submission[self.id_column] = submission[self.id_column].astype(int)
            
            # Survived should be 0 or 1
            if not set(submission[self.target_column].unique()).issubset({0, 1}):
                invalid_values = set(submission[self.target_column].unique()) - {0, 1}
                validation_errors.append(f"Invalid target values: {invalid_values}")
            
        except Exception as e:
            validation_errors.append(f"Data type validation error: {e}")
        
        # Check for duplicates
        if submission[self.id_column].duplicated().any():
            validation_errors.append("Duplicate PassengerId values found")
        
        # Check for missing values
        if submission.isnull().any().any():
            null_cols = submission.columns[submission.isnull().any()].tolist()
            validation_errors.append(f"Missing values in columns: {null_cols}")
        
        if validation_errors:
            for error in validation_errors:
                self.logger.error(error)
            return False
        
        self.logger.info("Submission validation passed")
        return True
    
    def _apply_postprocessing(self, submission: pd.DataFrame,
                            config: Dict[str, Any]) -> pd.DataFrame:
        """Apply post-processing rules to submission."""
        
        postprocessing_config = config.get("postprocessing", {})
        
        if not postprocessing_config:
            return submission
        
        self.logger.info("Applying post-processing rules")
        
        # Apply custom business rules if specified
        rules = postprocessing_config.get("rules", [])
        
        for rule in rules:
            rule_type = rule.get("type")
            
            if rule_type == "clip_predictions":
                # Ensure predictions are within valid range
                min_val = rule.get("min", 0)
                max_val = rule.get("max", 1)
                submission[self.target_column] = submission[self.target_column].clip(min_val, max_val)
                
            elif rule_type == "round_predictions":
                # Round to nearest integer
                submission[self.target_column] = submission[self.target_column].round().astype(int)
                
            else:
                self.logger.warning(f"Unknown post-processing rule: {rule_type}")
        
        return submission
    
    def save_submission(self, submission: pd.DataFrame, output_path: Path,
                       config: Dict[str, Any]) -> Path:
        """Save submission to CSV file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata comment if requested
        if config.get("add_metadata", True):
            self._add_metadata_to_csv(submission, output_path, config)
        else:
            submission.to_csv(output_path, index=False)
        
        self.logger.info(f"Submission saved to {output_path}")
        
        return output_path
    
    def _add_metadata_to_csv(self, submission: pd.DataFrame, output_path: Path,
                           config: Dict[str, Any]) -> None:
        """Add metadata comments to CSV file."""
        
        import datetime
        
        metadata = [
            "# Titanic ML Pipeline Submission",
            f"# Generated: {datetime.datetime.now().isoformat()}",
            f"# Model: {config.get('model_name', 'Unknown')}",
            f"# CV Score: {config.get('cv_score', 'Unknown')}",
            f"# Threshold: {config.get('threshold', 0.5)}",
            f"# Samples: {len(submission)}",
            ""
        ]
        
        with open(output_path, "w") as f:
            for line in metadata:
                f.write(line + "\n")
            
        # Append the actual CSV data
        submission.to_csv(output_path, mode="a", index=False)
    
    def create_ensemble_submission(self, submissions: Dict[str, pd.DataFrame],
                                 weights: Optional[Dict[str, float]] = None,
                                 config: Dict[str, Any] = None) -> pd.DataFrame:
        """Create ensemble submission from multiple submissions."""
        
        if len(submissions) < 2:
            raise ValueError("Need at least 2 submissions for ensembling")
        
        config = config or {}
        
        self.logger.info(f"Creating ensemble from {len(submissions)} submissions")
        
        # Align all submissions by PassengerId
        base_submission = list(submissions.values())[0]
        passenger_ids = base_submission[self.id_column].sort_values()
        
        # Collect predictions
        all_predictions = []
        submission_names = []
        
        for name, sub in submissions.items():
            sub_aligned = sub.set_index(self.id_column).loc[passenger_ids]
            all_predictions.append(sub_aligned[self.target_column].values)
            submission_names.append(name)
        
        # Calculate ensemble weights
        if weights is None:
            weights = {name: 1.0 for name in submission_names}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = [weights.get(name, 0) / total_weight for name in submission_names]
        
        # Create weighted ensemble
        ensemble_predictions = sum(
            pred * weight for pred, weight in zip(all_predictions, normalized_weights)
        )
        
        # Convert to binary predictions
        threshold = config.get("threshold", 0.5)
        binary_predictions = (ensemble_predictions >= threshold).astype(int)
        
        ensemble_submission = pd.DataFrame({
            self.id_column: passenger_ids,
            self.target_column: binary_predictions
        })
        
        self.logger.info(f"Ensemble submission created with weights: "
                        f"{dict(zip(submission_names, normalized_weights))}")
        
        return ensemble_submission


class SubmissionAnalyzer:
    """Analyzer for comparing and validating submissions."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
    
    def compare_submissions(self, submissions: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compare multiple submissions."""
        
        if len(submissions) < 2:
            return {"error": "Need at least 2 submissions to compare"}
        
        analysis = {
            "submission_names": list(submissions.keys()),
            "submission_stats": {},
            "agreement_matrix": {},
            "prediction_differences": {}
        }
        
        # Basic statistics for each submission
        for name, sub in submissions.items():
            analysis["submission_stats"][name] = {
                "n_predictions": len(sub),
                "positive_predictions": sub["Survived"].sum(),
                "positive_rate": sub["Survived"].mean(),
                "unique_passenger_ids": sub["PassengerId"].nunique()
            }
        
        # Pairwise agreement analysis
        submission_names = list(submissions.keys())
        
        for i, name1 in enumerate(submission_names):
            for j, name2 in enumerate(submission_names[i+1:], i+1):
                sub1 = submissions[name1].set_index("PassengerId")["Survived"]
                sub2 = submissions[name2].set_index("PassengerId")["Survived"]
                
                # Find common passenger IDs
                common_ids = sub1.index.intersection(sub2.index)
                
                if len(common_ids) > 0:
                    agreement = (sub1.loc[common_ids] == sub2.loc[common_ids]).mean()
                    
                    pair_key = f"{name1}_vs_{name2}"
                    analysis["agreement_matrix"][pair_key] = {
                        "agreement_rate": float(agreement),
                        "common_samples": len(common_ids),
                        "disagreements": len(common_ids) - (sub1.loc[common_ids] == sub2.loc[common_ids]).sum()
                    }
        
        return analysis
    
    def validate_against_sample_submission(self, submission: pd.DataFrame,
                                         sample_submission_path: Path) -> Dict[str, Any]:
        """Validate submission against Kaggle sample submission."""
        
        try:
            sample_sub = pd.read_csv(sample_submission_path)
        except Exception as e:
            return {"error": f"Could not load sample submission: {e}"}
        
        validation = {
            "is_valid": True,
            "issues": []
        }
        
        # Check structure
        if not set(submission.columns) == set(sample_sub.columns):
            validation["is_valid"] = False
            validation["issues"].append("Column mismatch with sample submission")
        
        # Check PassengerId alignment
        if not set(submission["PassengerId"]) == set(sample_sub["PassengerId"]):
            missing_ids = set(sample_sub["PassengerId"]) - set(submission["PassengerId"])
            extra_ids = set(submission["PassengerId"]) - set(sample_sub["PassengerId"])
            
            if missing_ids:
                validation["is_valid"] = False
                validation["issues"].append(f"Missing PassengerIds: {len(missing_ids)} IDs")
            
            if extra_ids:
                validation["issues"].append(f"Extra PassengerIds: {len(extra_ids)} IDs")
        
        # Check data types
        if submission["Survived"].dtype != sample_sub["Survived"].dtype:
            validation["issues"].append("Survived column data type mismatch")
        
        return validation
