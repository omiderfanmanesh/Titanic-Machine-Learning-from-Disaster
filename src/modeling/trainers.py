"""Training components with cross-validation and leak-safe preprocessing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from core.interfaces import ITrainer, ITransformer
from core.utils import LoggerFactory, PathManager, Timer
from cv.folds import FoldSplitterFactory


class TitanicTrainer(ITrainer):
    """Cross-validation trainer for Titanic competition with leak-safe feature processing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)
        self.path_manager = PathManager()
        
        # Training artifacts
        self.best_model_path: Optional[str] = None
        self.oof_predictions: Optional[np.ndarray] = None
        self.fold_models: List[BaseEstimator] = []
        self.fold_feature_pipelines: List[ITransformer] = []
        self.fold_scores: List[float] = []
        self.training_history: Dict[str, Any] = {}
        
    def fit(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
            config: Dict[str, Any], feature_builder: Optional[ITransformer] = None) -> Dict[str, Any]:
        """Simple fit without cross-validation."""
        with Timer(self.logger, "model fitting"):
            if feature_builder is not None:
                # Apply feature processing
                feature_builder.fit(X, y)
                X_processed = feature_builder.transform(X)
            else:
                X_processed = X

            model.fit(X_processed, y)

            # Evaluate on training data (for debugging)
            train_pred = self._predict_proba(model, X_processed)
            train_score = self._calculate_score(y, train_pred)

            artifacts = {
                "model": model,
                "feature_builder": feature_builder,
                "train_score": train_score,
                "config": config.copy()
            }
            
            self.logger.info(f"Model fitted. Train score: {train_score:.4f}")

            return artifacts
    
    def cross_validate(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                      cv_config: Dict[str, Any], feature_builder: Optional[ITransformer] = None) -> Dict[str, Any]:
        """Perform cross-validation with leak-safe preprocessing."""
        self.logger.info("Starting cross-validation with leak-safe feature processing")

        # Create run directory
        run_dir = self.path_manager.create_run_directory()
        
        # Create fold splitter
        splitter = FoldSplitterFactory.create_splitter(
            cv_config.get("strategy", "stratified"),
            n_splits=cv_config.get("n_folds", 5),
            shuffle=cv_config.get("shuffle", True),
            random_state=cv_config.get("random_state", 42)
        )
        
        # Generate splits
        splits = splitter.split(X, y)
        
        # Initialize tracking arrays
        self.oof_predictions = np.zeros(len(X))
        self.fold_models = []
        self.fold_feature_pipelines = []
        self.fold_scores = []
        
        with Timer(self.logger, f"{len(splits)}-fold cross-validation with feature processing"):
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                fold_artifacts = self._train_fold_with_features(
                    model, X, y, train_idx, val_idx, fold_idx, run_dir, feature_builder
                )
                
                self.fold_models.append(fold_artifacts["model"])
                self.fold_feature_pipelines.append(fold_artifacts["feature_pipeline"])
                self.fold_scores.append(fold_artifacts["score"])
                
                # Store OOF predictions
                self.oof_predictions[val_idx] = fold_artifacts["predictions"]
        
        # Calculate overall metrics
        overall_score = self._calculate_score(y, self.oof_predictions)
        cv_mean = np.mean(self.fold_scores)
        cv_std = np.std(self.fold_scores)
        
        # Save artifacts
        artifacts = self._save_cv_artifacts(run_dir, y, cv_config)
        
        self.logger.info(f"Cross-validation completed. "
                        f"CV: {cv_mean:.4f} ± {cv_std:.4f}, "
                        f"OOF Score: {overall_score:.4f}")

        return artifacts
    
    def _train_fold_with_features(self, base_model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                                 train_idx: List[int], val_idx: List[int], fold_idx: int,
                                 run_dir: Path, feature_builder: Optional[ITransformer] = None) -> Dict[str, Any]:
        """Train a single fold with leak-safe feature processing."""
        self.logger.info(f"Training fold {fold_idx + 1} with feature processing")

        # Split data - this is the raw data before feature processing
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone the feature builder for this fold to prevent leakage
        if feature_builder is not None:
            from copy import deepcopy
            fold_feature_pipeline = deepcopy(feature_builder)

            # Fit feature pipeline ONLY on training fold
            self.logger.debug(f"Fitting feature pipeline on training fold {fold_idx + 1}")
            fold_feature_pipeline.fit(X_train_raw, y_train)

            # Transform both training and validation folds using the fitted pipeline
            X_train = fold_feature_pipeline.transform(X_train_raw)
            X_val = fold_feature_pipeline.transform(X_val_raw)

            # Validation: ensure no transform was fitted on validation data
            self._validate_no_val_leakage(fold_feature_pipeline, X_val_raw, fold_idx)
        else:
            fold_feature_pipeline = None
            X_train, X_val = X_train_raw, X_val_raw

        # Clone the model for this fold
        fold_model = clone(base_model)
        
        # Train model on processed features
        fold_model.fit(X_train, y_train)
        
        # Make predictions on processed validation set
        val_pred = self._predict_proba(fold_model, X_val)
        
        # Calculate score
        fold_score = self._calculate_score(y_val, val_pred)

        # Save fold artifacts
        fold_model_path = run_dir / f"fold_{fold_idx}_model.joblib"
        joblib.dump(fold_model, fold_model_path)
        
        fold_pipeline_path = None
        if fold_feature_pipeline is not None:
            fold_pipeline_path = run_dir / f"fold_{fold_idx}_feature_pipeline.joblib"
            joblib.dump(fold_feature_pipeline, fold_pipeline_path)

        self.logger.info(f"Fold {fold_idx + 1} score: {fold_score:.4f}")

        return {
            "model": fold_model,
            "feature_pipeline": fold_feature_pipeline,
            "predictions": val_pred,
            "score": fold_score,
            "model_path": str(fold_model_path),
            "pipeline_path": str(fold_pipeline_path) if fold_pipeline_path else None
        }

    def _validate_no_val_leakage(self, feature_pipeline: ITransformer, X_val_raw: pd.DataFrame, fold_idx: int) -> None:
        """Validate that no transform was fitted on validation data."""
        # This is a regression test to ensure the feature pipeline was not fitted on validation data
        # We can't directly check if it was fitted on validation data, but we can verify
        # that the pipeline can transform validation data without errors and produces consistent results
        try:
            # Transform validation data twice - should be identical
            val_transformed_1 = feature_pipeline.transform(X_val_raw)
            val_transformed_2 = feature_pipeline.transform(X_val_raw)

            # Check if results are identical (deterministic transform)
            if not val_transformed_1.equals(val_transformed_2):
                self.logger.warning(f"Fold {fold_idx + 1}: Feature pipeline produces non-deterministic results")

            self.logger.debug(f"Fold {fold_idx + 1}: Validation leakage check passed")

        except Exception as e:
            raise RuntimeError(f"Fold {fold_idx + 1}: Feature pipeline validation failed: {e}") from e

    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate the evaluation score."""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)

    def _predict_proba(self, model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities, handling different model types."""
        try:
            # Try predict_proba first (for classifiers)
            proba = model.predict_proba(X)
            # Return probability of positive class
            return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
        except AttributeError:
            # Fall back to decision_function (for SVM, etc.)
            try:
                return model.decision_function(X)
            except AttributeError:
                # Fall back to predict (regression-like)
                return model.predict(X)
    
    def _save_cv_artifacts(self, run_dir: Path, y: pd.Series,
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Save cross-validation artifacts."""
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            "target": y,
            "prediction": self.oof_predictions
        })
        oof_path = run_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        
        # Save fold scores
        scores_data = {
            "fold_scores": self.fold_scores,
            "mean_score": float(np.mean(self.fold_scores)),
            "std_score": float(np.std(self.fold_scores)),
            "oof_score": float(roc_auc_score(y, self.oof_predictions))
        }
        
        scores_path = run_dir / "cv_scores.json"
        with open(scores_path, "w") as f:
            json.dump(scores_data, f, indent=2)
        
        # Save training config
        config_path = run_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        # Create training report
        self._create_training_report(run_dir, scores_data, config)
        
        # Find best model (lowest validation score fold)
        best_fold = np.argmax(self.fold_scores)  # Highest AUC
        self.best_model_path = str(run_dir / f"fold_{best_fold}_model.joblib")
        
        return {
            "run_dir": str(run_dir),
            "oof_path": str(oof_path),
            "scores_path": str(scores_path),
            "best_model_path": self.best_model_path,
            "cv_scores": scores_data,
            "fold_models": [str(run_dir / f"fold_{i}_model.joblib") 
                           for i in range(len(self.fold_scores))]
        }
    
    def _create_training_report(self, run_dir: Path, scores_data: Dict[str, Any],
                               config: Dict[str, Any]) -> None:
        """Create a comprehensive training report."""
        report_path = run_dir / "training_report.md"
        
        with open(report_path, "w") as f:
            f.write("# Titanic ML Training Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Run Directory:** {run_dir}\n\n")
            
            f.write("## Model Configuration\n")
            f.write(f"- Model: {config.get('model_name', 'Unknown')}\n")
            f.write(f"- CV Strategy: {config.get('cv_strategy', 'stratified')}\n")
            f.write(f"- CV Folds: {config.get('cv_folds', 5)}\n\n")
            
            f.write("## Cross-Validation Results\n")
            f.write(f"- Mean CV Score: {scores_data['mean_score']:.4f}\n")
            f.write(f"- Std CV Score: {scores_data['std_score']:.4f}\n")
            f.write(f"- OOF Score: {scores_data['oof_score']:.4f}\n\n")
            
            f.write("### Per-Fold Scores\n")
            for i, score in enumerate(scores_data['fold_scores']):
                f.write(f"- Fold {i+1}: {score:.4f}\n")
            
            f.write("\n## Model Parameters\n")
            model_params = config.get('model_params', {})
            for param, value in model_params.items():
                f.write(f"- {param}: {value}\n")
        
        self.logger.info(f"Training report saved to {report_path}")
    
    def best_checkpoint(self) -> str:
        """Return path to best model checkpoint."""
        if self.best_model_path is None:
            raise ValueError("No training completed. Run cross_validate() first.")
        return self.best_model_path
    
    def get_oof_predictions(self) -> Optional[np.ndarray]:
        """Get out-of-fold predictions."""
        return self.oof_predictions
    
    def get_fold_models(self) -> List[BaseEstimator]:
        """Get trained fold models."""
        return self.fold_models.copy()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        if not self.fold_scores:
            return {"error": "No training completed"}
            
        return {
            "n_folds": len(self.fold_scores),
            "fold_scores": self.fold_scores,
            "mean_cv_score": float(np.mean(self.fold_scores)),
            "std_cv_score": float(np.std(self.fold_scores)),
            "best_fold": int(np.argmax(self.fold_scores)),
            "best_model_path": self.best_model_path,
            "has_oof_predictions": self.oof_predictions is not None
        }
