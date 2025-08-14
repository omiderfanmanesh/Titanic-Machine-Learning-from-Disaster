"""Hyperparameter tuning using Optuna."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

from core.interfaces import ITrainer
from core.utils import LoggerFactory, PathManager, Timer
from cv.folds import FoldSplitterFactory
from modeling.model_registry import ModelRegistry


class OptunaTuner:
    """Optuna-based hyperparameter tuning with cross-validation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)
        self.path_manager = PathManager()

        # Optuna study configuration
        self.study_config = config.get("study", {})
        self.n_trials = self.study_config.get("n_trials", 50)
        self.pruning = self.study_config.get("pruning", True)
        self.cv_folds = config.get("cv", {}).get("n_folds", 5)

        # Tracking
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.study: Optional[optuna.Study] = None

    def tune(self, model_name: str, X: pd.DataFrame, y: pd.Series,
             cv_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hyperparameter tuning with cross-validation."""
        self.logger.info(f"Starting hyperparameter tuning for {model_name}")

        # Create run directory
        run_dir = self.path_manager.create_run_directory()

        # Create study
        study_name = f"{model_name}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        sampler = TPESampler(seed=cv_config.get("random_state", 42))
        pruner = MedianPruner() if self.pruning else optuna.pruners.NopPruner()

        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )

        # Suppress optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create objective function
        objective = self._create_objective(model_name, X, y, cv_config, run_dir)

        # Run optimization
        with Timer(self.logger, f"hyperparameter tuning ({self.n_trials} trials)"):
            self.study.optimize(objective, n_trials=self.n_trials)

        # Extract best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        # Save tuning results
        results = self._save_tuning_results(run_dir, model_name)

        self.logger.info(f"Tuning completed. Best score: {self.best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")

        return results

    def _create_objective(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                         cv_config: Dict[str, Any], run_dir: Path):
        """Create Optuna objective function."""
        from .search_spaces import SearchSpaceFactory

        def objective(trial: optuna.Trial) -> float:
            # Get search space for model
            search_space = SearchSpaceFactory.get_search_space(model_name)

            # Sample hyperparameters
            if model_name == "logistic":
                # Use special sampling for logistic regression to avoid incompatible combinations
                params = SearchSpaceFactory.sample_logistic_params(trial)
            else:
                # Standard sampling for other models
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config["type"] == "float":
                        if param_config.get("log", False):
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config["low"],
                                param_config["high"],
                                log=True
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config["low"],
                                param_config["high"]
                            )
                    elif param_config["type"] == "int":
                        if param_config.get("log", False):
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config["low"],
                                param_config["high"],
                                log=True
                            )
                        else:
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config["low"],
                                param_config["high"]
                            )
                    elif param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config["choices"]
                        )

            # Create model with sampled parameters
            model_registry = ModelRegistry()
            # Fixed: Pass params as dictionary argument, not as kwargs
            model_wrapper = model_registry.create_model(model_name, params=params)
            model = model_wrapper.build({"model_params": params})

            # Perform cross-validation
            cv_scores = self._cross_validate_trial(model, X, y, cv_config, trial)

            # Return mean CV score
            mean_score = np.mean(cv_scores)

            # Report intermediate value for pruning
            trial.report(mean_score, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return mean_score

        return objective

    def _cross_validate_trial(self, model: BaseEstimator, X: pd.DataFrame,
                             y: pd.Series, cv_config: Dict[str, Any],
                             trial: optuna.Trial) -> List[float]:
        """Perform cross-validation for a single trial."""
        # Create fold splitter
        splitter = FoldSplitterFactory.create_splitter(
            cv_config.get("strategy", "stratified"),
            n_splits=cv_config.get("n_folds", 5),
            shuffle=cv_config.get("shuffle", True),
            random_state=cv_config.get("random_state", 42)
        )

        # Generate splits
        splits = splitter.split(X, y)

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone and fit model
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            # Make predictions
            val_pred = self._predict_proba(fold_model, X_val)

            # Calculate score
            fold_score = roc_auc_score(y_val, val_pred)
            fold_scores.append(fold_score)

            # Report intermediate value for pruning (optional early stopping per fold)
            if self.pruning and fold_idx > 0:
                intermediate_score = np.mean(fold_scores)
                trial.report(intermediate_score, step=fold_idx)

                if trial.should_prune():
                    raise optuna.TrialPruned()

        return fold_scores

    def _predict_proba(self, model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities, handling different model types."""
        try:
            # Try predict_proba first (for classifiers)
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
        except AttributeError:
            # Fall back to decision_function (for SVM, etc.)
            try:
                return model.decision_function(X)
            except AttributeError:
                # Last resort - use predict (for regression models used as classifiers)
                return model.predict(X)

    def _save_tuning_results(self, run_dir: Path, model_name: str) -> Dict[str, Any]:
        """Save tuning results and artifacts."""
        # Save study object
        study_path = run_dir / "study.pkl"
        joblib.dump(self.study, study_path)

        # Save best parameters
        best_params_path = run_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(self.best_params, f, indent=2)

        # Save optimization history
        trials_df = self.study.trials_dataframe()
        trials_path = run_dir / "trials.csv"
        trials_df.to_csv(trials_path, index=False)

        # Save summary results
        results = {
            "model_name": model_name,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "n_trials": len(self.study.trials),
            "study_path": str(study_path),
            "best_params_path": str(best_params_path),
            "trials_path": str(trials_path),
            "run_dir": str(run_dir)
        }

        results_path = run_dir / "tuning_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Tuning results saved to {run_dir}")

        return results


class HyperparameterTuner:
    """High-level interface for hyperparameter tuning."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tuner = OptunaTuner(config)
        self.logger = LoggerFactory.get_logger(__name__)

    def tune_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                   cv_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Tune hyperparameters for a specific model."""
        if cv_config is None:
            cv_config = self.config.get("cv", {})

        return self.tuner.tune(model_name, X, y, cv_config)

    def get_best_model(self, model_name: str) -> BaseEstimator:
        """Get model with best hyperparameters."""
        if self.tuner.best_params is None:
            raise ValueError("No tuning results available. Run tune_model first.")

        model_registry = ModelRegistry()
        return model_registry.create_model(model_name, **self.tuner.best_params)
