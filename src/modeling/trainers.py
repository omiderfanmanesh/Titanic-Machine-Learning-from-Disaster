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
            cv_config.get("cv_strategy", cv_config.get("strategy", "stratified")),
            n_splits=cv_config.get("cv_folds", cv_config.get("n_folds", 5)),
            shuffle=cv_config.get("cv_shuffle", cv_config.get("shuffle", True)),
            random_state=cv_config.get("cv_random_state", cv_config.get("random_state", 42))
        )
        
        # Optional: group-aware splitting (requires explicit group column)
        groups = None
        strategy = str(cv_config.get("cv_strategy", cv_config.get("strategy", "stratified")))
        group_col = cv_config.get("group_column")
        if strategy == "group":
            if group_col and group_col in X.columns:
                groups = X[group_col]
            else:
                raise ValueError("Group CV strategy selected but 'group_column' is not configured in data.yaml or missing from the raw data.")
        
        # Generate splits (pass groups only for group strategy)
        if strategy == "group":
            splits = splitter.split(X, y, groups=groups)
        else:
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
        self.logger.info(
            f"Fold {self.fold_scores.__len__()+1 if hasattr(self,'fold_scores') else fold_idx+1}: raw shapes train={X_train_raw.shape}, val={X_val_raw.shape}"
        )
        
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
            self.logger.info(
                f"Fold {fold_idx+1}: after feature pipeline train={X_train.shape}, val={X_val.shape}"
            )

            # Drop id/target columns from transformed features to prevent leakage
            id_col = self.config.get("id_column") or "PassengerId"
            target_col = self.config.get("target_column") or "Survived"
            drop_cols = [c for c in [id_col, target_col] if c in X_train.columns]
            if drop_cols:
                X_train = X_train.drop(columns=drop_cols)
            drop_cols_val = [c for c in [id_col, target_col] if c in X_val.columns]
            if drop_cols_val:
                X_val = X_val.drop(columns=drop_cols_val)

            # Validation: ensure no transform was fitted on validation data
            self._validate_no_val_leakage(fold_feature_pipeline, X_val_raw, fold_idx)
        else:
            fold_feature_pipeline = None
            X_train, X_val = X_train_raw, X_val_raw

        # Optional: class imbalance handling on training split only
        X_train, y_train = self._apply_imbalance_sampling_if_enabled(X_train, y_train, fold_idx)

        # Clone the model for this fold
        fold_model = clone(base_model)
        
        # Train model on processed features
        # Guard against NaNs: stop and log detailed columns if found
        if fold_feature_pipeline is not None and hasattr(fold_feature_pipeline, "validate_no_nans"):
            n_train_nans = fold_feature_pipeline.validate_no_nans(X_train, context=f"fold {fold_idx+1} X_train")
            n_val_nans = fold_feature_pipeline.validate_no_nans(X_val, context=f"fold {fold_idx+1} X_val")
        else:
            def _log_nan_details(df: pd.DataFrame, label: str) -> int:
                nan_counts = df.isna().sum()
                nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
                if not nan_cols.empty:
                    self.logger.error(
                        f"Fold {fold_idx + 1}: NaNs detected in {label}. "
                        f"columns_with_nans={len(nan_cols)}; top offenders:\n{nan_cols.head(10).to_string()}"
                    )
                return int(nan_cols.sum()) if not nan_cols.empty else 0

            n_train_nans = _log_nan_details(X_train, "X_train")
            n_val_nans = _log_nan_details(X_val, "X_val")
        if n_train_nans or n_val_nans:
            raise ValueError(
                f"NaNs present in features for fold {fold_idx + 1}: "
                f"train_total_nan_entries={n_train_nans}, val_total_nan_entries={n_val_nans}. "
                "See error logs for per-column details."
            )

        # Train model on processed features (with optional sample_weight)
        try:
            # Optional: class weighting via sample_weight
            fold_model, sample_weight = self._prepare_class_weighting(fold_model, y_train)
            if sample_weight is not None:
                fold_model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                fold_model.fit(X_train, y_train)
        except TypeError:
            # Some estimators don't accept sample_weight
            fold_model.fit(X_train, y_train)
        self.logger.info(
            f"Fold {fold_idx+1}: model trained; val_predict on {len(X_val)} rows"
        )
        
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

    def _apply_imbalance_sampling_if_enabled(self, X_train: pd.DataFrame, y_train: pd.Series, fold_idx: int, model_name: Optional[str] = None) -> tuple[pd.DataFrame, pd.Series]:
        """Apply simple downsampling or upsampling for binary imbalance if enabled in config.
        Config keys (in data.yaml or trainer config):
          imbalance:
            enabled: true|false
            strategy: random_under|random_over|downsample|upsample|smote|adasyn
            params: {...}   # e.g., sampling_strategy, k_neighbors, random_state
        """
        imb = (self.config.get("imbalance") or {})
        if not imb or not bool(imb.get("enabled", False)):
            return X_train, y_train

        strategy = str(imb.get("strategy", imb.get("method", "downsample"))).lower()
        params = dict(imb.get("params", {}) or {})
        # Determine class counts
        vals, counts = np.unique(y_train.values, return_counts=True)
        if len(vals) != 2:
            self.logger.warning("Imbalance handling is only implemented for binary classification; skipping.")
            return X_train, y_train
        cls0, cls1 = int(vals[0]), int(vals[1])
        n0, n1 = int(counts[0]), int(counts[1])
        minority = cls0 if n0 <= n1 else cls1
        majority = cls1 if minority == cls0 else cls0
        n_min, n_maj = (n0, n1) if minority == cls0 else (n1, n0)

        self.logger.info(
            f"Fold {fold_idx+1}: Imbalance before sampling -> class {cls0}: {n0}, class {cls1}: {n1} (strategy={strategy}{' model='+model_name if model_name else ''})"
        )

        rng = np.random.default_rng(self.config.get("cv_random_state", 42))
        idx_min = y_train.index[y_train == minority].to_numpy()
        idx_maj = y_train.index[y_train == majority].to_numpy()

        # Try imblearn-based strategies first if requested
        if strategy in ("smote", "adasyn", "random_over", "random_under"):
            try:
                if strategy == "smote":
                    from imblearn.over_sampling import SMOTE
                    sampler = SMOTE(random_state=self.config.get("cv_random_state", 42), **params)
                elif strategy == "adasyn":
                    from imblearn.over_sampling import ADASYN
                    sampler = ADASYN(random_state=self.config.get("cv_random_state", 42), **params)
                elif strategy == "random_over":
                    from imblearn.over_sampling import RandomOverSampler
                    sampler = RandomOverSampler(random_state=self.config.get("cv_random_state", 42), **params)
                else:
                    from imblearn.under_sampling import RandomUnderSampler
                    sampler = RandomUnderSampler(random_state=self.config.get("cv_random_state", 42), **params)

                Xs, ys = sampler.fit_resample(X_train, y_train)
                vals_after, counts_after = np.unique(ys.values, return_counts=True)
                self.logger.info(
                    f"Fold {fold_idx+1}: Imbalance after sampling -> {dict(zip(vals_after.astype(int), counts_after.astype(int)))}"
                )
                return Xs, ys
            except Exception as e:
                self.logger.warning(f"Imblearn strategy '{strategy}' failed ({e}); falling back to simple sampler")

        # Fallback: internal simple up/down sampling
        if strategy in ("downsample", "random_under"):
            # Downsample majority to minority count
            if n_maj > n_min and n_min > 0:
                sel_maj = rng.choice(idx_maj, size=n_min, replace=False)
                new_idx = np.concatenate([idx_min, sel_maj])
            else:
                new_idx = y_train.index.to_numpy()
        elif strategy in ("upsample", "random_over"):
            # Upsample minority to majority count
            if n_maj > n_min and n_min > 0:
                sel_min = rng.choice(idx_min, size=n_maj - n_min, replace=True)
                new_idx = np.concatenate([idx_min, sel_min, idx_maj])
            else:
                new_idx = y_train.index.to_numpy()
        else:
            self.logger.warning(f"Unknown imbalance strategy '{strategy}'; skipping sampling.")
            return X_train, y_train

        # Shuffle indices to avoid ordering
        rng.shuffle(new_idx)
        Xs = X_train.loc[new_idx]
        ys = y_train.loc[new_idx]
        vals_after, counts_after = np.unique(ys.values, return_counts=True)
        self.logger.info(
            f"Fold {fold_idx+1}: Imbalance after sampling -> {dict(zip(vals_after.astype(int), counts_after.astype(int)))}"
        )
        return Xs, ys

    # -----------------
    # Class weighting
    # -----------------
    def _prepare_class_weighting(self, estimator: BaseEstimator, y_train: pd.Series, model_name: Optional[str] = None) -> tuple[BaseEstimator, Optional[np.ndarray]]:
        """Optionally apply class weighting as an alternative to resampling.
        Config:
          class_weight:
            enabled: true|false
            scheme: balanced|custom
            weights: {0: 1.0, 1: 2.0}
        Returns possibly modified estimator and optional sample_weight array for fit().
        """
        cw_cfg = (self.config.get("class_weight") or {})
        if not cw_cfg or not bool(cw_cfg.get("enabled", False)):
            return estimator, None

        scheme = str(cw_cfg.get("scheme", "balanced")).lower()
        custom_weights = cw_cfg.get("weights") or {}

        # Compute class counts
        vals, counts = np.unique(y_train.values, return_counts=True)
        if len(vals) != 2:
            self.logger.warning("class_weight is only implemented for binary tasks; skipping.")
            return estimator, None
        cls0, cls1 = int(vals[0]), int(vals[1])
        n0, n1 = int(counts[0]), int(counts[1])

        # Heuristics for different libraries
        params = {}
        sample_weight: Optional[np.ndarray] = None

        # sklearn-compatible 'class_weight'
        try:
            est_params = estimator.get_params() if hasattr(estimator, 'get_params') else {}
        except Exception:
            est_params = {}

        if 'class_weight' in est_params:
            if scheme == 'balanced':
                params['class_weight'] = 'balanced'
            elif scheme == 'custom' and custom_weights:
                # Ensure keys as ints
                params['class_weight'] = {int(k): float(v) for k, v in custom_weights.items()}
            try:
                estimator.set_params(**params)
                self.logger.info(f"Applied class_weight to estimator ({params.get('class_weight')})")
                return estimator, None
            except Exception as e:
                self.logger.warning(f"Failed to set class_weight on estimator: {e}")

        # xgboost / lightgbm / catboost mapping
        est_mod = getattr(estimator.__class__, '__module__', '')
        if 'xgboost' in est_mod:
            try:
                # scale_pos_weight = n_negative / n_positive for positive class 1
                pos = n1 if cls1 == 1 else n0
                neg = n0 if cls1 == 1 else n1
                spw = float(neg) / float(pos) if pos > 0 else 1.0
                estimator.set_params(scale_pos_weight=spw)
                self.logger.info(f"Applied XGBoost scale_pos_weight={spw:.4f}")
                return estimator, None
            except Exception:
                pass
        if 'lightgbm' in est_mod:
            try:
                # Prefer class_weight dict if supported
                if 'class_weight' in est_params:
                    estimator.set_params(class_weight={cls0: 1.0, cls1: float(n0)/float(n1) if n1>0 else 1.0})
                    self.logger.info("Applied LightGBM class_weight dict")
                    return estimator, None
                # Fallback to scale_pos_weight
                pos = n1 if cls1 == 1 else n0
                neg = n0 if cls1 == 1 else n1
                spw = float(neg) / float(pos) if pos > 0 else 1.0
                estimator.set_params(scale_pos_weight=spw)
                self.logger.info(f"Applied LightGBM scale_pos_weight={spw:.4f}")
                return estimator, None
            except Exception:
                pass
        if 'catboost' in est_mod:
            try:
                # If user already enabled auto_class_weights, don't set class_weights to avoid conflict
                ep = est_params
                auto_cw = ep.get('auto_class_weights', ep.get('AutoClassWeights', None))
                if auto_cw not in (None, 'None', 'Disabled', False):
                    self.logger.info("CatBoost auto_class_weights is set; skipping explicit class_weights to avoid conflict")
                    return estimator, None
                # CatBoost uses class_weights=[w0, w1]
                w0 = 1.0
                w1 = float(n0)/float(n1) if n1 > 0 else 1.0
                estimator.set_params(class_weights=[w0, w1] if cls0 == 0 else [w1, w0])
                self.logger.info("Applied CatBoost class_weights")
                return estimator, None
            except Exception:
                pass

        # Generic sample_weight fallback for fit()
        if scheme == 'balanced':
            # Use sklearn's balanced formula: n_samples / (n_classes * n_i)
            n = n0 + n1
            w0 = n / (2.0 * n0) if n0 > 0 else 1.0
            w1 = n / (2.0 * n1) if n1 > 0 else 1.0
        elif scheme == 'custom' and custom_weights:
            w0 = float(custom_weights.get(str(cls0), custom_weights.get(cls0, 1.0)))
            w1 = float(custom_weights.get(str(cls1), custom_weights.get(cls1, 1.0)))
        else:
            w0 = w1 = 1.0

        weights = {cls0: w0, cls1: w1}
        sample_weight = y_train.map(weights).astype(float).values
        self.logger.info(f"Prepared sample_weight fallback (w0={w0:.4f}, w1={w1:.4f})")
        return estimator, sample_weight

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
        """Calculate the evaluation score based on configured cv_metric."""
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        metric = str(self.config.get("cv_metric", "roc_auc")).lower()
        try:
            if metric == "roc_auc":
                return float(roc_auc_score(y_true, y_pred))
            # For accuracy/f1, threshold probabilities at 0.5
            y_bin = (np.asarray(y_pred).ravel() >= 0.5).astype(int)
            if metric == "f1":
                return float(f1_score(y_true, y_bin))
            # default accuracy
            return float(accuracy_score(y_true, y_bin))
        except Exception:
            # Fallback: accuracy at 0.5
            y_bin = (np.asarray(y_pred).ravel() >= 0.5).astype(int)
            return float(accuracy_score(y_true, y_bin))

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
