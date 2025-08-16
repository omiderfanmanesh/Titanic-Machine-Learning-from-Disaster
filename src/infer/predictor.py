"""Inference and prediction components with ensembling support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from core.interfaces import IPredictor
from core.utils import LoggerFactory


class TitanicPredictor(IPredictor):
    """Main predictor for Titanic competition with ensemble support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[neg])
        out[neg] = exp_x / (1.0 + exp_x)
        return out

    def _normalize_scores_to_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert model outputs to probabilities in [0,1].
        - If already in [0,1], keep as-is (clip tiny numeric drift).
        - If outside [0,1], assume logit-like scores and apply sigmoid.
        - If it's {0,1} hard labels, cast to float.
        """
        scores = np.asarray(scores, dtype=float).ravel()
        if np.isnan(scores).any():
            self.logger.warning("NaNs in model scores; replacing with 0.5")
            scores = np.where(np.isnan(scores), 0.5, scores)

        min_s, max_s = scores.min(), scores.max()
        if (min_s >= 0.0) and (max_s <= 1.0):
            return np.clip(scores, 0.0, 1.0)
        return np.clip(self._sigmoid(scores), 0.0, 1.0)

    def _read_threshold_from_report(self, report_path: Path, method: str) -> Optional[float]:
        """Try reading a threshold from a CSV report with columns ['method','threshold']."""
        try:
            if report_path.exists():
                df = pd.read_csv(report_path)
                if {"method", "threshold"}.issubset(df.columns):
                    row = df.loc[df["method"].str.lower() == method.lower()]
                    if not row.empty:
                        thr = float(row.iloc[0]["threshold"])
                        self.logger.info(f"Using threshold from report ({method}) at {report_path}: {thr:.6f}")
                        return thr
        except Exception as e:
            self.logger.warning(f"Failed reading threshold report '{report_path}': {e}")
        return None

    def _resolve_threshold(self, config: Dict[str, Any]) -> float:
        """
        Resolve threshold priority:
          1) config['threshold']['file'] or config['best_threshold_file']
          2) {run_dir}/best_threshold.txt
          3) threshold report (config['threshold']['report_path'] or {run_dir}/threshold_report.csv)
          4) config['threshold']['value']
          5) default 0.5
        """
        th_cfg = config.get("threshold", {}) or {}
        run_dir = config.get("run_dir")  # <- let the CLI set this key
        method = str(th_cfg.get("method", "accuracy")).lower()

        # 1) explicit file
        file_path = th_cfg.get("file") or config.get("best_threshold_file")
        if file_path:
            try:
                p = Path(file_path)
                if p.exists():
                    val = float(p.read_text().strip())
                    self.logger.info(f"Using saved threshold from {p}: {val:.6f}")
                    return float(val)
                else:
                    self.logger.warning(f"Threshold file not found: {p}; searching alternatives.")
            except Exception as e:
                self.logger.warning(f"Failed to read threshold file '{file_path}': {e}; searching alternatives.")

        # 2) auto-detect best_threshold.txt in run_dir
        if run_dir:
            p = Path(run_dir) / "best_threshold.txt"
            if p.exists():
                try:
                    val = float(p.read_text().strip())
                    self.logger.info(f"Using saved threshold from {p}: {val:.6f}")
                    return float(val)
                except Exception as e:
                    self.logger.warning(f"Failed reading {p}: {e}")

        # 3) read from threshold report
        #    a) explicit path from config
        report_from_cfg = th_cfg.get("report_path")
        if report_from_cfg:
            thr = self._read_threshold_from_report(Path(report_from_cfg), method)
            if thr is not None:
                return thr
        #    b) default report in run_dir
        if run_dir:
            thr = self._read_threshold_from_report(Path(run_dir) / "threshold_report.csv", method)
            if thr is not None:
                return thr

        # 4) fallback static value from config
        if isinstance(th_cfg, dict) and "value" in th_cfg:
            return float(th_cfg.get("value", 0.5))

        # 5) final default
        return 0.5

    # ---------------------------
    # Public API
    # ---------------------------
    def predict(
        self,
        X: pd.DataFrame,
        models: List[BaseEstimator],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Generate binary predictions (and include probabilities)."""
        proba_df = self.predict_proba(X, models, config)
        threshold = self._resolve_threshold(config)
        y_bin = (proba_df["prediction_proba"].values >= threshold).astype(int)

        result = pd.DataFrame(
            {
                "PassengerId": X.index,
                "prediction_proba": proba_df["prediction_proba"].values,
                "prediction": y_bin,
            }
        )

        # Apply postprocessing rules
        result = self._apply_postprocessing(result, config)

        self.logger.info(
            f"Generated binary predictions with threshold={threshold:.4f} (n={len(result)})"
        )
        return result

    def predict_proba(
        self,
        X: pd.DataFrame,
        models: List[BaseEstimator],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Generate probability predictions with ensembling."""
        if not models:
            raise ValueError("No models provided for prediction")

        self.logger.info(f"Making predictions with {len(models)} models")

        # Apply TTA if enabled
        all_predictions = self._apply_tta(X, models, config)

        if not all_predictions:
            raise RuntimeError("All model predictions failed")

        # Ensemble predictions
        ensemble_method = config.get("ensemble_method", "average")
        ensemble_weights = config.get("ensemble_weights")

        final_predictions = self._ensemble_predictions(
            all_predictions, ensemble_method, ensemble_weights
        )
        final_predictions = np.clip(final_predictions, 0.0, 1.0)

        result = pd.DataFrame(
            {
                "PassengerId": X.index,
                "prediction_proba": final_predictions,
            }
        )

        self.logger.info(f"Generated probability predictions for {len(result)} samples")
        return result

    def _predict_single_model(self, model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """Get raw predictions from a single model (may be proba, logits, or labels)."""
        try:
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
        except Exception:
            pass
        try:
            return model.decision_function(X)
        except Exception:
            pass
        return model.predict(X)

    def _ensemble_predictions(
        self,
        predictions: List[np.ndarray],
        method: str,
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Ensemble multiple probability vectors."""
        preds = np.vstack([np.asarray(p, dtype=float).ravel() for p in predictions])  # (m, n)

        if method == "single_best":
            # Return predictions from the first model only (assumes models are ordered by performance)
            self.logger.info("Using single best model (no ensemble)")
            return preds[0]

        elif method == "average":
            if weights is not None:
                weights = np.asarray(weights, dtype=float).ravel()
                if weights.size != preds.shape[0]:
                    self.logger.warning("Weights length mismatch; using equal weights.")
                    return preds.mean(axis=0)
                weights = weights / weights.sum()
                return np.average(preds, axis=0, weights=weights)
            return preds.mean(axis=0)

        elif method == "rank_average":
            ranks = np.vstack([pd.Series(p).rank(pct=True).to_numpy() for p in preds])
            return ranks.mean(axis=0)

        elif method == "geometric_mean":
            preds_clip = np.clip(preds, 1e-8, 1 - 1e-8)
            return np.exp(np.log(preds_clip).mean(axis=0))

        elif method == "median":
            return np.median(preds, axis=0)
        elif method == "max":
            return np.max(preds, axis=0)
        elif method == "min":
            return np.min(preds, axis=0)
        else:
            self.logger.warning(f"Unknown ensemble method: {method}; using average.")
            return preds.mean(axis=0)

    def _apply_tta(self, X: pd.DataFrame, models: List[BaseEstimator], config: Dict[str, Any]) -> List[np.ndarray]:
        """Apply Test-Time Augmentation if enabled."""
        use_tta = config.get("use_tta", False)
        if not use_tta:
            return self._predict_all_models(X, models)

        tta_rounds = config.get("tta_rounds", 5)
        tta_noise_scale = float(config.get("tta_noise_scale", 0.01))
        tta_noise_scale = config.get("tta_noise_scale", 0.01)

        self.logger.info(f"Applying TTA with {tta_rounds} rounds, noise scale {tta_noise_scale}")

        all_tta_predictions = []

        # Original predictions
        original_preds = self._predict_all_models(X, models)
        all_tta_predictions.extend(original_preds)

        # TTA rounds with noise
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for round_idx in range(tta_rounds):
            X_augmented = X.copy()
            if len(numeric_cols) > 0:
                noise = np.random.normal(0, tta_noise_scale, size=(len(X), len(numeric_cols)))
                X_augmented[numeric_cols] += noise

            tta_preds = self._predict_all_models(X_augmented, models)
            all_tta_predictions.extend(tta_preds)

        return all_tta_predictions

    def _predict_all_models(self, X: pd.DataFrame, models: List[BaseEstimator]) -> List[np.ndarray]:
        """Get predictions from all models for given input."""
        predictions = []
        for i, model in enumerate(models):
            try:
                raw = self._predict_single_model(model, X)
                proba = self._normalize_scores_to_proba(raw)
                predictions.append(proba)
            except Exception as e:
                self.logger.error(f"Model {i+1} prediction failed: {e}")
                continue
        return predictions

    def _apply_postprocessing(self, predictions: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply postprocessing rules to predictions."""
        postproc_config = config.get("postprocessing", {})
        rules = postproc_config.get("rules", [])

        if not rules:
            return predictions

        result = predictions.copy()

        for rule in rules:
            rule_type = rule.get("type")
            if rule_type == "round_predictions":
                result["prediction"] = result["prediction"].round().astype(int)
                result["prediction_proba"] = result["prediction_proba"].round(6)
                self.logger.info("Applied round_predictions postprocessing")
            else:
                self.logger.warning(f"Unknown postprocessing rule: {rule_type}")

        return result


class ModelLoader:
    """Utility for loading trained models from various sources."""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    def load_models_from_paths(self, model_paths: List[Union[str, Path]]) -> List[BaseEstimator]:
        """Load models from file paths."""
        models = []
        for path in model_paths:
            try:
                model = joblib.load(path)
                models.append(model)
                self.logger.info(f"Loaded model from {path}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {path}: {e}")

        if not models:
            raise RuntimeError("No models successfully loaded")
        return models

    def load_fold_models(self, run_dir: Union[str, Path]) -> List[BaseEstimator]:
        """Load all fold models from a training run directory."""
        run_dir = Path(run_dir)
        fold_model_paths = sorted(run_dir.glob("fold_*_model.joblib"))
        if not fold_model_paths:
            raise FileNotFoundError(f"No fold models found in {run_dir}")
        model_paths = [str(p) for p in fold_model_paths]
        return self.load_models_from_paths(model_paths)

    def load_best_model(self, run_dir: Union[str, Path]) -> BaseEstimator:
        """Load the best model from a training run."""
        run_dir = Path(run_dir)
        scores_path = run_dir / "cv_scores.json"

        if scores_path.exists():
            import json
            with open(scores_path, "r") as f:
                scores_data = json.load(f)
            fold_scores = scores_data.get("fold_scores", [])
            if fold_scores:
                best_fold = int(np.argmax(fold_scores))
                best_model_path = run_dir / f"fold_{best_fold}_model.joblib"
                if best_model_path.exists():
                    return joblib.load(best_model_path)

        # Fallback: load first fold model
        fold_models = sorted(run_dir.glob("fold_*_model.joblib"))
        if fold_models:
            return joblib.load(fold_models[0])

        raise FileNotFoundError(f"No models found in {run_dir}")


class TTAPredictor(TitanicPredictor):
    """Test-Time Augmentation predictor (for future extension)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def predict_proba(
        self,
        X: pd.DataFrame,
        models: List[BaseEstimator],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Predict with test-time augmentation."""
        tta_rounds = int(config.get("tta_rounds", 5))
        if tta_rounds <= 1 or not config.get("use_tta", False):
            return super().predict_proba(X, models, config)

        self.logger.info(f"Using TTA with {tta_rounds} rounds")

        all_tta_predictions = []
        for round_idx in range(tta_rounds):
            X_augmented = self._augment_data(X, round_idx, config)
            round_predictions = super().predict_proba(X_augmented, models, config)
            all_tta_predictions.append(round_predictions["prediction_proba"].values)

        final_predictions = np.mean(all_tta_predictions, axis=0)
        result = pd.DataFrame({"PassengerId": X.index, "prediction_proba": final_predictions})
        return result

    def _augment_data(self, X: pd.DataFrame, round_idx: int, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply test-time augmentation to data."""
        np.random.seed(round_idx)  # reproducible
        X_aug = X.copy()

        noise_scale = float(config.get("tta_noise_scale", 0.01))
        numeric_cols = X_aug.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col == "PassengerId":
                continue
            col_std = float(np.nan_to_num(X_aug[col].std(ddof=0), nan=0.0))
            if col_std == 0.0:
                continue
            noise = np.random.normal(0.0, noise_scale * col_std, len(X_aug))
            X_aug[col] = X_aug[col].astype(float) + noise

        return X_aug


def create_predictor(config: Dict[str, Any]) -> IPredictor:
    """Factory function to create predictor."""
    if config.get("use_tta", False):
        return TTAPredictor(config)
    else:
        return TitanicPredictor(config)
