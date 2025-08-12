"""Model evaluation with comprehensive metrics and analysis."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from eval.utils import ThresholdOptimizer
from core.interfaces import IEvaluator
from core.utils import LoggerFactory


class TitanicEvaluator(IEvaluator):
    """Comprehensive evaluator for Titanic binary classification."""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    # ---------------------------
    # Helper methods (new)
    # ---------------------------
    def _point_metrics(self, y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
        """Compute point metrics and CM pieces from hard labels."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        return {
            "accuracy": float(accuracy_score(y_true, y_hat)),
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        }

    def compute_at_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Compute point-metrics at each provided threshold (keys are method names)."""
        rows: List[Dict[str, Any]] = []
        for method, thr in thresholds.items():
            thr = float(thr)
            y_hat = (y_proba >= thr).astype(int)
            row = {"method": method, "threshold": thr}
            row.update(self._point_metrics(y_true, y_hat))
            rows.append(row)
        return rows

    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        th_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run ThresholdOptimizer and return:
        {
            "optimal_thresholds": {method: threshold, ...},
            "chosen_method": <str>,
            "chosen_threshold": <float or None>
        }
        """
        if np.unique(y_true).size < 2:
            return {
                "optimal_thresholds": {},
                "chosen_method": str(th_cfg.get("method", "accuracy")).lower(),
                "chosen_threshold": None,
            }

        opt = ThresholdOptimizer(y_true, y_proba)
        all_methods = {
            "f1": opt.best_f1,
            "accuracy": opt.best_accuracy,
            "youdenj": opt.best_youdenj,
            "cost": opt.best_cost,
        }
        optimal_thresholds = {m: float(all_methods[m]()[0]) for m in all_methods}
        chosen_method = str(th_cfg.get("method", "accuracy")).lower()
        chosen_threshold = optimal_thresholds.get(chosen_method)

        return {
            "optimal_thresholds": optimal_thresholds,
            "chosen_method": chosen_method,
            "chosen_threshold": chosen_threshold,
        }

    def per_fold_threshold_analysis(self, oof_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Expect oof_df to contain columns: ['target', 'prediction', 'fold'].
        Returns dict with JSON-friendly lists:
        {
          "per_fold_rows": [ {fold, method, threshold, objective, accuracy, precision, recall, f1}, ... ],
          "summary": [ {method, threshold_mean, threshold_std, ...}, ... ]
        }
        """
        if "fold" not in oof_df.columns:
            return {"per_fold_rows": [], "summary": []}

        rows: List[Dict[str, Any]] = []
        for fold_id, g in oof_df.groupby("fold"):
            y_t = g["target"].to_numpy()
            y_p = g["prediction"].to_numpy(dtype=float)
            if np.unique(y_t).size < 2:
                rows.append({"fold": int(fold_id), "note": "single class – skipped"})
                continue

            opt = ThresholdOptimizer(y_t, y_p)
            methods = {
                "f1": opt.best_f1,
                "accuracy": opt.best_accuracy,
                "youdenj": opt.best_youdenj,
                "cost": opt.best_cost,
            }
            for m, fn in methods.items():
                thr, score = fn()
                y_hat = (y_p >= float(thr)).astype(int)
                pm = self._point_metrics(y_t, y_hat)
                rows.append({
                    "fold": int(fold_id),
                    "method": m,
                    "threshold": float(thr),
                    "objective": float(score),
                    **pm,
                })

        # Summaries (mean/std per method) without pulling pandas into return types
        from collections import defaultdict
        agg = defaultdict(list)
        for r in rows:
            if "method" in r:
                agg[r["method"]].append(r)

        def mean_std(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"mean": None, "std": None}
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        summary: List[Dict[str, Any]] = []
        for m, lst in agg.items():
            summary.append({
                "method": m,
                **{f"threshold_{k}": v for k, v in mean_std([r["threshold"] for r in lst]).items()},
                **{f"objective_{k}": v for k, v in mean_std([r["objective"] for r in lst]).items()},
                **{f"accuracy_{k}": v for k, v in mean_std([r["accuracy"] for r in lst]).items()},
                **{f"precision_{k}": v for k, v in mean_std([r["precision"] for r in lst]).items()},
                **{f"recall_{k}": v for k, v in mean_std([r["recall"] for r in lst]).items()},
                **{f"f1_{k}": v for k, v in mean_std([r["f1"] for r in lst]).items()},
            })

        return {"per_fold_rows": rows, "summary": summary}

    # ---------------------------
    # Main evaluation methods
    # ---------------------------
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        config: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate predictions with comprehensive metrics."""

        # --- Handle probability vs binary predictions ---
        if self._is_probability(y_pred):
            y_pred_proba = y_pred.values.astype(float)
            th_cfg = config.get("threshold", {})
            threshold = th_cfg.get("value", 0.5) if isinstance(th_cfg, dict) else float(th_cfg)
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred_proba = None
            y_pred_binary = y_pred.values.astype(int)

        metrics: Dict[str, Any] = {}

        # --- Basic classification metrics ---
        metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)
        metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred_binary, zero_division=0)

        # --- Confusion matrix elements (stable 2x2) ---
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

        # Derived
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0
        metrics["npv"] = tn / (tn + fn) if (tn + fn) else 0.0

        # --- Probability-based metrics & analysis ---
        has_both_classes = np.unique(y_true).size == 2
        if y_pred_proba is not None and has_both_classes:
            metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
            metrics["log_loss"] = log_loss(y_true, y_pred_proba)
            metrics["brier_score"] = float(np.mean((y_pred_proba - y_true) ** 2))

            # ROC quick stats (YoudenJ at native ROC thresholds)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            youden_scores = tpr - fpr
            j_idx = int(np.argmax(youden_scores))
            metrics.update(
                {
                    "youden_index": float(youden_scores[j_idx]),
                    "optimal_tpr": float(tpr[j_idx]),
                    "optimal_fpr": float(fpr[j_idx]),
                    # leave threshold selection to the optimizer below
                }
            )

            # Calibration
            frac_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
            calibration_error = float(np.mean(np.abs(frac_pos - mean_pred)))
            metrics.update(
                {
                    "calibration_error": calibration_error,
                    "is_well_calibrated": calibration_error < 0.1,
                    "fraction_of_positives": frac_pos.tolist(),
                    "mean_predicted_values": mean_pred.tolist(),
                }
            )

            # Prediction distribution
            metrics.update(
                {
                    "mean_prediction": float(np.mean(y_pred_proba)),
                    "std_prediction": float(np.std(y_pred_proba)),
                    "min_prediction": float(np.min(y_pred_proba)),
                    "max_prediction": float(np.max(y_pred_proba)),
                    "p25": float(np.percentile(y_pred_proba, 25)),
                    "p50": float(np.percentile(y_pred_proba, 50)),
                    "p75": float(np.percentile(y_pred_proba, 75)),
                }
            )

            # Keep raw arrays for downstream optimization/reporting
            metrics["y_true"] = np.asarray(y_true)
            metrics["y_proba"] = y_pred_proba

            # ---- Optimized-threshold point metrics ----
            th_cfg = config.get("threshold", {})
            if th_cfg.get("optimizer", False):
                opt = ThresholdOptimizer(metrics["y_true"], metrics["y_proba"])
                method = str(th_cfg.get("method", "accuracy")).lower()
                chooser = {
                    "f1": opt.best_f1,
                    "accuracy": opt.best_accuracy,
                    "youdenj": opt.best_youdenj,
                    "cost": opt.best_cost,
                }.get(method, opt.best_accuracy)

                best_thresh, best_score = chooser()
                metrics["optimal_threshold"] = float(best_thresh)
                metrics["optimal_threshold_score"] = float(best_score)

                # Recompute point metrics at the optimized threshold
                y_opt = (y_pred_proba >= best_thresh).astype(int)
                pm_opt = self._point_metrics(np.asarray(y_true), y_opt)
                # attach with _opt suffix for readability
                metrics.update({k + "_opt": v for k, v in pm_opt.items()
                                if k in ("accuracy", "precision", "recall", "f1", "specificity", "npv")})
                metrics["true_negatives_opt"] = pm_opt["tn"]
                metrics["false_positives_opt"] = pm_opt["fp"]
                metrics["false_negatives_opt"] = pm_opt["fn"]
                metrics["true_positives_opt"] = pm_opt["tp"]

        # --- Robust logging (AUC may be missing) ---
        auc = metrics.get("auc")
        auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else "N/A"
        self.logger.info(
            f"Evaluation completed. AUC: {auc_str}, Accuracy: {metrics['accuracy']:.4f}"
        )

        return metrics

    def evaluate_cv(
        self,
        oof_predictions: pd.Series,
        y_true: pd.Series,
        fold_scores: List[float],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate cross-validation results."""

        # OOF evaluation
        oof_metrics = self.evaluate(y_true, oof_predictions, config)

        # CV stats
        cv_stats = {
            "cv_mean": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
            "cv_min": float(np.min(fold_scores)),
            "cv_max": float(np.max(fold_scores)),
            "n_folds": len(fold_scores),
            "fold_scores": fold_scores,
        }
        cv_cov = cv_stats["cv_std"] / cv_stats["cv_mean"] if cv_stats["cv_mean"] > 0 else float("inf")

        evaluation_report: Dict[str, Any] = {
            "oof_metrics": oof_metrics,
            "cv_statistics": cv_stats,
            "stability": {
                "coefficient_of_variation": cv_cov,
                "is_stable": cv_cov < 0.1,
                "score_range": cv_stats["cv_max"] - cv_stats["cv_min"],
            },
        }

        # Performance assessment
        evaluation_report["assessment"] = self._assess_performance(oof_metrics, cv_stats, config)

        # Global threshold optimization + per-method metrics (JSON-friendly)
        th_cfg = config.get("threshold", {}) or {}
        y_true_arr = oof_metrics.get("y_true")
        y_proba_arr = oof_metrics.get("y_proba")
        if th_cfg.get("optimizer", False) and (y_true_arr is not None) and (y_proba_arr is not None) and (np.unique(y_true_arr).size == 2):
            opt_pack = self.optimize_thresholds(y_true_arr, y_proba_arr, th_cfg)
            evaluation_report["optimal_thresholds"] = opt_pack["optimal_thresholds"]
            evaluation_report["chosen_method"] = opt_pack["chosen_method"]
            evaluation_report["chosen_threshold"] = opt_pack["chosen_threshold"]

            if opt_pack["optimal_thresholds"]:
                evaluation_report["threshold_method_metrics"] = self.compute_at_thresholds(
                    y_true_arr, y_proba_arr, opt_pack["optimal_thresholds"]
                )

        auc = oof_metrics.get("auc")
        auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else "N/A"
        self.logger.info(
            f"CV Evaluation - Mean: {cv_stats['cv_mean']:.4f} ± {cv_stats['cv_std']:.4f}, OOF AUC: {auc_str}"
        )
        return evaluation_report

    def evaluate_with_analysis(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extended evaluation with detailed analysis."""

        basic_metrics = self.evaluate(y_true, y_pred, config)
        if not self._is_probability(y_pred):
            return {"basic_metrics": basic_metrics, "detailed_analysis": {}}

        y_proba = np.asarray(basic_metrics.get("y_proba"))
        y_true_arr = np.asarray(y_true)
        if np.unique(y_true_arr).size < 2:
            return {
                "basic_metrics": basic_metrics,
                "detailed_analysis": {"note": "Only one class present; ROC/calibration skipped."},
            }

        opt = ThresholdOptimizer(y_true_arr, y_proba)
        fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true_arr, y_proba)
        best_thresh, best_j = opt.best_youdenj()
        idx = int(np.argmin(np.abs(thresh_arr - best_thresh)))
        roc_analysis = {
            "optimal_threshold": float(best_thresh),
            "optimal_tpr": float(tpr_arr[idx]),
            "optimal_fpr": float(fpr_arr[idx]),
            "youden_index": float(best_j),
        }

        frac_pos, mean_pred = calibration_curve(y_true_arr, y_proba, n_bins=10)
        calib_err = float(np.mean(np.abs(frac_pos - mean_pred)))
        calibration = {
            "calibration_error": calib_err,
            "is_well_calibrated": calib_err < 0.1,
            "fraction_of_positives": frac_pos.tolist(),
            "mean_predicted_values": mean_pred.tolist(),
        }

        percentiles = {p: float(np.percentile(y_proba, p)) for p in (25, 50, 75)}
        pred_dist = {
            "mean_prediction": float(np.mean(y_proba)),
            "std_prediction": float(np.std(y_proba)),
            "min_prediction": float(np.min(y_proba)),
            "max_prediction": float(np.max(y_proba)),
            "percentiles": percentiles,
        }

        return {
            "basic_metrics": basic_metrics,
            "detailed_analysis": {
                "roc_analysis": roc_analysis,
                "calibration": calibration,
                "prediction_distribution": pred_dist,
            },
        }

    def compare_models(self, evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model evaluations."""
        if len(evaluations) < 2:
            return {"error": "Need at least 2 models to compare"}

        comparison = {
            "models": list(evaluations.keys()),
            "metrics_comparison": {},
            "rankings": {},
            "best_model": {},
        }

        metrics_to_compare = ["auc", "accuracy", "f1", "precision", "recall", "log_loss"]

        for metric in metrics_to_compare:
            metric_values = {}
            for model_name, eval_result in evaluations.items():
                metrics = eval_result["oof_metrics"] if isinstance(eval_result, dict) and "oof_metrics" in eval_result else eval_result
                if metric in metrics and metrics[metric] is not None:
                    metric_values[model_name] = metrics[metric]

            if metric_values:
                comparison["metrics_comparison"][metric] = metric_values
                if metric == "log_loss":
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
                else:
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                comparison["rankings"][metric] = [name for name, _ in sorted_models]

        primary_metric = "auc" if "auc" in comparison["metrics_comparison"] else "accuracy"
        if primary_metric in comparison["rankings"] and comparison["rankings"][primary_metric]:
            best_model_name = comparison["rankings"][primary_metric][0]
            comparison["best_model"] = {
                "name": best_model_name,
                "primary_metric": primary_metric,
                "primary_score": comparison["metrics_comparison"][primary_metric][best_model_name],
            }

        return comparison

    def _is_probability(self, y_pred: pd.Series) -> bool:
        """Check if predictions are probabilities (float in [0,1])."""
        return np.issubdtype(y_pred.dtype, np.floating) and (y_pred.min() >= 0) and (y_pred.max() <= 1)

    def _assess_performance(
        self,
        oof_metrics: Dict[str, float],
        cv_stats: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        assessment = {
            "performance_level": "unknown",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
        }

        auc = oof_metrics.get("auc")
        if isinstance(auc, (int, float)):
            if auc >= 0.9:
                assessment["performance_level"] = "excellent"
                assessment["strengths"].append("Very high AUC score")
            elif auc >= 0.8:
                assessment["performance_level"] = "good"
                assessment["strengths"].append("Good AUC score")
            elif auc >= 0.7:
                assessment["performance_level"] = "moderate"
            else:
                assessment["performance_level"] = "poor"
                assessment["weaknesses"].append("Low AUC score")
                assessment["recommendations"].append("Consider feature engineering or different models")

        # Stability
        if cv_stats["cv_mean"] > 0 and (cv_stats["cv_std"] / cv_stats["cv_mean"]) > 0.1:
            assessment["weaknesses"].append("High cross-validation variance")
            assessment["recommendations"].append("Model may be unstable - consider regularization")
        else:
            assessment["strengths"].append("Stable cross-validation performance")

        # Precision vs recall balance
        precision = oof_metrics.get("precision", 0.0) or 0.0
        recall = oof_metrics.get("recall", 0.0) or 0.0
        if abs(precision - recall) > 0.2:
            if precision > recall:
                assessment["weaknesses"].append("Low recall relative to precision")
                assessment["recommendations"].append("Consider lowering classification threshold")
            else:
                assessment["weaknesses"].append("Low precision relative to recall")
                assessment["recommendations"].append("Consider raising classification threshold")

        return assessment
