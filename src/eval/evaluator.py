"""Model evaluation with comprehensive metrics and analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from core.interfaces import IEvaluator
from core.utils import LoggerFactory


class TitanicEvaluator(IEvaluator):
    """Comprehensive evaluator for Titanic binary classification."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series,
                config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate predictions with comprehensive metrics."""
        
        # Handle probability vs binary predictions
        if self._is_probability(y_pred):
            y_pred_proba = y_pred.values
            threshold = config.get("threshold", 0.5)
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred_binary = y_pred.values
            y_pred_proba = None
            
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)
        metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Probability-based metrics (if available)
        if y_pred_proba is not None:
            metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
            metrics["log_loss"] = log_loss(y_true, y_pred_proba, eps=1e-15)
            
            # Brier score (mean squared error for probabilities)
            metrics["brier_score"] = np.mean((y_pred_proba - y_true) ** 2)
            
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
        
        # Additional derived metrics
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        self.logger.info(f"Evaluation completed. AUC: {metrics.get('auc', 'N/A'):.4f}, "
                        f"Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
        
    def evaluate_cv(self, oof_predictions: pd.Series, y_true: pd.Series,
                   fold_scores: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate cross-validation results."""
        
        # Basic OOF evaluation
        oof_metrics = self.evaluate(y_true, oof_predictions, config)
        
        # Cross-validation statistics
        cv_stats = {
            "cv_mean": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
            "cv_min": float(np.min(fold_scores)),
            "cv_max": float(np.max(fold_scores)),
            "n_folds": len(fold_scores),
            "fold_scores": fold_scores
        }
        
        # Stability analysis
        cv_coefficient_of_variation = cv_stats["cv_std"] / cv_stats["cv_mean"] if cv_stats["cv_mean"] > 0 else float('inf')
        
        # Combine results
        evaluation_report = {
            "oof_metrics": oof_metrics,
            "cv_statistics": cv_stats,
            "stability": {
                "coefficient_of_variation": cv_coefficient_of_variation,
                "is_stable": cv_coefficient_of_variation < 0.1,  # Less than 10% variation
                "score_range": cv_stats["cv_max"] - cv_stats["cv_min"]
            }
        }
        
        # Performance assessment
        evaluation_report["assessment"] = self._assess_performance(
            oof_metrics, cv_stats, config
        )
        
        self.logger.info(f"CV Evaluation - Mean: {cv_stats['cv_mean']:.4f} ± {cv_stats['cv_std']:.4f}, "
                        f"OOF AUC: {oof_metrics.get('auc', 'N/A'):.4f}")
        
        return evaluation_report
        
    def evaluate_with_analysis(self, y_true: pd.Series, y_pred: pd.Series,
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Extended evaluation with detailed analysis."""
        
        # Basic metrics
        metrics = self.evaluate(y_true, y_pred, config)
        
        analysis = {
            "basic_metrics": metrics,
            "detailed_analysis": {}
        }
        
        if self._is_probability(y_pred):
            y_pred_proba = y_pred.values
            
            # ROC curve analysis
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            
            # Find optimal threshold (Youden's Index)
            youden_scores = tpr - fpr
            optimal_threshold_idx = np.argmax(youden_scores)
            optimal_threshold = thresholds[optimal_threshold_idx]
            
            analysis["detailed_analysis"]["roc_analysis"] = {
                "optimal_threshold": float(optimal_threshold),
                "optimal_tpr": float(tpr[optimal_threshold_idx]),
                "optimal_fpr": float(fpr[optimal_threshold_idx]),
                "youden_index": float(youden_scores[optimal_threshold_idx])
            }
            
            # Calibration analysis
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_pred_proba, n_bins=10
                )
                
                # Calculate calibration error
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                
                analysis["detailed_analysis"]["calibration"] = {
                    "calibration_error": float(calibration_error),
                    "is_well_calibrated": calibration_error < 0.1,
                    "fraction_of_positives": fraction_of_positives.tolist(),
                    "mean_predicted_values": mean_predicted_value.tolist()
                }
            except Exception as e:
                self.logger.warning(f"Calibration analysis failed: {e}")
                
            # Prediction distribution analysis
            analysis["detailed_analysis"]["prediction_distribution"] = {
                "mean_prediction": float(np.mean(y_pred_proba)),
                "std_prediction": float(np.std(y_pred_proba)),
                "min_prediction": float(np.min(y_pred_proba)),
                "max_prediction": float(np.max(y_pred_proba)),
                "percentiles": {
                    "p25": float(np.percentile(y_pred_proba, 25)),
                    "p50": float(np.percentile(y_pred_proba, 50)),
                    "p75": float(np.percentile(y_pred_proba, 75))
                }
            }
        
        # Class imbalance analysis
        class_distribution = y_true.value_counts(normalize=True).to_dict()
        analysis["detailed_analysis"]["class_distribution"] = class_distribution
        
        return analysis
        
    def compare_models(self, evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model evaluations."""
        
        if len(evaluations) < 2:
            return {"error": "Need at least 2 models to compare"}
            
        comparison = {
            "models": list(evaluations.keys()),
            "metrics_comparison": {},
            "rankings": {},
            "best_model": {}
        }
        
        # Extract metrics for comparison
        metrics_to_compare = ["auc", "accuracy", "f1", "precision", "recall", "log_loss"]
        
        for metric in metrics_to_compare:
            metric_values = {}
            for model_name, eval_result in evaluations.items():
                if isinstance(eval_result, dict) and "oof_metrics" in eval_result:
                    metrics = eval_result["oof_metrics"]
                else:
                    metrics = eval_result
                    
                if metric in metrics:
                    metric_values[model_name] = metrics[metric]
                    
            if metric_values:
                comparison["metrics_comparison"][metric] = metric_values
                
                # Rank models (higher is better, except for log_loss)
                if metric == "log_loss":
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
                else:
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                    
                comparison["rankings"][metric] = [model[0] for model in sorted_models]
        
        # Determine overall best model (based on AUC if available, otherwise accuracy)
        primary_metric = "auc" if "auc" in comparison["metrics_comparison"] else "accuracy"
        
        if primary_metric in comparison["rankings"]:
            best_model_name = comparison["rankings"][primary_metric][0]
            comparison["best_model"] = {
                "name": best_model_name,
                "primary_metric": primary_metric,
                "primary_score": comparison["metrics_comparison"][primary_metric][best_model_name]
            }
        
        return comparison
        
    def _is_probability(self, y_pred: pd.Series) -> bool:
        """Check if predictions are probabilities."""
        return (y_pred.min() >= 0) and (y_pred.max() <= 1) and not set(y_pred.unique()).issubset({0, 1})
        
    def _assess_performance(self, oof_metrics: Dict[str, float], 
                          cv_stats: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall performance and provide recommendations."""
        
        assessment = {
            "performance_level": "unknown",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        auc = oof_metrics.get("auc")
        if auc is not None:
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
        
        # Check stability
        if cv_stats["cv_std"] / cv_stats["cv_mean"] > 0.1:
            assessment["weaknesses"].append("High cross-validation variance")
            assessment["recommendations"].append("Model may be unstable - consider regularization")
        else:
            assessment["strengths"].append("Stable cross-validation performance")
            
        # Check balance between precision and recall
        precision = oof_metrics.get("precision", 0)
        recall = oof_metrics.get("recall", 0)
        
        if abs(precision - recall) > 0.2:
            if precision > recall:
                assessment["weaknesses"].append("Low recall relative to precision")
                assessment["recommendations"].append("Consider lowering classification threshold")
            else:
                assessment["weaknesses"].append("Low precision relative to recall") 
                assessment["recommendations"].append("Consider raising classification threshold")
        
        return assessment
