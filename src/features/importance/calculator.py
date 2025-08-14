"""Feature importance calculator using multiple algorithms."""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from core.utils import LoggerFactory


class FeatureImportanceCalculator:
    """Calculate feature importance using multiple algorithms."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("feature_importance_config", {})
        self.logger = LoggerFactory.get_logger(__name__)

        self.enabled = self.config.get("enabled", True)
        self.algorithms = self.config.get("algorithms", ["random_forest"])
        self.output_dir = Path(self.config.get("output_dir", "artifacts/feature_importance"))
        self.top_k = self.config.get("top_k_features", 20)
        self.cross_validate = self.config.get("cross_validate", True)
        self.cv_folds = self.config.get("cv_folds", 5)
        self.random_state = self.config.get("random_state", 42)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store results
        self.importance_results: Dict[str, pd.DataFrame] = {}
        self.model_scores: Dict[str, float] = {}

    def calculate_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.DataFrame]:
        """Calculate feature importance using specified algorithms."""
        if not self.enabled:
            self.logger.info("Feature importance calculation disabled")
            return {}

        self.logger.info(f"Calculating feature importance using algorithms: {self.algorithms}")

        # Store feature names
        feature_names = X.columns.tolist()

        results = {}

        for algorithm in self.algorithms:
            try:
                self.logger.info(f"Computing importance with {algorithm}")

                if algorithm == "random_forest":
                    importance_df, score = self._calculate_rf_importance(X, y, feature_names)
                elif algorithm == "xgboost":
                    importance_df, score = self._calculate_xgb_importance(X, y, feature_names)
                elif algorithm == "permutation":
                    importance_df, score = self._calculate_permutation_importance(X, y, feature_names)
                else:
                    self.logger.warning(f"Unknown algorithm: {algorithm}")
                    continue

                results[algorithm] = importance_df
                self.model_scores[algorithm] = score

                self.logger.info(f"{algorithm} - CV Score: {score:.4f}")

            except Exception as e:
                self.logger.error(f"Error calculating {algorithm} importance: {e}")
                continue

        self.importance_results = results

        # Save results
        if self.config.get("save_results", True):
            self._save_results(results)

        return results

    def _calculate_rf_importance(self, X: pd.DataFrame, y: pd.Series,
                                feature_names: List[str]) -> Tuple[pd.DataFrame, float]:
        """Calculate Random Forest feature importance."""
        params = self.config.get("algorithm_params", {}).get("random_forest", {})

        rf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=params.get("random_state", self.random_state),
            n_jobs=-1
        )

        # Fit model
        rf.fit(X, y)

        # Get importance scores
        importances = rf.feature_importances_

        # Calculate cross-validation score
        cv_score = 0.0
        if self.cross_validate:
            cv_scores = cross_val_score(
                rf, X, y, cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring="accuracy", n_jobs=-1
            )
            cv_score = cv_scores.mean()

        # Create results DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
            "rank": range(1, len(feature_names) + 1)
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df, cv_score

    def _calculate_xgb_importance(self, X: pd.DataFrame, y: pd.Series,
                                 feature_names: List[str]) -> Tuple[pd.DataFrame, float]:
        """Calculate XGBoost feature importance."""
        if xgb is None:
            raise ImportError("XGBoost not installed")

        params = self.config.get("algorithm_params", {}).get("xgboost", {})

        xgb_model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=params.get("random_state", self.random_state),
            n_jobs=-1,
            eval_metric="logloss"
        )

        # Fit model
        xgb_model.fit(X, y)

        # Get importance scores (gain-based)
        importances = xgb_model.feature_importances_

        # Calculate cross-validation score
        cv_score = 0.0
        if self.cross_validate:
            cv_scores = cross_val_score(
                xgb_model, X, y, cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring="accuracy", n_jobs=-1
            )
            cv_score = cv_scores.mean()

        # Create results DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
            "rank": range(1, len(feature_names) + 1)
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df, cv_score

    def _calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series,
                                        feature_names: List[str]) -> Tuple[pd.DataFrame, float]:
        """Calculate permutation importance."""
        params = self.config.get("algorithm_params", {}).get("permutation", {})

        # Use Random Forest as base estimator for permutation importance
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Fit base estimator
        base_estimator.fit(X, y)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            base_estimator, X, y,
            n_repeats=params.get("n_repeats", 10),
            random_state=params.get("random_state", self.random_state),
            scoring=params.get("scoring", "accuracy"),
            n_jobs=-1
        )

        # Calculate cross-validation score
        cv_score = 0.0
        if self.cross_validate:
            cv_scores = cross_val_score(
                base_estimator, X, y, cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring="accuracy", n_jobs=-1
            )
            cv_score = cv_scores.mean()

        # Create results DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std,
            "rank": range(1, len(feature_names) + 1)
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df, cv_score

    def _save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """Save feature importance results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual algorithm results
        for algorithm, df in results.items():
            filename = f"feature_importance_{algorithm}_{timestamp}.csv"
            filepath = self.output_dir / filename
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {algorithm} importance to {filepath}")

        # Save combined summary
        self._save_combined_summary(results, timestamp)

        # Save model scores
        scores_df = pd.DataFrame([
            {"algorithm": alg, "cv_score": score}
            for alg, score in self.model_scores.items()
        ])
        scores_file = self.output_dir / f"model_scores_{timestamp}.csv"
        scores_df.to_csv(scores_file, index=False)

    def _save_combined_summary(self, results: Dict[str, pd.DataFrame], timestamp: str) -> None:
        """Create and save a combined importance summary."""
        if not results:
            return

        # Get top features from each algorithm
        combined_features = set()
        for df in results.values():
            top_features = df.head(self.top_k)["feature"].tolist()
            combined_features.update(top_features)

        # Create summary DataFrame
        summary_data = []
        for feature in combined_features:
            row = {"feature": feature}
            avg_rank = 0
            count = 0

            for algorithm, df in results.items():
                feature_row = df[df["feature"] == feature]
                if not feature_row.empty:
                    importance = feature_row.iloc[0]["importance"]
                    rank = feature_row.iloc[0]["rank"]
                    row[f"{algorithm}_importance"] = importance
                    row[f"{algorithm}_rank"] = rank
                    avg_rank += rank
                    count += 1
                else:
                    row[f"{algorithm}_importance"] = 0.0
                    row[f"{algorithm}_rank"] = len(df) + 1
                    avg_rank += len(df) + 1
                    count += 1

            row["avg_rank"] = avg_rank / count if count > 0 else float("inf")
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data).sort_values("avg_rank").reset_index(drop=True)
        summary_file = self.output_dir / f"feature_importance_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Saved combined summary to {summary_file}")

    def get_top_features(self, algorithm: str = None, k: int = None) -> List[str]:
        """Get top k features from specified algorithm or combined ranking."""
        if k is None:
            k = self.top_k

        if algorithm and algorithm in self.importance_results:
            return self.importance_results[algorithm].head(k)["feature"].tolist()

        # Return combined top features if no specific algorithm
        if not self.importance_results:
            return []

        # Combine rankings from all algorithms
        all_features = set()
        for df in self.importance_results.values():
            all_features.update(df["feature"].tolist())

        feature_scores = {}
        for feature in all_features:
            total_score = 0
            for df in self.importance_results.values():
                feature_row = df[df["feature"] == feature]
                if not feature_row.empty:
                    # Use normalized importance (1/rank)
                    rank = feature_row.iloc[0]["rank"]
                    total_score += 1.0 / rank
            feature_scores[feature] = total_score

        # Sort by combined score and return top k
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in sorted_features[:k]]
