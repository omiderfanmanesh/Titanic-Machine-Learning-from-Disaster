"""Feature importance visualization module."""

from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from core.utils import LoggerFactory


class FeatureImportanceVisualizer:
    """Create visualizations for feature importance results."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("feature_importance_config", {})
        self.logger = LoggerFactory.get_logger(__name__)

        self.output_dir = Path(self.config.get("output_dir", "artifacts/feature_importance"))
        self.top_k = self.config.get("top_k_features", 20)

        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_all_plots(self, importance_results: Dict[str, pd.DataFrame],
                        model_scores: Dict[str, float]) -> None:
        """Create all visualization plots."""
        if not self.config.get("plot_importance", True):
            return

        self.logger.info("Creating feature importance visualizations")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Individual algorithm plots
        for algorithm, df in importance_results.items():
            self._plot_single_algorithm(algorithm, df, timestamp)

        # Comparison plots
        if len(importance_results) > 1:
            self._plot_algorithm_comparison(importance_results, timestamp)
            self._plot_feature_ranking_heatmap(importance_results, timestamp)

        # Model performance comparison
        self._plot_model_performance(model_scores, timestamp)

        self.logger.info(f"All plots saved to {self.output_dir}")

    def _plot_single_algorithm(self, algorithm: str, df: pd.DataFrame, timestamp: str) -> None:
        """Create bar plot for single algorithm importance."""
        top_features = df.head(self.top_k)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features["importance"])

        # Color bars by importance value - fix colormap reference
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {self.top_k} Features - {algorithm.title()} Importance")
        plt.gca().invert_yaxis()

        # Add value labels on bars - robust float conversion
        for i, (idx, row) in enumerate(top_features.iterrows()):
            try:
                importance_val = float(row["importance"])
            except Exception:
                importance_val = 0.0
            plt.text(importance_val + 0.001, i, f'{importance_val:.3f}',
                    va='center', fontsize=9)

        plt.tight_layout()
        filename = f"importance_{algorithm}_{timestamp}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_algorithm_comparison(self, importance_results: Dict[str, pd.DataFrame], 
                                  timestamp: str) -> None:
        """Create comparison plot across algorithms."""
        # Get features that appear in top K for any algorithm
        all_top_features = set()
        for df in importance_results.values():
            top_features = df.head(self.top_k)["feature"].tolist()
            all_top_features.update(top_features)

        # Create comparison DataFrame
        comparison_data = []
        for feature in all_top_features:
            row = {"feature": feature}
            for algorithm, df in importance_results.items():
                feature_row = df[df["feature"] == feature]
                if not feature_row.empty:
                    row[algorithm] = feature_row.iloc[0]["importance"]
                else:
                    row[algorithm] = 0.0
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Calculate average importance for sorting
        algorithm_cols = [col for col in comparison_df.columns if col != "feature"]
        comparison_df["avg_importance"] = comparison_df[algorithm_cols].mean(axis=1)
        comparison_df = comparison_df.sort_values("avg_importance", ascending=True)

        # Create grouped bar plot - fix figure size calculation
        fig, ax = plt.subplots(figsize=(14, max(8, int(len(comparison_df) * 0.4))))

        x_pos = np.arange(len(comparison_df))
        width = 0.8 / len(algorithm_cols)

        for i, algorithm in enumerate(algorithm_cols):
            values = comparison_df[algorithm].values
            ax.barh(x_pos + i * width, values, width, label=algorithm.title(), alpha=0.8)

        ax.set_yticks(x_pos + width * (len(algorithm_cols) - 1) / 2)
        ax.set_yticklabels(comparison_df["feature"])
        ax.set_xlabel("Feature Importance")
        ax.set_title("Feature Importance Comparison Across Algorithms")
        ax.legend()

        plt.tight_layout()
        filename = f"importance_comparison_{timestamp}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_feature_ranking_heatmap(self, importance_results: Dict[str, pd.DataFrame],
                                     timestamp: str) -> None:
        """Create heatmap of feature rankings across algorithms."""
        # Get top features from all algorithms
        all_top_features = set()
        for df in importance_results.values():
            top_features = df.head(self.top_k)["feature"].tolist()
            all_top_features.update(top_features)

        # Create ranking matrix
        ranking_data = []
        for feature in all_top_features:
            row = {"feature": feature}
            for algorithm, df in importance_results.items():
                feature_row = df[df["feature"] == feature]
                if not feature_row.empty:
                    row[algorithm] = feature_row.iloc[0]["rank"]
                else:
                    row[algorithm] = len(df) + 1  # Worst possible rank
            ranking_data.append(row)

        ranking_df = pd.DataFrame(ranking_data)

        # Calculate average rank for sorting
        algorithm_cols = [col for col in ranking_df.columns if col != "feature"]
        ranking_df["avg_rank"] = ranking_df[algorithm_cols].mean(axis=1)
        ranking_df = ranking_df.sort_values("avg_rank").head(self.top_k)

        # Create heatmap
        plt.figure(figsize=(max(8, len(algorithm_cols) * 2), max(8, int(len(ranking_df) * 0.4))))

        heatmap_data = ranking_df.set_index("feature")[algorithm_cols]

        # Invert colormap so lower ranks (better) are darker
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Feature Rank'})

        plt.title("Feature Ranking Heatmap Across Algorithms\n(Lower numbers = higher importance)")
        plt.xlabel("Algorithm")
        plt.ylabel("Feature")
        plt.tight_layout()

        filename = f"ranking_heatmap_{timestamp}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_model_performance(self, model_scores: Dict[str, float], timestamp: str) -> None:
        """Create bar plot of model cross-validation scores."""
        if not model_scores:
            return

        algorithms = list(model_scores.keys())
        scores = list(model_scores.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(algorithms, scores, alpha=0.8)

        # Color bars by performance - fix colormap reference
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(algorithms)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.ylabel("Cross-Validation Accuracy")
        plt.title("Model Performance Comparison\n(5-Fold Cross-Validation)")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = f"model_performance_{timestamp}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def create_summary_report(self, importance_results: Dict[str, pd.DataFrame],
                            model_scores: Dict[str, float], timestamp: str = None) -> str:
        """Create a text summary report."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_lines = []
        report_lines.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Model performance summary
        if model_scores:
            report_lines.append("MODEL PERFORMANCE (Cross-Validation Accuracy):")
            report_lines.append("-" * 45)
            for algorithm, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"{algorithm.ljust(15)}: {score:.4f}")
            report_lines.append("")

        # Top features by algorithm
        for algorithm, df in importance_results.items():
            report_lines.append(f"TOP 10 FEATURES - {algorithm.upper()}:")
            report_lines.append("-" * 35)
            for idx, row in df.head(10).iterrows():
                report_lines.append(f"{str(row['rank']).rjust(2)}. {row['feature'].ljust(20)} ({row['importance']:.4f})")
            report_lines.append("")

        # Combined ranking
        if len(importance_results) > 1:
            # Calculate consensus ranking
            all_features = set()
            for df in importance_results.values():
                all_features.update(df["feature"].tolist())

            feature_consensus = {}
            for feature in all_features:
                total_inv_rank = 0
                for df in importance_results.values():
                    feature_row = df[df["feature"] == feature]
                    if not feature_row.empty:
                        rank = feature_row.iloc[0]["rank"]
                        total_inv_rank += 1.0 / rank
                feature_consensus[feature] = total_inv_rank

            sorted_consensus = sorted(feature_consensus.items(), key=lambda x: x[1], reverse=True)

            report_lines.append("CONSENSUS TOP 10 FEATURES (Combined Ranking):")
            report_lines.append("-" * 45)
            for i, (feature, score) in enumerate(sorted_consensus[:10], 1):
                report_lines.append(f"{str(i).rjust(2)}. {feature.ljust(20)} (score: {score:.4f})")

        report_content = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / f"feature_importance_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(report_content)

        self.logger.info(f"Summary report saved to {report_file}")
        return report_content
