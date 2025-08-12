"""Command-line interface for Titanic ML pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import subprocess

from core.utils import (
    ExperimentConfig,
    DataConfig,
    InferenceConfig,
    LoggerFactory,
    PathManager,
    SeedManager,
    ConfigManager,
)
from data.loader import TitanicDataLoader
try:
    from data.validate import TitanicDataValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    TitanicDataValidator = None  # Ensure it's defined in the except block
from features.build import create_feature_builder
from modeling.model_registry import ModelRegistry
from modeling.trainers import TitanicTrainer
from eval.evaluator import TitanicEvaluator
from infer.predictor import create_predictor, ModelLoader
from submit.build_submission import TitanicSubmissionBuilder


# Global objects
path_manager = PathManager()
config_manager = ConfigManager(path_manager.config_dir)
logger = LoggerFactory.get_logger("titanic_ml.cli")


@click.group()
@click.option("--config-dir", type=click.Path(exists=True),
              help="Configuration directory path")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(config_dir: Optional[str], debug: bool):
    """Titanic ML Pipeline - Professional ML pipeline for Kaggle competition."""
    global path_manager, config_manager

    if config_dir:
        path_manager.config_dir = Path(config_dir)
        config_manager = ConfigManager(path_manager.config_dir)

    # Ensure directories exist
    path_manager.ensure_directories()

    if debug:
        logger.info("Debug mode enabled")


@cli.command()
@click.option("--competition", default="titanic", help="Kaggle competition name")
@click.option("--output-dir", type=click.Path(), help="Output directory for data")
def download(competition: str, output_dir: Optional[str]):
    """Download data from Kaggle competition."""

    if output_dir:
        data_dir = Path(output_dir)
    else:
        data_dir = path_manager.data_dir / "raw"

    try:
        from data.loader import KaggleDataLoader
        loader = KaggleDataLoader(competition, path_manager.data_dir)
        loader.download_competition_data()

        click.echo(f"✅ Data downloaded to {data_dir}")

    except ImportError:
        click.echo("❌ Kaggle API not installed. Install with: pip install kaggle")
        click.echo("   Also ensure ~/.kaggle/kaggle.json contains your API credentials")
    except Exception as e:
        click.echo(f"❌ Download failed: {e}")


@cli.command()
@click.option("--config", "-c", default="configs/data.yaml", help="Data configuration file")
def validate(config: str) -> None:
    """Validate data quality and check for leakage."""
    if not VALIDATION_AVAILABLE:
        logger.error("Data validation requires 'pandera' package. Install with: pip install pandera")
        return

    logger.info("Starting data validation...")

    # Load configuration
    path_manager = PathManager()
    config_manager = ConfigManager(path_manager.config_dir)
    data_config = config_manager.load_config(config)

    # Load data
    loader = TitanicDataLoader(
        train_file=data_config["train_path"],
        test_file=data_config["test_path"]
    )
    train_df, test_df = loader.load()

    # Validate data
    validator = TitanicDataValidator()

    try:
        validator.validate_train(train_df)
        logger.info("✅ Training data validation passed")

        validator.validate_test(test_df)
        logger.info("✅ Test data validation passed")

        validator.validate_consistency(train_df, test_df)
        logger.info("✅ Data consistency validation passed")

        logger.info("🎉 All data validation checks passed!")

    except Exception as e:
        logger.error(f"❌ Data validation failed: {e}")
        raise


@cli.command()
@click.option("--experiment-config", default="experiment", help="Experiment configuration name")
@click.option("--data-config", default="data", help="Data configuration name")
def features(experiment_config: str, data_config: str):
    """Build features for training and test data."""

    try:
        # Load configurations
        exp_config = config_manager.load_config(experiment_config)
        data_config_dict = config_manager.load_config(data_config)

        experiment_cfg = ExperimentConfig(**exp_config)
        data_cfg = DataConfig(**data_config_dict)

        # Set seed
        SeedManager.set_seed(experiment_cfg.seed)

        # Load data
        # Resolve paths (fallback if raw paths missing)
        train_path = Path(data_cfg.train_path)
        test_path = Path(data_cfg.test_path)
        if not train_path.exists():
            alt_train = Path('data') / train_path.name
            if alt_train.exists():
                train_path = alt_train
        if not test_path.exists():
            alt_test = Path('data') / test_path.name
            if alt_test.exists():
                test_path = alt_test

        loader = TitanicDataLoader(train_path, test_path)
        train_df, test_df = loader.load()

        # Apply debug mode if needed
        if experiment_cfg.debug_mode and experiment_cfg.debug_n_rows:
            train_df = train_df.head(experiment_cfg.debug_n_rows)
            click.echo(f"🐛 Debug mode: Using {len(train_df)} training samples")

        # Build features
        feature_config = {
            "add_family_features": True,
            "add_title_features": True,
            "add_deck_features": True,
            "add_ticket_features": True,
            "transform_fare": True,
            "add_missing_indicators": True
        }

        feature_builder = create_feature_builder(feature_config, debug=experiment_cfg.debug_mode)

        # Fit on training data
        X_train = train_df.drop(columns=[data_cfg.target_column])
        y_train = train_df[data_cfg.target_column]

        feature_builder.fit(X_train, y_train)

        # Transform both train and test
        X_train_processed = feature_builder.transform(X_train)
        X_test_processed = feature_builder.transform(test_df)

        # Save processed data
        processed_dir = path_manager.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)

        train_processed_path = processed_dir / "train_features.csv"
        test_processed_path = processed_dir / "test_features.csv"

        # Add target back to training data
        X_train_processed[data_cfg.target_column] = y_train

        X_train_processed.to_csv(train_processed_path, index=False)
        X_test_processed.to_csv(test_processed_path, index=False)

        click.echo(f"✅ Features built and saved:")
        click.echo(f"   📁 Train: {train_processed_path} ({X_train_processed.shape})")
        click.echo(f"   📁 Test: {test_processed_path} ({X_test_processed.shape})")

    except Exception as e:
        click.echo(f"❌ Feature building failed: {e}")
        raise


@cli.command()
@click.option("--experiment-config", default="experiment", help="Experiment configuration name")
@click.option("--data-config", default="data", help="Data configuration name")
def train(experiment_config: str, data_config: str):
    """Train model with cross-validation."""

    try:
        # Load configurations
        exp_config = config_manager.load_config(experiment_config)
        data_config_dict = config_manager.load_config(data_config)

        experiment_cfg = ExperimentConfig(**exp_config)
        data_cfg = DataConfig(**data_config_dict)

        # Set seed
        SeedManager.set_seed(experiment_cfg.seed)

        # Load processed data
        processed_dir = path_manager.data_dir / "processed"
        train_path = processed_dir / "train_features.csv"

        if not train_path.exists():
            click.echo("❌ Processed training data not found. Run 'features' command first.")
            return

        train_df = pd.read_csv(train_path)

        # Apply debug mode if needed
        if experiment_cfg.debug_mode and experiment_cfg.debug_n_rows:
            train_df = train_df.head(experiment_cfg.debug_n_rows)
            click.echo(f"🐛 Debug mode: Using {len(train_df)} training samples")

        # Prepare data
        # Drop target and ID column for modeling
        drop_for_training = [data_cfg.target_column]
        if data_cfg.id_column in train_df.columns:
            drop_for_training.append(data_cfg.id_column)
        X = train_df.drop(columns=drop_for_training)
        # Drop raw text / identifier columns not yet encoded if present
        drop_cols = [c for c in ['Name', 'Ticket', 'Cabin'] if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        y = train_df[data_cfg.target_column]

        # Create model
        model_config = {
            "model_name": experiment_cfg.model_name,
            "model_params": experiment_cfg.model_params
        }

        registry = ModelRegistry()
        model_wrapper = registry.create_model(experiment_cfg.model_name, experiment_cfg.model_params)
        estimator = model_wrapper.build(model_config)

        # Create trainer
        trainer_config = {
            "strategy": experiment_cfg.cv_strategy,
            "n_folds": experiment_cfg.cv_folds,
            "shuffle": experiment_cfg.cv_shuffle,
            "random_state": experiment_cfg.cv_random_state,
            # Include model info for downstream artifact naming
            "model_name": experiment_cfg.model_name,
            "model_params": experiment_cfg.model_params
        }

        trainer = TitanicTrainer(trainer_config)

        # Train with cross-validation
        click.echo(f"🚀 Starting {experiment_cfg.cv_folds}-fold cross-validation...")
        click.echo(f"   Model: {experiment_cfg.model_name}")
        click.echo(f"   Strategy: {experiment_cfg.cv_strategy}")

        cv_results = trainer.cross_validate(estimator, X, y, trainer_config)

        # Display results
        scores = cv_results["cv_scores"]
        click.echo(f"✅ Training completed!")
        click.echo(f"   📊 CV Score: {scores['mean_score']:.4f} ± {scores['std_score']:.4f}")
        click.echo(f"   📊 OOF Score: {scores['oof_score']:.4f}")
        click.echo(f"   📁 Artifacts: {cv_results['run_dir']}")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        raise


@cli.command()
@click.option("--run-dir", type=click.Path(exists=True), required=True,
              help="Training run directory")
def evaluate(run_dir: str):
    """Evaluate cross-validation results."""
    try:
        # Local imports used only here
        import numpy as np

        run_path = Path(run_dir)

        # Load OOF predictions and scores
        oof_path = run_path / "oof_predictions.csv"
        scores_path = run_path / "cv_scores.json"

        if not oof_path.exists() or not scores_path.exists():
            click.echo("❌ Required files not found in run directory")
            return

        # Load data
        oof_df = pd.read_csv(oof_path)
        with open(scores_path, "r") as f:
            scores_data = json.load(f)

        # Create evaluator
        evaluator = TitanicEvaluator()

        # Load inference configuration
        inference_cfg = config_manager.load_config("inference")
        th_cfg = inference_cfg.get("threshold", {}) or {}

        # Perform evaluation (now returns optimal_thresholds + per-method metrics if enabled)
        evaluation_results = evaluator.evaluate_cv(
            oof_df["prediction"],
            oof_df["target"],
            scores_data["fold_scores"],
            config=inference_cfg,
        )

        # Helper for safe float formatting
        def fmt4(x):
            return f"{x:.4f}" if isinstance(x, (int, float, np.floating)) and x is not None else "N/A"

        oof_metrics = evaluation_results.get("oof_metrics", {}) or {}
        cv_stats = evaluation_results.get("cv_statistics", {}) or {}
        stability = evaluation_results.get("stability", {}) or {}

        # --- Base metrics ---
        click.echo("📊 Evaluation Results")
        click.echo(f"   AUC: {fmt4(oof_metrics.get('auc'))}")
        click.echo(f"   Accuracy: {fmt4(oof_metrics.get('accuracy'))}")
        click.echo(f"   F1 Score: {fmt4(oof_metrics.get('f1'))}")

        # --- Optimized-threshold metrics (from evaluator) ---
        if th_cfg.get("optimizer", False) and th_cfg.get("print", True):
            opt_thr = oof_metrics.get("optimal_threshold")
            opt_scr = oof_metrics.get("optimal_threshold_score")
            has_point_opt = any(k in oof_metrics for k in ("f1_opt", "accuracy_opt", "precision_opt", "recall_opt"))

            if opt_thr is not None or has_point_opt:
                click.echo("\n🎯 Optimized Threshold (per config)")
                if opt_thr is not None:
                    click.echo(f"   Threshold: {fmt4(opt_thr)}")
                if opt_scr is not None:
                    click.echo(f"   Objective score: {fmt4(opt_scr)}")

                if has_point_opt:
                    click.echo(f"   Accuracy@opt:  {fmt4(oof_metrics.get('accuracy_opt'))}")
                    click.echo(f"   Precision@opt: {fmt4(oof_metrics.get('precision_opt'))}")
                    click.echo(f"   Recall@opt:    {fmt4(oof_metrics.get('recall_opt'))}")
                    click.echo(f"   F1@opt:        {fmt4(oof_metrics.get('f1_opt'))}")

                    tn = oof_metrics.get("true_negatives_opt")
                    fp = oof_metrics.get("false_positives_opt")
                    fn = oof_metrics.get("false_negatives_opt")
                    tp = oof_metrics.get("true_positives_opt")
                    if None not in (tn, fp, fn, tp):
                        click.echo(f"   CM@opt (tn, fp, fn, tp): {tn}, {fp}, {fn}, {tp}")

        # --- CV stats ---
        click.echo("\n📈 Cross-Validation Statistics")
        click.echo(f"   Mean:  {fmt4(cv_stats.get('cv_mean'))}")
        click.echo(f"   Std:   {fmt4(cv_stats.get('cv_std'))}")
        rng = None
        if "cv_max" in cv_stats and "cv_min" in cv_stats:
            rng = (cv_stats["cv_max"] - cv_stats["cv_min"])
        click.echo(f"   Range: {fmt4(rng)}")

        # --- Stability ---
        click.echo("\n🎯 Model Stability")
        is_stable = stability.get("is_stable")
        click.echo(f"   Stable: {'✅ Yes' if is_stable else '❌ No'}")
        click.echo(f"   CV of Variation: {fmt4(stability.get('coefficient_of_variation'))}")

        # --- Per-method thresholds & point-metrics (already computed in evaluator) ---
        if th_cfg.get("optimizer", False) and th_cfg.get("print", True):
            all_opts = evaluation_results.get("optimal_thresholds") or {}
            if all_opts:
                click.echo("\n🎯 Optimal Thresholds (by method)")
                for method, threshold in all_opts.items():
                    click.echo(f"   {method.capitalize()}: {fmt4(threshold)}")

            rows = evaluation_results.get("threshold_method_metrics") or []
            if rows:
                click.echo("\n📌 Metrics at each method's optimal threshold")
                for r in rows:
                    click.echo(
                        f"   {r['method'].capitalize():9s} thr={fmt4(r['threshold'])}  "
                        f"Acc={fmt4(r['accuracy'])}  Prec={fmt4(r['precision'])}  "
                        f"Rec={fmt4(r['recall'])}  F1={fmt4(r['f1'])}  "
                        f"CM=({int(r['tn'])},{int(r['fp'])},{int(r['fn'])},{int(r['tp'])})"
                    )

                # Save per-method report CSV
                report_path_cfg = th_cfg.get("report_path")
                report_path = (run_path / report_path_cfg) if report_path_cfg else (run_path / "threshold_report.csv")
                report_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(report_path, index=False)
                evaluation_results["threshold_report_path"] = str(report_path)
                click.echo(f"\n📝 Threshold report saved to {report_path}")

            # Persist chosen threshold
            chosen_thr = evaluation_results.get("chosen_threshold")
            chosen_method = evaluation_results.get("chosen_method")
            if chosen_thr is not None:
                chosen_file = run_path / "best_threshold.txt"
                chosen_file.write_text(f"{float(chosen_thr):.6f}\n")
                evaluation_results["best_threshold_file"] = str(chosen_file)
                click.echo(f"🔖 Chosen threshold ({chosen_method}) saved to {chosen_file}")

        # --- Optional: per-fold threshold analysis (if fold column exists) ---
        if th_cfg.get("optimizer", False) and "fold" in oof_df.columns:
            click.echo("\n🧩 Per-fold threshold analysis")
            pf = evaluator.per_fold_threshold_analysis(oof_df)
            evaluation_results["per_fold_thresholds"] = pf.get("per_fold_rows", [])
            evaluation_results["per_fold_thresholds_summary"] = pf.get("summary", [])

            if evaluation_results["per_fold_thresholds"]:
                pf_path = run_path / "thresholds_per_fold.csv"
                pf_sum_path = run_path / "thresholds_per_fold_summary.csv"
                pd.DataFrame(evaluation_results["per_fold_thresholds"]).to_csv(pf_path, index=False)
                pd.DataFrame(evaluation_results["per_fold_thresholds_summary"]).to_csv(pf_sum_path, index=False)
                evaluation_results["per_fold_thresholds_path"] = str(pf_path)
                evaluation_results["per_fold_thresholds_summary_path"] = str(pf_sum_path)
                click.echo(f"   • Saved per-fold thresholds to {pf_path}")
                click.echo(f"   • Saved per-fold summary to {pf_sum_path}")
            else:
                click.echo("   • No eligible folds for per-fold analysis.")

        # Save detailed evaluation (full JSON with everything embedded)
        eval_path = run_path / "evaluation_report.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        click.echo(f"\n📄 Detailed evaluation saved to {eval_path}")

    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}")

@cli.command()
@click.option("--run-dir", type=click.Path(exists=True), required=True,
              help="Training run directory")
@click.option("--inference-config", default="inference", help="Inference configuration name")
@click.option("--output-path", type=click.Path(), help="Output path for predictions")
@click.option("--threshold-file", type=click.Path(exists=True),
              help="Path to a file with a single float threshold (e.g., best_threshold.txt)")
@click.option("--threshold", "threshold_value", type=float,
              help="Numeric threshold override (e.g., 0.61)")
def predict(run_dir: str, inference_config: str, output_path: Optional[str],
            threshold_file: Optional[str], threshold_value: Optional[float]):
    """Generate predictions on test data."""
    try:
        from pathlib import Path
        import numpy as np
        import pandas as pd  # <-- ensure imported ONCE at top of function (or at module level)

        run_path = Path(run_dir)

        # Load inference config
        inference_cfg = config_manager.load_config(inference_config)
        th_cfg = inference_cfg.get("threshold", {}) or {}

        # Make run_dir available to the predictor for auto-discovery of artifacts
        inference_cfg["run_dir"] = str(run_path)

        # Ensure model paths
        if not inference_cfg.get("model_paths"):
            fold_models = sorted(run_path.glob("fold_*_model.joblib"))
            inference_cfg["model_paths"] = [str(p) for p in fold_models]

        # --- Resolve threshold source (file > numeric > best_threshold.txt > report > config.value) ---
        used_src = "config"
        if threshold_file:
            th_cfg["file"] = str(threshold_file)
            used_src = "cli:file"
        elif threshold_value is not None:
            th_cfg["value"] = float(threshold_value)
            th_cfg.pop("file", None)
            used_src = "cli:value"
        else:
            auto_file = run_path / "best_threshold.txt"
            if auto_file.exists():
                th_cfg["file"] = str(auto_file)
                used_src = "auto:best_threshold.txt"
            else:
                # optional: read from threshold report in run_dir
                report_file = run_path / "threshold_report.csv"
                if report_file.exists():
                    try:
                        report_df = pd.read_csv(report_file)
                        method = (th_cfg.get("method") or "accuracy").lower()
                        if {"method", "threshold"}.issubset(report_df.columns):
                            row = report_df.loc[report_df["method"].str.lower() == method]
                            if not row.empty:
                                th_cfg["value"] = float(row.iloc[0]["threshold"])
                                used_src = f"auto:report[{method}]"
                    except Exception as e:
                        click.echo(f"⚠️ Could not read threshold from report: {e}")

        inference_cfg["threshold"] = th_cfg

        # Load processed test data
        processed_dir = path_manager.data_dir / "processed"
        test_path = processed_dir / "test_features.csv"
        if not test_path.exists():
            click.echo("❌ Processed test data not found. Run 'features' command first.")
            return
        test_df = pd.read_csv(test_path)
        click.echo(f"📥 Loaded test data: {test_df.shape}")

        # Prepare model input
        drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in test_df.columns]
        test_model_df = test_df.drop(columns=drop_cols) if drop_cols else test_df.copy()
        if "PassengerId" in test_model_df.columns:
            test_model_df = test_model_df.set_index("PassengerId")

        # Load models
        model_loader = ModelLoader()
        models = model_loader.load_fold_models(run_dir)
        click.echo(f"🔄 Loaded {len(models)} fold models")

        # Predict (binary + proba)
        predictor = create_predictor(inference_cfg)
        click.echo("🔮 Generating predictions...")
        predictions = predictor.predict(test_model_df, models, inference_cfg)

        # Save predictions
        pred_path = Path(output_path) if output_path else (run_path / "predictions.csv")
        predictions.to_csv(pred_path, index=False)
        click.echo(f"✅ Predictions saved to {pred_path}")

        # Report threshold actually used
        try:
            used_thr = predictor._resolve_threshold(inference_cfg)
            click.echo(f"   📊 Threshold source: {used_src} | value: {used_thr:.4f}")
        except Exception:
            pass

        # Distributions
        dist_bin = predictions["prediction"].astype(float)
        dist_proba = predictions["prediction_proba"].astype(float)
        click.echo("   📊 Binary prediction distribution:")
        click.echo(f"      Mean: {dist_bin.mean():.3f} | Std: {dist_bin.std():.3f} | "
                   f"Min: {dist_bin.min():.3f} | Max: {dist_bin.max():.3f}")
        click.echo("   📊 Probability distribution:")
        click.echo(f"      Mean: {dist_proba.mean():.3f} | Std: {dist_proba.std():.3f} | "
                   f"Min: {dist_proba.min():.3f} | Max: {dist_proba.max():.3f}")

    except Exception as e:
        click.echo(f"❌ Prediction failed: {e}")

@cli.command()
@click.option("--experiment-config", default="experiment", help="Experiment configuration name")
@click.option("--data-config", default="data", help="Data configuration name")
@click.option("--inference-config", default="inference", help="Inference configuration name")
@click.option("--competition", default="titanic", show_default=True, help="Kaggle competition slug")
@click.option("--remote", is_flag=True, help="If set, perform remote Kaggle submission")
@click.option("--message", "-m", default="Auto pipeline run", show_default=True, help="Submission message")
def autopipeline(experiment_config: str, data_config: str, inference_config: str, competition: str, remote: bool, message: str):
    """Run end-to-end: features -> train -> predict -> submit (optional remote)."""
    try:
        click.echo("🛠  Building features...")
        ctx = click.get_current_context()
        ctx.invoke(features, experiment_config=experiment_config, data_config=data_config)

        click.echo("🧪 Training model...")
        # Capture stdout from train by invoking and parsing run_dir from artifacts listing afterwards
        before = set(p.name for p in path_manager.artifacts_dir.glob('20*'))
        ctx.invoke(train, experiment_config=experiment_config, data_config=data_config)
        after = sorted([p for p in path_manager.artifacts_dir.glob('20*') if p.name not in before], key=lambda x: x.stat().st_mtime, reverse=True)
        if not after:
            click.echo("❌ Could not determine training run directory")
            return
        run_dir = str(after[0])
        click.echo(f"📁 Using run_dir: {run_dir}")

        click.echo("🔮 Predicting on test set...")
        ctx.invoke(predict, run_dir=run_dir, inference_config=inference_config, output_path=None)
        pred_path = Path(run_dir) / 'predictions.csv'
        if not pred_path.exists():
            click.echo("❌ Predictions file not found; aborting submission step")
            return

        click.echo("📄 Building submission file...")
        # Local submission first
        submission_args = {
            'predictions_path': str(pred_path),
            'output_path': None,
            'threshold': 0.5,
            'competition': competition,
            'remote': remote,
            'message': message
        }
        ctx.invoke(submit, **submission_args)
        click.echo("✅ Auto pipeline completed")
    except Exception as e:
        click.echo(f"❌ Auto pipeline failed: {e}")


@cli.command()
@click.option("--predictions-path", type=click.Path(exists=True), required=True,
              help="Path to predictions CSV (from the predict step)")
@click.option("--output-path", type=click.Path(), help="Output path for submission CSV")
@click.option("--threshold", type=float, default=None,
              help="Only used if predictions.csv does NOT have a binary 'prediction' column")
@click.option("--competition", default="titanic", show_default=True, help="Kaggle competition slug")
@click.option("--remote", is_flag=True, help="If set, submit to Kaggle after building the local submission file")
@click.option("--descriptive/--no-descriptive", default=True, show_default=True,
              help="Use descriptive filename with model+scores+timestamp (short form)")
@click.option("--message", "-m", default="Automated submission", show_default=True, help="Kaggle submission message")
def submit(predictions_path: str, output_path: Optional[str], threshold: Optional[float],
           competition: str, remote: bool, descriptive: bool, message: str):
    """Build (and optionally remotely submit) a Kaggle submission file from predictions.csv."""
    try:
        import json
        import shutil
        import subprocess
        from pathlib import Path

        import pandas as pd

        pred_path = Path(predictions_path)
        run_dir = pred_path.parent

        # 1) Load predictions
        df = pd.read_csv(predictions_path)

        # 2) Decide labels without re-deriving thresholds
        used_src = "predictions.csv"
        if "prediction" in df.columns and set(pd.unique(df["prediction"])) <= {0, 1}:
            y = df["prediction"].astype(int)
        elif "prediction_proba" in df.columns:
            # Only threshold if we must (no binary column present)
            thr = None
            if threshold is not None:
                thr = float(threshold)
                used_src = "cli:threshold"
            else:
                # Best effort: try the run artifact (but ONLY because we must convert proba)
                best_thr_file = run_dir / "best_threshold.txt"
                if best_thr_file.exists():
                    try:
                        thr = float(best_thr_file.read_text().strip())
                        used_src = "artifact:best_threshold.txt"
                    except Exception:
                        thr = None
                if thr is None:
                    # fall back to 0.5 if nothing else
                    thr = 0.5
                    used_src = "default:0.5"

            y = (df["prediction_proba"].astype(float) >= thr).astype(int)
        else:
            raise ValueError(
                "Predictions file must contain either a binary 'prediction' column "
                "or a 'prediction_proba' column."
            )

        # 3) Build submission (PassengerId + Survived)
        if "PassengerId" in df.columns:
            pid = df["PassengerId"].astype(int)
        elif df.index.name == "PassengerId":
            pid = df.index.to_series().astype(int)
        else:
            raise ValueError("Could not find PassengerId column or index in predictions.")

        submission = pd.DataFrame({"PassengerId": pid, "Survived": y.astype(int)})

        # 4) Validate (basic checks)
        if submission.isna().any().any():
            click.echo("❌ Submission has NaNs.")
            return
        if submission["PassengerId"].duplicated().any():
            click.echo("❌ Duplicate PassengerId in submission.")
            return

        # 5) Decide save path
        if output_path:
            sub_path = Path(output_path)
        else:
            if descriptive:
                # Compact descriptive name using available artifacts (best-effort)
                model_name = "unknown"
                mean_score = None
                oof_score = None
                try:
                    tc_path = run_dir / "training_config.json"
                    if tc_path.exists():
                        with open(tc_path, "r") as f:
                            tc_data = json.load(f)
                            model_name = tc_data.get("model_name", model_name)
                    scores_path = run_dir / "cv_scores.json"
                    if scores_path.exists():
                        with open(scores_path, "r") as f:
                            scores_data = json.load(f)
                            mean_score = scores_data.get("mean_score")
                            oof_score = scores_data.get("oof_score")
                except Exception:
                    pass

                def _fmt(val, digits):
                    try:
                        return f"{val:.{digits}f}".replace(".", "")
                    except Exception:
                        return "na"

                parts = ["sub", str(model_name).replace(" ", "-").replace("/", "-")]
                if mean_score is not None:
                    parts.append(f"cv{_fmt(mean_score,4)}")
                if oof_score is not None:
                    parts.append(f"oof{_fmt(oof_score,4)}")

                # If we used a threshold source during this call (only when proba->binary), annotate it
                if used_src != "predictions.csv":
                    # include short form of source and value if we have it
                    parts.append(f"thrsrc-{used_src.split(':',1)[0]}")

                parts.append(run_dir.name)
                filename = "_".join(parts) + ".csv"
                sub_path = run_dir / filename

                # Remove legacy generic if present
                generic = run_dir / "submission.csv"
                if generic.exists():
                    try:
                        generic.unlink()
                    except Exception:
                        pass
            else:
                sub_path = run_dir / "submission.csv"

        # 6) Save submission
        submission.to_csv(sub_path, index=False)

        # 7) Summary
        click.echo(f"✅ Submission created: {sub_path}")
        click.echo(f"   📊 Samples: {len(submission)}")
        click.echo(f"   📊 Positive rate: {submission['Survived'].mean():.3f}")
        if used_src != "predictions.csv":
            click.echo(f"   📊 Labels derived from proba using: {used_src}")
        else:
            click.echo(f"   ✅ Using binary predictions already present in predictions.csv")

        # 8) Optional remote submit
        if remote:
            click.echo("🌐 Remote submission requested -- preparing Kaggle submission...")
            kaggle_cli = shutil.which("kaggle")
            if kaggle_cli is None:
                click.echo("❌ Kaggle CLI not found. Install with: pip install kaggle")
                click.echo("   Then place your API token at ~/.kaggle/kaggle.json (chmod 600). Skipping remote submit.")
                return

            creds_path = Path.home() / ".kaggle" / "kaggle.json"
            if not creds_path.exists():
                click.echo("❌ Kaggle credentials file ~/.kaggle/kaggle.json not found. Skipping remote submit.")
                return
            if oct(creds_path.stat().st_mode)[-3:] not in {"600", "640"}:
                click.echo("⚠️  Warning: kaggle.json permissions should be 600 (chmod 600 ~/.kaggle/kaggle.json)")

            cmd = ["kaggle", "competitions", "submit", "-c", competition, "-f", str(sub_path), "-m", message]
            click.echo(f"🚀 Running: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    click.echo("❌ Kaggle submission failed:")
                    click.echo(result.stderr.strip())
                else:
                    click.echo("🎉 Kaggle submission uploaded successfully!")
                    if result.stdout.strip():
                        click.echo(result.stdout.strip())
            except Exception as sub_e:
                click.echo(f"❌ Error invoking Kaggle CLI: {sub_e}")

    except Exception as e:
        click.echo(f"❌ Submission creation failed: {e}")

@cli.command()
def info():
    """Show pipeline information and available models."""

    click.echo("🚢 Titanic ML Pipeline")
    click.echo(f"   📁 Project root: {path_manager.project_root}")
    click.echo(f"   📁 Config dir: {path_manager.config_dir}")
    click.echo(f"   📁 Data dir: {path_manager.data_dir}")
    click.echo(f"   📁 Artifacts dir: {path_manager.artifacts_dir}")

    click.echo(f"\n🤖 Available models:")
    registry = ModelRegistry()
    available_models = registry.get_available_models()
    for model in sorted(available_models):
        click.echo(f"   - {model}")

    click.echo(f"\n📋 Configuration files:")
    if path_manager.config_dir.exists():
        config_files = list(path_manager.config_dir.glob("*.yaml"))
        for config_file in sorted(config_files):
            click.echo(f"   - {config_file.name}")
    else:
        click.echo("   No config directory found")


@cli.command()
@click.option("--config-name", required=True, help="Configuration name to create")
@click.option("--template", type=click.Choice(["experiment", "data", "inference"]),
              default="experiment", help="Configuration template")
def create_config(config_name: str, template: str):
    """Create configuration file from template."""

    templates = {
        "experiment": {
            "name": "titanic_experiment",
            "seed": 42,
            "debug_mode": False,
            "debug_n_rows": None,
            "model_name": "random_forest",
            "model_params": {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": 42
            },
            "cv_folds": 5,
            "cv_strategy": "stratified",
            "cv_shuffle": True,
            "cv_random_state": 42,
            "early_stopping_rounds": None,
            "logging_level": "INFO"
        },
        "data": {
            "train_path": "data/raw/train.csv",
            "test_path": "data/raw/test.csv",
            "target_column": "Survived",
            "id_column": "PassengerId",
            "task_type": "binary",
            "required_columns": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age"],
            "numeric_columns": ["Age", "SibSp", "Parch", "Fare"],
            "categorical_columns": ["Sex", "Embarked", "Pclass"],
            "handle_missing": True,
            "scale_features": True,
            "encode_categoricals": True
        },
        "inference": {
            "model_paths": ["artifacts/latest/fold_0_model.joblib"],
            "ensemble_method": "average",
            "ensemble_weights": None,
            "use_tta": False,
            "tta_rounds": 5,
            "output_path": "artifacts/predictions.csv",
            "submission_path": "artifacts/submission.csv"
        }
    }

    config_path = path_manager.config_dir / f"{config_name}.yaml"

    import yaml
    with open(config_path, "w") as f:
        yaml.dump(templates[template], f, default_flow_style=False, indent=2)

    click.echo(f"✅ Configuration created: {config_path}")


if __name__ == "__main__":
    cli()
