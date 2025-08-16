"""Command-line interface for Titanic ML pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from core.utils import (
    ExperimentConfig,
    DataConfig,
    LoggerFactory,
    PathManager,
    SeedManager,
    ConfigManager,
    Timer,
)
from data.loader import TitanicDataLoader
try:
    from data.validate import TitanicDataValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    TitanicDataValidator = None  # Ensure it's defined in the except block
from features import create_feature_builder
from modeling.model_registry import ModelRegistry
from modeling.trainers import TitanicTrainer
from eval.evaluator import TitanicEvaluator
from infer.predictor import create_predictor, ModelLoader


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
def diagnose():
    """Diagnose environment, data availability, and config toggles."""
    try:
        import importlib
        from pathlib import Path

        click.echo("🩺 Environment & Pipeline Diagnosis")

        # Optional dependencies
        for mod in ["category_encoders", "xgboost", "lightgbm"]:
            try:
                importlib.import_module(mod)
                click.echo(f"   ✅ {mod} installed")
            except Exception:
                click.echo(f"   ⚠️  {mod} NOT installed")

        # Data paths
        train_raw = path_manager.data_dir / "raw" / "train.csv"
        test_raw = path_manager.data_dir / "raw" / "test.csv"
        click.echo(f"   📄 Train raw: {'OK' if train_raw.exists() else 'MISSING'} → {train_raw}")
        click.echo(f"   📄 Test  raw: {'OK' if test_raw.exists() else 'MISSING'} → {test_raw}")

        # Processed
        train_proc = path_manager.data_dir / "processed" / "train_features.csv"
        test_proc = path_manager.data_dir / "processed" / "test_features.csv"
        click.echo(f"   📁 Train processed: {'OK' if train_proc.exists() else 'MISSING'} → {train_proc}")
        click.echo(f"   📁 Test  processed: {'OK' if test_proc.exists() else 'MISSING'} → {test_proc}")

        # Config toggles
        try:
            data_cfg_dict = config_manager.load_config("data")
            click.echo("   ⚙️  Data config toggles:")
            for k in ["handle_missing", "encode_categorical", "scale_features", "feature_importance", "add_original_columns"]:
                v = data_cfg_dict.get(k, None)
                click.echo(f"      {k}: {v}")
            fe = data_cfg_dict.get("feature_engineering", {})
            toggles = data_cfg_dict.get("feature_toggles", {})
            click.echo(f"   🔧 Enabled transforms (pre_impute): {fe.get('pre_impute', [])}")
            click.echo(f"   🔧 Enabled transforms (post_impute): {fe.get('post_impute', [])}")
            if toggles:
                on = [k for k, v in toggles.items() if v]
                off = [k for k, v in toggles.items() if not v]
                click.echo(f"   🔌 Toggles ON: {on}")
                click.echo(f"   🔌 Toggles OFF: {off}")
        except Exception as e:
            click.echo(f"   ⚠️  Could not load data config: {e}")

        # Latest artifacts
        latest = path_manager.artifacts_dir / "latest"
        if latest.exists():
            click.echo(f"   🧭 Latest run: {latest.resolve()} -> contains models and reports")
        else:
            click.echo("   🧭 No 'latest' artifacts symlink yet")

        click.echo("✅ Diagnosis complete")
    except Exception as e:
        click.echo(f"❌ Diagnose failed: {e}")


@cli.command("suggest-columns")
@click.option("--top", type=int, default=20, show_default=True, help="Number of columns to suggest")
def suggest_columns(top: int):
    """Suggest a compact training column set using RandomForest importance on processed features."""
    try:
        import pandas as pd
        from features.importance.calculator import FeatureImportanceCalculator

        proc_dir = path_manager.data_dir / "processed"
        train_path = proc_dir / "train_features.csv"
        if not train_path.exists():
            click.echo("❌ Processed training features not found. Run 'features' first.")
            return

        df = pd.read_csv(train_path)
        id_col = "PassengerId" if "PassengerId" in df.columns else None
        target_col = "Survived" if "Survived" in df.columns else None

        X = df.drop(columns=[c for c in [id_col, target_col] if c])
        # numeric/bool only to avoid string leakage
        X = X.select_dtypes(include=["number", "bool"]) 
        y = df[target_col] if target_col else None
        if y is None:
            click.echo("❌ Target column 'Survived' not found in processed data.")
            return

        calc = FeatureImportanceCalculator({
            "feature_importance_config": {
                "enabled": True,
                "algorithms": ["random_forest"],
                "cross_validate": False,
                "save_results": False,
            }
        })
        res = calc.calculate_importance(X, y)
        rf = res.get("random_forest")
        if rf is None or rf.empty:
            click.echo("❌ Could not compute feature importance.")
            return

        topk = rf.head(top)["feature"].tolist()
        click.echo("✅ Suggested training columns (top importance):")
        for c in topk:
            click.echo(f"  - {c}")
        click.echo("\nPaste into configs/data.yaml under train_columns or remove from exclude list.")
    except Exception as e:
        click.echo(f"❌ Suggest-columns failed: {e}")
@click.option("--input-path", type=click.Path(exists=True), required=True,
              help="Path to the CSV to profile")
@click.option("--output-dir", type=click.Path(), required=True,
              help="Directory to save the profiling report")
@click.option("--minimal/--full", default=True, show_default=True,
              help="Use minimal mode (faster) vs full analysis")
def analyze(input_path: str, output_dir: str, minimal: bool):
    """Generate a data profiling HTML report."""
    import pandas as pd
    from pathlib import Path

    # --- robust import for both package names ---
    ProfileReport = None
    try:
        from ydata_profiling import ProfileReport  # modern package
    except Exception:
        try:
            from pandas_profiling import ProfileReport  # legacy name
        except Exception as e:
            click.echo(
                "❌ ydata-profiling is not importable. Install with:\n"
                "   pip install ydata-profiling\n"
                "or legacy:\n"
                "   pip install pandas-profiling"
            )
            return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)

    # Build report
    title = f"Data Profile — {Path(input_path).name}"
    profile = ProfileReport(
        df,
        title=title,
        minimal=minimal,          # fast overview
        explorative=not minimal,  # richer interactions in full mode
        progress_bar=True,
    )

    # Save report
    out_file = out_dir / "profile.html"
    profile.to_file(out_file)

    click.echo(f"✅ Profiling report saved to {out_file}")
    click.echo(f"   Rows: {len(df):,} | Columns: {df.shape[1]}")

@cli.command()
@click.option("--competition", default="titanic", help="Kaggle competition name")
@click.option("--output-dir", type=click.Path(), help="Output directory for data")
def download(competition: str, output_dir: Optional[str]):
    """Download data from Kaggle competition."""

    # Single source of truth for destination directory
    data_dir = Path(output_dir) if output_dir else (path_manager.data_dir / "raw")

    try:
        from data.loader import KaggleDataLoader
        # Pass the chosen destination to the loader
        loader = KaggleDataLoader(competition, data_dir)
        loader.download_competition_data()

        click.echo(f"✅ Data downloaded to {data_dir.resolve()}")
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
@click.option("--features-config", type=click.Path(exists=True), help="Path to features.yaml configuration file")
@click.option("--profile", type=click.Choice(["fast","standard","full"]), help="Optional profile to merge")
@click.option("--set", "set_overrides", multiple=True, help="Override config values, e.g. key=value. Supports dot paths.")
def features(experiment_config: str, data_config: str, features_config: Optional[str], profile: Optional[str], set_overrides: tuple[str]):
    """Build features for training and test data."""

    try:
        # Load configurations
        exp_config = config_manager.load_config(experiment_config)
        data_config_dict = config_manager.load_config(data_config)

        # Merge profile if provided (applies to both exp and data dicts)
        if profile:
            prof_path = path_manager.config_dir / "profiles" / f"{profile}.yaml"
            if prof_path.exists():
                prof = config_manager.load_config(str(prof_path))
                def deep_update(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and isinstance(d.get(k), dict):
                            deep_update(d[k], v)
                        else:
                            d[k] = v
                deep_update(exp_config, prof)
                deep_update(data_config_dict, prof)

        # Apply inline overrides
        if set_overrides:
            def _parse_val(v: str):
                if v.lower() in ("true", "false"): return v.lower() == "true"
                try:
                    if "." in v: return float(v)
                    return int(v)
                except Exception:
                    return v
            def _set(d: dict, path: str, value):
                parts = path.split(".")
                cur = d
                for p in parts[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p] = {}
                    cur = cur[p]
                cur[parts[-1]] = value
            for kv in set_overrides:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    _set(exp_config, k, _parse_val(v))
                    _set(data_config_dict, k, _parse_val(v))

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
        feature_config_path = Path(features_config) if features_config else path_manager.config_dir / "features.yaml"

        # Load feature configuration properly
        if features_config:
            # Load custom features config if provided
            feature_config_dict = config_manager.load_config_from_path(feature_config_path)
            feature_cfg = DataConfig(**feature_config_dict)
        else:
            # Use the data config for feature building (standard approach)
            feature_cfg = data_cfg

        feature_builder = create_feature_builder(feature_cfg, debug=experiment_cfg.debug_mode)

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
@click.option("--profile", type=click.Choice(["fast","standard","full"]), help="Optional profile to merge")
@click.option("--set", "set_overrides", multiple=True, help="Override config values, e.g. key=value. Supports dot paths.")
def train(experiment_config: str, data_config: str, profile: Optional[str], set_overrides: tuple[str]):
    """Train model with cross-validation."""

    try:
        # Load configurations
        exp_config = config_manager.load_config(experiment_config)
        data_config_dict = config_manager.load_config(data_config)

        # Merge profile if provided
        if profile:
            prof_path = path_manager.config_dir / "profiles" / f"{profile}.yaml"
            if prof_path.exists():
                prof = config_manager.load_config(str(prof_path))
                def deep_update(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and isinstance(d.get(k), dict):
                            deep_update(d[k], v)
                        else:
                            d[k] = v
                deep_update(exp_config, prof)
                deep_update(data_config_dict, prof)

        # Apply inline overrides
        if set_overrides:
            def _parse_val(v: str):
                if v.lower() in ("true", "false"): return v.lower() == "true"
                try:
                    if "." in v: return float(v)
                    return int(v)
                except Exception:
                    return v
            def _set(d: dict, path: str, value):
                parts = path.split(".")
                cur = d
                for p in parts[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p] = {}
                    cur = cur[p]
                cur[parts[-1]] = value
            for kv in set_overrides:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    _set(exp_config, k, _parse_val(v))
                    _set(data_config_dict, k, _parse_val(v))

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
        # Drop target and ID columns for modeling
        drop_for_training = [data_cfg.target_column]
        if data_cfg.id_column in train_df.columns:
            drop_for_training.append(data_cfg.id_column)

        selected_cols: list[str] = []
        # If train_columns is provided in data config, honor it
        if getattr(data_cfg, 'train_columns', None):
            requested = list(data_cfg.train_columns or [])
            selected_cols = [c for c in requested if c not in drop_for_training and c in train_df.columns]
            missing = [c for c in requested if c not in train_df.columns]
            if missing:
                click.echo(f"⚠️ Some requested training columns not found and will be ignored: {missing}")

        if selected_cols:
            X = train_df[selected_cols].copy()
        else:
            X = train_df.drop(columns=drop_for_training)
            # Drop raw text / identifier columns not yet encoded if present
            drop_cols = [c for c in ['Name', 'Ticket', 'Cabin'] if c in X.columns]
            if drop_cols:
                X = X.drop(columns=drop_cols)

            # If exclusion list is provided, drop those columns if present
            if getattr(data_cfg, 'exclude_column_for_training', None):
                excl = [c for c in (data_cfg.exclude_column_for_training or []) if c in X.columns]
                if excl:
                    X = X.drop(columns=excl)
                    click.echo(f"ℹ️ Dropped excluded training columns: {excl}")
        # Fallback safety: keep only numeric/bool columns (avoids raw objects when originals are kept)
        X = X.select_dtypes(include=["number", "bool"]) 

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

        # Update latest symlink
        try:
            run_dir_path = Path(cv_results["run_dir"]).resolve()
            latest = path_manager.artifacts_dir / "latest"
            if latest.exists() or latest.is_symlink():
                try:
                    latest.unlink()
                except Exception:
                    pass
            latest.symlink_to(run_dir_path)
            click.echo(f"   🔗 Updated artifacts/latest -> {run_dir_path}")
        except Exception:
            pass

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
@click.option("--set", "set_overrides", multiple=True, help="Override inference/data config values, e.g. key=value")
def predict(run_dir: str, inference_config: str, output_path: Optional[str],
            threshold_file: Optional[str], threshold_value: Optional[float], set_overrides: tuple[str] = ()): 
    """Generate predictions on test data."""
    try:
        from pathlib import Path
        import numpy as np
        import pandas as pd  # <-- ensure imported ONCE at top of function (or at module level)

        run_path = Path(run_dir)

        # Load inference config
        inference_cfg = config_manager.load_config(inference_config)
        # Inline overrides for inference/data
        if set_overrides:
            def _parse_val(v: str):
                if isinstance(v, str) and v.lower() in ("true", "false"):
                    return v.lower() == "true"
                try:
                    if isinstance(v, str) and "." in v:
                        return float(v)
                    return int(v)
                except Exception:
                    return v
            def _set(d: dict, path: str, value):
                parts = path.split(".")
                cur = d
                for p in parts[:-1]:
                    if p not in cur or not isinstance(cur.get(p), dict):
                        cur[p] = {}
                    cur = cur[p]
                cur[parts[-1]] = value
            # Try apply to inference first; user can still target data.* keys but we don't merge here
            for kv in set_overrides:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    _set(inference_cfg, k, _parse_val(v))
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
                                threshold_value = row.iloc[0]["threshold"]
                                if isinstance(threshold_value, pd.Series):
                                    th_cfg["value"] = float(threshold_value.values[0])
                                else:
                                    th_cfg["value"] = float(threshold_value)
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

        # Prepare model input (use training column selection if configured)
        test_model_df = test_df.copy()
        try:
            data_config_dict = config_manager.load_config("data")
            data_cfg = DataConfig(**data_config_dict)
        except Exception:
            data_cfg = None

        if data_cfg and getattr(data_cfg, 'train_columns', None):
            requested = list(data_cfg.train_columns or [])
            keep_cols = [c for c in requested if c in test_model_df.columns]
            missing = [c for c in requested if c not in test_model_df.columns]
            if missing:
                click.echo(f"⚠️ Some train_columns missing in test data and will be ignored: {missing}")
            if keep_cols:
                test_model_df = test_model_df[keep_cols]
            else:
                drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in test_model_df.columns]
                if drop_cols:
                    test_model_df = test_model_df.drop(columns=drop_cols)
        else:
            drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in test_model_df.columns]
            if drop_cols:
                test_model_df = test_model_df.drop(columns=drop_cols)

        # Apply exclusion list to test as well (after optional selection/drop)
        if data_cfg and getattr(data_cfg, 'exclude_column_for_training', None):
            excl = [c for c in (data_cfg.exclude_column_for_training or []) if c in test_model_df.columns]
            if excl:
                test_model_df = test_model_df.drop(columns=excl)

        if "PassengerId" in test_model_df.columns:
            test_model_df = test_model_df.set_index("PassengerId")
        # Fallback safety: keep only numeric/bool feature columns
        test_model_df = test_model_df.select_dtypes(include=["number", "bool"]) 

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
        th_cfg = inference_cfg.get("threshold", {}) or {}
        try:
            used_thr = predictor._resolve_threshold(inference_cfg)
            click.echo(f"   📊 Threshold source: {used_src} | value: {used_thr:.4f}")

            # Print threshold details if requested
            if th_cfg.get("print", False):
                click.echo(f"   📊 Threshold configuration:")
                click.echo(f"      Method: {th_cfg.get('method', 'accuracy')}")
                click.echo(f"      Optimizer: {th_cfg.get('optimizer', False)}")
                if th_cfg.get('method') == 'cost':
                    click.echo(f"      Cost FP: {th_cfg.get('cost_fp', 1.0)}")
                    click.echo(f"      Cost FN: {th_cfg.get('cost_fn', 1.0)}")
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

        # Report TTA and postprocessing usage
        if inference_cfg.get("use_tta", False):
            click.echo(f"   🔄 TTA enabled: {inference_cfg.get('tta_rounds', 5)} rounds, "
                      f"noise scale: {inference_cfg.get('tta_noise_scale', 0.01)}")

        postproc_rules = inference_cfg.get("postprocessing", {}).get("rules", [])
        if postproc_rules:
            rule_types = [rule.get("type") for rule in postproc_rules]
            click.echo(f"   ⚙️ Postprocessing applied: {', '.join(rule_types)}")

    except Exception as e:
        click.echo(f"❌ Prediction failed: {e}")

@cli.command()
@click.option("--experiment-config", default="experiment", help="Experiment configuration name")
@click.option("--data-config", default="data", help="Data configuration name")
@click.option("--inference-config", default="inference", help="Inference configuration name")
@click.option("--profile", type=click.Choice(["fast","standard","full"]), help="Optional profile to merge")
@click.option("--competition", default="titanic", show_default=True, help="Kaggle competition slug")
@click.option("--remote", is_flag=True, help="If set, perform remote Kaggle submission")
@click.option("--message", "-m", default="Auto pipeline run", show_default=True, help="Submission message")
@click.option("--set", "set_overrides", multiple=True, help="Override config values for all phases (key=value)")
def autopipeline(experiment_config: str, data_config: str, inference_config: str, profile: Optional[str], competition: str, remote: bool, message: str, set_overrides: tuple[str]):
    """Run end-to-end: features -> train -> predict -> submit (optional remote)."""
    try:
        click.echo("🛠  Building features...")
        ctx = click.get_current_context()
        ctx.invoke(features, experiment_config=experiment_config, data_config=data_config, profile=profile, set_overrides=set_overrides)

        click.echo("🧪 Training model...")
        # Capture stdout from train by invoking and parsing run_dir from artifacts listing afterwards
        before = set(p.name for p in path_manager.artifacts_dir.glob('20*'))
        ctx.invoke(train, experiment_config=experiment_config, data_config=data_config, profile=profile, set_overrides=set_overrides)
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

    click.echo(f"��� Configuration created: {config_path}")


@cli.command()
@click.option("--strategy", default="standard",
              type=click.Choice(["quick", "standard", "comprehensive"]),
              help="Tuning strategy: quick (25 trials), standard (100 trials), comprehensive (200 trials)")
@click.option("--model", "-m", "model_name",
              help="Specific model to tune (e.g., logistic, random_forest). If not specified, tunes all enabled models")
@click.option("--n-trials", type=int,
              help="Number of trials to run (overrides strategy default)")
@click.option("--timeout", type=int,
              help="Maximum time in seconds (overrides strategy default)")
@click.option("--experiment-config", default="experiment", help="Experiment configuration name")
@click.option("--data-config", default="data", help="Data configuration name")
@click.option("--tuning-config", default="tuning", help="Tuning configuration name")
@click.option("--features-config", type=click.Path(exists=True), help="Path to features.yaml configuration file")
def tune(strategy: str, model_name: Optional[str], n_trials: Optional[int],
         timeout: Optional[int], experiment_config: str, data_config: str,
         tuning_config: str, features_config: Optional[str]):
    """Perform hyperparameter tuning using Optuna."""
    try:
        from tuning import HyperparameterTuner, SearchSpaceFactory

        click.echo("🔧 Starting hyperparameter tuning...")

        # Load configurations
        exp_config = config_manager.load_config(experiment_config)
        data_config_dict = config_manager.load_config(data_config)
        tuning_config_dict = config_manager.load_config(tuning_config)

        experiment_cfg = ExperimentConfig(**exp_config)
        data_cfg = DataConfig(**data_config_dict)

        # Set seed
        SeedManager.set_seed(experiment_cfg.seed)

        # Load and prepare data - use processed data if available
        processed_dir = path_manager.data_dir / "processed"
        train_processed_path = processed_dir / "train_features.csv"

        if train_processed_path.exists():
            # Use existing processed data
            click.echo(f"📊 Using existing processed data from {train_processed_path}")
            train_df = pd.read_csv(train_processed_path)

            # Prepare data for modeling
            drop_for_training = [data_cfg.target_column]
            if data_cfg.id_column in train_df.columns:
                drop_for_training.append(data_cfg.id_column)
            X = train_df.drop(columns=drop_for_training)

            # Drop raw text / identifier columns not yet encoded if present
            drop_cols = [c for c in ['Name', 'Ticket', 'Cabin'] if c in X.columns]
            if drop_cols:
                X = X.drop(columns=drop_cols)
            y = train_df[data_cfg.target_column]

        else:
            # Fallback to raw data + feature engineering if processed data doesn't exist
            train_path = Path(data_cfg.train_path)
            if not train_path.exists():
                alt_train = Path('data') / train_path.name
                if alt_train.exists():
                    train_path = alt_train

            loader = TitanicDataLoader(str(train_path), None)
            result = loader.load()

            # Handle the loader return value properly
            if isinstance(result, tuple):
                train_df, _ = result
            else:
                train_df = result

            # Build features
            feature_config_path = Path(features_config) if features_config else path_manager.config_dir / "features.yaml"

            # Load feature configuration properly
            if features_config:
                # Load custom features config if provided
                feature_config_dict = config_manager.load_config_from_path(feature_config_path)
                feature_cfg = DataConfig(**feature_config_dict)
            else:
                # Use the data config for feature building (standard approach)
                feature_cfg = data_cfg

            feature_builder = create_feature_builder(feature_cfg, debug=experiment_cfg.debug_mode)

            with Timer(logger, "feature engineering"):
                # Use the correct fit/transform interface instead of build_features
                feature_builder.fit(train_df.drop(columns=["Survived"]), train_df["Survived"])
                X = feature_builder.transform(train_df.drop(columns=["Survived"]))
                y = train_df["Survived"]

        click.echo(f"📊 Training data prepared: {X.shape}")

        # Apply strategy overrides
        strategy_config = tuning_config_dict.get("strategies", {}).get(strategy, {})
        if strategy_config:
            if n_trials is None:
                n_trials = strategy_config.get("n_trials")
            if timeout is None:
                timeout = strategy_config.get("timeout")

            # Filter models based on strategy
            if model_name is None:
                enabled_models = strategy_config.get("models", [])
            else:
                enabled_models = [model_name] if model_name in strategy_config.get("models", [model_name]) else [model_name]
        else:
            # Use all enabled models from config
            if model_name is None:
                enabled_models = [name for name, config in tuning_config_dict.get("models", {}).items()
                                if config.get("enabled", True)]
            else:
                enabled_models = [model_name]

        # Override config with CLI parameters
        if n_trials:
            tuning_config_dict["study"]["n_trials"] = n_trials
        if timeout:
            tuning_config_dict["study"]["timeout"] = timeout

        click.echo(f"🎯 Strategy: {strategy}")
        click.echo(f"🔢 Trials: {tuning_config_dict['study']['n_trials']}")
        click.echo(f"⏱️  Timeout: {tuning_config_dict['study'].get('timeout', 'unlimited')}s")
        click.echo(f"🤖 Models: {', '.join(enabled_models)}")

        # Create tuner
        tuner = HyperparameterTuner(tuning_config_dict)

        # CV configuration
        cv_config = {
            "strategy": tuning_config_dict.get("cv", {}).get("strategy", "stratified"),
            "n_folds": tuning_config_dict.get("cv", {}).get("n_folds", 5),
            "shuffle": tuning_config_dict.get("cv", {}).get("shuffle", True),
            "random_state": tuning_config_dict.get("cv", {}).get("random_state", 42)
        }

        # Tune each model
        all_results = {}
        for model in enabled_models:
            if model not in SearchSpaceFactory.get_available_models():
                click.echo(f"⚠️  Skipping {model}: no search space defined")
                continue

            click.echo(f"\n🚀 Tuning {model}...")

            try:
                results = tuner.tune_model(model, X, y, cv_config)
                all_results[model] = results

                click.echo(f"✅ {model} tuning completed!")
                click.echo(f"   📊 Best score: {results['best_score']:.4f}")
                click.echo(f"   🔧 Best params: {results['best_params']}")
                click.echo(f"   📁 Results: {results['run_dir']}")

            except Exception as e:
                click.echo(f"❌ {model} tuning failed: {e}")
                logger.error(f"Tuning failed for {model}: {e}")
                continue

        # Summary
        if all_results:
            click.echo(f"\n🎉 Tuning completed for {len(all_results)} models!")
            click.echo("\n📊 Summary:")

            # Sort by score
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_score'], reverse=True)

            for model, results in sorted_results:
                click.echo(f"   {model}: {results['best_score']:.4f}")

            # Best overall model
            best_model, best_results = sorted_results[0]
            click.echo(f"\n🏆 Best model: {best_model} (score: {best_results['best_score']:.4f})")

            # Save summary
            summary_path = path_manager.artifacts_dir / f"tuning_summary_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, "w") as f:
                summary_data = {
                    "strategy": strategy,
                    "timestamp": datetime.now().isoformat(),
                    "results": {k: {
                        "best_score": v["best_score"],
                        "best_params": v["best_params"],
                        "n_trials": v["n_trials"],
                        "run_dir": v["run_dir"]
                    } for k, v in all_results.items()},
                    "best_model": best_model,
                    "cv_config": cv_config
                }
                json.dump(summary_data, f, indent=2)

            click.echo(f"📄 Summary saved to: {summary_path}")
        else:
            click.echo("❌ No models were successfully tuned")

    except ImportError as e:
        click.echo("❌ Tuning requires 'optuna' package. Install with: pip install optuna")
        logger.error(f"Import error: {e}")
    except Exception as e:
        click.echo(f"❌ Tuning failed: {e}")
        logger.error(f"Tuning failed: {e}")
        raise


if __name__ == "__main__":
    cli()
