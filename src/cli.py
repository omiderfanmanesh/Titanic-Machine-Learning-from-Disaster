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
from infer.predictor import create_predictor, TitanicPredictor
from cv.folds import create_splits_with_validation
from sklearn.base import clone


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
            # Dimensionality reduction summary
            dr = (data_cfg_dict.get("dimensionality_reduction") or {})
            if dr:
                click.echo("   🔽 Dimensionality reduction:")
                click.echo(f"      enabled: {dr.get('enabled', False)}")
                click.echo(f"      method: {dr.get('method', 'pca')}")
                click.echo(f"      n_components: {dr.get('n_components')}" )
                if dr.get('keep_variance') is not None:
                    click.echo(f"      keep_variance: {dr.get('keep_variance')}")
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
                "cross_validate": True,
                "save_results": True,
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
        except Exception:
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

        click.echo(f"🔧 Feature building: fitting on X_train={X_train.shape}, y_train={y_train.shape}")
        feature_builder.fit(X_train, y_train)

        # Transform both train and test
        click.echo("🔄 Transforming training and test data...")
        X_train_processed = feature_builder.transform(X_train)
        X_test_processed = feature_builder.transform(test_df)
        click.echo(f"✅ Transformed: train={X_train_processed.shape}, test={X_test_processed.shape}")

        # Save processed data
        processed_dir = path_manager.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)

        train_processed_path = processed_dir / "train_features.csv"
        test_processed_path = processed_dir / "test_features.csv"

        # Add target back to training data
        X_train_processed[data_cfg.target_column] = y_train

        X_train_processed.to_csv(train_processed_path, index=False)
        X_test_processed.to_csv(test_processed_path, index=False)

        click.echo("✅ Features built and saved:")
        click.echo(f"   📁 Train: {train_processed_path} ({X_train_processed.shape})")
        click.echo(f"   📁 Test: {test_processed_path} ({X_test_processed.shape})")
        try:
            id_col = data_cfg.id_column
            target_col = data_cfg.target_column
            cols = list(X_train_processed.columns)
            click.echo(f"   🔎 Columns: total={len(cols)}; id_present={id_col in cols}; target_present={target_col in cols}")
        except Exception:
            pass

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

        # Load RAW training data for leak-safe per-fold feature processing
        # Resolve raw path
        raw_train_path = Path(data_cfg.train_path)
        if not raw_train_path.exists():
            alt = Path('data') / raw_train_path.name
            if alt.exists():
                raw_train_path = alt
        if not raw_train_path.exists():
            click.echo(f"❌ Raw training data not found at {data_cfg.train_path}")
            return

        train_df = pd.read_csv(raw_train_path)

        # Apply debug mode if needed
        if experiment_cfg.debug_mode and experiment_cfg.debug_n_rows:
            train_df = train_df.head(experiment_cfg.debug_n_rows)
            click.echo(f"🐛 Debug mode: Using {len(train_df)} training samples")

        # Prepare raw X/y (builder handles transforms per fold)
        y = train_df[data_cfg.target_column]
        X = train_df.drop(columns=[data_cfg.target_column])

        # If ensemble config is provided, run multi-model training in one run
        ens_cfg = (getattr(experiment_cfg, "ensemble", None) or {})
        model_list = ens_cfg.get("model_list") if isinstance(ens_cfg, dict) else getattr(ens_cfg, "model_list", [])
        use_ensemble = bool(ens_cfg.get("use", False)) if isinstance(ens_cfg, dict) else bool(getattr(ens_cfg, "use", False))
        if use_ensemble and model_list:
            click.echo("🤝 Ensemble mode detected — training multiple models in one run")

            # Create run directory up-front
            run_path = path_manager.create_run_directory()
            click.echo(f"   📁 Artifacts dir: {run_path}")

            # Build per-fold pipelines once (reused across models)
            feature_builder = create_feature_builder(data_cfg, debug=experiment_cfg.debug_mode)

            # Create CV splits
            splits, _ = create_splits_with_validation(X, y, {
                "cv_strategy": data_config_dict.get("cv_strategy", experiment_cfg.cv_strategy),
                "cv_folds": data_config_dict.get("cv_folds", experiment_cfg.cv_folds),
                "cv_shuffle": data_config_dict.get("cv_shuffle", experiment_cfg.cv_shuffle),
                "cv_random_state": data_config_dict.get("cv_random_state", experiment_cfg.cv_random_state),
            })

            import numpy as np
            import joblib
            from copy import deepcopy

            n_samples = len(X)
            # number of folds not used directly; fold_pipes carries fold info
            id_col = data_cfg.id_column
            target_col = data_cfg.target_column

            # Prepare structures for OOF per model
            per_model_oof: dict[str, np.ndarray] = {}
            per_model_fold_scores: dict[str, list[float]] = {}
            fold_assign = np.full(n_samples, -1, dtype=int)

            # Fit and save per-fold pipelines once
            fold_pipes = []
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                X_tr_raw, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_va_raw, y_va = X.iloc[va_idx], y.iloc[va_idx]

                pipe = deepcopy(feature_builder)
                pipe.fit(X_tr_raw, y_tr)
                X_tr = pipe.transform(X_tr_raw)
                X_va = pipe.transform(X_va_raw)

                # Drop id/target if present
                for col in (id_col, target_col):
                    if col and col in X_tr.columns:
                        X_tr = X_tr.drop(columns=[col])
                    if col and col in X_va.columns:
                        X_va = X_va.drop(columns=[col])

                # Persist pipeline
                pipe_path = run_path / f"fold_{fold_idx}_feature_pipeline.joblib"
                joblib.dump(pipe, pipe_path)
                fold_pipes.append((pipe_path, X_tr, y_tr, X_va, y_va, va_idx))
                fold_assign[va_idx] = fold_idx

            # Helper: scoring per configured metric
            trainer_for_metric = TitanicTrainer(data_config_dict)

            # Train each model on the same folds/pipelines
            registry = ModelRegistry()
            # Normalize specs for iteration and metadata
            def _spec_to_pair(s):
                if hasattr(s, 'name'):
                    return s.name, dict(getattr(s, 'params', {}) or {})
                elif isinstance(s, dict):
                    return s.get('name'), dict(s.get('params', {}) or {})
                else:
                    return str(s), {}

            model_specs_for_meta = []
            for spec in model_list:
                m_name, m_params = _spec_to_pair(spec)
                model_specs_for_meta.append({"name": m_name, "params": m_params})
                click.echo(f"\n🚀 Training model: {m_name}")

                # Build estimator
                wrapper = registry.create_model(m_name, m_params)
                estimator = wrapper.build({"model_name": m_name, "model_params": m_params})

                oof = np.zeros(n_samples, dtype=float)
                fold_scores: list[float] = []
                skip_model_due_to_nan = False

                for fold_idx, (_, X_tr, y_tr, X_va, y_va, va_idx) in enumerate(fold_pipes):
                    # Optional: imbalance handling on this fold's training split
                    try:
                        X_tr, y_tr = trainer_for_metric._apply_imbalance_sampling_if_enabled(X_tr, y_tr, fold_idx, model_name=m_name)
                    except Exception as e:
                        click.echo(f"   ⚠️  Imbalance sampling failed on fold {fold_idx}: {e}")
                    # Detailed NaN logging and skip if model can't handle NaNs
                    def _nan_report(df, lbl):
                        counts = df.isna().sum()
                        bad = counts[counts > 0].sort_values(ascending=False)
                        if not bad.empty:
                            click.echo(
                                f"   ❌ NaNs detected for model '{m_name}' on fold {fold_idx}: {lbl} "
                                f"cols_with_nans={len(bad)}\n{bad.head(10).to_string()}"
                            )
                        return not bad.empty

                    has_nan_tr = _nan_report(X_tr, "X_train")
                    has_nan_va = _nan_report(X_va, "X_valid")
                    # Conservative: most sklearn classic estimators don't support NaN; if present, stop this model
                    if has_nan_tr or has_nan_va:
                        skip_model_due_to_nan = True
                        click.echo(
                            f"   ⏭️  Skipping model '{m_name}' due to NaNs in features on fold {fold_idx}. "
                            "Consider enabling/adjusting imputation or switching to HistGradientBoosting/XGBoost/LightGBM/CatBoost."
                        )
                        break
                    # Clone estimator per fold
                    est_fold = clone(estimator)
                    # Optional: class weighting
                    try:
                        est_fold, cw_sw = trainer_for_metric._prepare_class_weighting(est_fold, y_tr, model_name=m_name)
                    except Exception:
                        cw_sw = None
                    # Fit with optional sample_weight
                    try:
                        if cw_sw is not None:
                            est_fold.fit(X_tr, y_tr, sample_weight=cw_sw)
                        else:
                            est_fold.fit(X_tr, y_tr)
                    except TypeError:
                        est_fold.fit(X_tr, y_tr)

                    # Predict proba/logits
                    try:
                        proba = est_fold.predict_proba(X_va)
                        fold_pred = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
                    except Exception:
                        try:
                            fold_pred = est_fold.decision_function(X_va)
                        except Exception:
                            fold_pred = est_fold.predict(X_va)

                    # Save fold model
                    model_path = run_path / f"fold_{fold_idx}_model_{m_name}.joblib"
                    joblib.dump(est_fold, model_path)

                    # Score
                    score = trainer_for_metric._calculate_score(y_va, fold_pred)
                    fold_scores.append(float(score))

                    # Fill OOF
                    oof[va_idx] = fold_pred

                if skip_model_due_to_nan:
                    click.echo(f"   ⚠️  Model '{m_name}' was not trained due to NaNs; continuing with remaining models")
                    continue
                per_model_oof[m_name] = oof
                per_model_fold_scores[m_name] = fold_scores

                # Save OOF CSV for this model
                oof_df = pd.DataFrame({
                    "target": y.values,
                    "prediction": oof,
                    "fold": fold_assign
                })
                oof_df.to_csv(run_path / f"oof_{m_name}.csv", index=False)
                click.echo(f"   📁 Saved OOF for {m_name}: oof_{m_name}.csv | CV {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

            # Save ensemble config + metadata
            ens_method = (ens_cfg.get("method") if isinstance(ens_cfg, dict) else getattr(ens_cfg, "method", "average")) or "average"
            ens_weights = (ens_cfg.get("weights") if isinstance(ens_cfg, dict) else getattr(ens_cfg, "weights", None))
            meta = {
                "run_dir": str(run_path),
                "timestamp": datetime.now().isoformat(),
                "cv": {
                    "strategy": data_config_dict.get("cv_strategy", experiment_cfg.cv_strategy),
                    "folds": data_config_dict.get("cv_folds", experiment_cfg.cv_folds),
                    "shuffle": data_config_dict.get("cv_shuffle", experiment_cfg.cv_shuffle),
                    "random_state": data_config_dict.get("cv_random_state", experiment_cfg.cv_random_state),
                    "metric": data_config_dict.get("cv_metric", experiment_cfg.cv_metric),
                },
                "models": model_specs_for_meta,
                "ensemble": {
                    "method": ens_method,
                    "weights": ens_weights,
                    "model_order": [ms["name"] for ms in model_specs_for_meta],
                },
            }
            (run_path / "training_config.json").write_text(json.dumps(meta, indent=2))
            (run_path / "ensemble_config.json").write_text(json.dumps(meta["ensemble"], indent=2))

            # Compute ensemble OOF fold-wise (combine models within fold, then aggregate across folds implicitly via OOF)
            predictor = TitanicPredictor({"ensemble_method": ens_method, "ensemble_weights": ens_weights})
            # Build a matrix of shape (n_models, n_samples) with NaNs where sample not in a model? OOFs are complete
            model_order = meta["ensemble"]["model_order"]
            # stack not used directly; combine within each fold for OOF
            # For each fold validation segment, combine across models
            ens_oof = np.zeros(n_samples, dtype=float)
            for fold_idx, (_, _, _, _, _, va_idx) in enumerate(fold_pipes):
                fold_preds = [per_model_oof[m][va_idx] for m in model_order]
                ens_fold = predictor._ensemble_predictions(fold_preds, ens_method, ens_weights)
                ens_oof[va_idx] = np.clip(ens_fold, 0.0, 1.0)

            # Save ensemble OOF CSV
            ens_oof_df = pd.DataFrame({
                "target": y.values,
                "prediction": ens_oof,
                "fold": fold_assign
            })
            ens_oof_df.to_csv(run_path / "oof_ensemble.csv", index=False)

            # Compute ensemble CV scores using same metric; accuracy/F1 use 0.5 threshold inside trainer
            fold_scores_ens = []
            for fold_idx, (_, _, _, _, y_va, va_idx) in enumerate(fold_pipes):
                s = trainer_for_metric._calculate_score(y_va, ens_oof[va_idx])
                fold_scores_ens.append(float(s))

            # Compute OOF score: if metric is AUC use roc_auc_score on full OOF; for accuracy/f1 threshold at 0.5 via trainer helper
            from sklearn.metrics import roc_auc_score as _auc
            metric_name = str(data_config_dict.get("cv_metric", experiment_cfg.cv_metric)).lower()
            oof_score_val = float(_auc(y, ens_oof)) if metric_name == "roc_auc" else float(trainer_for_metric._calculate_score(y, ens_oof))

            scores_payload = {
                "fold_scores": fold_scores_ens,
                "mean_score": float(np.mean(fold_scores_ens)),
                "std_score": float(np.std(fold_scores_ens)),
                "oof_score": oof_score_val
            }
            (run_path / "ensemble_cv_scores.json").write_text(json.dumps(scores_payload, indent=2))

            # Optional: Stacking (meta-learner on OOF)
            stacking_cfg = exp_config.get("stacking") or {}
            use_stacking = bool(stacking_cfg.get("use", False))
            if use_stacking:
                # Build meta features (n_samples, n_models) in model_order
                model_order = meta["ensemble"]["model_order"]
                import numpy as _np
                X_meta = _np.column_stack([per_model_oof[m] for m in model_order])
                y_meta = y.values

                # Save meta OOF features
                import pandas as _pd
                meta_oof_df = _pd.DataFrame(X_meta, columns=[f"oof_{m}" for m in model_order])
                meta_oof_df['target'] = y_meta
                meta_oof_df.to_csv(run_path / "meta_features_oof.csv", index=False)

                # Create meta learner
                meta_spec = stacking_cfg.get("meta_model") or {"name": "logistic", "params": {}}
                meta_name = meta_spec.get("name", "logistic")
                meta_params = meta_spec.get("params", {})
                registry = ModelRegistry()
                meta_wrapper = registry.create_model(meta_name, meta_params)
                meta_est = meta_wrapper.build({"model_name": meta_name, "model_params": meta_params})
                meta_est.fit(X_meta, y_meta)
                # Save
                import joblib as _joblib
                _joblib.dump(meta_est, run_path / "meta_model.joblib")
                (run_path / "meta_config.json").write_text(json.dumps({
                    "meta_model": {"name": meta_name, "params": meta_params},
                    "columns": model_order,
                }, indent=2))
                click.echo("   🧠 Stacking enabled: saved meta_model.joblib and meta_features_oof.csv")

            # Update latest symlink
            try:
                latest = path_manager.artifacts_dir / "latest"
                if latest.exists() or latest.is_symlink():
                    try:
                        latest.unlink()
                    except Exception:
                        pass
                latest.symlink_to(run_path.resolve())
                click.echo(f"   🔗 Updated artifacts/latest -> {run_path}")
            except Exception:
                pass

            # Summary
            click.echo("\n✅ Ensemble training completed!")
            click.echo(f"   📁 Artifacts: {run_path}")
            click.echo(f"   🤖 Models: {', '.join(model_order)}")
            click.echo(f"   🧪 Ensemble OOF mean: {scores_payload['mean_score']:.4f} ± {scores_payload['std_score']:.4f}")
            return

        # Create model
        model_config = {
            "model_name": experiment_cfg.model_name,
            "model_params": experiment_cfg.model_params
        }

        registry = ModelRegistry()
        model_wrapper = registry.create_model(experiment_cfg.model_name, experiment_cfg.model_params)
        estimator = model_wrapper.build(model_config)

        # Create trainer
        # Allow training-related keys to come from data.yaml to keep configs in one place
        # If present in data.yaml, these override experiment values
        dc = data_config_dict  # original dict for raw access
        # CV/train knobs are passed via trainer_config directly; no local copies needed

        # Pass data.yaml as CV/train config (config-driven, no inline trainer_config)
        trainer_config = data_config_dict

        trainer = TitanicTrainer(trainer_config)

        # Train with cross-validation
        click.echo(f"🚀 Starting {experiment_cfg.cv_folds}-fold cross-validation...")
        click.echo(f"   Model: {experiment_cfg.model_name}")
        click.echo(f"   Strategy: {experiment_cfg.cv_strategy}")

        # Build feature builder for leak-safe per-fold processing
        feature_builder = create_feature_builder(data_cfg, debug=experiment_cfg.debug_mode)
        cv_results = trainer.cross_validate(estimator, X, y, trainer_config, feature_builder=feature_builder)

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
        click.echo("✅ Training completed!")
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

        # Load OOF predictions and scores (support both single-model and ensemble artifacts)
        oof_path = run_path / "oof_predictions.csv"
        scores_path = run_path / "cv_scores.json"
        if not (oof_path.exists() and scores_path.exists()):
            # Fallback to ensemble files if present
            ens_oof = run_path / "oof_ensemble.csv"
            ens_scores = run_path / "ensemble_cv_scores.json"
            if ens_oof.exists() and ens_scores.exists():
                oof_path, scores_path = ens_oof, ens_scores
                click.echo("ℹ️ Using ensemble OOF and scores for evaluation")
            else:
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
@click.option("--run-dir", type=click.Path(exists=True), multiple=True, required=False,
              help="Training run directory (can be provided multiple times for cross-run ensembling)")
@click.option("--inference-config", default="inference", help="Inference configuration name")
@click.option("--output-path", type=click.Path(), help="Output path for predictions")
@click.option("--threshold-file", type=click.Path(exists=True),
              help="Path to a file with a single float threshold (e.g., best_threshold.txt)")
@click.option("--threshold", "threshold_value", type=float,
              help="Numeric threshold override (e.g., 0.61)")
@click.option("--set", "set_overrides", multiple=True, help="Override inference/data config values, e.g. key=value")
@click.option("--verbose", is_flag=True, help="Print extra diagnostics (NaN counts, column summary)")
def predict(run_dir: tuple[str], inference_config: str, output_path: Optional[str],
            threshold_file: Optional[str], threshold_value: Optional[float], set_overrides: tuple[str] = (), verbose: bool = False): 
    """Generate predictions on test data."""
    try:
        from pathlib import Path
        import numpy as np
        import pandas as pd  # <-- ensure imported ONCE at top of function (or at module level)

        run_dirs_cli = list(run_dir) if run_dir else []

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

        # Build run list: CLI has priority; otherwise use inference_cfg.runs
        run_specs = []
        if run_dirs_cli:
            run_specs = [{"path": p} for p in run_dirs_cli]
        else:
            runs_cfg = inference_cfg.get("runs") or []
            if isinstance(runs_cfg, list):
                for item in runs_cfg:
                    if isinstance(item, dict) and item.get("path"):
                        run_specs.append({"path": item["path"], "weight": item.get("weight")})
        if not run_specs:
            click.echo("❌ No run directories specified. Provide --run-dir or define runs in inference.yaml.")
            return

        # Helper: compute final proba vector for a single run
        def _predict_for_run(run_path: Path, predictor, inference_cfg_local: dict) -> np.ndarray:
            # Make run_dir available only for single-run threshold discovery; we won't use it for multi-run threshold
            inference_cfg_local = dict(inference_cfg_local)
            inference_cfg_local["run_dir"] = str(run_path)
            # Ensure model paths (single-model-per-fold fallback)
            if not inference_cfg_local.get("model_paths"):
                fold_models = sorted(run_path.glob("fold_*_model.joblib"))
                inference_cfg_local["model_paths"] = [str(p) for p in fold_models]

            fold_pipes = sorted(run_path.glob("fold_*_feature_pipeline.joblib"))
            multi_model_detected = any(run_path.glob("fold_*_model_*.joblib"))
            if multi_model_detected:
                click.echo("🤝 Found multi-model-per-fold artifacts; using per-fold ensembling")

            if fold_pipes and multi_model_detected:
                # Load raw test
                try:
                    data_cfg_dict = config_manager.load_config("data")
                    data_cfg_local = DataConfig(**data_cfg_dict)
                except Exception as e:
                    raise RuntimeError(f"Could not load data config for raw test path: {e}")
                raw_test_path = Path(data_cfg_local.test_path)
                if not raw_test_path.exists():
                    alt = Path('data') / raw_test_path.name
                    if alt.exists():
                        raw_test_path = alt
                if not raw_test_path.exists():
                    raise FileNotFoundError(f"Raw test data not found at {data_cfg_local.test_path}")
                test_raw_df = pd.read_csv(raw_test_path)

                # If a meta model exists and not disabled, use stacking
                use_stacking = (inference_cfg_local.get("stacking", {}) or {}).get("use", True)
                import joblib as _joblib
                meta_model_path = run_path / "meta_model.joblib"
                if use_stacking and meta_model_path.exists():
                    # We need per-model averaged predictions across folds
                    # Load model order if available
                    model_order = None
                    try:
                        import json as _json
                        ecfg = _json.loads((run_path / "ensemble_config.json").read_text())
                        model_order = ecfg.get("model_order")
                    except Exception:
                        model_order = None

                    # Collect per-model predictions for each fold, then average
                    per_model_fold_preds: dict[str, list[np.ndarray]] = {}
                    n_folds = len(fold_pipes)
                    for i in range(n_folds):
                        pipe_path = run_path / f"fold_{i}_feature_pipeline.joblib"
                        pipe = _joblib.load(pipe_path)
                        Xt = pipe.transform(test_raw_df)
                        for col in (data_cfg_local.id_column, data_cfg_local.target_column):
                            if col and col in Xt.columns:
                                Xt = Xt.drop(columns=[col])
                        Xt = Xt.select_dtypes(include=["number", "bool"]).replace([np.inf, -np.inf], np.nan).fillna(0)

                        for mpath in sorted(run_path.glob(f"fold_{i}_model_*.joblib")):
                            mname = mpath.stem.split("_model_")[-1]
                            model = _joblib.load(mpath)
                            raw = predictor._predict_single_model(model, Xt)
                            proba = predictor._normalize_scores_to_proba(raw)
                            per_model_fold_preds.setdefault(mname, []).append(proba)

                    if not per_model_fold_preds:
                        raise RuntimeError("No base model predictions produced for stacking")

                    # Average across folds per model
                    import numpy as _np
                    model_names = model_order or sorted(per_model_fold_preds.keys())
                    X_meta = _np.column_stack([
                        _np.mean(_np.vstack(per_model_fold_preds[name]), axis=0) for name in model_names
                    ])
                    meta_model = _joblib.load(meta_model_path)
                    # Predict probabilities
                    try:
                        proba = meta_model.predict_proba(X_meta)
                        final = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
                    except Exception:
                        try:
                            final = meta_model.decision_function(X_meta)
                        except Exception:
                            final = meta_model.predict(X_meta)
                    return predictor._normalize_scores_to_proba(final)
                else:
                    # Per-fold transform and predict across model types, then average across folds
                    per_fold_ensembled = []
                    for i in range(len(fold_pipes)):
                        pipe_path = run_path / f"fold_{i}_feature_pipeline.joblib"
                        pipe = _joblib.load(pipe_path)
                        Xt = pipe.transform(test_raw_df)
                        # Drop ID/target if present
                        for col in (data_cfg_local.id_column, data_cfg_local.target_column):
                            if col and col in Xt.columns:
                                Xt = Xt.drop(columns=[col])
                        Xt = Xt.select_dtypes(include=["number", "bool"]).replace([np.inf, -np.inf], np.nan).fillna(0)

                        fold_model_paths = sorted(run_path.glob(f"fold_{i}_model_*.joblib"))
                        if not fold_model_paths:
                            continue
                        fold_model_preds = []
                        for mpath in fold_model_paths:
                            model = _joblib.load(mpath)
                            raw = predictor._predict_single_model(model, Xt)
                            proba = predictor._normalize_scores_to_proba(raw)
                            fold_model_preds.append(proba)
                        if not fold_model_preds:
                            continue
                        fold_ens = predictor._ensemble_predictions(
                            fold_model_preds,
                            inference_cfg_local.get("ensemble_method", "average"),
                            inference_cfg_local.get("ensemble_weights"),
                        )
                        per_fold_ensembled.append(np.clip(fold_ens, 0.0, 1.0))
                    if not per_fold_ensembled:
                        raise RuntimeError("All fold predictions failed with per-fold pipelines")
                    return np.mean(np.vstack(per_fold_ensembled), axis=0)

            elif fold_pipes:
                # Single-model-per-fold path
                from infer.predictor import ModelLoader as _ML
                model_loader = _ML()
                models = model_loader.load_fold_models(str(run_path))
                # Load raw test
                try:
                    data_cfg_dict = config_manager.load_config("data")
                    data_cfg_local = DataConfig(**data_cfg_dict)
                except Exception as e:
                    raise RuntimeError(f"Could not load data config for raw test path: {e}")
                raw_test_path = Path(data_cfg_local.test_path)
                if not raw_test_path.exists():
                    alt = Path('data') / raw_test_path.name
                    if alt.exists():
                        raw_test_path = alt
                if not raw_test_path.exists():
                    raise FileNotFoundError(f"Raw test data not found at {data_cfg_local.test_path}")
                test_raw_df = pd.read_csv(raw_test_path)
                import joblib as _joblib
                per_fold_probs = []
                for i, model in enumerate(models):
                    pipe_path = run_path / f"fold_{i}_feature_pipeline.joblib"
                    pipe = _joblib.load(pipe_path)
                    Xt = pipe.transform(test_raw_df)
                    for col in (data_cfg_local.id_column, data_cfg_local.target_column):
                        if col and col in Xt.columns:
                            Xt = Xt.drop(columns=[col])
                    Xt = Xt.select_dtypes(include=["number", "bool"]).replace([np.inf, -np.inf], np.nan).fillna(0)
                    raw = predictor._predict_single_model(model, Xt)
                    proba = predictor._normalize_scores_to_proba(raw)
                    per_fold_probs.append(proba)
                if not per_fold_probs:
                    raise RuntimeError("All fold predictions failed with per-fold pipelines")
                return np.clip(predictor._ensemble_predictions(per_fold_probs, inference_cfg_local.get("ensemble_method", "average"), inference_cfg_local.get("ensemble_weights")), 0.0, 1.0)

            else:
                # Fallback to processed features
                processed_dir = path_manager.data_dir / "processed"
                test_path = processed_dir / "test_features.csv"
                if not test_path.exists():
                    raise FileNotFoundError("Processed test data not found. Run 'features' command first.")
                test_df = pd.read_csv(test_path)
                test_model_df = test_df.copy()
                try:
                    data_config_dict = config_manager.load_config("data")
                    data_cfg_local = DataConfig(**data_config_dict)
                except Exception:
                    data_cfg_local = None
                if data_cfg_local and getattr(data_cfg_local, 'train_columns', None):
                    requested = list(data_cfg_local.train_columns or [])
                    keep_cols = [c for c in requested if c in test_model_df.columns]
                    if keep_cols:
                        test_model_df = test_model_df[keep_cols]
                    else:
                        drop_cols = [c for c in ['Name','Ticket','Cabin'] if c in test_model_df.columns]
                        if drop_cols:
                            test_model_df = test_model_df.drop(columns=drop_cols)
                else:
                    drop_cols = [c for c in ['Name','Ticket','Cabin'] if c in test_model_df.columns]
                    if drop_cols:
                        test_model_df = test_model_df.drop(columns=drop_cols)
                if data_cfg_local and getattr(data_cfg_local, 'exclude_column_for_training', None):
                    excl = [c for c in (data_cfg_local.exclude_column_for_training or []) if c in test_model_df.columns]
                    if excl:
                        test_model_df = test_model_df.drop(columns=excl)
                if "PassengerId" in test_model_df.columns:
                    test_model_df = test_model_df.set_index("PassengerId")
                test_model_df = test_model_df.select_dtypes(include=["number", "bool"]).replace([np.inf, -np.inf], np.nan).fillna(0)
                # Load models
                from infer.predictor import ModelLoader as _ML
                model_loader = _ML()
                models = model_loader.load_fold_models(str(run_path))
                proba_df = predictor.predict_proba(test_model_df, models, inference_cfg_local)
                return proba_df["prediction_proba"].values

        # Update inference_cfg with (possibly) overridden threshold config
        # Note: in multi-run mode the chosen threshold will be resolved later without relying on a single run_dir
        inference_cfg["threshold"] = th_cfg

        # Predictor instance for ensembling helper
        predictor = create_predictor(inference_cfg)

        # Compute proba per run
        run_probas = []
        for r in run_specs:
            rp = Path(r["path"]).resolve()
            if not rp.exists():
                click.echo(f"⚠️ Skipping missing run directory: {rp}")
                continue
            proba_vec = _predict_for_run(rp, predictor, inference_cfg)
            run_probas.append(proba_vec)
        if not run_probas:
            click.echo("❌ No predictions produced from provided runs")
            return

        # Cross-run ensemble
        run_weights = None
        if all(isinstance(r, dict) and r.get("weight") is not None for r in run_specs):
            run_weights = [float(r.get("weight")) for r in run_specs if Path(r["path"]).exists()]
        final_proba = predictor._ensemble_predictions(run_probas, inference_cfg.get("ensemble_method", "average"), run_weights or inference_cfg.get("ensemble_weights"))
        final_proba = np.clip(final_proba, 0.0, 1.0)

        # Choose threshold: if exactly one run, allow artifact discovery; else use config/CLI-provided
        if len(run_probas) == 1 and run_specs:
            inference_cfg["run_dir"] = str(Path(run_specs[0]["path"]))
        else:
            inference_cfg.pop("run_dir", None)
        thr = TitanicPredictor(inference_cfg)._resolve_threshold(inference_cfg)

        # Build output dataframe (need PassengerId from raw test)
        # Load raw test once (assumes all runs used same data config)
        try:
            data_cfg_dict = config_manager.load_config("data")
            data_cfg_local = DataConfig(**data_cfg_dict)
        except Exception as e:
            click.echo(f"❌ Could not load data config for raw test path: {e}")
            return
        raw_test_path = Path(data_cfg_local.test_path)
        if not raw_test_path.exists():
            alt = Path('data') / raw_test_path.name
            if alt.exists(): raw_test_path = alt
        if not raw_test_path.exists():
            click.echo(f"❌ Raw test data not found at {data_cfg_local.test_path}")
            return
        test_raw_df = pd.read_csv(raw_test_path)
        predictions = pd.DataFrame({
            "PassengerId": test_raw_df["PassengerId"].values if "PassengerId" in test_raw_df.columns else np.arange(len(final_proba)),
            "prediction_proba": final_proba,
            "prediction": (final_proba >= thr).astype(int)
        })

        # Save predictions
        # Save predictions
        if output_path:
            pred_path = Path(output_path)
        else:
            # If one run → save inside that run; else create combined file next to first run
            if run_specs:
                base = Path(run_specs[0]["path"]).resolve()
                fname = "predictions.csv" if len(run_probas) == 1 else "predictions_ensemble.csv"
                pred_path = base / fname
            else:
                pred_path = Path("artifacts/predictions.csv")
        predictions.to_csv(pred_path, index=False)
        click.echo(f"✅ Predictions saved to {pred_path}")

        # Report threshold actually used
        th_cfg = inference_cfg.get("threshold", {}) or {}
        try:
            used_thr = predictor._resolve_threshold(inference_cfg)
            click.echo(f"   📊 Threshold: {used_thr:.4f}")

            # Print threshold details if requested
            if th_cfg.get("print", False):
                click.echo("   📊 Threshold configuration:")
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
            click.echo("   ✅ Using binary predictions already present in predictions.csv")

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

    click.echo("\n🤖 Available models:")
    registry = ModelRegistry()
    available_models = registry.get_available_models()
    for model in sorted(available_models):
        click.echo(f"   - {model}")

    click.echo("\n📋 Configuration files:")
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
