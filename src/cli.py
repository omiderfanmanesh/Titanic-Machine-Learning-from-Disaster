"""Command-line interface for Titanic ML pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from core.utils import (
    ConfigManager,
    ExperimentConfig,
    DataConfig,
    InferenceConfig,
    LoggerFactory,
    PathManager,
    SeedManager,
)
from data.loader import TitanicDataLoader
try:
    from data.validate import TitanicDataValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
from features.build import create_feature_builder
from modeling.model_registry import ModelRegistry
from modeling.trainers import TitanicTrainer
from eval.evaluator import TitanicEvaluator
from infer.predictor import create_predictor, ModelLoader
from submit.build_submission import TitanicSubmissionBuilder
from cv.folds import create_splits_with_validation


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
        loader = TitanicDataLoader(data_config_dict)
        train_df = loader.load(data_cfg.train_path)
        test_df = loader.load(data_cfg.test_path)
        
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
        X = train_df.drop(columns=[data_cfg.target_column])
        y = train_df[data_cfg.target_column]
        
        # Create model
        model_config = {
            "model_name": experiment_cfg.model_name,
            "model_params": experiment_cfg.model_params
        }
        
        model = ModelRegistry.create_model(experiment_cfg.model_name)
        estimator = model.build(model_config)
        
        # Create trainer
        trainer_config = {
            "strategy": experiment_cfg.cv_strategy,
            "n_folds": experiment_cfg.cv_folds,
            "shuffle": experiment_cfg.cv_shuffle,
            "random_state": experiment_cfg.cv_random_state
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
        
        # Comprehensive evaluation
        evaluation_results = evaluator.evaluate_cv(
            oof_df["prediction"], 
            oof_df["target"], 
            scores_data["fold_scores"],
            config={}
        )
        
        # Display results
        click.echo("📊 Evaluation Results")
        click.echo(f"   AUC: {evaluation_results['oof_metrics']['auc']:.4f}")
        click.echo(f"   Accuracy: {evaluation_results['oof_metrics']['accuracy']:.4f}")
        click.echo(f"   F1 Score: {evaluation_results['oof_metrics']['f1']:.4f}")
        
        click.echo(f"\n📈 Cross-Validation Statistics")
        click.echo(f"   Mean: {evaluation_results['cv_statistics']['cv_mean']:.4f}")
        click.echo(f"   Std: {evaluation_results['cv_statistics']['cv_std']:.4f}")
        click.echo(f"   Range: {evaluation_results['cv_statistics']['cv_max'] - evaluation_results['cv_statistics']['cv_min']:.4f}")
        
        # Stability assessment
        stability = evaluation_results['stability']
        click.echo(f"\n🎯 Model Stability")
        click.echo(f"   Stable: {'✅ Yes' if stability['is_stable'] else '❌ No'}")
        click.echo(f"   CV of Variation: {stability['coefficient_of_variation']:.3f}")
        
        # Save detailed evaluation
        eval_path = run_path / "evaluation_report.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)
            
        click.echo(f"📄 Detailed evaluation saved to {eval_path}")
        
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}")


@cli.command()
@click.option("--run-dir", type=click.Path(exists=True), required=True,
              help="Training run directory")
@click.option("--inference-config", default="inference", help="Inference configuration name")
@click.option("--output-path", type=click.Path(), help="Output path for predictions")
def predict(run_dir: str, inference_config: str, output_path: Optional[str]):
    """Generate predictions on test data."""
    
    try:
        # Load configurations
        inference_config_dict = config_manager.load_config(inference_config)
        inference_cfg = InferenceConfig(**inference_config_dict)
        
        # Load test data
        processed_dir = path_manager.data_dir / "processed"
        test_path = processed_dir / "test_features.csv"
        
        if not test_path.exists():
            click.echo("❌ Processed test data not found. Run 'features' command first.")
            return
            
        test_df = pd.read_csv(test_path)
        click.echo(f"📥 Loaded test data: {test_df.shape}")
        
        # Load models
        model_loader = ModelLoader()
        models = model_loader.load_fold_models(run_dir)
        click.echo(f"🔄 Loaded {len(models)} fold models")
        
        # Create predictor
        predictor = create_predictor(inference_config_dict)
        
        # Generate predictions
        click.echo("🔮 Generating predictions...")
        predictions = predictor.predict_proba(test_df, models, inference_config_dict)
        
        # Save predictions
        if output_path:
            pred_path = Path(output_path)
        else:
            pred_path = path_manager.artifacts_dir / "predictions.csv"
            
        predictions.to_csv(pred_path, index=False)
        
        click.echo(f"✅ Predictions saved to {pred_path}")
        click.echo(f"   📊 Prediction distribution:")
        click.echo(f"      Mean: {predictions['prediction'].mean():.3f}")
        click.echo(f"      Std: {predictions['prediction'].std():.3f}")
        click.echo(f"      Min: {predictions['prediction'].min():.3f}")
        click.echo(f"      Max: {predictions['prediction'].max():.3f}")
        
    except Exception as e:
        click.echo(f"❌ Prediction failed: {e}")


@cli.command()
@click.option("--predictions-path", type=click.Path(exists=True), required=True,
              help="Path to predictions CSV")
@click.option("--output-path", type=click.Path(), help="Output path for submission")
@click.option("--threshold", type=float, default=0.5, help="Classification threshold")
def submit(predictions_path: str, output_path: Optional[str], threshold: float):
    """Build Kaggle submission file."""
    
    try:
        # Load predictions
        predictions_df = pd.read_csv(predictions_path)
        
        # Create submission builder
        builder = TitanicSubmissionBuilder()
        
        # Build submission
        config = {
            "threshold": threshold,
            "add_metadata": True
        }
        
        submission = builder.build_submission(predictions_df, config)
        
        # Validate submission
        if not builder.validate_submission(submission):
            click.echo("❌ Submission validation failed")
            return
        
        # Save submission
        if output_path:
            sub_path = Path(output_path)
        else:
            sub_path = path_manager.artifacts_dir / "submission.csv"
            
        builder.save_submission(submission, sub_path, config)
        
        # Summary statistics
        positive_rate = submission["Survived"].mean()
        
        click.echo(f"✅ Submission created: {sub_path}")
        click.echo(f"   📊 Samples: {len(submission)}")
        click.echo(f"   📊 Positive rate: {positive_rate:.3f}")
        click.echo(f"   📊 Threshold used: {threshold}")
        
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
