#!/usr/bin/env python3
"""
Simple pipeline runner to train model and create Kaggle submission.
This bypasses CLI issues and runs the pipeline directly.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from data.loader import TitanicDataLoader
from features.build import create_feature_builder
from modeling.model_registry import ModelRegistry
from cv.folds import create_fold_strategy
from eval.evaluator import ModelEvaluator
from core.utils import PathManager, SeedManager, LoggerFactory
from sklearn.model_selection import cross_val_score
import joblib
from datetime import datetime

def main():
    """Run the complete ML pipeline."""
    
    # Setup
    logger = LoggerFactory.get_logger(__name__)
    path_manager = PathManager()
    SeedManager.set_seed(42)
    
    logger.info("🚀 Starting Titanic ML Pipeline")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = path_manager.artifacts_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data
        logger.info("📊 Loading data...")
        loader = TitanicDataLoader(
            train_file="data/train.csv",
            test_file="data/test.csv"
        )
        train_df, test_df = loader.load()
        logger.info(f"✅ Loaded train: {train_df.shape}, test: {test_df.shape}")
        
        # 2. Build features
        logger.info("🔧 Building features...")
        feature_builder = create_feature_builder()
        
        # Fit on training data and transform both sets
        X_train = feature_builder.fit_transform(train_df)
        X_test = feature_builder.transform(test_df)
        y_train = train_df['Survived']
        
        logger.info(f"✅ Features built - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 3. Train model with cross-validation
        logger.info("🤖 Training Random Forest model...")
        registry = ModelRegistry()
        model = registry.create_model('random_forest', 
                                    n_estimators=200,
                                    max_depth=8,
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    random_state=42,
                                    n_jobs=-1)
        
        # Perform cross-validation
        cv_strategy = create_fold_strategy('stratified', n_splits=5, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        logger.info(f"✅ Cross-validation AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Train final model on all data
        logger.info("🎯 Training final model on all training data...")
        model.fit(X_train, y_train)
        
        # 4. Make predictions on test set
        logger.info("🔮 Making predictions on test set...")
        test_predictions = model.predict(X_test)
        
        # 5. Create submission file
        logger.info("📄 Creating Kaggle submission...")
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': test_predictions
        })
        
        submission_path = output_dir / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        logger.info(f"✅ Submission saved to: {submission_path}")
        
        # 6. Save model artifacts
        logger.info("💾 Saving model artifacts...")
        
        # Save trained model
        model_path = output_dir / 'model.joblib'
        joblib.dump(model, model_path)
        
        # Save feature builder
        features_path = output_dir / 'feature_builder.joblib'
        joblib.dump(feature_builder, features_path)
        
        # Save cross-validation results
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'model_params': model.get_params(),
            'timestamp': timestamp
        }
        
        with open(output_dir / 'cv_results.json', 'w') as f:
            import json
            json.dump(cv_results, f, indent=2)
        
        # Print summary
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📊 CV AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
        logger.info(f"🎯 Test predictions: {len(test_predictions)} samples")
        logger.info(f"📄 Submission file: {submission_path}")
        
        # Display first few predictions
        print("\n📋 First 10 predictions:")
        print(submission.head(10).to_string(index=False))
        
        print(f"\n🏆 Survival rate in predictions: {test_predictions.mean():.1%}")
        
        # Print feature importance if available
        if hasattr(model, 'feature_importances_'):
            print(f"\n🔍 Top 10 Most Important Features:")
            feature_names = X_train.columns
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(10).to_string(index=False))
        
        return str(submission_path)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    submission_file = main()
    print(f"\n🚢 Ready for Kaggle submission: {submission_file}")
