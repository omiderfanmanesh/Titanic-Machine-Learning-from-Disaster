"""Integration tests for the complete ML pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from data.loader import TitanicDataLoader
from features.build import TitanicFeatureBuilder
from modeling.model_registry import ModelRegistry
from modeling.trainers import TitanicTrainer
from eval.evaluator import TitanicEvaluator
from infer.predictor import TitanicPredictor
from submit.build_submission import TitanicSubmissionBuilder
from cv.folds import StratifiedKFoldSplitter
from core.utils import ConfigManager


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return {
        "experiment": {
            "name": "integration_test",
            "seed": 42,
            "cv_folds": 3,
            "scoring": "accuracy"
        },
        "model": {
            "name": "logistic",
            "params": {
                "C": 1.0,
                "random_state": 42,
                "max_iter": 1000
            }
        },
        "features": {
            "numeric_features": ["Age", "Fare"],
            "categorical_features": ["Sex", "Pclass"],
            "use_family_features": True,
            "use_title_features": True,
            "use_deck_features": True
        }
    }


@pytest.fixture
def synthetic_titanic_data():
    """Generate synthetic Titanic data for integration testing."""
    np.random.seed(42)
    n_samples = 200
    
    # Create synthetic training data
    train_data = pd.DataFrame({
        "PassengerId": range(1, n_samples + 1),
        "Survived": np.random.randint(0, 2, n_samples),
        "Pclass": np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        "Name": [f"Person{i}, Mr. John" for i in range(n_samples)],
        "Sex": np.random.choice(["male", "female"], n_samples, p=[0.6, 0.4]),
        "Age": np.random.normal(30, 15, n_samples).clip(0, 80),
        "SibSp": np.random.poisson(0.5, n_samples).clip(0, 5),
        "Parch": np.random.poisson(0.3, n_samples).clip(0, 3),
        "Ticket": [f"TICKET{i//5}" for i in range(n_samples)],  # Some shared tickets
        "Fare": np.random.lognormal(3, 1, n_samples).clip(0, 500),
        "Cabin": [f"C{np.random.randint(1, 100)}" if np.random.rand() > 0.7 else None 
                 for _ in range(n_samples)],
        "Embarked": np.random.choice(["S", "C", "Q"], n_samples, p=[0.7, 0.2, 0.1])
    })
    
    # Add some realistic patterns
    # Higher class passengers more likely to survive
    survival_prob = 0.8 - 0.2 * (train_data["Pclass"] - 1)
    # Females more likely to survive
    survival_prob = np.where(train_data["Sex"] == "female", 
                            survival_prob + 0.3, survival_prob - 0.1)
    # Clip probabilities
    survival_prob = survival_prob.clip(0.1, 0.9)
    
    # Generate survival based on probabilities
    train_data["Survived"] = np.random.binomial(1, survival_prob)
    
    # Create synthetic test data (no Survived column)
    n_test = 50
    test_data = pd.DataFrame({
        "PassengerId": range(n_samples + 1, n_samples + n_test + 1),
        "Pclass": np.random.choice([1, 2, 3], n_test, p=[0.2, 0.3, 0.5]),
        "Name": [f"TestPerson{i}, Mrs. Jane" for i in range(n_test)],
        "Sex": np.random.choice(["male", "female"], n_test, p=[0.6, 0.4]),
        "Age": np.random.normal(30, 15, n_test).clip(0, 80),
        "SibSp": np.random.poisson(0.5, n_test).clip(0, 5),
        "Parch": np.random.poisson(0.3, n_test).clip(0, 3),
        "Ticket": [f"TEST_TICKET{i//3}" for i in range(n_test)],
        "Fare": np.random.lognormal(3, 1, n_test).clip(0, 500),
        "Cabin": [f"D{np.random.randint(1, 100)}" if np.random.rand() > 0.7 else None 
                 for _ in range(n_test)],
        "Embarked": np.random.choice(["S", "C", "Q"], n_test, p=[0.7, 0.2, 0.1])
    })
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    train_data.loc[missing_indices, "Age"] = np.nan
    
    missing_indices_test = np.random.choice(n_test, size=int(0.2 * n_test), replace=False)
    test_data.loc[missing_indices_test, "Age"] = np.nan
    
    return train_data, test_data


class TestEndToEndPipeline:
    """Test complete end-to-end ML pipeline."""
    
    def test_complete_pipeline_workflow(self, synthetic_titanic_data, integration_config):
        """Test complete pipeline from data loading to submission."""
        train_data, test_data = synthetic_titanic_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save synthetic data
            train_file = temp_path / "train.csv"
            test_file = temp_path / "test.csv"
            train_data.to_csv(train_file, index=False)
            test_data.to_csv(test_file, index=False)
            
            # 1. Data Loading
            loader = TitanicDataLoader(train_file, test_file)
            train_df, test_df = loader.load()
            
            assert len(train_df) == len(train_data)
            assert len(test_df) == len(test_data)
            assert "Survived" in train_df.columns
            assert "Survived" not in test_df.columns
            
            # 2. Feature Engineering
            feature_builder = TitanicFeatureBuilder()
            feature_builder.fit(train_df)
            
            X_train = feature_builder.transform(train_df)
            X_test = feature_builder.transform(test_df)
            y_train = train_df["Survived"]
            
            # Check feature engineering worked
            assert len(X_train) == len(train_df)
            assert len(X_test) == len(test_df)
            assert X_train.shape[1] == X_test.shape[1]  # Same number of features
            
            # Should have engineered features
            expected_features = ["FamilySize", "IsAlone", "Title"]
            for feature in expected_features:
                assert feature in X_train.columns, f"Missing feature: {feature}"
            
            # 3. Model Training
            model_registry = ModelRegistry()
            model = model_registry.create_model(
                integration_config["model"]["name"],
                integration_config["model"]["params"]
            )
            
            fold_splitter = StratifiedKFoldSplitter(
                n_splits=integration_config["experiment"]["cv_folds"],
                shuffle=True,
                random_state=integration_config["experiment"]["seed"]
            )
            
            trainer = TitanicTrainer(
                config=integration_config,
                output_dir=temp_path
            )
            
            training_result = trainer.train(
                model=model,
                X=X_train,
                y=y_train,
                fold_splitter=fold_splitter
            )
            
            # Check training results
            assert "cv_scores" in training_result
            assert "oof_predictions" in training_result
            assert len(training_result["oof_predictions"]) == len(y_train)
            
            # 4. Model Evaluation
            evaluator = TitanicEvaluator()
            eval_result = evaluator.evaluate(
                y_true=y_train,
                oof_predictions=training_result["oof_predictions"],
                cv_scores=training_result["cv_scores"]
            )
            
            assert "accuracy" in eval_result
            assert "precision" in eval_result
            assert "recall" in eval_result
            assert "f1" in eval_result
            
            # Performance should be reasonable for synthetic data
            assert eval_result["accuracy"] > 0.5  # Better than random
            
            # 5. Inference
            predictor = TitanicPredictor()
            test_predictions = predictor.predict(
                model=training_result["model"],
                X=X_test
            )
            
            assert len(test_predictions) == len(test_df)
            assert all(pred in [0, 1] for pred in test_predictions)
            
            # 6. Submission Building
            submission_builder = TitanicSubmissionBuilder()
            submission = submission_builder.build_submission(
                passenger_ids=test_df["PassengerId"],
                predictions=test_predictions,
                output_file=temp_path / "submission.csv"
            )
            
            # Check submission format
            assert len(submission) == len(test_df)
            assert list(submission.columns) == ["PassengerId", "Survived"]
            assert submission["PassengerId"].tolist() == test_df["PassengerId"].tolist()
            
            # Check submission file was created
            submission_file = temp_path / "submission.csv"
            assert submission_file.exists()
            
            # Verify saved submission
            saved_submission = pd.read_csv(submission_file)
            pd.testing.assert_frame_equal(saved_submission, submission)
            
    def test_pipeline_reproducibility(self, synthetic_titanic_data, integration_config):
        """Test that pipeline produces reproducible results."""
        train_data, test_data = synthetic_titanic_data
        
        results = []
        
        for run in range(2):  # Run twice
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save data
                train_file = temp_path / "train.csv"
                test_file = temp_path / "test.csv"
                train_data.to_csv(train_file, index=False)
                test_data.to_csv(test_file, index=False)
                
                # Run pipeline
                loader = TitanicDataLoader(train_file, test_file)
                train_df, test_df = loader.load()
                
                feature_builder = TitanicFeatureBuilder()
                feature_builder.fit(train_df)
                X_train = feature_builder.transform(train_df)
                y_train = train_df["Survived"]
                
                model_registry = ModelRegistry()
                model = model_registry.create_model(
                    integration_config["model"]["name"],
                    integration_config["model"]["params"]
                )
                
                fold_splitter = StratifiedKFoldSplitter(
                    n_splits=3,
                    shuffle=True,
                    random_state=42  # Fixed seed
                )
                
                trainer = TitanicTrainer(
                    config=integration_config,
                    output_dir=temp_path
                )
                
                training_result = trainer.train(
                    model=model,
                    X=X_train,
                    y=y_train,
                    fold_splitter=fold_splitter
                )
                
                results.append(training_result["cv_scores"])
        
        # Results should be identical (within numerical precision)
        np.testing.assert_array_almost_equal(
            results[0]["mean"],
            results[1]["mean"],
            decimal=10
        )
        
    def test_pipeline_with_missing_data(self, integration_config):
        """Test pipeline handles missing data gracefully."""
        # Create data with extensive missing values
        np.random.seed(42)
        n_samples = 100
        
        train_data = pd.DataFrame({
            "PassengerId": range(1, n_samples + 1),
            "Survived": np.random.randint(0, 2, n_samples),
            "Pclass": np.random.choice([1, 2, 3], n_samples),
            "Name": [f"Person{i}, Mr. John" for i in range(n_samples)],
            "Sex": np.random.choice(["male", "female"], n_samples),
            "Age": np.where(np.random.rand(n_samples) > 0.5, 
                           np.random.normal(30, 15, n_samples), np.nan),  # 50% missing
            "SibSp": np.random.poisson(0.5, n_samples),
            "Parch": np.random.poisson(0.3, n_samples),
            "Ticket": [f"TICKET{i//5}" for i in range(n_samples)],
            "Fare": np.where(np.random.rand(n_samples) > 0.3,
                           np.random.lognormal(3, 1, n_samples), np.nan),  # 30% missing
            "Cabin": [None] * n_samples,  # All missing
            "Embarked": np.where(np.random.rand(n_samples) > 0.1,
                               np.random.choice(["S", "C", "Q"], n_samples), None)  # 10% missing
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            train_file = temp_path / "train.csv"
            train_data.to_csv(train_file, index=False)
            
            # Should handle missing data without crashing
            loader = TitanicDataLoader(train_file, train_file)  # Use same file for test
            train_df, _ = loader.load()
            
            feature_builder = TitanicFeatureBuilder()
            feature_builder.fit(train_df)
            X_train = feature_builder.transform(train_df)
            
            # Should not have any infinite or extremely large values
            assert np.isfinite(X_train.select_dtypes(include=[np.number]).values).all()
            
            # Should have handled missing values (no NaN in final features)
            assert not X_train.select_dtypes(include=[np.number]).isnull().any().any()
            
    def test_pipeline_error_handling(self, integration_config):
        """Test pipeline error handling for invalid data."""
        # Create invalid data
        invalid_train = pd.DataFrame({
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 2],  # Invalid value (should be 0 or 1)
            "InvalidColumn": ["a", "b", "c"]  # Missing required columns
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            train_file = temp_path / "invalid_train.csv"
            invalid_train.to_csv(train_file, index=False)
            
            loader = TitanicDataLoader(train_file, train_file)
            train_df, _ = loader.load()
            
            # Feature builder should handle missing columns gracefully
            feature_builder = TitanicFeatureBuilder()
            
            # Should raise informative error for missing required columns
            with pytest.raises(Exception):  # Could be KeyError or custom error
                feature_builder.fit(train_df)
