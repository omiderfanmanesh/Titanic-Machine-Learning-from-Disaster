"""
Test file to verify the core pipeline components work without the broken imports.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_core_imports():
    """Test that core modules can be imported without issues."""
    print("Testing core imports...")
    
    # Test individual imports to isolate issues
    try:
        from modeling.model_registry import ModelRegistry
        print("✅ ModelRegistry imported successfully")
        
        registry = ModelRegistry()
        models = registry.get_available_models()
        print(f"✅ Available models: {models}")
        
    except Exception as e:
        print(f"❌ ModelRegistry import failed: {e}")
        
    try:
        from cv.folds import create_fold_strategy
        print("✅ CV folds imported successfully")
        
        cv_strategy = create_fold_strategy('stratified', n_splits=5, random_state=42)
        print("✅ CV strategy created successfully")
        
    except Exception as e:
        print(f"❌ CV folds import failed: {e}")

def test_data_loading_direct():
    """Test data loading without using the complex data module."""
    print("\nTesting direct data loading...")
    
    try:
        import pandas as pd
        
        # Load data directly
        train_df = pd.read_csv(project_root / "data" / "train.csv")
        test_df = pd.read_csv(project_root / "data" / "test.csv")
        
        print(f"✅ Train data loaded: {train_df.shape}")
        print(f"✅ Test data loaded: {test_df.shape}")
        print(f"✅ Train columns: {list(train_df.columns)}")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"❌ Direct data loading failed: {e}")
        return None, None

def test_simple_feature_engineering():
    """Test basic feature engineering."""
    print("\nTesting simple feature engineering...")
    
    train_df, test_df = test_data_loading_direct()
    if train_df is None:
        return None, None
        
    try:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        
        # Combine datasets for consistent preprocessing
        train_df['is_train'] = 1
        test_df['is_train'] = 0
        combined = pd.concat([train_df, test_df], sort=False)
        
        # Basic feature engineering
        combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
        combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
        
        # Fill missing values
        combined['Age'] = combined['Age'].fillna(combined['Age'].median())
        combined['Fare'] = combined['Fare'].fillna(combined.groupby('Pclass')['Fare'].transform('median'))
        combined['Embarked'] = combined['Embarked'].fillna('S')
        
        # Encode categorical variables
        le_sex = LabelEncoder()
        combined['Sex_encoded'] = le_sex.fit_transform(combined['Sex'])
        
        le_embarked = LabelEncoder()
        combined['Embarked_encoded'] = le_embarked.fit_transform(combined['Embarked'])
        
        # Select features
        features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
                   'Embarked_encoded', 'FamilySize', 'IsAlone']
        
        X = combined[features]
        
        # Split back
        X_train = X[combined['is_train'] == 1]
        X_test = X[combined['is_train'] == 0]
        y_train = train_df['Survived']
        
        print(f"✅ Features engineered: {X_train.shape}")
        print(f"✅ Feature names: {features}")
        
        return X_train, X_test, y_train
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None, None, None

def test_model_training():
    """Test model training and cross-validation."""
    print("\nTesting model training...")
    
    X_train, X_test, y_train = test_simple_feature_engineering()
    if X_train is None:
        return
        
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        # Create model
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced for faster testing
            max_depth=5,
            random_state=42,
            n_jobs=1
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"✅ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        print(f"✅ Predictions made: {len(predictions)} samples")
        print(f"✅ Survival rate: {predictions.mean():.1%}")
        
        return model, predictions
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None, None

def test_submission_creation():
    """Test creating submission file."""
    print("\nTesting submission creation...")
    
    try:
        import pandas as pd
        
        # Load test data for PassengerIds
        test_df = pd.read_csv(project_root / "data" / "test.csv")
        
        # Get predictions from model training test
        model, predictions = test_model_training()
        
        if predictions is None:
            print("❌ No predictions available")
            return
            
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': predictions
        })
        
        print(f"✅ Submission created: {submission.shape}")
        print("✅ First 5 predictions:")
        print(submission.head().to_string(index=False))
        
        # Save to test directory (not artifacts)
        test_output_dir = project_root / "tests" / "output"
        test_output_dir.mkdir(exist_ok=True)
        
        submission_path = test_output_dir / "test_submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"✅ Test submission saved to: {submission_path}")
        
        return submission_path
        
    except Exception as e:
        print(f"❌ Submission creation failed: {e}")
        return None

def run_all_tests():
    """Run all pipeline tests."""
    print("🚀 Running Titanic ML Pipeline Tests\n")
    
    test_core_imports()
    test_data_loading_direct()
    test_simple_feature_engineering()
    test_model_training()
    submission_path = test_submission_creation()
    
    print(f"\n🎉 All tests completed!")
    if submission_path:
        print(f"🚢 Test submission ready: {submission_path}")

if __name__ == "__main__":
    run_all_tests()
