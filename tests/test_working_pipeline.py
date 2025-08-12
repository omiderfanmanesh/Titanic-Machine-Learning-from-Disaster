"""
Working end-to-end pipeline test that creates a Kaggle submission.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_end_to_end_pipeline():
    """Test complete pipeline and create submission."""
    print("🚀 Testing End-to-End Titanic ML Pipeline")
    
    # Create test output directory
    test_output_dir = project_root / "tests" / "output"
    test_output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Load data
        print("\n📊 Step 1: Loading data...")
        train_df = pd.read_csv(project_root / "data" / "train.csv")
        test_df = pd.read_csv(project_root / "data" / "test.csv")
        print(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")
        
        # 2. Feature Engineering
        print("\n🔧 Step 2: Feature engineering...")
        
        # Combine for consistent preprocessing
        train_df['is_train'] = 1
        test_df['is_train'] = 0
        combined = pd.concat([train_df, test_df], sort=False)
        
        # Basic feature engineering
        combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
        combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
        
        # Extract title
        combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        combined['Title'] = combined['Title'].map(title_mapping).fillna('Mr')
        
        # Fill missing values
        combined['Age'] = combined.groupby('Title')['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        combined['Fare'] = combined.groupby('Pclass')['Fare'].transform(
            lambda x: x.fillna(x.median())
        )
        combined['Embarked'] = combined['Embarked'].fillna('S')
        
        # Create age and fare bins
        combined['AgeBin'] = pd.cut(combined['Age'], bins=5, labels=False)
        combined['FareBin'] = pd.cut(combined['Fare'], bins=5, labels=False)
        
        # Encode categorical variables
        le_sex = LabelEncoder()
        combined['Sex_encoded'] = le_sex.fit_transform(combined['Sex'])
        
        le_embarked = LabelEncoder()
        combined['Embarked_encoded'] = le_embarked.fit_transform(combined['Embarked'])
        
        le_title = LabelEncoder()
        combined['Title_encoded'] = le_title.fit_transform(combined['Title'])
        
        # Select final features
        features = [
            'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_encoded', 'FamilySize', 'IsAlone', 'Title_encoded',
            'AgeBin', 'FareBin'
        ]
        
        X = combined[features]
        
        # Split back to train and test
        X_train = X[combined['is_train'] == 1].reset_index(drop=True)
        X_test = X[combined['is_train'] == 0].reset_index(drop=True)
        y_train = train_df['Survived'].reset_index(drop=True)
        
        print(f"✅ Features: {X_train.shape}")
        print(f"✅ Feature names: {features}")
        
        # 3. Model Training and Cross-Validation
        print("\n🤖 Step 3: Model training...")
        
        # Test with ModelRegistry
        try:
            from modeling.model_registry import ModelRegistry
            registry = ModelRegistry()
            model = registry.create_model('random_forest',
                                        n_estimators=100,
                                        max_depth=8,
                                        min_samples_split=5,
                                        random_state=42,
                                        n_jobs=-1)
            print("✅ Using ModelRegistry for Random Forest")
        except Exception as e:
            print(f"⚠️  ModelRegistry failed ({e}), using sklearn directly")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        print(f"✅ CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        print("✅ Final model trained")
        
        # 4. Predictions
        print("\n🔮 Step 4: Making predictions...")
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        print(f"✅ Predictions: {len(predictions)} samples")
        print(f"✅ Survival rate: {predictions.mean():.1%}")
        
        # 5. Create Submission
        print("\n📄 Step 5: Creating submission...")
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': predictions
        })
        
        # Save submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = test_output_dir / f"titanic_submission_{timestamp}.csv"
        submission.to_csv(submission_path, index=False)
        
        print(f"✅ Submission saved: {submission_path}")
        
        # 6. Results Summary
        print(f"\n🎉 Pipeline Test Completed Successfully!")
        print(f"📊 Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"🎯 Test predictions: {len(predictions)} samples")
        print(f"🏆 Predicted survival rate: {predictions.mean():.1%}")
        print(f"📁 Submission file: {submission_path}")
        
        # Show first few predictions
        print(f"\n📋 First 10 predictions:")
        print(submission.head(10).to_string(index=False))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 5 Important Features:")
            print(importance_df.head().to_string(index=False))
        
        return submission_path, cv_scores.mean()
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    submission_path, cv_score = test_end_to_end_pipeline()
    
    if submission_path:
        print(f"\n🚢 Ready for Kaggle submission!")
        print(f"📄 File: {submission_path}")
        print(f"📊 Expected AUC: ~{cv_score:.4f}")
        print(f"\n💡 To submit to Kaggle:")
        print(f"   1. Go to https://www.kaggle.com/c/titanic/submit")
        print(f"   2. Upload: {submission_path}")
        print(f"   3. Add description: 'Random Forest with feature engineering (CV AUC: {cv_score:.4f})'")
    else:
        print(f"\n❌ Pipeline test failed. Check the errors above.")
