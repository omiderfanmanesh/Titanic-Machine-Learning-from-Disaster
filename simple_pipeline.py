#!/usr/bin/env python3
"""
Minimal pipeline runner for Kaggle submission.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime

def preprocess_titanic_data(train_df, test_df):
    """Simple feature engineering for Titanic data."""
    
    # Combine train and test for consistent preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined = pd.concat([train_df, test_df], sort=False)
    
    # Basic feature engineering
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # Extract title from name
    combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Mlle': 'Miss',
        'Major': 'Rare', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    combined['Title'] = combined['Title'].map(title_mapping).fillna('Mr')
    
    # Fill missing ages by title
    for title in combined['Title'].unique():
        combined.loc[combined['Title'] == title, 'Age'] = combined.loc[combined['Title'] == title, 'Age'].fillna(
            combined.loc[combined['Title'] == title, 'Age'].median()
        )
    
    # Fill missing Embarked
    combined['Embarked'] = combined['Embarked'].fillna('S')
    
    # Fill missing Fare
    combined['Fare'] = combined['Fare'].fillna(combined.groupby('Pclass')['Fare'].transform('median'))
    
    # Create age bins
    combined['AgeBin'] = pd.cut(combined['Age'], bins=5, labels=False)
    
    # Create fare bins
    combined['FareBin'] = pd.cut(combined['Fare'], bins=5, labels=False)
    
    # Select features for modeling
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin']
    
    X = combined[features].copy()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    X['Sex'] = le_sex.fit_transform(X['Sex'])
    
    le_embarked = LabelEncoder()
    X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
    
    le_title = LabelEncoder()
    X['Title'] = le_title.fit_transform(X['Title'])
    
    # Split back to train and test
    X_train = X[combined['is_train'] == 1].copy()
    X_test = X[combined['is_train'] == 0].copy()
    y_train = train_df['Survived'].copy()
    
    return X_train, X_test, y_train

def main():
    """Run simple ML pipeline."""
    print("🚀 Starting Simple Titanic ML Pipeline")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("artifacts") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📊 Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"✅ Loaded train: {train_df.shape}, test: {test_df.shape}")
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    X_train, X_test, y_train = preprocess_titanic_data(train_df, test_df)
    print(f"✅ Features: {X_train.shape}")
    
    # Train model with cross-validation
    print("🤖 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"✅ Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train final model
    print("🎯 Training final model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print("🔮 Making predictions...")
    predictions = model.predict(X_test)
    
    # Create submission
    print("📄 Creating submission...")
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    submission_path = output_dir / 'submission.csv'
    submission.to_csv(submission_path, index=False)
    
    # Save model
    model_path = output_dir / 'model.joblib'
    joblib.dump(model, model_path)
    
    print("🎉 Pipeline completed!")
    print(f"📁 Output: {output_dir}")
    print(f"📊 CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"🏆 Survival rate: {predictions.mean():.1%}")
    print(f"📄 Submission: {submission_path}")
    
    # Show first few predictions
    print("\n📋 First 10 predictions:")
    print(submission.head(10).to_string(index=False))
    
    # Feature importance
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                     'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin']
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 5 Important Features:")
    print(importance_df.head().to_string(index=False))
    
    return str(submission_path)

if __name__ == "__main__":
    submission_file = main()
    print(f"\n🚢 Ready for Kaggle: {submission_file}")
