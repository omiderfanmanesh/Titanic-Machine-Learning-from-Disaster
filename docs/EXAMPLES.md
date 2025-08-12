# Examples and Use Cases

Practical examples and common recipes for the Titanic ML Pipeline.

## 🎯 Common Use Cases

### 1. Quick Model Comparison

Compare multiple models with the same features:

```bash
# Train Random Forest
python src/cli.py train --experiment-config rf_config --data-config data

# Train XGBoost  
python src/cli.py train --experiment-config xgb_config --data-config data

# Train Logistic Regression
python src/cli.py train --experiment-config lr_config --data-config data

# Compare results
python src/cli.py evaluate --run-dir artifacts/20250812-150000  # RF results
python src/cli.py evaluate --run-dir artifacts/20250812-151000  # XGB results  
python src/cli.py evaluate --run-dir artifacts/20250812-152000  # LR results
```

**Config files:**

`configs/rf_config.yaml`:
```yaml
name: "random_forest_experiment"
model_name: "random_forest"
model_params:
  n_estimators: 200
  max_depth: 10
  min_samples_split: 5
cv_folds: 5
```

`configs/xgb_config.yaml`:
```yaml
name: "xgboost_experiment"
model_name: "xgboost"
model_params:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 6
cv_folds: 5
```

### 2. Feature Engineering Experiments

Test different feature combinations:

```bash
# Minimal features
python src/cli.py features --experiment-config minimal_features
python src/cli.py train --experiment-config minimal_features

# All features  
python src/cli.py features --experiment-config all_features
python src/cli.py train --experiment-config all_features

# Custom features
python src/cli.py features --experiment-config custom_features  
python src/cli.py train --experiment-config custom_features
```

**Feature configs:**

`configs/minimal_features.yaml`:
```yaml
features:
  add_family_features: true
  add_title_features: false
  add_deck_features: false
  add_ticket_features: false
  transform_fare: true
  log_transform_fare: false
  add_missing_indicators: false
model_name: "random_forest"
```

`configs/all_features.yaml`:
```yaml
features:
  add_family_features: true
  add_title_features: true
  add_deck_features: true
  add_ticket_features: true
  transform_fare: true
  log_transform_fare: true
  add_age_bins: true
  age_bins: 5
  add_missing_indicators: true
model_name: "random_forest"
```

### 3. Ensemble Methods

Create powerful ensemble predictions:

```bash
# Train multiple models
python src/cli.py train --experiment-config rf_ensemble
python src/cli.py train --experiment-config xgb_ensemble  
python src/cli.py train --experiment-config lgb_ensemble

# Create ensemble prediction
python src/cli.py predict --run-dir artifacts/rf_run --output-path rf_predictions.csv
python src/cli.py predict --run-dir artifacts/xgb_run --output-path xgb_predictions.csv
python src/cli.py predict --run-dir artifacts/lgb_run --output-path lgb_predictions.csv

# Weighted ensemble submission
python src/cli.py submit --ensemble-predictions rf_predictions.csv,xgb_predictions.csv,lgb_predictions.csv \
                         --ensemble-weights 0.4,0.4,0.2 \
                         --output-path ensemble_submission.csv
```

### 4. Cross-Validation Strategies

Experiment with different CV approaches:

```bash
# Standard Stratified K-Fold
python src/cli.py train --experiment-config standard_cv

# Group K-Fold (prevent family leakage)
python src/cli.py train --experiment-config group_cv

# More folds for robust estimation
python src/cli.py train --experiment-config robust_cv
```

**CV configs:**

`configs/group_cv.yaml`:
```yaml
name: "group_cv_experiment"
model_name: "random_forest"
cv_strategy: "group"
cv_folds: 5
group_column: "Ticket"  # Prevent ticket/family leakage
```

`configs/robust_cv.yaml`:
```yaml
name: "robust_cv_experiment" 
model_name: "random_forest"
cv_strategy: "stratified"
cv_folds: 10  # More folds for stable estimates
cv_shuffle: true
cv_random_state: 42
```

## 🔬 Advanced Examples

### 1. Hyperparameter Optimization

Manual grid search using configuration files:

```bash
# Create multiple configs with different parameters
for lr in 0.01 0.1 0.2; do
  for depth in 3 6 9; do
    cat > "configs/xgb_lr${lr}_d${depth}.yaml" << EOF
name: "xgb_lr${lr}_depth${depth}"
model_name: "xgboost"
model_params:
  learning_rate: ${lr}
  max_depth: ${depth}
  n_estimators: 100
cv_folds: 5
EOF
    python src/cli.py train --experiment-config "xgb_lr${lr}_d${depth}"
  done
done

# Find best configuration by comparing CV scores
ls artifacts/*/cv_scores.json | xargs -I {} sh -c 'echo {} && cat {} | grep "mean_score"'
```

### 2. Data Leakage Detection

Comprehensive data validation:

```bash
# Enable all validation checks
python src/cli.py validate --config comprehensive_validation

# Custom validation with specific checks
python src/cli.py validate --config custom_validation
```

`configs/comprehensive_validation.yaml`:
```yaml
validation:
  check_temporal_leakage: true
  check_target_leakage: true
  check_duplicate_leakage: true
  correlation_threshold: 0.99
  duplicate_threshold: 0.95
  
data:
  train_path: "data/raw/train.csv"
  test_path: "data/raw/test.csv" 
  target_column: "Survived"
  exclude_from_duplicate_check: ["PassengerId", "Name", "Ticket"]
```

### 3. Custom Feature Engineering

Programmatic feature creation:

```python
# custom_features.py
from features.transforms import BaseTransform
import pandas as pd

class CustomTitleTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.title_groups = {
            'Mr': 'Mr',
            'Miss': 'Miss', 
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'VIP',
            'Rev': 'VIP',
            'Col': 'VIP',
            'Major': 'VIP',
            'Mlle': 'Miss',
            'Countess': 'VIP',
            'Ms': 'Miss',
            'Lady': 'VIP',
            'Jonkheer': 'VIP',
            'Don': 'VIP',
            'Dona': 'VIP',
            'Mme': 'Mrs',
            'Capt': 'VIP',
            'Sir': 'VIP'
        }
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        # Extract title from name
        X['Title_Raw'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        # Group titles
        X['Title_Grouped'] = X['Title_Raw'].map(self.title_groups).fillna('Other')
        return X

# Use in config
features:
  custom_transforms:
    - class: "custom_features.CustomTitleTransform"
```

### 4. Production Pipeline

Complete production workflow:

```bash
#!/bin/bash
# production_pipeline.sh

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TITANIC_ENV="production"

# Download latest data
python src/cli.py download --competition titanic --output-dir data/raw/$(date +%Y%m%d)

# Validate data quality
python src/cli.py validate --config production_validation

# Feature engineering
python src/cli.py features --experiment-config production_features --data-config production_data

# Train ensemble of models
models=("random_forest" "xgboost" "lightgbm" "catboost")
for model in "${models[@]}"; do
  echo "Training $model..."
  cp configs/production_base.yaml "configs/production_${model}.yaml"
  sed -i "s/MODEL_NAME/${model}/g" "configs/production_${model}.yaml"
  python src/cli.py train --experiment-config "production_${model}" --data-config production_data
done

# Generate ensemble predictions
python src/cli.py predict --ensemble-models artifacts/*/model_*.joblib --output-path production_predictions.csv

# Create submission
python src/cli.py submit --predictions-path production_predictions.csv \
                         --output-path "submissions/$(date +%Y%m%d_%H%M%S)_submission.csv" \
                         --descriptive \
                         --remote -m "Production ensemble"

echo "Production pipeline completed successfully!"
```

## 🧪 Testing and Debugging

### 1. Quick Debug Run

```bash
# Fast debugging with small data
python src/cli.py train --experiment-config debug_config --data-config debug_data
```

`configs/debug_config.yaml`:
```yaml
name: "debug_run"
debug_mode: true
debug_n_rows: 100
model_name: "logistic"
model_params:
  max_iter: 100
cv_folds: 2
logging_level: "DEBUG"
```

### 2. Component Testing

Test individual components:

```python
# test_components.py
from data.loader import TitanicDataLoader
from features.build import create_feature_builder  
from modeling.model_registry import ModelRegistry

# Test data loading
loader = TitanicDataLoader("data/raw/train.csv", "data/raw/test.csv")
train_df, test_df = loader.load()
print(f"✅ Loaded train: {train_df.shape}, test: {test_df.shape}")

# Test feature engineering
builder = create_feature_builder()
X_train = builder.fit_transform(train_df)
print(f"✅ Features built: {X_train.shape}")

# Test model registry
registry = ModelRegistry()
models = registry.get_available_models()
print(f"✅ Available models: {models}")

# Test model creation
model = registry.create_model("random_forest")
print(f"✅ Model created: {type(model).__name__}")
```

### 3. Performance Profiling

```bash
# Profile training time
time python src/cli.py train --experiment-config profiling_config

# Memory usage monitoring
/usr/bin/time -v python src/cli.py train --experiment-config memory_config

# Detailed profiling
python -m cProfile -s cumtime src/cli.py train --experiment-config profile_config 2> profile.log
```

## 📊 Result Analysis

### 1. Cross-Validation Analysis

```python
# analyze_cv.py
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load CV results
with open('artifacts/20250812-150000/cv_scores.json') as f:
    cv_results = json.load(f)

# Analyze fold stability
fold_scores = cv_results['fold_scores']
print(f"CV Mean: {cv_results['mean_score']:.4f}")
print(f"CV Std: {cv_results['std_score']:.4f}")
print(f"CV Range: {max(fold_scores) - min(fold_scores):.4f}")

# Plot fold scores
plt.figure(figsize=(10, 6))
plt.bar(range(len(fold_scores)), fold_scores)
plt.axhline(y=cv_results['mean_score'], color='r', linestyle='--', label='Mean')
plt.title('Cross-Validation Scores by Fold')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.legend()
plt.show()
```

### 2. Feature Importance Analysis

```python
# feature_importance.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('artifacts/20250812-150000/fold_0_model.joblib')

# Get feature importance
if hasattr(model.model, 'feature_importances_'):
    importance = model.model.feature_importances_
    feature_names = model.feature_names  # if available
    
    # Create importance DataFrame
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    imp_df.head(20).plot(x='feature', y='importance', kind='bar')
    plt.title('Top 20 Feature Importances')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

## 🚀 Production Deployment

### 1. Model Serving API

```python
# app.py - Flask API for model serving
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from features.build import create_feature_builder

app = Flask(__name__)

# Load model and feature builder
model = joblib.load('production_model.joblib')
feature_builder = joblib.load('production_features.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        df = pd.DataFrame([data])
        
        # Transform features
        X = feature_builder.transform(df)
        
        # Make prediction
        prediction = model.predict_proba(X)[0]
        
        return jsonify({
            'survival_probability': float(prediction[1]),
            'prediction': int(prediction[1] > 0.5)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. Batch Prediction

```bash
# batch_predict.sh
#!/bin/bash

INPUT_FILE=$1
OUTPUT_FILE=$2

python src/cli.py predict \
  --model-path production_model.joblib \
  --input-path $INPUT_FILE \
  --output-path $OUTPUT_FILE \
  --batch-size 1000 \
  --config production_inference.yaml

echo "Batch prediction completed: $OUTPUT_FILE"
```

---

**💡 Pro Tips:**
- Start with simple examples and gradually add complexity
- Use configuration files to make experiments reproducible  
- Monitor cross-validation stability for reliable models
- Combine multiple approaches for robust ensemble methods
- Profile your pipeline to identify bottlenecks

Happy experimenting! 🧪✨
