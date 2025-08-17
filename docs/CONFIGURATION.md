# Configuration Guide

Complete guide to configuring the Titanic ML Pipeline for your specific needs.

## 📁 Configuration Files

The pipeline uses YAML configuration files stored in the `configs/` directory:

```
configs/
├── experiment.yaml     # Main experiment configuration
├── data.yaml          # Data loading and validation
└── inference.yaml     # Inference and prediction settings
```

## 🎛️ Experiment Configuration

### Basic Structure (`configs/experiment.yaml`)

```yaml
# Experiment Metadata
name: "titanic_experiment"
seed: 42
debug_mode: false
debug_n_rows: null

# Model Configuration
model_name: "random_forest"
model_params:
  n_estimators: 100
  max_depth: null
  random_state: 42

# Cross-Validation
cv_folds: 5
cv_strategy: "stratified"  # stratified, kfold, group, timeseries
cv_shuffle: true
cv_random_state: 42

# Training
early_stopping_rounds: null
logging_level: "INFO"
```

### Available Models

```yaml
model_name: "logistic"           # Logistic Regression
model_name: "random_forest"      # Random Forest
model_name: "gradient_boosting"  # Gradient Boosting
model_name: "svm"               # Support Vector Machine
model_name: "xgboost"           # XGBoost (if installed)
model_name: "catboost"          # CatBoost (if installed)
model_name: "lightgbm"          # LightGBM (if installed)
```

### Model-Specific Parameters

#### Random Forest
```yaml
model_params:
  n_estimators: 100
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  max_features: "sqrt"
  random_state: 42
```

#### XGBoost
```yaml
model_params:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 6
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
```

#### Logistic Regression
```yaml
model_params:
  C: 1.0
  max_iter: 1000
  solver: "liblinear"
  random_state: 42
```

## 📊 Data Configuration

### Basic Structure (`configs/data.yaml`)

```yaml
# File Paths
train_path: "data/raw/train.csv"
test_path: "data/raw/test.csv"
target_column: "Survived"
id_column: "PassengerId"

# Task Configuration
task_type: "binary"

# Schema Validation
required_columns: 
  - "PassengerId"
  - "Survived"
  - "Pclass"
  - "Name" 
  - "Sex"
  - "Age"

numeric_columns:
  - "Age"
  - "SibSp" 
  - "Parch"
  - "Fare"

categorical_columns:
  - "Sex"
  - "Embarked"
  - "Pclass"

# Preprocessing Options
handle_missing: true
scale_features: true
encode_categoricals: true
```

### Data Loading Options

```yaml
# For Kaggle competition data
data_source: "kaggle"
competition: "titanic"
download_path: "data/raw"

# For custom CSV files  
data_source: "csv"
train_path: "path/to/train.csv"
test_path: "path/to/test.csv"

# For cached data (faster repeated runs)
use_cache: true
cache_path: "data/cache/processed_data.pkl"
```

## 🔧 Feature Engineering

### Feature Configuration

```yaml
# Feature Engineering Options
features:
  # Family Features
  add_family_features: true
  
  # Title Extraction
  add_title_features: true
  title_rare_threshold: 10
  
  # Cabin/Deck Features  
  add_deck_features: true
  
  # Ticket Group Features
  add_ticket_features: true
  
  # Fare Transformations
  transform_fare: true
  log_transform_fare: false
  
  # Age Processing
  add_age_bins: false
  age_bins: 5
  
  # Missing Value Indicators
  add_missing_indicators: true
  missing_threshold: 0.01
  
  # Scaling and Encoding
  scale_features: true
  encode_categoricals: true
```

### Advanced Feature Options

```yaml
features:
  # Interaction Features
  add_interaction_features: true
  interaction_pairs:
    - ["Age", "Pclass"]
    - ["Fare", "Pclass"]
  
  # Polynomial Features
  add_polynomial_features: false
  poly_degree: 2
  
  # Custom Transformations
  custom_transforms:
    - name: "age_squared"
      type: "polynomial" 
      column: "Age"
      degree: 2
```

## 🎯 Inference Configuration

### Basic Structure (`configs/inference.yaml`)

```yaml
# Model Paths
model_paths:
  - "artifacts/20250812-150000/fold_0_model.joblib"
  - "artifacts/20250812-150000/fold_1_model.joblib"
  - "artifacts/20250812-150000/fold_2_model.joblib"

# Ensemble Method
ensemble_method: "average"  # average, weighted, rank_average, geometric_mean
ensemble_weights: null      # [0.4, 0.3, 0.3] for weighted ensemble

# Test-Time Augmentation
use_tta: false
tta_rounds: 5
tta_noise_scale: 0.01

# Output Paths
output_path: "artifacts/predictions.csv"
submission_path: "artifacts/submission.csv"
```

### Ensemble Methods

```yaml
# Simple Average
ensemble_method: "average"

# Weighted Average  
ensemble_method: "weighted"
ensemble_weights: [0.4, 0.3, 0.3]
```

## ✅ Practical Configuration (This Repo)

This repository exposes additional, convenient knobs beyond the generic examples above.

### Feature Engineering Pipeline (staged)

```yaml
feature_engineering:
  pre_impute:
    - TitleTransform           # → Title + name-derived fields
    - FamilySizeTransform      # → FamilySize, IsAlone
    - TicketParseTransform     # → Ticket_prefix, Ticket_number
    - DeckTransform            # → Deck
    - TicketGroupTransform     # → TicketGroupSize
  post_impute:
    - FareTransform            # optional log transform
    - AgeBinningTransform      # → AgeBin

# Enable/disable any transform by class name
feature_toggles:
  TitleTransform: true
  FamilySizeTransform: true
  TicketParseTransform: true
  DeckTransform: true
  TicketGroupTransform: true
  FareTransform: true
  AgeBinningTransform: true
```

### Encoding, Scaling, Missing

```yaml
handle_missing: true           # run imputation orchestrator
encode_categorical: true       # run encoding orchestrator
scale_features: true           # run scaling orchestrator

# Imputation order — prevents Embarked __MISSING__ dummies
imputation:
  order: [Fare, Embarked, Age]
```

### Keep Originals (off by default)

```yaml
# If true, original raw columns are retained alongside engineered/encoded ones
add_original_columns: false
```

### Include/Exclude Columns for Training

Two ways to control which columns go into the model:

```yaml
# 1) Exclusion mode (recommended): dropped BEFORE encoding (so no dummies are created)
exclude_column_for_training:
  - Ticket_prefix      # base col (prevents generating any Ticket_prefix_* dummies)
  - Ticket_number
  - Surname
  - First_Middle
  - Title_First_Middle
  - Title_Raw

# 2) Inclusion mode (exact feature list): takes precedence over exclusion
train_columns:
  - AgeBin
  - FamilySize
  - IsAlone
  - TicketGroupSize
  - Sex_female
  - Sex_male
```

Notes:
- Exclusions are applied before encoding and also respected by feature importance. This prevents “extra” encoded columns from appearing.
- If you supply `train_columns`, the training/prediction will use that exact set (and drop ID/target automatically).

### Feature Importance

Feature importance runs at the end of the `features` step when `feature_importance: true`.
It writes CSVs, plots, and a text report to `artifacts/feature_importance/`.

Robust behaviors:
- Uses only numeric/bool features (sanitized; inf/NaN handled; constant columns dropped).
- Cross-validated with accuracy; falls back to train accuracy if CV fails.
- Algorithms: start with `random_forest` (add `xgboost`/`permutation` as desired).

### Profiles and Inline Overrides

```bash
# Use a profile (merges configs/profiles/{fast,standard,full}.yaml)
python src/cli.py train --profile fast

# Inline override any key (supports dot-paths)
python src/cli.py predict --set threshold.method=f1 --set threshold.optimizer=true
```

Profiles provide quick presets (e.g., fewer folds for fast iterations). Inline `--set` helps tweak without editing YAML.


# Rank Average (more robust to outliers)
ensemble_method: "rank_average"

# Geometric Mean (for probabilities)
ensemble_method: "geometric_mean"

# Take Maximum/Minimum
ensemble_method: "max"  # or "min"
```

## 🔍 Cross-Validation Strategies

### Stratified K-Fold (Default)
```yaml
cv_strategy: "stratified"
cv_folds: 5
cv_shuffle: true
cv_random_state: 42
```

### Group K-Fold (Prevent leakage)
```yaml
cv_strategy: "group"
cv_folds: 5
group_column: "Ticket"  # Group by ticket to prevent family leakage
```

### Time Series Split
```yaml
cv_strategy: "timeseries"
cv_folds: 5
test_size: null  # Use default test size
```

### Custom Stratified
```yaml
cv_strategy: "custom_stratified"
cv_folds: 5
min_samples_per_class: 2
```

## 🐛 Debug Configuration

### Debug Mode Settings

```yaml
# Enable debug mode
debug_mode: true
debug_n_rows: 1000        # Use subset of data
logging_level: "DEBUG"     # Verbose logging

# Fast testing
cv_folds: 2               # Fewer folds
model_params:
  n_estimators: 10        # Smaller models
```

### Logging Levels

```yaml
logging_level: "DEBUG"    # Most verbose
logging_level: "INFO"     # Default
logging_level: "WARNING"  # Warnings only
logging_level: "ERROR"    # Errors only
```

## 📝 Configuration Templates

### Quick Experimentation
```bash
python src/cli.py create-config --config-name quick_test --template experiment
# Then edit configs/quick_test.yaml with:
# - cv_folds: 3
# - debug_mode: true
# - model with fast parameters
```

### Production Training
```bash
python src/cli.py create-config --config-name production --template experiment
# Then edit configs/production.yaml with:
# - cv_folds: 10
# - Multiple ensemble models
# - Comprehensive feature engineering
```

### Hyperparameter Tuning
```bash
# Create config for systematic parameter search
# Use with external tools like Optuna or GridSearch
```

## 🔧 Environment Variables

```bash
# Override config paths
export TITANIC_CONFIG_DIR="path/to/configs"
export TITANIC_DATA_DIR="path/to/data"
export TITANIC_ARTIFACTS_DIR="path/to/artifacts"

# Kaggle API credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

## 📚 Configuration Best Practices

1. **Start Simple**: Begin with default configs, then customize
2. **Version Control**: Keep configs in git for reproducibility  
3. **Environment-Specific**: Use different configs for dev/prod
4. **Validation**: Test configs with `debug_mode: true` first
5. **Documentation**: Comment complex configurations

---

**💡 Pro Tip**: Use configuration inheritance by creating base configs and overriding specific parameters for different experiments!
