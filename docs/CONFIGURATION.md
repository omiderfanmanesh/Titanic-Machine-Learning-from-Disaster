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

### Multi‑Model Ensemble (single run)

You can train multiple models within a single run using a shared per‑fold feature pipeline. Add an `ensemble` block to `configs/experiment.yaml`:

```yaml
ensemble:
  model_list:
    - name: random_forest
      params: { n_estimators: 400, max_depth: 8, random_state: 42 }
    - name: xgboost
      params: { n_estimators: 300, max_depth: 4, learning_rate: 0.08, subsample: 0.8, colsample_bytree: 0.8, random_state: 42 }
  method: average     # average | weighted | rank_average | geometric_mean | median | max | min
  weights: null       # optional weights aligned with model_list
```

Artifacts per run will include one pipeline per fold (`fold_i_feature_pipeline.joblib`), one model per type and fold (`fold_i_model_<name>.joblib`), OOF per model (`oof_<name>.csv`), an optional ensemble OOF (`oof_ensemble.csv`), and metadata (`ensemble_config.json`, `training_config.json`).

### Stacking (Level‑2 Meta‑Learner)

Enable a meta‑learner trained strictly on concatenated OOF probabilities:

```yaml
stacking:
  use: true
  meta_model:
    name: logistic
    params: { C: 1.0, max_iter: 1000 }
```

Training artifacts additionally include:
- `meta_features_oof.csv` — columns `oof_<modelname>` per base model + target
- `meta_model.joblib` — fitted meta‑learner
- `meta_config.json` — meta model spec + base model order

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

### Cross‑Run Ensembling (optional)

To ensemble predictions from multiple independent training runs, specify a runs manifest in `configs/inference.yaml`:

```yaml
runs:
  - path: artifacts/20250817-120713
    weight: 0.6
  - path: artifacts/20250817-121058
    weight: 0.4

ensemble_method: average      # average | weighted | rank_average | geometric_mean | median | max | min
ensemble_weights: null        # used if runs[].weight not provided
```

CLI usage:

```bash
# With manifest
python src/cli.py predict --inference-config inference

# Or pass multiple run dirs explicitly
python src/cli.py predict \
  --run-dir artifacts/20250817-120713 \
  --run-dir artifacts/20250817-121058 \
  --inference-config inference
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
```

Prefer keeping most knobs in a single file (configs/data.yaml). The train command now reads
these training-related keys from data.yaml if present and uses them over experiment.yaml:

```yaml
# In configs/data.yaml
cv_strategy: group           # stratified (default), group, kfold, timeseries
cv_folds: 5
cv_shuffle: true
cv_random_state: 42
cv_metric: accuracy          # accuracy, f1, or roc_auc
group_column: FamilyID       # optional; if absent and strategy=group, groups are derived from Name+Ticket
```

You can still use `--set` for ad-hoc tweaks, but it’s optional now that the common
training knobs live in data.yaml.


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
