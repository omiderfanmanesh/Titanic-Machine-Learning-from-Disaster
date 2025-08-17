# Quick Start Guide

Get up and running with the Titanic ML Pipeline in minutes.

## Prerequisites

- Python 3.11+
- pip or conda
- Kaggle data in `data/raw/train.csv` and `data/raw/test.csv` (or use the `download` command if configured)

## Option A — Run via Python (no install)

```bash
# Build features (uses configs/experiment.yaml and configs/data.yaml)
python src/cli.py features --experiment-config experiment --data-config data

# Train (creates artifacts/<run> and updates artifacts/latest)
python src/cli.py train --experiment-config experiment --data-config data

# Predict using the latest run
python src/cli.py predict --run-dir artifacts/latest

# Diagnose environment & data
python src/cli.py diagnose
```

## Option B — Install local CLI

```bash
pip install -e .                 # base deps
# Optional extras:
# pip install -e .[boosting]     # xgboost/lightgbm/catboost
# pip install -e .[encoders]     # category-encoders

# Then use the CLI
titanic diagnose
titanic features --experiment-config experiment --data-config data
titanic train --experiment-config experiment --data-config data
titanic predict --run-dir artifacts/latest
```

## Profiles and Overrides

```bash
# Use a fast profile (fewer folds/features)
python src/cli.py features --profile fast
python src/cli.py train --profile fast

# Inline overrides (dot-paths supported)
python src/cli.py train --set cv_folds=3 --set model_name=random_forest
python src/cli.py predict --set threshold.method=f1 --set threshold.optimizer=true
```

## Single‑Run Multi‑Model Ensemble

Train multiple models in one run by adding `ensemble.model_list` to `configs/experiment.yaml` (see `configs/experiment_ensemble.yaml` for an example). Then run:

```bash
# Train all models listed under ensemble.model_list in one run
python src/cli.py train --experiment-config experiment --data-config data

# Predict (per-fold pipelines applied once; per-fold model types ensembled; then averaged across folds)
python src/cli.py predict --run-dir artifacts/latest
```

Artifacts include: `fold_i_feature_pipeline.joblib`, `fold_i_model_<name>.joblib`, `oof_<name>.csv`, `oof_ensemble.csv`, and `ensemble_config.json`.

## Cross‑Run Ensembling

Combine predictions from multiple independent runs:

```bash
# Option A: Provide multiple run dirs
python src/cli.py predict \
  --run-dir artifacts/run_random_forest \
  --run-dir artifacts/run_xgboost \
  --inference-config inference

# Option B: Use an inference manifest (configs/inference.yaml)
# runs:
#   - path: artifacts/run_random_forest
#     weight: 0.6
#   - path: artifacts/run_xgboost
#     weight: 0.4
python src/cli.py predict --inference-config inference
```

## Stacking (Optional)

Train a meta‑learner on base models’ OOF predictions by enabling `stacking` in `configs/experiment.yaml`:

```yaml
stacking:
  use: true
  meta_model:
    name: logistic
    params: { C: 1.0 }
```

During predict, if a `meta_model.joblib` exists in the run directory, the system builds meta features from per‑model per‑fold predictions on RAW test, averages per model across folds, and feeds them to the meta model. Disable via `--set stacking.use=false` if you want to fall back to direct ensembling.

## Training Columns & Exclusions

In `configs/data.yaml`:

```yaml
# Exclusions (applied before encoding; avoids generating dummies)
exclude_column_for_training:
  - Ticket_prefix
  - Ticket_number
  - Surname
  - First_Middle
  - Title_First_Middle
  - Title_Raw

# Or exact inclusion list (takes precedence)
train_columns:
  - AgeBin
  - FamilySize
  - IsAlone
  - TicketGroupSize
  - Sex_female
  - Sex_male
```

## Feature Importance

- Runs automatically at the end of `features` if `feature_importance: true`.
- Outputs saved to `artifacts/feature_importance/` (CSVs, plots, text report).
- Use the helper: `python src/cli.py suggest-columns --top 20` to print a compact training column list.


### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python src/cli.py --help
```

You should see the pipeline commands listed.

## 🎯 First Run - 5 Minutes to Predictions!

### Step 1: Download Data (1 min)
```bash
# Download Titanic competition data
python src/cli.py download --competition titanic

# Or use your own CSV files
# python src/cli.py info  # to see expected format
```

### Step 2: Validate Data (30 seconds)
```bash
python src/cli.py validate --config configs/data.yaml
```

### Step 3: Build Features (1 min)
```bash
python src/cli.py features --experiment-config experiment --data-config data
```

### Step 4: Train Model (2 minutes)
```bash
python src/cli.py train --experiment-config experiment --data-config data
```

### Step 5: Generate Predictions (30 seconds)
```bash
# Find your latest training run directory
ls artifacts/

# Use the latest run (e.g., 20250812-150000)
python src/cli.py predict --run-dir artifacts/20250812-150000
```

### Step 6: Create Submission (30 seconds)
```bash
# Basic (default descriptive filename inside run dir)
python src/cli.py submit --predictions-path artifacts/20250812-150000/predictions.csv

# Force simple name (submission.csv)
python src/cli.py submit --predictions-path artifacts/20250812-150000/predictions.csv --no-descriptive

# Remote Kaggle submit (no metadata lines added)
python src/cli.py submit --predictions-path artifacts/20250812-150000/predictions.csv --remote -m "First CV run"
```

Descriptive filenames look like:
`sub_random_forest_cv08769_oof08753_thr050_20250812-150000.csv`

Pattern parts:
- `sub` prefix
- model name
- cv + mean CV score (dots removed) 
- oof + OOF AUC (dots removed) 
- thr + threshold (dots removed)
- run directory timestamp

## 🎉 Congratulations!

You now have:
- ✅ A trained ML model
- ✅ Test predictions
- ✅ A Kaggle-ready submission file

## 🔧 Customization

### Change Model Type
```bash
# Edit configs/experiment.yaml
model_name: "random_forest"  # or "xgboost", "lightgbm", etc.
```

### Add More Features
```bash
# Edit the feature config in configs/experiment.yaml
features:
  add_family_features: true
  add_title_features: true
  add_deck_features: true
  add_ticket_features: true  # Enable more features
```

### Adjust Cross-Validation
```bash
# In configs/experiment.yaml
cv_folds: 10        # More folds for better validation
cv_strategy: "group"  # Different CV strategy
```

## 📊 Monitor Your Results

### View Training Results
```bash
python src/cli.py evaluate --run-dir artifacts/YOUR_RUN_DIR
```

### Check Model Performance
Look for these files in your run directory:
- `cv_scores.json` - Cross-validation scores
- `oof_predictions.csv` - Out-of-fold predictions  
- `training_report.md` - Detailed training report

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project root
cd /path/to/Titanic-Machine-Learning-from-Disaster
python src/cli.py info
```

**Missing Dependencies**
```bash
pip install pandas scikit-learn click pydantic
# For optional models:
pip install xgboost lightgbm catboost
```

**Data Not Found**
```bash
# Check data directory
ls data/raw/
# Should contain train.csv and test.csv
```

### Getting Help

```bash
# Get help for any command
python src/cli.py COMMAND --help

# Get pipeline information
python src/cli.py info

# Check available models
python src/cli.py info  # Lists all available models
```

## 🎓 Next Steps

1. **Experiment**: Try different models and feature combinations
2. **Optimize**: Use the hyperparameter tuning features
3. **Ensemble**: Combine multiple models for better performance
4. **Deploy**: Use the inference API for production predictions

## 📚 Learn More

- [Configuration Guide](CONFIGURATION.md) - Detailed config options
- [Examples](EXAMPLES.md) - Common use cases and recipes
- [Architecture](ARCHITECTURE.md) - How the pipeline works
- [API Reference](API.md) - Programmatic usage

---

**💡 Pro Tip**: Start with the default configuration and gradually customize as you learn the system!

Happy modeling! 🚀
