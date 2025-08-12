# Quick Start Guide

Get up and running with the Titanic ML Pipeline in just 5 minutes!

## Prerequisites

- Python 3.8+
- pip or conda package manager
- Git (optional, for cloning)

## 🚀 Installation

### 1. Clone the Repository (Optional)
```bash
git clone https://github.com/omiderfanmanesh/Titanic-Machine-Learning-from-Disaster.git
cd Titanic-Machine-Learning-from-Disaster
```

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
