# Titanic ML Pipeline

A professional, SOLID-principle-based machine learning pipeline for the Kaggle Titanic competition, featuring comprehensive testing, configuration-driven architecture, and production-ready code quality.

## 🏗️ Architecture Overview

This pipeline follows **SOLID principles** with a modular, interface-based design:

- **Single Responsibility**: Each component has one clear purpose
- **Open/Closed**: Easy to extend with new models, features, and evaluators  
- **Liskov Substitution**: All implementations are interchangeable via interfaces
- **Interface Segregation**: Clean, focused interfaces for each component type
- **Dependency Inversion**: High-level modules depend on abstractions, not concretions

## 🚀 Quick Start

Option 1 — Run via Python (no install):

```bash
# Build features
python src/cli.py features --experiment-config experiment --data-config data

# Train (creates artifacts/<run> and artifacts/latest symlink)
python src/cli.py train --experiment-config experiment --data-config data

# Predict using the latest run
python src/cli.py predict --run-dir artifacts/latest

# Diagnose environment & data
python src/cli.py diagnose
```

Option 2 — Install local CLI:

```bash
pip install -e .                 # base deps
# Optional extras:
# pip install -e .[boosting]     # xgboost/lightgbm/catboost
# pip install -e .[encoders]     # category-encoders

# Use the CLI
titanic diagnose
titanic features --experiment-config experiment --data-config data
titanic train --experiment-config experiment --data-config data
titanic predict --run-dir artifacts/latest
```

Makefile shortcuts:

```bash
make quickstart   # features -> train -> predict
make fast         # same, using the 'fast' profile
```

## 📊 Pipeline Components

### Data Pipeline (`data/`)
- **`loader.py`**: Data loading with Kaggle API integration and caching
- **`validate.py`**: Comprehensive data validation and leakage detection

### Feature Engineering (`features/`)
- **`transforms/`**: Atomic, composable transformations (family size, titles, deck extraction)
- **`build.py`**: Main feature builder with leak-safe preprocessing

### Cross-Validation (`cv/`)
- **`folds.py`**: Multiple CV strategies (Stratified, Group, Time Series)

### Modeling (`modeling/`)
- **`model_registry.py`**: Factory for sklearn and boosting models
- **`trainers.py`**: Cross-validation training with artifact management

### Evaluation (`eval/`)
- **`evaluator.py`**: Comprehensive metrics, calibration analysis, stability assessment

### Inference (`infer/`)
- **`predictor.py`**: Single model and ensemble prediction with TTA support

### Submission (`submit/`)
- **`build_submission.py`**: Kaggle submission formatting and validation

## ⚙️ Configuration

The pipeline is fully configuration-driven using YAML files:

### `configs/experiment.yaml` - Main experiment configuration
```yaml
experiment:
  name: "titanic_baseline"
  seed: 42
  cv_folds: 5
  scoring: "accuracy"
  
model:
  name: "logistic"  # or "random_forest", "xgboost", etc.
  params:
    C: 1.0
    random_state: 42
    
features:
  use_family_features: true
  use_title_features: true
  impute_age: true
```

### `configs/data.yaml` - Data configuration
```yaml
data:
  train_file: "data/train.csv"
  test_file: "data/test.csv"
  cache_processed: true
  validation_split: 0.2
```

### Dimensionality Reduction (optional)

You can enable PCA or TruncatedSVD after encoding and scaling by adding this block to `configs/data.yaml`:

```yaml
dimensionality_reduction:
  enabled: true       # set to true to use it
  method: pca         # pca | svd (TruncatedSVD)
  n_components: 8     # or null + keep_variance: 0.95 for PCA
  keep_variance: null # e.g., 0.95 to keep 95% variance (PCA only)
  whiten: false       # PCA whitening (optional)
  random_state: 42
```

When enabled, the feature builder trains the reducer on scaled features and returns reduced components (columns `dim_1..dim_k`) to the model. This integrates cleanly with the existing cross-validation trainer without leaking validation data.

## 🧪 Testing

Comprehensive test suite with 80%+ coverage:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Fast unit tests
pytest tests/integration/   # End-to-end tests

# Run with coverage
pytest --cov=titanic_ml --cov-report=html

# Run specific test file
pytest tests/unit/test_features.py -v
```

### Test Categories

- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end pipeline testing with synthetic data
- **Property Tests**: Edge case and invariant testing

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code  
ruff check src/ tests/

# Type checking
mypy src/

# Run all quality checks
make quality
```

### Adding New Components

1. **New Model**: Implement `IModel` interface in `modeling/model_registry.py`
2. **New Features**: Create transformer in `features/transforms/` 
3. **New Evaluator**: Implement `IEvaluator` interface in `eval/`

Example - Adding XGBoost model:

```python
class XGBoostModel(BaseModel):
    def __init__(self, **params):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

## 📈 Performance Tracking

- **Cross-Validation**: Stratified K-Fold with stability metrics
- **Leakage Prevention**: Proper train/validation/test separation
- **Model Comparison**: Automated benchmarking across model types
- **Feature Importance**: Integrated feature analysis and selection

## 🏆 Production Features

### Error Handling
- Comprehensive exception handling with informative messages
- Graceful degradation for missing optional dependencies
- Data validation with clear failure descriptions

### Logging
- Structured logging with configurable levels
- Run tracking with unique experiment IDs
- Performance monitoring and profiling

### Reproducibility  
- Complete seed management across all random components
- Configuration versioning and artifact tracking
- Deterministic pipeline execution

### Scalability
- Memory-efficient data processing with chunking
- Parallel cross-validation training
- Cached intermediate results

## 📋 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure package is installed in development mode
pip install -e .
```

**Kaggle API Issues**
```bash
# Set up Kaggle credentials
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Memory Issues**
```bash
# Reduce data chunk size in configs/data.yaml
data:
  chunk_size: 1000  # Reduce from default 5000
```

### Performance Optimization

- Use `CachedDataLoader` for repeated experiments
- Enable feature caching in `TitanicFeatureBuilder`
- Use ensemble prediction caching for multiple submissions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-component`
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Run quality checks: `make quality`
6. Submit pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle Titanic Competition for the dataset
- Scikit-learn and pandas communities for excellent libraries
- SOLID principles and clean architecture patterns

---

**Happy Modeling! 🚢⚓**
