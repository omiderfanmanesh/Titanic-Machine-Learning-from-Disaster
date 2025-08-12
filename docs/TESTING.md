# Testing Guide

Comprehensive testing strategy and practices for the Titanic ML Pipeline.

## 🧪 Testing Philosophy

Our testing strategy follows the testing pyramid:
- **Unit Tests (70%)**: Fast, isolated tests for individual components
- **Integration Tests (20%)**: Test component interactions and data flow  
- **End-to-End Tests (10%)**: Full pipeline validation

## 🏗️ Test Structure

```
tests/
├── conftest.py              # Shared test fixtures and configuration
├── conftest_additional.py   # Additional fixtures for complex scenarios
├── test_smoke.py           # Quick smoke tests for basic functionality
├── unit/                   # Unit tests for individual components
│   ├── test_data/          # Data loading and validation tests
│   ├── test_features/      # Feature engineering tests
│   ├── test_modeling/      # Model training and prediction tests
│   ├── test_cv/           # Cross-validation tests
│   └── test_eval/         # Evaluation and metrics tests
├── integration/           # Integration tests
│   ├── test_pipeline/     # End-to-end pipeline tests
│   ├── test_cli/         # CLI interface tests
│   └── test_workflows/   # Multi-component workflow tests
└── data/                 # Test data and fixtures
    ├── sample_train.csv  # Small training dataset
    ├── sample_test.csv   # Small test dataset
    └── fixtures/         # Data fixtures for specific test cases
```

## 🚀 Running Tests

### Quick Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test categories
python -m pytest tests/unit/           # Unit tests only
python -m pytest tests/integration/   # Integration tests only
python -m pytest tests/test_smoke.py  # Smoke tests only

# Run tests for specific component
python -m pytest tests/unit/test_features/
python -m pytest tests/unit/test_modeling/
```

### Verbose Testing

```bash
# Detailed output
python -m pytest tests/ -v

# Show print statements
python -m pytest tests/ -s

# Stop on first failure
python -m pytest tests/ -x

# Run specific test
python -m pytest tests/unit/test_data/test_loader.py::TestTitanicDataLoader::test_load_valid_data
```

### Performance Testing

```bash
# Profile test execution time
python -m pytest tests/ --durations=10

# Parallel test execution
python -m pytest tests/ -n auto  # Requires pytest-xdist
```

## 🔧 Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --disable-warnings
    --cov-report=term-missing:skip-covered
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests requiring longer execution
    external: Tests requiring external resources
    cli: CLI interface tests
```

### conftest.py
```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_train_data():
    """Small sample of training data for testing."""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived': [0, 1, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 
                'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath',
                'Allen, Mr. William Henry'],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
        'Fare': [7.25, 71.28, 7.92, 53.10, 8.05],
        'Cabin': [None, 'C85', None, 'C123', None],
        'Embarked': ['S', 'C', 'S', 'S', 'S']
    })

@pytest.fixture
def sample_test_data():
    """Small sample of test data for testing."""
    return pd.DataFrame({
        'PassengerId': [892, 893, 894],
        'Pclass': [3, 3, 2],
        'Name': ['Kelly, Mr. James', 'Wilkes, Mrs. James', 'Myles, Mr. Thomas Francis'],
        'Sex': ['male', 'female', 'male'],
        'Age': [34.5, 47.0, 62.0],
        'SibSp': [0, 1, 0],
        'Parch': [0, 0, 0],
        'Ticket': ['330911', '363272', '240276'],
        'Fare': [7.83, 7.00, 9.69],
        'Cabin': [None, None, None],
        'Embarked': ['Q', 'S', 'Q']
    })

@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path

@pytest.fixture
def mock_model():
    """Mock model for testing."""
    from unittest.mock import Mock
    model = Mock()
    model.fit.return_value = model
    model.predict.return_value = np.array([0, 1, 1])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
    return model
```

## 📝 Unit Testing Examples

### Data Loading Tests

```python
# tests/unit/test_data/test_loader.py
import pytest
import pandas as pd
from data.loader import TitanicDataLoader

class TestTitanicDataLoader:
    def test_load_valid_data(self, sample_train_data, sample_test_data, temp_dir):
        """Test loading valid CSV files."""
        # Save sample data to temporary files
        train_path = temp_dir / "train.csv"
        test_path = temp_dir / "test.csv"
        
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        
        # Test loader
        loader = TitanicDataLoader(str(train_path), str(test_path))
        train_df, test_df = loader.load()
        
        # Assertions
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) == len(sample_train_data)
        assert len(test_df) == len(sample_test_data)
        assert 'Survived' in train_df.columns
        assert 'Survived' not in test_df.columns
    
    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        loader = TitanicDataLoader("nonexistent.csv", "also_nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_load_invalid_csv(self, temp_dir):
        """Test error handling for invalid CSV files."""
        invalid_path = temp_dir / "invalid.csv"
        invalid_path.write_text("this is not valid csv content")
        
        loader = TitanicDataLoader(str(invalid_path), str(invalid_path))
        
        with pytest.raises((pd.errors.ParserError, pd.errors.EmptyDataError)):
            loader.load()

    @pytest.mark.parametrize("missing_column", ["PassengerId", "Name", "Sex"])
    def test_load_missing_required_columns(self, sample_train_data, temp_dir, missing_column):
        """Test error handling for missing required columns."""
        # Remove a required column
        invalid_data = sample_train_data.drop(columns=[missing_column])
        invalid_path = temp_dir / "invalid.csv"
        invalid_data.to_csv(invalid_path, index=False)
        
        loader = TitanicDataLoader(str(invalid_path), str(invalid_path))
        
        with pytest.raises(KeyError):
            loader.load()
```

### Feature Engineering Tests

```python
# tests/unit/test_features/test_transforms.py
import pytest
import pandas as pd
import numpy as np
from features.transforms import FamilySizeTransform, TitleTransform

class TestFamilySizeTransform:
    def test_family_size_calculation(self, sample_train_data):
        """Test family size calculation."""
        transformer = FamilySizeTransform()
        result = transformer.fit_transform(sample_train_data)
        
        # Verify family size = SibSp + Parch + 1
        expected_family_size = sample_train_data['SibSp'] + sample_train_data['Parch'] + 1
        
        assert 'FamilySize' in result.columns
        pd.testing.assert_series_equal(result['FamilySize'], expected_family_size, check_names=False)
    
    def test_is_alone_feature(self, sample_train_data):
        """Test is_alone feature creation."""
        transformer = FamilySizeTransform()
        result = transformer.fit_transform(sample_train_data)
        
        expected_is_alone = (sample_train_data['SibSp'] + sample_train_data['Parch'] == 0)
        
        assert 'IsAlone' in result.columns
        pd.testing.assert_series_equal(result['IsAlone'], expected_is_alone, check_names=False)
    
    def test_fit_transform_idempotent(self, sample_train_data):
        """Test that multiple fit_transform calls produce same result."""
        transformer = FamilySizeTransform()
        result1 = transformer.fit_transform(sample_train_data)
        result2 = transformer.fit_transform(sample_train_data)
        
        pd.testing.assert_frame_equal(result1, result2)

class TestTitleTransform:
    def test_title_extraction(self):
        """Test title extraction from names."""
        data = pd.DataFrame({
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John Bradley',
                'Heikkinen, Miss. Laina',
                'Futrelle, Mrs. Jacques Heath',
                'Allen, Mr. William Henry'
            ]
        })
        
        transformer = TitleTransform()
        result = transformer.fit_transform(data)
        
        expected_titles = ['Mr', 'Mrs', 'Miss', 'Mrs', 'Mr']
        
        assert 'Title' in result.columns
        assert result['Title'].tolist() == expected_titles
    
    def test_rare_title_grouping(self):
        """Test grouping of rare titles."""
        data = pd.DataFrame({
            'Name': [
                'Connolly, Miss. Kate',
                'Kelly, Dr. Ernest',
                'Williams, Rev. John',
                'Smith, Col. James'
            ]
        })
        
        transformer = TitleTransform()
        result = transformer.fit_transform(data)
        
        # All rare titles should be grouped as 'Rare'
        assert all(title in ['Miss', 'Rare'] for title in result['Title'])
    
    @pytest.mark.parametrize("name,expected_title", [
        ('Smith, Mr. John', 'Mr'),
        ('Jones, Mrs. Mary', 'Mrs'),
        ('Brown, Miss. Sarah', 'Miss'),
        ('Wilson, Master. Tommy', 'Master'),
        ('Davis, Dr. Robert', 'Rare'),
        ('Invalid Name Format', 'Mr'),  # Default fallback
    ])
    def test_title_extraction_cases(self, name, expected_title):
        """Test title extraction for various name formats."""
        data = pd.DataFrame({'Name': [name]})
        transformer = TitleTransform()
        result = transformer.fit_transform(data)
        
        assert result['Title'].iloc[0] == expected_title
```

### Model Tests

```python
# tests/unit/test_modeling/test_model_registry.py
import pytest
from modeling.model_registry import ModelRegistry
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class TestModelRegistry:
    def test_available_models(self):
        """Test getting list of available models."""
        registry = ModelRegistry()
        models = registry.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'random_forest' in models
        assert 'logistic' in models
    
    def test_create_valid_model(self):
        """Test creating valid models."""
        registry = ModelRegistry()
        
        rf_model = registry.create_model('random_forest')
        assert isinstance(rf_model, RandomForestClassifier)
        
        lr_model = registry.create_model('logistic')
        assert isinstance(lr_model, LogisticRegression)
    
    def test_create_invalid_model(self):
        """Test error handling for invalid model names."""
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="Unknown model"):
            registry.create_model('nonexistent_model')
    
    def test_create_model_with_params(self):
        """Test creating models with custom parameters."""
        registry = ModelRegistry()
        
        params = {'n_estimators': 200, 'max_depth': 10}
        model = registry.create_model('random_forest', **params)
        
        assert model.n_estimators == 200
        assert model.max_depth == 10
    
    @pytest.mark.parametrize("model_name", [
        'random_forest', 'logistic', 'xgboost', 'lightgbm'
    ])
    def test_all_models_creatable(self, model_name):
        """Test that all registered models can be created."""
        registry = ModelRegistry()
        
        try:
            model = registry.create_model(model_name)
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
        except ImportError:
            pytest.skip(f"{model_name} dependencies not installed")
```

## 🔗 Integration Testing Examples

### Pipeline Tests

```python
# tests/integration/test_pipeline/test_full_pipeline.py
import pytest
import pandas as pd
from pathlib import Path
from data.loader import TitanicDataLoader
from features.build import create_feature_builder
from modeling.model_registry import ModelRegistry
from cv.folds import create_fold_strategy

class TestFullPipeline:
    def test_complete_training_pipeline(self, sample_train_data, sample_test_data, temp_dir):
        """Test complete training pipeline from data to predictions."""
        # Setup data files
        train_path = temp_dir / "train.csv"
        test_path = temp_dir / "test.csv"
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        
        # 1. Load data
        loader = TitanicDataLoader(str(train_path), str(test_path))
        train_df, test_df = loader.load()
        
        # 2. Build features
        feature_builder = create_feature_builder()
        X_train = feature_builder.fit_transform(train_df)
        X_test = feature_builder.transform(test_df)
        y_train = train_df['Survived']
        
        # 3. Create cross-validation folds
        fold_strategy = create_fold_strategy('stratified', n_splits=2, random_state=42)
        folds = list(fold_strategy.split(X_train, y_train))
        
        # 4. Train model
        registry = ModelRegistry()
        model = registry.create_model('logistic', max_iter=100)
        
        # Train on first fold
        train_idx, val_idx = folds[0]
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        
        # 5. Make predictions
        val_preds = model.predict(X_train.iloc[val_idx])
        test_preds = model.predict(X_test)
        
        # Assertions
        assert len(val_preds) == len(val_idx)
        assert len(test_preds) == len(test_df)
        assert all(pred in [0, 1] for pred in val_preds)
        assert all(pred in [0, 1] for pred in test_preds)
    
    def test_feature_pipeline_consistency(self, sample_train_data, sample_test_data):
        """Test that feature pipeline produces consistent results."""
        feature_builder = create_feature_builder()
        
        # Transform multiple times
        X1 = feature_builder.fit_transform(sample_train_data)
        X2 = feature_builder.transform(sample_train_data)
        
        # Should be identical after fitting
        pd.testing.assert_frame_equal(X1, X2)
        
        # Test data should have same columns (different values OK)
        X_test = feature_builder.transform(sample_test_data)
        assert list(X1.columns) == list(X_test.columns)
```

### CLI Tests

```python
# tests/integration/test_cli/test_cli_commands.py
import pytest
import subprocess
import json
from pathlib import Path

class TestCLICommands:
    def test_cli_info_command(self):
        """Test CLI info command."""
        result = subprocess.run(
            ['python', 'src/cli.py', 'info'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Titanic ML Pipeline' in result.stdout
    
    def test_cli_validate_command(self, sample_train_data, sample_test_data, temp_dir):
        """Test CLI validate command."""
        # Setup data files
        train_path = temp_dir / "train.csv"
        test_path = temp_dir / "test.csv"
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        
        # Create config file
        config_path = temp_dir / "test_config.yaml"
        config_content = f"""
data:
  train_path: "{train_path}"
  test_path: "{test_path}"
  target_column: "Survived"
"""
        config_path.write_text(config_content)
        
        # Run validate command
        result = subprocess.run(
            ['python', 'src/cli.py', 'validate', '--config', str(config_path)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Validation completed' in result.stdout
    
    @pytest.mark.slow
    def test_cli_train_command(self, sample_train_data, temp_dir):
        """Test CLI train command."""
        # Setup data files
        train_path = temp_dir / "train.csv"
        sample_train_data.to_csv(train_path, index=False)
        
        # Create config file
        config_path = temp_dir / "train_config.yaml"
        config_content = f"""
name: "test_experiment"
data:
  train_path: "{train_path}"
  target_column: "Survived"
model_name: "logistic"
model_params:
  max_iter: 100
cv_folds: 2
output_dir: "{temp_dir}"
"""
        config_path.write_text(config_content)
        
        # Run train command
        result = subprocess.run(
            ['python', 'src/cli.py', 'train', '--experiment-config', str(config_path)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Training completed' in result.stdout
        
        # Check output files exist
        output_files = list(temp_dir.glob("*/"))
        assert len(output_files) > 0  # At least one run directory created
```

## 🎯 Test Best Practices

### 1. Test Organization

```python
# Good: Descriptive test names
def test_family_size_calculation_with_siblings_and_parents():
    pass

def test_title_extraction_handles_missing_titles():
    pass

def test_model_training_with_stratified_cv_produces_consistent_scores():
    pass

# Bad: Vague test names  
def test_features():
    pass

def test_model():
    pass
```

### 2. Parameterized Tests

```python
@pytest.mark.parametrize("model_name,expected_type", [
    ('random_forest', RandomForestClassifier),
    ('logistic', LogisticRegression),
    ('xgboost', XGBClassifier),
])
def test_model_creation(model_name, expected_type):
    registry = ModelRegistry()
    model = registry.create_model(model_name)
    assert isinstance(model, expected_type)

@pytest.mark.parametrize("cv_strategy,n_splits", [
    ('stratified', 5),
    ('group', 3),
    ('time_series', 4),
])
def test_cv_strategies(cv_strategy, n_splits):
    fold_strategy = create_fold_strategy(cv_strategy, n_splits=n_splits)
    assert fold_strategy.n_splits == n_splits
```

### 3. Fixtures for Complex Setup

```python
@pytest.fixture
def trained_model_pipeline(sample_train_data):
    """Complete trained pipeline for testing."""
    # Feature engineering
    feature_builder = create_feature_builder()
    X = feature_builder.fit_transform(sample_train_data)
    y = sample_train_data['Survived']
    
    # Model training
    registry = ModelRegistry()
    model = registry.create_model('logistic', max_iter=100)
    model.fit(X, y)
    
    return {
        'model': model,
        'feature_builder': feature_builder,
        'X': X,
        'y': y
    }

def test_prediction_with_trained_pipeline(trained_model_pipeline, sample_test_data):
    """Test predictions using trained pipeline."""
    pipeline = trained_model_pipeline
    
    # Transform test data
    X_test = pipeline['feature_builder'].transform(sample_test_data)
    
    # Make predictions
    predictions = pipeline['model'].predict(X_test)
    
    assert len(predictions) == len(sample_test_data)
    assert all(pred in [0, 1] for pred in predictions)
```

### 4. Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=10), 
       st.integers(min_value=0, max_value=10))
def test_family_size_calculation_property(sibsp, parch):
    """Property-based test for family size calculation."""
    data = pd.DataFrame({
        'SibSp': [sibsp],
        'Parch': [parch]
    })
    
    transformer = FamilySizeTransform()
    result = transformer.fit_transform(data)
    
    # Family size should always be SibSp + Parch + 1
    expected_family_size = sibsp + parch + 1
    assert result['FamilySize'].iloc[0] == expected_family_size
    
    # IsAlone should be True only when no siblings/parents
    expected_is_alone = (sibsp + parch == 0)
    assert result['IsAlone'].iloc[0] == expected_is_alone
```

## 🚨 Testing Anti-Patterns to Avoid

### ❌ Bad Practices

```python
# DON'T: Test implementation details
def test_model_uses_correct_algorithm():
    model = create_model('random_forest')
    assert 'RandomForest' in str(type(model))  # Too brittle

# DON'T: Multiple unrelated assertions
def test_everything():
    # Tests data loading, feature engineering, modeling all in one
    # Hard to debug when it fails
    pass

# DON'T: Hardcoded paths
def test_load_data():
    loader = TitanicDataLoader('/Users/me/data/train.csv', '/Users/me/data/test.csv')
    # Won't work on other machines

# DON'T: Tests that depend on external state
def test_model_training():
    # Assumes specific files exist in specific locations
    # Assumes previous test ran successfully
    pass
```

### ✅ Good Practices

```python
# DO: Test behavior, not implementation
def test_model_produces_valid_predictions():
    model = create_model('random_forest')
    predictions = model.predict(X_test)
    assert all(pred in [0, 1] for pred in predictions)

# DO: Single responsibility per test
def test_data_loader_handles_missing_file():
    # Only tests error handling for missing files
    pass

def test_feature_builder_creates_family_size():
    # Only tests family size feature creation
    pass

# DO: Use fixtures and temporary directories
def test_load_data(sample_data, temp_dir):
    data_path = temp_dir / "test.csv"
    sample_data.to_csv(data_path)
    loader = TitanicDataLoader(str(data_path))
    # Test will work anywhere

# DO: Independent tests
def test_model_training(sample_data):
    # All setup is done within the test
    # Doesn't depend on other tests
    pass
```

## 🏃‍♂️ Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

---

**🎯 Testing Checklist:**
- [ ] All components have unit tests
- [ ] Integration tests cover main workflows  
- [ ] CLI commands have tests
- [ ] Edge cases and error conditions tested
- [ ] Tests are fast and independent
- [ ] Coverage > 80%
- [ ] Tests run in CI/CD pipeline

**Remember:** Good tests are your safety net for refactoring and extending the codebase! 🛡️✨
