# API Reference

Complete API documentation for the Titanic ML Pipeline components.

## 📦 Core Modules

### data.loader

#### TitanicDataLoader

```python
class TitanicDataLoader:
    """Handles loading and basic validation of Titanic dataset."""
    
    def __init__(self, train_path: str, test_path: str):
        """
        Initialize data loader.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file
        """
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets.
        
        Returns:
            Tuple of (train_df, test_df)
        
        Raises:
            FileNotFoundError: If data files don't exist
            ValidationError: If required columns are missing
        """
    
    def get_column_info(self) -> Dict[str, Any]:
        """Get information about dataset columns and types."""
```

### data.validate

#### DataValidator

```python
class DataValidator:
    """Validates data quality and checks for potential issues."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration.
        
        Args:
            config: Validation configuration dictionary
        """
    
    def validate_schema(self, df: pd.DataFrame, dataset_type: str) -> ValidationResult:
        """
        Validate dataframe schema.
        
        Args:
            df: DataFrame to validate
            dataset_type: 'train' or 'test'
            
        Returns:
            ValidationResult with passed/failed checks
        """
    
    def check_data_leakage(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> ValidationResult:
        """Check for potential data leakage between train and test sets."""
    
    def check_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """Check for duplicate rows in dataset."""
    
    def check_missing_patterns(self, df: pd.DataFrame) -> ValidationResult:
        """Analyze missing data patterns."""
```

## 🔧 Feature Engineering

### features.transforms

#### BaseTransform

```python
class BaseTransform:
    """Base class for all feature transformations."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'BaseTransform':
        """
        Fit transformer to data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Self
        """
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform data in one step."""
```

#### FamilySizeTransform

```python
class FamilySizeTransform(BaseTransform):
    """Creates family size related features."""
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add family size features.
        
        New columns:
            - FamilySize: SibSp + Parch + 1
            - IsAlone: 1 if no family members, 0 otherwise
        """
```

#### TitleTransform

```python
class TitleTransform(BaseTransform):
    """Extracts and groups passenger titles from names."""
    
    def __init__(self, rare_threshold: int = 10):
        """
        Args:
            rare_threshold: Minimum count to keep title separate
        """
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract titles from Name column.
        
        New columns:
            - Title: Extracted and grouped title (Mr, Mrs, Miss, Master, Rare)
        """
```

#### AgeImputeTransform

```python
class AgeImputeTransform(BaseTransform):
    """Imputes missing age values using multiple strategies."""
    
    def __init__(self, strategy: str = 'median_by_title'):
        """
        Args:
            strategy: 'median', 'mean', 'median_by_title', 'model_based'
        """
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'AgeImputeTransform':
        """Learn imputation values from training data."""
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing ages based on fitted strategy."""
```

#### FareTransform

```python
class FareTransform(BaseTransform):
    """Transforms fare values and handles missing fares."""
    
    def __init__(self, log_transform: bool = True, fill_missing: bool = True):
        """
        Args:
            log_transform: Apply log1p transformation
            fill_missing: Fill missing fares with median by Pclass
        """
```

### features.build

```python
def create_feature_builder(config: Dict[str, Any] = None) -> Pipeline:
    """
    Create feature engineering pipeline.
    
    Args:
        config: Feature configuration dictionary
        
    Returns:
        sklearn Pipeline with all transformations
    """

class FeatureBuilder:
    """High-level interface for feature engineering."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
    
    def fit(self, df: pd.DataFrame, target: pd.Series = None) -> 'FeatureBuilder':
        """Fit all transformations to training data."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformations."""
    
    def fit_transform(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step."""
    
    def get_feature_names(self) -> List[str]:
        """Get names of all output features."""
    
    def get_feature_importance_data(self, model, feature_names: List[str] = None) -> pd.DataFrame:
        """Extract feature importance from trained model."""
```

## 🤖 Modeling

### modeling.model_registry

#### ModelRegistry

```python
class ModelRegistry:
    """Registry for all available models."""
    
    def get_available_models(self) -> List[str]:
        """Get list of all available model names."""
    
    def create_model(self, model_name: str, **params) -> BaseEstimator:
        """
        Create model instance.
        
        Args:
            model_name: Name of model to create
            **params: Model hyperparameters
            
        Returns:
            Configured sklearn estimator
            
        Raises:
            ValueError: If model_name is not recognized
        """
    
    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for a model."""
    
    def get_param_space(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter search space for a model."""
```

### modeling.trainers

#### BaseTrainer

```python
class BaseTrainer:
    """Base class for model training."""
    
    def __init__(self, model, cv_strategy, evaluator):
        """
        Args:
            model: sklearn estimator
            cv_strategy: Cross-validation strategy
            evaluator: Model evaluator
        """
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """
        Train model with cross-validation.
        
        Returns:
            TrainingResult with trained models, scores, and metadata
        """
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained models."""
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
```

#### TrainingResult

```python
class TrainingResult:
    """Container for training results."""
    
    def __init__(self):
        self.models: List[BaseEstimator] = []
        self.cv_scores: Dict[str, Any] = {}
        self.oof_predictions: np.ndarray = None
        self.feature_importance: pd.DataFrame = None
        self.training_time: float = 0.0
        self.metadata: Dict[str, Any] = {}
    
    def save(self, output_dir: str) -> None:
        """Save training results to directory."""
    
    @classmethod
    def load(cls, run_dir: str) -> 'TrainingResult':
        """Load training results from directory."""
    
    def get_mean_cv_score(self) -> float:
        """Get mean cross-validation score."""
    
    def get_std_cv_score(self) -> float:
        """Get standard deviation of CV scores."""
```

## 🔄 Cross-Validation

### cv.folds

```python
def create_fold_strategy(strategy: str, **params) -> BaseCrossValidator:
    """
    Create cross-validation strategy.
    
    Args:
        strategy: 'stratified', 'kfold', 'group', 'time_series'
        **params: Strategy-specific parameters
        
    Returns:
        sklearn cross-validator
    """

class StratifiedKFoldCV:
    """Stratified K-Fold cross-validation."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = None):
        """Initialize stratified K-fold."""
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits."""

class GroupKFoldCV:
    """Group K-Fold to prevent data leakage."""
    
    def __init__(self, n_splits: int = 5, group_column: str = 'group'):
        """
        Args:
            n_splits: Number of folds
            group_column: Column to use for grouping
        """
```

### cv.validation

```python
class CrossValidator:
    """High-level cross-validation interface."""
    
    def __init__(self, cv_strategy, evaluator, random_state: int = None):
        """Initialize cross-validator."""
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> CVResult:
        """
        Perform cross-validation.
        
        Returns:
            CVResult with scores, predictions, and fold information
        """
    
    def compare_models(self, models: Dict[str, BaseEstimator], 
                      X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compare multiple models using cross-validation."""
```

## 📊 Evaluation

### eval.evaluator

#### ModelEvaluator

```python
class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, metrics: List[str] = None):
        """
        Args:
            metrics: List of metrics to compute
                   ('accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss')
        """
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Returns:
            Dictionary of metric name -> score
        """
    
    def evaluate_cv(self, y_true: np.ndarray, oof_predictions: np.ndarray,
                   fold_scores: List[Dict[str, float]]) -> Dict[str, Any]:
        """Evaluate cross-validation results."""
    
    def create_classification_report(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> str:
        """Create detailed classification report."""
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: str = None) -> None:
        """Plot confusion matrix."""
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      save_path: str = None) -> None:
        """Plot ROC curve."""
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   save_path: str = None) -> None:
        """Plot precision-recall curve."""
```

### eval.metrics

```python
def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""

def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, 
                       average: str = 'binary') -> float:
    """Calculate precision score."""

def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray,
                    average: str = 'binary') -> float:
    """Calculate recall score."""

def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray,
                average: str = 'binary') -> float:
    """Calculate F1 score."""

def calculate_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate AUC-ROC score."""

def calculate_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate log loss."""

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray = None) -> Dict[str, float]:
    """Calculate all available metrics."""
```

## 🔮 Inference

### infer.predictor

#### Predictor

```python
class Predictor:
    """High-level prediction interface."""
    
    def __init__(self, model_path: str = None, feature_builder_path: str = None):
        """
        Load trained model and feature builder.
        
        Args:
            model_path: Path to saved model
            feature_builder_path: Path to saved feature builder
        """
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Raw input data (same format as training)
            
        Returns:
            Binary predictions (0/1)
        """
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Returns:
            Probability array with shape (n_samples, 2)
        """
    
    def predict_with_uncertainty(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Returns:
            Dictionary with 'predictions', 'probabilities', 'uncertainty'
        """
    
    @classmethod
    def from_run_directory(cls, run_dir: str) -> 'Predictor':
        """Create predictor from training run directory."""
    
    def batch_predict(self, data_path: str, output_path: str, 
                     batch_size: int = 1000) -> None:
        """Process large datasets in batches."""
```

### infer.ensemble

#### EnsemblePredictor

```python
class EnsemblePredictor:
    """Ensemble multiple models for better predictions."""
    
    def __init__(self, predictors: List[Predictor], weights: List[float] = None):
        """
        Args:
            predictors: List of individual predictors
            weights: Weights for ensemble averaging (optional)
        """
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction using majority vote or weighted average."""
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Ensemble probability prediction."""
    
    def add_predictor(self, predictor: Predictor, weight: float = 1.0) -> None:
        """Add a new predictor to the ensemble."""
    
    def optimize_weights(self, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        """Optimize ensemble weights using validation data."""
```

## 📄 Submission

### submit.build_submission

```python
def create_submission(predictions: np.ndarray, test_ids: np.ndarray,
                     output_path: str, metadata: Dict[str, Any] = None) -> None:
    """
    Create Kaggle submission file.
    
    Args:
        predictions: Binary predictions
        test_ids: PassengerId values for test set
        output_path: Path to save submission CSV
        metadata: Additional metadata to include
    """

def validate_submission(submission_path: str) -> ValidationResult:
    """
    Validate submission file format.
    
    Returns:
        ValidationResult indicating if submission is valid
    """

class SubmissionBuilder:
    """High-level submission creation interface."""
    
    def __init__(self, test_data_path: str):
        """Initialize with test data path."""
    
    def from_predictions(self, predictions: np.ndarray, 
                        output_path: str) -> None:
        """Create submission from prediction array."""
    
    def from_predictor(self, predictor: Predictor, 
                      output_path: str) -> None:
        """Create submission using predictor."""
    
    def from_ensemble(self, predictors: List[Predictor], 
                     weights: List[float], output_path: str) -> None:
        """Create ensemble submission."""
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to submission."""
```

## 🖥️ CLI Interface

### cli

```python
@click.group()
def cli():
    """Titanic ML Pipeline CLI."""
    pass

@cli.command()
@click.option('--competition', default='titanic', help='Kaggle competition name')
@click.option('--output-dir', default='data/raw', help='Output directory')
def download(competition: str, output_dir: str):
    """Download competition data from Kaggle."""

@cli.command()
@click.option('--config', required=True, help='Data validation config file')
def validate(config: str):
    """Validate data quality and check for issues."""

@cli.command()  
@click.option('--experiment-config', required=True, help='Feature config file')
@click.option('--data-config', default='data', help='Data config file')
def features(experiment_config: str, data_config: str):
    """Build features using configuration."""

@cli.command()
@click.option('--experiment-config', required=True, help='Experiment config file')
@click.option('--data-config', default='data', help='Data config file')
def train(experiment_config: str, data_config: str):
    """Train model with cross-validation."""

@cli.command()
@click.option('--run-dir', required=True, help='Training run directory')
def evaluate(run_dir: str):
    """Evaluate trained model performance."""

@cli.command()
@click.option('--run-dir', help='Training run directory')
@click.option('--model-path', help='Path to trained model')
@click.option('--data-config', default='inference', help='Data config file')
@click.option('--output-path', required=True, help='Output predictions path')
def predict(run_dir: str, model_path: str, data_config: str, output_path: str):
    """Make predictions on test data."""

@cli.command()
@click.option('--predictions-path', help='Path to predictions CSV')
@click.option('--run-dir', help='Training run directory') 
@click.option('--output-path', required=True, help='Submission output path')
@click.option('--data-config', default='inference', help='Data config file')
def submit(predictions_path: str, run_dir: str, output_path: str, data_config: str):
    """Create Kaggle submission file."""

@cli.command()
def info():
    """Display pipeline information and status."""
```

## 🔧 Utilities

### core.utils

```python
def setup_logging(level: str = 'INFO', log_file: str = None) -> None:
    """Configure logging for the application."""

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""

def create_run_directory(base_dir: str = 'artifacts') -> str:
    """Create timestamped run directory."""

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if necessary."""

def get_git_commit_hash() -> str:
    """Get current git commit hash for reproducibility."""

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"{self.description} completed in {elapsed:.2f}s")
```

### core.interfaces

```python
class DataLoader(Protocol):
    """Interface for data loading classes."""
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data."""
        ...

class Transform(Protocol):
    """Interface for feature transformation classes."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'Transform':
        """Fit transformer to data."""
        ...
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        ...

class Trainer(Protocol):
    """Interface for model training classes."""
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train model on data."""
        ...

class Evaluator(Protocol):
    """Interface for model evaluation classes."""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions."""
        ...

class CVStrategy(Protocol):
    """Interface for cross-validation strategies."""
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits."""
        ...
```

## 🔍 Exception Classes

```python
class TitanicMLError(Exception):
    """Base exception for Titanic ML Pipeline."""
    pass

class DataLoadError(TitanicMLError):
    """Error loading data files."""
    pass

class ValidationError(TitanicMLError):
    """Data validation error."""
    pass

class FeatureEngineeringError(TitanicMLError):
    """Error in feature engineering process."""
    pass

class ModelTrainingError(TitanicMLError):
    """Error during model training."""
    pass

class PredictionError(TitanicMLError):
    """Error making predictions."""
    pass

class ConfigurationError(TitanicMLError):
    """Configuration file error."""
    pass
```

## 📋 Type Definitions

```python
from typing import Union, List, Dict, Any, Tuple, Optional, Callable, Protocol
import pandas as pd
import numpy as np

# Common type aliases
DataFrame = pd.DataFrame
Series = pd.Series
NDArray = np.ndarray

# Configuration types
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, float]
ParamsDict = Dict[str, Any]

# Data types
TrainTestData = Tuple[DataFrame, DataFrame]
Features = DataFrame
Target = Series
Predictions = NDArray

# Model types  
Model = Any  # sklearn estimator
ModelList = List[Model]
ModelDict = Dict[str, Model]

# Cross-validation types
FoldIndices = Tuple[NDArray, NDArray]
CVSplits = Iterator[FoldIndices]
CVScores = Dict[str, Any]
```

---

**📚 Usage Examples:**

```python
# Basic usage
from data.loader import TitanicDataLoader
from features.build import create_feature_builder
from modeling.model_registry import ModelRegistry

# Load data
loader = TitanicDataLoader('train.csv', 'test.csv')
train_df, test_df = loader.load()

# Build features
builder = create_feature_builder()
X_train = builder.fit_transform(train_df)
y_train = train_df['Survived']

# Train model
registry = ModelRegistry()
model = registry.create_model('random_forest')
model.fit(X_train, y_train)

# Make predictions
X_test = builder.transform(test_df)
predictions = model.predict(X_test)
```

**🔗 See Also:**
- [Configuration Guide](CONFIGURATION.md) for config file formats
- [Examples](EXAMPLES.md) for practical usage examples
- [Testing Guide](TESTING.md) for testing the API

---

*This API reference is auto-generated from code docstrings. For the most up-to-date information, refer to the source code.*
