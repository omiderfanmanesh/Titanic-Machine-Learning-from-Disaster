# Architecture Documentation

## Overview

The Titanic ML Pipeline follows **SOLID principles** and **Clean Architecture** patterns to create a maintainable, testable, and extensible machine learning system.

## Core Principles

### SOLID Design Principles

#### 1. Single Responsibility Principle (SRP)
Each class and module has a single, well-defined purpose:
- `TitanicDataLoader` - Only responsible for loading data
- `FamilySizeTransform` - Only creates family size features  
- `LogisticRegressionModel` - Only wraps logistic regression functionality

#### 2. Open/Closed Principle (OCP)
The system is open for extension but closed for modification:
- New models can be added by implementing `IModel` interface
- New features can be added by implementing `ITransformer` interface
- New evaluation metrics can be added by extending `IEvaluator`

#### 3. Liskov Substitution Principle (LSP)
All implementations are interchangeable through their interfaces:
- Any `IModel` implementation can replace another in the training pipeline
- Any `IDataLoader` can be swapped without changing downstream code
- Any `IFoldSplitter` provides the same cross-validation contract

#### 4. Interface Segregation Principle (ISP)
Interfaces are focused and specific to client needs:
- `IDataLoader` only defines data loading methods
- `ITransformer` only defines fit/transform methods
- `IModel` only defines fit/predict/predict_proba methods

#### 5. Dependency Inversion Principle (DIP)
High-level modules depend on abstractions, not concretions:
- `TitanicTrainer` depends on `IModel` interface, not specific model implementations
- Pipeline orchestration depends on interfaces, not concrete classes
- Configuration drives dependency injection

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │────│  Core Utils     │────│  Configuration  │
│   (Click)       │    │  (Logging,      │    │  (YAML + Pydantic)│
└─────────────────┘    │   Paths, Seeds) │    └─────────────────┘
                       └─────────────────┘                     
                                │                              
        ┌──────────────────────────────────────────────┐      
        │                                              │      
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │────│ Feature Layer   │────│  Model Layer    │
│   - Loaders     │    │ - Transforms    │    │  - Registry     │
│   - Validators  │    │ - Builders      │    │  - Trainers     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │       
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      CV Layer   │────│  Eval Layer     │────│ Inference Layer │
│   - Splitters   │    │ - Evaluators    │    │  - Predictors   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │       
                                               ┌─────────────────┐
                                               │ Submission Layer│
                                               │  - Builders     │
                                               └─────────────────┘
```

## Interface Contracts

### Core Interfaces

#### `IDataLoader`
```python
class IDataLoader(ABC):
    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data."""
        pass
```

#### `ITransformer`  
```python
class ITransformer(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'ITransformer':
        """Fit transformer on training data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformer."""
        pass
```

#### `IModel`
```python
class IModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'IModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        pass
```

### Design Patterns

#### Factory Pattern
Used extensively for object creation:
- `ModelRegistry` creates model instances based on configuration
- `FoldSplitterFactory` creates appropriate cross-validation splitters
- `TransformFactory` creates feature transformation pipelines

#### Strategy Pattern  
Used for algorithm selection:
- Different cross-validation strategies (`StratifiedKFoldSplitter`, `GroupKFoldSplitter`)
- Different ensemble methods (`AverageEnsemble`, `WeightedEnsemble`)
- Different preprocessing strategies (`StandardScaler`, `RobustScaler`)

#### Template Method Pattern
Used in base classes to define algorithm structure:
- `BaseTransform` defines fit/transform workflow
- `BaseModel` defines common model functionality
- `BaseEvaluator` defines evaluation workflow

#### Observer Pattern
Used for logging and monitoring:
- Training progress callbacks
- Evaluation metric observers
- Artifact tracking observers

## Data Flow

### Training Pipeline
1. **Data Loading**: `IDataLoader` loads raw data
2. **Validation**: `DataValidator` checks data quality and leakage
3. **Feature Engineering**: `ITransformer` pipeline creates features
4. **Cross-Validation**: `IFoldSplitter` creates train/validation splits  
5. **Model Training**: `ITrainer` trains models per fold
6. **Evaluation**: `IEvaluator` computes metrics and analysis
7. **Artifact Storage**: Models and results saved to timestamped directory

### Inference Pipeline
1. **Data Loading**: Load test data using same `IDataLoader`
2. **Feature Engineering**: Apply fitted transformations
3. **Model Loading**: Load trained model artifacts
4. **Prediction**: Generate predictions using `IPredictor`
5. **Submission**: Format results using `ISubmissionBuilder`

## Error Handling Strategy

### Hierarchical Exception Design
```python
TitanicMLError (Base)
├── DataError
│   ├── DataLoadingError
│   ├── DataValidationError
│   └── DataLeakageError
├── FeatureError
│   ├── TransformError
│   └── FeatureBuildError
├── ModelError
│   ├── TrainingError
│   └── PredictionError
└── ConfigError
    ├── ConfigLoadError
    └── ConfigValidationError
```

### Error Recovery Strategies
- **Graceful Degradation**: Continue with warnings when optional components fail
- **Retry Logic**: Retry transient failures (network, file system)
- **Fallback Options**: Use default configurations when custom configs fail
- **Detailed Diagnostics**: Provide actionable error messages with context

## Configuration Management

### Hierarchical Configuration
```yaml
# Base configuration
experiment:
  name: "titanic_baseline"  
  seed: 42

# Override for specific runs  
experiment:
  name: "titanic_hyperopt"
  seed: 42
  model:
    name: "xgboost"
    params:
      n_estimators: 1000
```

### Configuration Validation
- **Pydantic Models**: Strong typing and validation for all config sections
- **Schema Validation**: Ensure required fields and correct types
- **Cross-Field Validation**: Validate relationships between config sections
- **Environment Override**: Support for environment variable overrides

## Testing Architecture

### Test Pyramid Structure

#### Unit Tests (Fast, Isolated)
- Test individual components with mocks
- Validate business logic and edge cases
- High coverage of core utilities and transformations

#### Integration Tests (Medium, Realistic)  
- Test component interactions with real data
- Validate end-to-end workflows
- Test configuration-driven behavior

#### Property Tests (Comprehensive, Generated)
- Test invariants with generated data
- Validate mathematical properties
- Test edge cases and boundary conditions

### Test Fixtures and Utilities
- **Synthetic Data Generation**: Realistic test data with known properties
- **Mock Factories**: Reusable mocks for external dependencies
- **Configuration Builders**: Fluent API for test configuration creation
- **Assertion Helpers**: Domain-specific assertions for ML workflows

## Performance Considerations

### Memory Management
- **Streaming Processing**: Process large datasets in chunks
- **Feature Caching**: Cache expensive feature computations
- **Model Persistence**: Efficient serialization using joblib
- **Memory Profiling**: Built-in memory usage monitoring

### Computational Efficiency
- **Parallel Processing**: Parallel cross-validation training
- **Vectorized Operations**: Use pandas/numpy vectorization
- **Lazy Evaluation**: Defer expensive computations until needed
- **Caching Strategies**: Multi-level caching for data and features

### Scalability Patterns
- **Horizontal Scaling**: Support for distributed training
- **Vertical Scaling**: Efficient resource utilization
- **Cloud Integration**: Support for cloud storage and compute
- **Resource Monitoring**: Track CPU, memory, and disk usage

## Security Considerations

### Data Protection
- **Input Validation**: Sanitize all external inputs
- **Path Traversal Protection**: Validate file paths
- **Credential Management**: Secure handling of API keys
- **Data Anonymization**: Remove PII when logging

### Code Security
- **Dependency Scanning**: Regular security vulnerability checks
- **Code Analysis**: Static analysis for security issues
- **Secrets Management**: No hardcoded secrets
- **Access Controls**: Proper file permissions and access controls

## Extension Points

### Adding New Models
1. Implement `IModel` interface
2. Add to `ModelRegistry`
3. Add configuration schema
4. Add unit tests
5. Update documentation

### Adding New Features  
1. Create transform class implementing `ITransformer`
2. Add to feature pipeline
3. Add configuration options
4. Add unit tests
5. Update feature documentation

### Adding New Evaluation Metrics
1. Extend `IEvaluator` interface
2. Implement metric calculation
3. Add visualization if needed
4. Add to evaluation reports
5. Add tests and documentation

## Future Enhancements

### Planned Improvements
- **AutoML Integration**: Automated feature selection and hyperparameter optimization
- **MLOps Integration**: Integration with MLflow, Weights & Biases, etc.
- **Real-time Inference**: Support for streaming/real-time prediction
- **Model Monitoring**: Production model monitoring and drift detection
- **A/B Testing Framework**: Support for model comparison in production

### Technical Debt Management
- **Regular Refactoring**: Scheduled code review and refactoring cycles
- **Dependency Updates**: Regular dependency version updates
- **Performance Optimization**: Continuous performance profiling and optimization  
- **Documentation Maintenance**: Keep documentation synchronized with code

## Conclusion

This architecture provides a solid foundation for machine learning pipelines that is:
- **Maintainable**: Clear separation of concerns and SOLID principles
- **Testable**: Comprehensive test coverage with dependency injection
- **Extensible**: Easy to add new models, features, and evaluation methods
- **Reproducible**: Complete configuration and seed management
- **Production-Ready**: Error handling, logging, monitoring, and security considerations
