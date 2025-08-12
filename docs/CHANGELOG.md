# CHANGELOG - Detected Issues & Fixes

## Identified Issues & Refactoring Changes

### 1. **Architecture Violations**
**Before**: Monolithic CLI commands with business logic embedded directly
**After**: Clean separation of concerns with interfaces, dependency injection, and modular components
**Rationale**: Violates Single Responsibility Principle; hard to test and extend

### 2. **Missing Core Interfaces**
**Before**: Concrete classes tightly coupled, no polymorphic interfaces
**After**: Abstract base classes for IDataLoader, ITransformer, IModel, ITrainer, IEvaluator
**Rationale**: Violates Dependency Inversion Principle; prevents extensibility

### 3. **Data Leakage Risk**
**Before**: Pipeline fitting on full dataset before CV splits in some paths
**After**: Strict fit-per-fold with isolated preprocessing pipelines
**Rationale**: Training data leakage can inflate CV scores and hurt generalization

### 4. **Poor Configuration Management**
**Before**: Hard-coded parameters scattered throughout codebase
**After**: Centralized YAML-driven configuration with Pydantic validation
**Rationale**: Reduces repeatability and makes experimentation difficult

### 5. **Inadequate Testing**
**Before**: Only basic smoke test, no unit tests for core logic
**After**: Comprehensive test suite with unit tests, integration tests, and fixtures
**Rationale**: No confidence in code correctness, difficult to refactor safely

### 6. **Inconsistent Error Handling**
**Before**: Silent failures, generic exceptions, poor error messages
**After**: Custom exception hierarchy with actionable error messages
**Rationale**: Debugging and troubleshooting is difficult

### 7. **Missing Reproducibility Guarantees**
**Before**: Inconsistent seeding, no deterministic pipeline execution
**After**: Global seed management, deterministic caching, version tracking
**Rationale**: Results not reproducible across runs

### 8. **No Model Registry Pattern**
**Before**: Model instantiation scattered, hard to add new models
**After**: Registry pattern with factory methods for clean extensibility
**Rationale**: Violates Open/Closed Principle

### 9. **Inadequate Validation**
**Before**: No schema validation, silent data issues
**After**: Comprehensive data validation with Pandera schemas
**Rationale**: Data quality issues propagate silently through pipeline

### 10. **Poor Logging & Observability**
**Before**: Print statements, no structured logging
**After**: Structured logging with correlation IDs, metrics tracking
**Rationale**: Difficult to debug issues in production runs
