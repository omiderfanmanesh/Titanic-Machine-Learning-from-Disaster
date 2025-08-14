
Based on my exploration of the /features module, I can see it's a comprehensive preprocessing pipeline that works with YAML configuration. The system has multiple components including transforms, imputation, encoding, and scaling. Here's my test plan and to-dos:
TEST PLAN FOR /features MODULE
1. Core Components to Test
A. Feature Builder (TitanicFeatureBuilder)
Configuration loading and validation
Pipeline orchestration (5-stage process)
Fit/transform workflow
State management (fitted vs unfitted)
B. Transform Pipeline
Pre-imputation transforms (FamilySize, Deck, TicketGroup)
Post-imputation transforms (Fare, AgeBinning)
Config-driven pipeline building
C. Imputation System
Per-column strategies (median, mean, constant, model-based)
Ordered imputation (Fare → Age)
Missing indicator generation
Model-based imputation for Age
D. Encoding Orchestrator
Per-column encoding strategies (onehot, catboost, target encoding)
Default vs custom configurations
Handle missing/unknown values
Feature name generation
E. Scaling System
Feature scaling orchestration
Column alignment and schema freezing
2. Integration Testing
A. End-to-End Pipeline
Full preprocessing workflow with YAML config
Train/test data consistency
Feature schema alignment
B. Configuration Validation
YAML parsing and validation
Invalid configuration handling
Missing configuration defaults
3. Edge Cases & Error Handling
A. Data Quality
Missing columns handling
Unexpected data types
Empty datasets
Outlier handling
B. Configuration Issues
Invalid transform names
Circular dependencies
Missing required parameters
<hr></hr>
TO-DOS
Let me create comprehensive tests for the features module: