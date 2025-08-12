# Model Training Report

**Run Directory:** artifacts/20250812-101559

## Best Model
- Name: `catboost`
- Parameters: `{'iterations': 800, 'depth': 4, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0, 'random_seed': 42, 'verbose': False}`

## Cross-Validation Scores
- Fold scores: [0.88972, 0.88643, 0.8629, 0.86424, 0.87608]
- Mean CV score: 0.87587

## Config Snapshot
### base.yaml
```competition_name: Titanic
cv:
  group_column: null
  n_splits: 5
  random_state: 42
  scheme: stratified
  shuffle: true
  time_column: null
id_column: PassengerId
kaggle:
  competition: titanic
metric_name: auc
paths:
  test_csv: data/test.csv
  train_csv: data/train.csv
target: Survived
task_type: binary
```

### models.yaml
```models:
- name: xgb
  params:
    colsample_bytree: 0.9
    learning_rate: 0.05
    max_depth: 4
    n_estimators: 500
    random_state: 42
    reg_lambda: 1.0
    subsample: 0.9
- name: catboost
  params:
    depth: 4
    iterations: 800
    l2_leaf_reg: 3.0
    learning_rate: 0.05
    random_seed: 42
    verbose: false
- name: hgb
  params:
    learning_rate: 0.06
    max_depth: 3
    random_state: 42
- name: gbc
  params:
    learning_rate: 0.05
    max_depth: 3
    n_estimators: 300
    random_state: 42
- name: logistic
  params:
    C: 1.0
    random_state: 42
```
