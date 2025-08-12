# Model Training Report

**Run Directory:** artifacts/20250811-121558

## Best Model
- Name: `logistic`
- Parameters: `{'C': 1.0, 'random_state': 42}`

## Cross-Validation Scores
- Fold scores: [0.90593, 0.88743, 0.86858, 0.86631, 0.90347]
- Mean CV score: 0.88635

## Features Used
- scale_numeric: True
- encode_categorical: True
- impute_numeric: median
- impute_categorical: most_frequent
- add_family: True
- add_is_alone: True
- add_title: True
- add_deck: False
- add_ticket_group_size: False
- log_fare: True
- bin_age: False
- rare_title_threshold: 10

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

### features.yaml
```features:
  add_deck: false
  add_family: true
  add_is_alone: true
  add_ticket_group_size: false
  add_title: true
  bin_age: false
  encode_categorical: true
  impute_categorical: most_frequent
  impute_numeric: median
  log_fare: true
  rare_title_threshold: 10
  scale_numeric: true
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
