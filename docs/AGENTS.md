# Codex Global Guidance

## Coding Values
- Favor **clean, idiomatic Python**, small objects with single responsibility (SOLID), and **full type hints**.
- Prefer **composition over inheritance**; inject dependencies via constructors or function params.
- Always add **docstrings** (NumPy or Google style) and **structured logging** (no bare prints).
- Determinism first: seed randomness; never rely on global state.

## Defaults
- Python ≥ 3.9; use `pathlib`, `dataclasses`, `typing`, `typing_extensions` if needed.
- Use `typer` for CLIs, `pydantic` or YAML for configs (config-first mindset).
- Use `sklearn` Pipelines, `ColumnTransformer`; fit transforms **inside CV folds** only.

## Safety & Secrets
- Never hardcode API keys or tokens. Read from env vars or user config files.
- When the internet might be unavailable, implement graceful fallbacks (e.g., if profiling lib missing → CSV summary).

## Tests & Tooling
- Include at least **smoke tests** for imports/CLI.
- Keep external deps minimal and pinned sensibly.
- Prefer readable error messages with custom exceptions.

## Git Hygiene
- Small, purposeful commits with imperative messages: `Add`, `Fix`, `Refactor`, `Docs`.
- No generated artifacts or data in version control by default (except tiny fixtures).


# Kaggle Tabular Lab — Project AGENT

## Vision
A **config-first, leak-safe, OOP** pipeline for tabular Kaggle comps:
EDA → preprocessing → CV training → tuning → ensembling → inference → optional Kaggle submit.

## Non-Negotiables
- **Leak safety**: split first, fit transforms **per fold**. Never fit on full train before splitting.
- **OOF everywhere**: save out-of-fold preds + per-fold metrics.
- **Reproducibility**: global seed; log library versions; artifacts under `./artifacts/<timestamp>/`.

## Architecture Contracts
- `utils/validation.py`: `CVConfig` + `SplitterFactory.build(...)`.
- `features/preprocess.py`: `FeaturesConfig` + `PreprocessorBuilder.build(df, target, cfg) -> (ColumnTransformer, num_cols, cat_cols, dt_cols)`.
- `models/metrics.py`: `TaskType`, `MetricDirection`, `Metric` + `get_metric(task, name)`.
- `models/model_zoo.py`: `ModelFactory.make(name, task, params)` returning an estimator with `fit`, `predict`, optional `predict_proba`.
- `models/train.py`: `Trainer.run(train_csv)`: runs CV, saves fold pipelines, `oof.csv`, `cv_summary.json`.
- `models/tune.py`: `Tuner.optimize(model_name)` using Optuna; saves best params/value.
- `models/ensembling.py`: `Ensembler.simple_average`, `Ensembler.weighted_average`.
- `inference/predict.py`: `Predictor.predict_latest(run_dir, test_csv, id_column, task_type, out_path)`.

## Coding Standards
- Full **type hints** (use `numpy.typing` where helpful).  
- Docstrings for all public classes/methods; raise `KTLException` subclasses on user-facing errors.  
- Prefer `pathlib` and `typer`. No `print()` in library code.

## Config-First Rules
- `config/base.yaml`: `competition_name`, `target`, `id_column`, `task_type`, `metric_name`, `cv`, `paths`, `kaggle`.
- `features.yaml`: numeric/categorical imputation+scaling/encoding, rare-category combiner, optional datetime expansion & text TF-IDF+SVD, feature selection toggles.
- `models.yaml`: list of models (`lgbm`, `xgb`, `catboost`, `logistic` for binary, `ridge` for regression) with params.
- `tuning.yaml`: Optuna spaces for `lgbm/xgb`, trials, pruning.

## Edge Cases to Handle
- High-cardinality categoricals → combine rare categories or hashing (if added).
- Time series → `TimeSeriesSplit`, prevent future leakage (no future-derived features).
- Profiling not installed → fallback to CSV summary.
- Kaggle API missing → skip download/submit gracefully.

## Acceptance Checklist
- `pip install -e .` works.
- `ktl --help` shows commands.
- `ktl eda` produces HTML or CSV fallback.
- `ktl train` saves artifacts, OOF, and `cv_summary.json`.
- `ktl tune --model lgbm` returns best params/value and saves summary.
- `ktl predict` creates `submission.csv` with correct `id_column` order.

## Memory: Fill Before First Run
- Competition: `<NAME>`
- Target: `<COLUMN>`
- Task: `<regression|binary|multiclass>`
- Metric: `<rmse|auc|...>`
- CV: `<kfold|stratified|group|timeseries>`, `n_splits`, `group/time column` if applicable.


# Models Module AGENT

## Scope
Model abstractions, metrics, training, tuning, and ensembling. No CLI, no file I/O outside artifacts helpers.

## Contracts & Interfaces
- `ModelFactory.make(name, task, params)` → returns estimator with:
  - `fit(X, y, **fit_params)`
  - `predict(X)` for all tasks
  - `predict_proba(X)` for classifiers (binary returns prob of positive class or (n,2) array).
- `Metric` is callable: `__call__(y_true, preds) -> float` and exposes `direction`.

## Invariants
- **Never** fit preprocessors on full train; fold-only inside `Trainer`.
- For binary classification, prefer probability outputs; AUC is default local metric.
- Early stopping: pass eval sets via estimator-native hooks when available, but keep training deterministic.

## Extending
- Add a model: implement branch in `ModelFactory`; document expected params; update `models.yaml` example; add a tiny smoke test.
- Add a metric: extend `get_metric` and update docstring with direction + expectations (max/min).

## Logging & Errors
- Use `LoggerFactory.get_logger("ktl.models")`.
- Wrap third-party errors in `TrainingError` with actionable hints.

## Done When
- Per-fold metrics logged.
- `oof.csv` and `cv_summary.json` written by `Trainer`.
- Fold pipelines (`joblib`) saved and reloadable by `Predictor`.

