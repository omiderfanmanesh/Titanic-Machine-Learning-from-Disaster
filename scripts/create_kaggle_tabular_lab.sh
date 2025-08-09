#!/usr/bin/env bash
set -euo pipefail

# Kaggle Tabular Lab — repo bootstrap script
# Creates a clean, config-first, leak-safe tabular ML project skeleton
# in the project root. By default, the root is the parent directory of this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${1:-$DEFAULT_ROOT}"
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

mkdir -p \
  artifacts \
  config \
  src/ktl/{utils,features,models,inference} \
  tests \
  .github/workflows \
  scripts

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
.venv/
venv/
ENV/
env/
.env
.mypy_cache/
.pytest_cache/
.coverage
coverage.xml

# Project
artifacts/
data/
*.ipynb_checkpoints
.DS_Store
EOF

cat > README.md << 'EOF'
# Kaggle Tabular Lab (ktl)

Config-first, leak-safe, reproducible pipeline for Kaggle tabular competitions.

Key ideas:
- Split first, fit transforms per-fold only.
- OOF predictions and per-fold metrics.
- Reproducible seeds, artifacted runs under `./artifacts/<timestamp>/`.

Quick start:
- Create and activate a virtualenv, then:
  - `pip install -e .`
  - `ktl --help`

Repo layout follows the provided architecture contracts. See `config/` for examples.
EOF

cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kaggle-tabular-lab"
version = "0.1.0"
description = "Config-first, leak-safe tabular ML pipeline for Kaggle"
authors = [{ name = "KTL", email = "noreply@example.com" }]
requires-python = ">=3.9"
dependencies = [
  "typer>=0.9",
  "pydantic>=2.5",
  "PyYAML>=6.0",
  "pandas>=2.0",
  "numpy>=1.26",
  "scikit-learn>=1.3",
  "joblib>=1.3"
]

[project.optional-dependencies]
extras = [
  "optuna>=3.4",
  "xgboost>=2.0",
  "lightgbm>=4.3",
  "catboost>=1.2"
]

[project.scripts]
ktl = "ktl.cli:app"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
EOF

cat > src/ktl/__init__.py << 'EOF'
"""Kaggle Tabular Lab (ktl).

Lightweight, config-first pipeline scaffolding for tabular ML.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
EOF

cat > src/ktl/cli.py << 'EOF'
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ktl.utils.logger import LoggerFactory
from ktl.models.train import Trainer

app = typer.Typer(help="Kaggle Tabular Lab CLI")
log = LoggerFactory.get_logger("ktl.cli")


@app.callback()
def main_callback() -> None:
    """CLI entry callback to initialize logging."""
    log.debug("CLI initialized")


@app.command()
def eda(
    train_csv: Path = typer.Option(..., exists=True, readable=True, help="Training CSV path"),
    out_dir: Path = typer.Option(Path("artifacts"), help="Output directory for EDA report"),
) -> None:
    """Produce a quick EDA CSV summary (no heavy deps)."""
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eda_summary.csv"
    df = pd.read_csv(train_csv)
    desc = df.describe(include="all").transpose()
    desc.to_csv(summary_path)
    log.info("EDA summary written: %s", summary_path)


@app.command()
def train(
    train_csv: Path = typer.Option(..., exists=True, readable=True, help="Training CSV path"),
    config_dir: Path = typer.Option(Path("config"), help="Directory with YAML configs"),
) -> None:
    """Run cross-validation training and save artifacts."""
    Trainer(config_dir=config_dir).run(train_csv)


@app.command()
def tune(
    model: str = typer.Option("ridge", help="Model name to tune"),
    config_dir: Path = typer.Option(Path("config"), help="Directory with YAML configs"),
) -> None:
    """Stub for tuning; to be implemented with Optuna."""
    log.warning("Tuning not yet implemented. Requested model: %s", model)


@app.command()
def predict(
    test_csv: Path = typer.Option(..., exists=True, readable=True, help="Test CSV path"),
    run_dir: Optional[Path] = typer.Option(None, help="Artifacts run dir; defaults to latest"),
    out_path: Path = typer.Option(Path("submission.csv"), help="Output submission CSV"),
) -> None:
    """Stub for inference; to be implemented."""
    log.warning("Predict not yet implemented. test_csv=%s out=%s", test_csv, out_path)


if __name__ == "__main__":
    app()
EOF

cat > src/ktl/utils/logger.py << 'EOF'
from __future__ import annotations

import logging
from logging import Logger


class LoggerFactory:
    """Structured logger factory for KTL.

    Ensures consistent formatting without relying on global state.
    """

    _configured = False

    @classmethod
    def _configure(cls) -> None:
        if cls._configured:
            return
        handler = logging.StreamHandler()
        fmt = (
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(logging.Formatter(fmt))
        root = logging.getLogger("ktl")
        root.setLevel(logging.INFO)
        if not root.handlers:
            root.addHandler(handler)
        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> Logger:
        cls._configure()
        return logging.getLogger(name)
EOF

cat > src/ktl/utils/exceptions.py << 'EOF'
from __future__ import annotations


class KTLException(Exception):
    """Base exception for KTL user-facing errors."""


class TrainingError(KTLException):
    """Raised for training-related errors with actionable hints."""
EOF

cat > src/ktl/utils/paths.py << 'EOF'
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_dir(base: Path = Path("artifacts")) -> Path:
    """Create a timestamped run directory under artifacts.

    Parameters
    ----------
    base : Path
        Base artifacts directory.

    Returns
    -------
    Path
        Newly created run directory.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
EOF

cat > src/ktl/utils/validation.py << 'EOF'
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


SplitScheme = Literal["kfold", "stratified", "group", "timeseries"]


@dataclass
class CVConfig:
    """Cross-validation configuration.

    Attributes
    ----------
    scheme : SplitScheme
        CV strategy.
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle (where applicable).
    random_state : int
        Seed for reproducibility.
    group_column : Optional[str]
        Group column name for group CV.
    time_column : Optional[str]
        Time column name for time series CV.
    """

    scheme: SplitScheme = "kfold"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    group_column: Optional[str] = None
    time_column: Optional[str] = None


class SplitterFactory:
    """Factory for building sklearn splitters from CVConfig."""

    @staticmethod
    def build(cfg: CVConfig):  # type: ignore[override]
        if cfg.scheme == "kfold":
            return KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state)
        if cfg.scheme == "stratified":
            return StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state)
        if cfg.scheme == "group":
            return GroupKFold(n_splits=cfg.n_splits)
        if cfg.scheme == "timeseries":
            return TimeSeriesSplit(n_splits=cfg.n_splits)
        raise ValueError(f"Unknown CV scheme: {cfg.scheme}")
EOF

cat > src/ktl/features/preprocess.py << 'EOF'
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeaturesConfig:
    """Configuration for feature preprocessing.

    Attributes
    ----------
    scale_numeric : bool
        Whether to scale numeric features.
    encode_categorical : bool
        Whether to one-hot encode categoricals.
    impute_numeric : str
        Strategy for numeric imputation.
    impute_categorical : str
        Strategy for categorical imputation.
    """

    scale_numeric: bool = True
    encode_categorical: bool = True
    impute_numeric: str = "median"
    impute_categorical: str = "most_frequent"


class PreprocessorBuilder:
    """Builds a ColumnTransformer given a dataframe and config."""

    @staticmethod
    def build(df: pd.DataFrame, target: str, cfg: Optional[FeaturesConfig] = None) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
        cfg = cfg or FeaturesConfig()
        feats = df.drop(columns=[target]) if target in df.columns else df.copy()
        num_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = feats.select_dtypes(exclude=[np.number]).columns.tolist()
        dt_cols: List[str] = []  # reserved for future

        num_steps = []
        if num_cols:
            num_steps.append(("imputer", SimpleImputer(strategy=cfg.impute_numeric)))
            if cfg.scale_numeric:
                num_steps.append(("scaler", StandardScaler(with_mean=True)))

        cat_steps = []
        if cat_cols and cfg.encode_categorical:
            cat_steps.append(("imputer", SimpleImputer(strategy=cfg.impute_categorical)))
            cat_steps.append(("ohe", OneHotEncoder(handle_unknown="ignore")))

        transformers = []
        if num_cols:
            from sklearn.pipeline import Pipeline

            transformers.append(("num", Pipeline(num_steps), num_cols))
        if cat_cols:
            from sklearn.pipeline import Pipeline

            transformers.append(("cat", Pipeline(cat_steps), cat_cols))

        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        return ct, num_cols, cat_cols, dt_cols
EOF

cat > src/ktl/models/metrics.py << 'EOF'
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal

import numpy as np
from sklearn import metrics as skm


TaskType = Literal["regression", "binary", "multiclass"]


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Metric:
    name: str
    direction: MetricDirection
    func: Callable[[np.ndarray, np.ndarray], float]

    def __call__(self, y_true: np.ndarray, preds: np.ndarray) -> float:
        return float(self.func(y_true, preds))


def _rmse(y_true: np.ndarray, preds: np.ndarray) -> float:
    return float(np.sqrt(skm.mean_squared_error(y_true, preds)))


def _auc(y_true: np.ndarray, preds: np.ndarray) -> float:
    return float(skm.roc_auc_score(y_true, preds))


def get_metric(task: TaskType, name: str) -> Metric:
    key = (task, name.lower())
    if key == ("regression", "rmse"):
        return Metric("rmse", MetricDirection.MINIMIZE, _rmse)
    if key == ("binary", "auc"):
        return Metric("auc", MetricDirection.MAXIMIZE, _auc)
    # sensible defaults
    if task == "regression":
        return Metric("rmse", MetricDirection.MINIMIZE, _rmse)
    if task == "binary":
        return Metric("auc", MetricDirection.MAXIMIZE, _auc)
    raise ValueError(f"Unsupported metric for task={task}: {name}")
EOF

cat > src/ktl/models/model_zoo.py << 'EOF'
from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import BaseEstimator

from ktl.models.metrics import TaskType
from ktl.utils.exceptions import TrainingError


class ModelFactory:
    """Factory for instantiating models by name.

    Supported core models: 'logistic' (binary), 'ridge' (regression).
    Optional: 'lgbm', 'xgb', 'catboost' (require extras installed).
    """

    @staticmethod
    def make(name: str, task: TaskType, params: Dict[str, Any]) -> BaseEstimator:
        key = name.lower()
        if key == "logistic":
            return LogisticRegression(**{**{"max_iter": 1000, "n_jobs": -1}, **params})
        if key == "ridge":
            return Ridge(**params)
        if key in {"lgbm", "xgb", "catboost"}:
            raise TrainingError(
                f"Model '{name}' requires optional dependency. Install extras: `pip install -e .[extras]`"
            )
        raise TrainingError(f"Unknown model name: {name}")
EOF

cat > src/ktl/models/train.py << 'EOF'
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ktl.features.preprocess import FeaturesConfig, PreprocessorBuilder
from ktl.models.metrics import Metric, TaskType, get_metric
from ktl.models.model_zoo import ModelFactory
from ktl.utils.logger import LoggerFactory
from ktl.utils.paths import create_run_dir
from ktl.utils.validation import CVConfig, SplitterFactory

log = LoggerFactory.get_logger("ktl.models.train")


def _load_yaml(path: Path) -> Dict:
    import yaml

    with path.open("r") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Trainer:
    """Cross-validation trainer that saves artifacts and OOF predictions.

    Parameters
    ----------
    config_dir : Path
        Directory that contains `base.yaml`, `features.yaml`, `models.yaml`.
    """

    config_dir: Path

    def _load_configs(self) -> Tuple[Dict, FeaturesConfig, Dict]:
        base = _load_yaml(self.config_dir / "base.yaml")
        feats_cfg = _load_yaml(self.config_dir / "features.yaml")
        models_cfg = _load_yaml(self.config_dir / "models.yaml")
        features = FeaturesConfig(**feats_cfg.get("features", {}))
        return base, features, models_cfg

    def run(self, train_csv: Path) -> Path:
        base, features_cfg, models_cfg = self._load_configs()

        target = base.get("target")
        id_column = base.get("id_column")
        task: TaskType = base.get("task_type", "regression")
        metric_name: str = base.get("metric_name", "rmse")
        cv_cfg = base.get("cv", {})
        cv = CVConfig(
            scheme=cv_cfg.get("scheme", "kfold"),
            n_splits=int(cv_cfg.get("n_splits", 5)),
            shuffle=bool(cv_cfg.get("shuffle", True)),
            random_state=int(cv_cfg.get("random_state", 42)),
            group_column=cv_cfg.get("group_column"),
            time_column=cv_cfg.get("time_column"),
        )

        if not target:
            raise ValueError("'target' must be set in config/base.yaml")

        model_entries = models_cfg.get("models", [])
        if not model_entries:
            raise ValueError("No models found in config/models.yaml under 'models'")
        model_name: str = model_entries[0].get("name", "ridge")
        model_params: Dict = model_entries[0].get("params", {})

        df = pd.read_csv(train_csv)
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in training CSV")

        y = df[target].values
        X = df.drop(columns=[target])

        if task == "binary":
            splitter = StratifiedKFold(n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state)
        else:
            splitter = KFold(n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state)

        ct, _, _, _ = PreprocessorBuilder.build(df, target, features_cfg)
        est = ModelFactory.make(model_name, task, model_params)
        pipe = Pipeline([("preprocess", ct), ("model", est)])
        metric: Metric = get_metric(task, metric_name)

        run_dir = create_run_dir(Path("artifacts"))
        log.info("Run dir: %s", run_dir)

        oof = np.zeros(len(df), dtype=float)
        fold_scores: List[float] = []
        fold_paths: List[Path] = []

        for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = pipe
            model.fit(X_tr, y_tr)

            if task == "binary":
                # probability for positive class if available
                try:
                    proba = model.predict_proba(X_va)
                    preds = proba[:, 1] if proba.ndim == 2 else proba
                except Exception:
                    preds = model.predict(X_va)
            else:
                preds = model.predict(X_va)

            score = metric(y_va, preds)
            fold_scores.append(score)
            oof[va_idx] = preds

            fold_path = run_dir / f"pipeline_fold{fold}.joblib"
            joblib.dump(model, fold_path)
            fold_paths.append(fold_path)
            log.info("Fold %d score=%.6f saved=%s", fold, score, fold_path.name)

        # Save OOF predictions
        oof_df = pd.DataFrame({"oof": oof})
        if id_column and id_column in df.columns:
            oof_df.insert(0, id_column, df[id_column].values)
        oof_path = run_dir / "oof.csv"
        oof_df.to_csv(oof_path, index=False)

        summary = {
            "model": model_name,
            "metric": metric.name,
            "direction": str(metric.direction.value),
            "fold_scores": fold_scores,
            "score_mean": float(np.mean(fold_scores)),
            "score_std": float(np.std(fold_scores)),
            "pipelines": [str(p) for p in fold_paths],
        }
        (run_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Saved OOF and summary to %s", run_dir)
        return run_dir
EOF

cat > src/ktl/models/tune.py << 'EOF'
from __future__ import annotations

from dataclasses import dataclass

from ktl.utils.logger import LoggerFactory

log = LoggerFactory.get_logger("ktl.models.tune")


@dataclass
class Tuner:
    """Parameter tuning using Optuna (stub)."""

    def optimize(self, model_name: str) -> None:
        log.warning("Tuner.optimize not implemented for %s", model_name)
EOF

cat > src/ktl/models/ensembling.py << 'EOF'
from __future__ import annotations

from typing import List

import numpy as np


class Ensembler:
    """Simple ensembling utilities."""

    @staticmethod
    def simple_average(arrays: List[np.ndarray]) -> np.ndarray:
        if not arrays:
            raise ValueError("No arrays provided for ensembling")
        return np.mean(np.vstack(arrays), axis=0)

    @staticmethod
    def weighted_average(arrays: List[np.ndarray], weights: List[float]) -> np.ndarray:
        if not arrays or not weights or len(arrays) != len(weights):
            raise ValueError("Arrays and weights must be non-empty and same length")
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        stacked = np.vstack(arrays)
        return np.average(stacked, axis=0, weights=w)
EOF

cat > src/ktl/inference/predict.py << 'EOF'
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ktl.utils.logger import LoggerFactory

log = LoggerFactory.get_logger("ktl.inference.predict")


@dataclass
class Predictor:
    """Predictor that loads latest run and creates a submission (stub)."""

    def predict_latest(self, run_dir: Optional[Path], test_csv: Path, id_column: str, task_type: str, out_path: Path) -> None:
        log.warning("Predictor.predict_latest not implemented yet")
EOF

cat > config/base.yaml << 'EOF'
competition_name: <NAME>
target: <COLUMN>
id_column: id
task_type: regression  # or binary|multiclass
metric_name: rmse      # or auc for binary
cv:
  scheme: kfold        # kfold|stratified|group|timeseries
  n_splits: 5
  shuffle: true
  random_state: 42
  group_column: null
  time_column: null
paths:
  train_csv: data/train.csv
  test_csv: data/test.csv
kaggle:
  competition: null
EOF

cat > config/features.yaml << 'EOF'
features:
  scale_numeric: true
  encode_categorical: true
  impute_numeric: median
  impute_categorical: most_frequent
EOF

cat > config/models.yaml << 'EOF'
models:
  - name: ridge
    params: {alpha: 1.0, random_state: 42}
  - name: logistic
    params: {C: 1.0, random_state: 42}
EOF

cat > config/tuning.yaml << 'EOF'
study:
  model_spaces:
    ridge:
      alpha: {low: 0.01, high: 10.0, log: true}
    logistic:
      C: {low: 0.01, high: 10.0, log: true}
  n_trials: 25
  pruning: false
EOF

cat > tests/test_smoke.py << 'EOF'
from __future__ import annotations

from typer.testing import CliRunner

from ktl.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--help"]) 
    assert res.exit_code == 0
    assert "Kaggle Tabular Lab" in res.stdout
EOF

echo "Repo scaffold created in $(pwd)"
