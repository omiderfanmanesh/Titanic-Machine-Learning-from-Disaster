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
