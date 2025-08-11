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
from ktl.utils.validation import CVConfig

import warnings

warnings.filterwarnings("ignore")

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

        # Train all models and select the best
        best_score = None
        best_model = None
        best_model_name = None
        best_model_params = None
        best_pipe = None
        best_oof = None
        best_fold_scores = None
        for entry in model_entries:
            model_name = entry.get("name", "ridge")
            model_params = entry.get("params", {})
            est = ModelFactory.make(model_name, task, model_params)
            pipe = Pipeline([("preprocess", ct), ("model", est)])
            oof = np.zeros(len(df), dtype=float)
            fold_scores: List[float] = []
            for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y)):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                model = pipe
                model.fit(X_tr, y_tr)
                if task == "binary":
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
            avg_score = np.mean(fold_scores)
            log.info(f"Model {model_name} CV score: {avg_score:.5f}")
            if (best_score is None) or (avg_score > best_score):
                best_score = avg_score
                best_model = est
                best_model_name = model_name
                best_model_params = model_params
                best_pipe = pipe
                best_oof = oof.copy()
                best_fold_scores = fold_scores.copy()
        log.info(f"Best model: {best_model_name} with CV score: {best_score:.5f}")
        # Save best model and OOF predictions
        joblib.dump(best_pipe, run_dir / f"model_{best_model_name}.joblib")
        np.save(run_dir / f"oof_{best_model_name}.npy", best_oof)
        with (run_dir / f"cv_scores_{best_model_name}.json").open("w") as f:
            json.dump(best_fold_scores, f)

        # Generate a markdown report for this run
        report_path = run_dir / "report.md"
        with report_path.open("w") as f:
            f.write("# Model Training Report\n\n")
            f.write(f"**Run Directory:** {run_dir}\n\n")
            f.write("## Best Model\n")
            f.write(f"- Name: `{best_model_name}`\n")
            f.write(f"- Parameters: `{best_model_params}`\n\n")
            f.write("## Cross-Validation Scores\n")
            f.write(f"- Fold scores: {[round(s, 5) for s in best_fold_scores]}\n")
            f.write(f"- Mean CV score: {round(best_score, 5)}\n\n")
            # Features used
            f.write("## Features Used\n")
            import yaml
            feats_cfg = yaml.safe_load((self.config_dir / "features.yaml").read_text())
            for k, v in feats_cfg.get("features", feats_cfg).items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
            # Config snapshot
            f.write("## Config Snapshot\n")
            base_cfg = yaml.safe_load((self.config_dir / "base.yaml").read_text())
            models_cfg = yaml.safe_load((self.config_dir / "models.yaml").read_text())
            f.write("### base.yaml\n```")
            yaml.dump(base_cfg, f)
            f.write("```\n\n")
            f.write("### features.yaml\n```")
            yaml.dump(feats_cfg, f)
            f.write("```\n\n")
            f.write("### models.yaml\n```")
            yaml.dump(models_cfg, f)
            f.write("```\n")
        print(f"Report saved to: {report_path}")

        # Print summary to terminal
        print("\n=== Model Selection Report ===")
        print(f"Best model: {best_model_name}")
        print(f"CV scores: {[round(s, 5) for s in best_fold_scores]}")
        print(f"Mean CV score: {round(best_score, 5)}")
        print(f"Model params: {best_model_params}")
        print(f"Artifacts saved in: {run_dir}")

        return run_dir
