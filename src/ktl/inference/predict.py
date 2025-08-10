from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from ktl.utils.exceptions import KTLException, InferenceError
from ktl.utils.logger import LoggerFactory

log = LoggerFactory.get_logger("ktl.inference.predict")


def _find_latest_run_dir(base: Path = Path("artifacts")) -> Path:
    runs: List[Path] = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        raise InferenceError("No run directories found under 'artifacts/'. Run `ktl train` first.")
    # Sort by directory name (timestamp format YYYYMMDD-HHMMSS)
    latest = sorted(runs, key=lambda p: p.name)[-1]
    return latest


def _load_pipelines_from_summary(run_dir: Path) -> List[Path]:
    summary_path = run_dir / "cv_summary.json"
    if not summary_path.exists():
        raise InferenceError(f"Missing cv_summary.json in {run_dir}. Train first or specify a valid run_dir.")
    data = json.loads(summary_path.read_text())
    pipe_paths = [Path(p) for p in data.get("pipelines", [])]
    if not pipe_paths:
        # Fallback to glob
        pipe_paths = sorted(run_dir.glob("pipeline_fold*.joblib"))
    if not pipe_paths:
        raise InferenceError(f"No pipeline artifacts found in {run_dir}.")
    return pipe_paths


@dataclass
class Predictor:
    """Predictor that loads fold pipelines and writes a Kaggle submission.

    Methods
    -------
    predict_latest(run_dir, test_csv, id_column, task_type, out_path)
        Loads the specified or latest run, averages fold predictions, and writes
        a submission with columns `[id_column, 'Survived']` for Titanic.
    """

    def _predict_proba_binary(self, pipelines: Sequence, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Average fold probabilities for the positive class if available.

        Returns an array of shape (n_samples,) of probabilities, or None if
        no probability-like outputs are available.
        """
        n = len(X)
        probas: List[np.ndarray] = []
        for i, model in enumerate(pipelines):
            try:
                p = model.predict_proba(X)
                p1 = p[:, 1] if p.ndim == 2 else p
                probas.append(p1.astype(float))
                log.info("Fold %d: used predict_proba", i)
                continue
            except Exception:
                pass
            try:
                score = model.decision_function(X)
                # Convert to probability via sigmoid; threshold 0.5 equivalent to score > 0
                p1 = 1.0 / (1.0 + np.exp(-score))
                probas.append(p1.astype(float))
                log.info("Fold %d: used decision_function->sigmoid", i)
                continue
            except Exception:
                pass
        if not probas:
            return None
        return np.mean(np.vstack(probas), axis=0)

    def predict_latest(
        self,
        run_dir: Optional[Path],
        test_csv: Path,
        id_column: str,
        task_type: str,
        out_path: Path,
        threshold: float = 0.5,
    ) -> None:
        if task_type != "binary":
            raise InferenceError("This predictor currently supports binary tasks (Titanic).")

        run_path = Path(run_dir) if run_dir else _find_latest_run_dir()
        pipe_paths = _load_pipelines_from_summary(run_path)
        pipelines = [joblib.load(p) for p in pipe_paths]

        df_test = pd.read_csv(test_csv)
        if id_column not in df_test.columns:
            raise InferenceError(f"id_column '{id_column}' not found in test CSV.")

        X_test = df_test.copy()
        # Try to get probabilities and threshold them; else fall back to label voting
        p_test = self._predict_proba_binary(pipelines, X_test)
        if p_test is not None:
            y_pred = (p_test >= float(threshold)).astype(int)
        else:
            votes: List[np.ndarray] = []
            for i, m in enumerate(pipelines):
                yhat = m.predict(X_test)
                votes.append(yhat.astype(int))
                log.info("Fold %d: used predict() labels (no proba)", i)
            y_pred = (np.mean(np.vstack(votes), axis=0) >= 0.5).astype(int)
        if y_pred.ndim != 1 or len(y_pred) != len(df_test):
            raise InferenceError("Prediction shape mismatch with test data.")

        sub = pd.DataFrame({id_column: df_test[id_column].values, "Survived": y_pred.astype(int)})
        # Ensure only two columns and row count matches expectation (Titanic test has 418 rows)
        sub = sub[[id_column, "Survived"]]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        log.info("Submission written: %s (rows=%d)", out_path, len(sub))
