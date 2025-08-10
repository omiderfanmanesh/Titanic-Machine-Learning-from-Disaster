from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ktl.utils.logger import LoggerFactory
from ktl.inference.predict import Predictor, InferenceError
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
    threshold: float = typer.Option(0.5, min=0.0, max=1.0, help="Decision threshold for binary"),
) -> None:
    """Create a Kaggle submission CSV using the latest trained pipelines.

    Reads `config/base.yaml` to get `id_column` and `task_type`.
    """
    import yaml

    cfg_path = Path("config/base.yaml")
    if not cfg_path.exists():
        raise typer.Exit(code=1)

    base = yaml.safe_load(cfg_path.read_text()) or {}
    id_column = base.get("id_column", "PassengerId")
    task_type = base.get("task_type", "binary")

    try:
        Predictor().predict_latest(
            run_dir=run_dir,
            test_csv=test_csv,
            id_column=id_column,
            task_type=task_type,
            out_path=out_path,
            threshold=threshold,
        )
    except InferenceError as e:
        log.error("Prediction failed: %s", e)
        raise typer.Exit(code=2)


@app.command()
def eval_oof(
    run_dir: Optional[Path] = typer.Option(None, help="Artifacts run dir; defaults to latest"),
    train_csv: Optional[Path] = typer.Option(None, help="Training CSV path; defaults to config.paths.train_csv"),
    config_dir: Path = typer.Option(Path("config"), help="Directory with YAML configs"),
) -> None:
    """Evaluate OOF predictions with accuracy for binary tasks and save JSON.

    Loads the latest run's `oof.csv`, aligns with training labels, computes accuracy,
    and writes `oof_eval.json` under the run directory.
    """
    import json
    from sklearn.metrics import accuracy_score
    import yaml
    import pandas as pd

    base = yaml.safe_load((config_dir / "base.yaml").read_text()) or {}
    id_column = base.get("id_column")
    target = base.get("target")
    task_type = base.get("task_type", "binary")
    paths = base.get("paths", {})

    if task_type != "binary":
        log.error("eval_oof currently supports binary tasks only")
        raise typer.Exit(code=3)

    # Determine run_dir
    if run_dir is None:
        arts = Path("artifacts")
        runs = [p for p in arts.iterdir() if p.is_dir()]
        if not runs:
            log.error("No artifacts found. Run `ktl train` first.")
            raise typer.Exit(code=4)
        run_dir = sorted(runs, key=lambda p: p.name)[-1]

    oof_path = run_dir / "oof.csv"
    if not oof_path.exists():
        log.error("OOF file not found: %s", oof_path)
        raise typer.Exit(code=5)

    # Resolve train CSV
    if train_csv is None:
        default_train = paths.get("train_csv")
        if not default_train:
            log.error("Training CSV not provided and not set in config.paths.train_csv")
            raise typer.Exit(code=6)
        train_csv = Path(default_train)
    if not train_csv.exists():
        log.error("Training CSV not found: %s", train_csv)
        raise typer.Exit(code=7)

    df_train = pd.read_csv(train_csv)
    df_oof = pd.read_csv(oof_path)

    if target not in df_train.columns:
        log.error("Target '%s' not found in training CSV", target)
        raise typer.Exit(code=8)

    # Align by id_column if present, else by order
    if id_column and id_column in df_train.columns and id_column in df_oof.columns:
        merged = df_oof.merge(df_train[[id_column, target]], on=id_column, how="inner")
        if len(merged) != len(df_oof):
            log.warning("ID merge changed row count (oof=%d, merged=%d)", len(df_oof), len(merged))
        y_true = merged[target].values
        y_proba = merged["oof"].values
    else:
        if len(df_oof) != len(df_train):
            log.error("Row count mismatch (oof=%d, train=%d) and no id_column to merge", len(df_oof), len(df_train))
            raise typer.Exit(code=9)
        y_true = df_train[target].values
        y_proba = df_oof["oof"].values

    y_pred = (y_proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))

    out = {"metric": "accuracy", "value": acc, "n": int(len(y_true)), "run_dir": str(run_dir)}
    (run_dir / "oof_eval.json").write_text(json.dumps(out, indent=2))
    log.info("OOF accuracy: %.4f (n=%d). Saved to %s", acc, len(y_true), run_dir / "oof_eval.json")


@app.command("threshold-sweep")
def threshold_sweep(
    run_dir: Optional[Path] = typer.Option(None, help="Artifacts run dir; defaults to latest"),
    train_csv: Optional[Path] = typer.Option(None, help="Training CSV; defaults to config.paths.train_csv"),
    config_dir: Path = typer.Option(Path("config"), help="Directory with YAML configs"),
    start: float = typer.Option(0.0),
    stop: float = typer.Option(1.0),
    step: float = typer.Option(0.01),
) -> None:
    """Sweep thresholds on OOF probabilities and report best accuracy.

    Writes `oof_thresholds.csv` and `best_threshold.json` to the run directory.
    """
    import json
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import yaml

    base = yaml.safe_load((config_dir / "base.yaml").read_text()) or {}
    id_column = base.get("id_column")
    target = base.get("target")
    paths = base.get("paths", {})

    # Determine run_dir
    if run_dir is None:
        arts = Path("artifacts")
        runs = [p for p in arts.iterdir() if p.is_dir()]
        if not runs:
            log.error("No artifacts found. Run `ktl train` first.")
            raise typer.Exit(code=4)
        run_dir = sorted(runs, key=lambda p: p.name)[-1]

    oof_path = run_dir / "oof.csv"
    if not oof_path.exists():
        log.error("OOF file not found: %s", oof_path)
        raise typer.Exit(code=5)

    # Resolve train CSV
    if train_csv is None:
        default_train = paths.get("train_csv")
        if not default_train:
            log.error("Training CSV not provided and not set in config.paths.train_csv")
            raise typer.Exit(code=6)
        train_csv = Path(default_train)
    if not train_csv.exists():
        log.error("Training CSV not found: %s", train_csv)
        raise typer.Exit(code=7)

    df_train = pd.read_csv(train_csv)
    df_oof = pd.read_csv(oof_path)
    if target not in df_train.columns:
        log.error("Target '%s' not found in training CSV", target)
        raise typer.Exit(code=8)

    # Align by id if available
    if id_column and id_column in df_train.columns and id_column in df_oof.columns:
        merged = df_oof.merge(df_train[[id_column, target]], on=id_column, how="inner")
        y_true = merged[target].values
        y_proba = merged["oof"].values
    else:
        if len(df_oof) != len(df_train):
            log.error("Row count mismatch and no id_column; cannot align OOF with labels.")
            raise typer.Exit(code=9)
        y_true = df_train[target].values
        y_proba = df_oof["oof"].values

    thresholds = np.arange(start, stop + 1e-9, step)
    records = []
    best = (None, -1.0)
    for t in thresholds:
        acc = float(accuracy_score(y_true, (y_proba >= t).astype(int)))
        records.append({"threshold": float(t), "accuracy": acc})
        if acc > best[1]:
            best = (float(t), acc)

    df_thr = pd.DataFrame(records)
    df_thr.to_csv(run_dir / "oof_thresholds.csv", index=False)
    (run_dir / "best_threshold.json").write_text(json.dumps({"best_threshold": best[0], "accuracy": best[1]}, indent=2))
    log.info("Best threshold=%.3f, OOF accuracy=%.4f. Files saved in %s", best[0], best[1], run_dir)


@app.command()
def submit(
    file: Path = typer.Option(Path("artifacts/submission.csv"), exists=True, readable=True, help="Submission CSV path"),
    message: str = typer.Option("Baseline submission", help="Submission message"),
    competition: Optional[str] = typer.Option(None, help="Kaggle competition name; defaults to config.kaggle.competition or 'titanic'"),
) -> None:
    """Submit a CSV to Kaggle via the Kaggle CLI (if installed)."""
    import shutil
    import subprocess
    import yaml

    cfg_path = Path("config/base.yaml")
    base = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    comp = competition or (base.get("kaggle", {}) or {}).get("competition") or "titanic"

    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        log.error("Kaggle CLI not found. Install with `pip install kaggle` and configure API token.")
        raise typer.Exit(code=10)

    cmd = [kaggle_bin, "competitions", "submit", "-c", comp, "-f", str(file), "-m", message]
    log.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        log.info("Submission command completed.")
    except subprocess.CalledProcessError as e:
        log.error("Submission failed with exit code %s", e.returncode)
        raise typer.Exit(code=e.returncode)


if __name__ == "__main__":
    app()
