from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ktl.utils.logger import LoggerFactory
from ktl.inference.predict import Predictor, InferenceError
from ktl.models.train import Trainer
from ktl.features.preprocess import preprocess_cli

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
    """Produce a quick EDA CSV summary and a full HTML profiling report."""
    import pandas as pd
    from ydata_profiling import ProfileReport
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eda_summary.csv"
    report_path = out_dir / "eda_full_report.html"
    df = pd.read_csv(train_csv)
    desc = df.describe(include="all").transpose()
    desc.to_csv(summary_path)
    profile = ProfileReport(df, title="Full EDA Report", explorative=True)
    profile.to_file(report_path)
    log.info("EDA summary written: %s", summary_path)
    log.info("Full EDA HTML report written: %s", report_path)


@app.command()
def train(
    train_csv: Path = typer.Option(..., exists=True, readable=True, help="Training CSV path"),
    config_dir: Path = typer.Option(Path("config"), help="Directory with YAML configs"),
) -> None:
    """Run cross-validation training and save artifacts."""
    run_dir = Trainer(config_dir=config_dir).run(train_csv)
    print("\nTraining complete. See above for model selection report.")
    print(f"Artifacts saved in: {run_dir}")


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


@app.command("log-submission")
def log_submission(
    file: Path = typer.Option(..., exists=True, readable=True, help="Submission CSV path that was submitted"),
    leaderboard_score: float = typer.Option(..., help="Leaderboard score to record"),
    run_dir: Optional[Path] = typer.Option(None, help="Artifacts run dir associated with this submission"),
    message: Optional[str] = typer.Option(None, help="Submission message/notes"),
    threshold: Optional[float] = typer.Option(None, help="Threshold used to convert probabilities to labels, if applicable"),
    config_dir: Path = typer.Option(Path("config"), help="Directory with YAML configs"),
) -> None:
    """Record a submission result and metadata into artifacts for tracking.

    Appends an entry to `artifacts/submissions_ledger.jsonl` and CSV, and if
    `run_dir` is provided, also writes `run_dir/submissions.jsonl`.
    """
    import json
    from datetime import datetime
    import yaml
    import pandas as pd

    base_path = config_dir / "base.yaml"
    feats_path = config_dir / "features.yaml"
    models_path = config_dir / "models.yaml"
    base = yaml.safe_load(base_path.read_text()) if base_path.exists() else {}
    feats_raw = yaml.safe_load(feats_path.read_text()) if feats_path.exists() else {}
    models_raw = yaml.safe_load(models_path.read_text()) if models_path.exists() else {}
    competition = (base.get("kaggle", {}) or {}).get("competition") or "titanic"

    # Try to gather OOF metrics from run_dir
    oof_acc = None
    auc_mean = None
    cv_model = None
    if run_dir:
        oof_eval_path = run_dir / "oof_eval.json"
        if oof_eval_path.exists():
            try:
                oof = json.loads(oof_eval_path.read_text())
                if oof.get("metric") == "accuracy":
                    oof_acc = float(oof.get("value"))
            except Exception:
                pass
        cv_path = run_dir / "cv_summary.json"
        if cv_path.exists():
            try:
                cv = json.loads(cv_path.read_text())
                cv_model = cv.get("model")
                if cv.get("metric") == "auc":
                    auc_mean = float(cv.get("score_mean"))
            except Exception:
                pass

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Snapshot configs (JSON-encoded for CSV compatibility)
    features_cfg = feats_raw.get("features", feats_raw) if isinstance(feats_raw, dict) else feats_raw
    models_cfg = models_raw.get("models", models_raw) if isinstance(models_raw, dict) else models_raw
    cv_cfg = base.get("cv")

    entry = {
        "timestamp": ts,
        "competition": competition,
        "file": str(file),
        "message": message,
        "run_dir": str(run_dir) if run_dir else None,
        "threshold": threshold,
        "oof_accuracy": oof_acc,
        "oof_auc_mean": auc_mean,
        "cv_model": cv_model,
        "leaderboard_score": float(leaderboard_score),
        "features_cfg": json.dumps(features_cfg) if features_cfg is not None else None,
        "models_cfg": json.dumps(models_cfg) if models_cfg is not None else None,
        "cv_cfg": json.dumps(cv_cfg) if cv_cfg is not None else None,
    }

    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    # JSONL ledger
    jsonl_path = artifacts / "submissions_ledger.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    # CSV ledger (append or create)
    csv_path = artifacts / "submissions_ledger.csv"
    df = pd.DataFrame([entry])
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_all = pd.concat([df_existing, df], ignore_index=True)
        df_all.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

    # Per-run JSONL if applicable
    if run_dir:
        per_run = run_dir / "submissions.jsonl"
        with per_run.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    log.info("Saved submission record: score=%.5f file=%s", leaderboard_score, file)

    # Auto-refresh reports
    try:
        report_submissions()  # type: ignore[misc]
        report_markdown()  # type: ignore[misc]
        log.info("Updated submissions_report.csv and submissions_summary.md")
    except Exception as _e:  # pragma: no cover
        log.warning("Could not auto-generate reports: %s", _e)


@app.command("ensemble-predict")
def ensemble_predict(
    run_dir: list[Path] = typer.Option(..., help="One or more run directories to ensemble", exists=True, readable=True),
    test_csv: Path = typer.Option(..., exists=True, readable=True, help="Test CSV path"),
    out_path: Path = typer.Option(Path("submission.csv"), help="Output submission CSV"),
    threshold: Optional[float] = typer.Option(None, help="Decision threshold; if omitted, use prevalence-matched threshold from OOF"),
    config_dir: Path = typer.Option(Path("config"), help="Directory with YAML configs"),
) -> None:
    """Average probabilities from multiple runs; threshold via prevalence or provided value; write submission."""
    import json
    import numpy as np
    import pandas as pd
    import yaml

    base = yaml.safe_load((config_dir / "base.yaml").read_text()) or {}
    id_column = base.get("id_column", "PassengerId")
    target = base.get("target", "Survived")
    paths = base.get("paths", {})

    predictor = Predictor()
    df_test = pd.read_csv(test_csv)
    if id_column not in df_test.columns:
        log.error("id_column '%s' not found in test CSV", id_column)
        raise typer.Exit(code=11)

    # Compute probabilities on test for each run
    test_prob_list = []
    for rd in run_dir:
        p = predictor.proba_from_run(rd, df_test)
        test_prob_list.append(p)
        log.info("Loaded test probabilities from %s", rd)
    p_test_mean = np.mean(np.vstack(test_prob_list), axis=0)

    # Derive threshold if not provided: prevalence-matched using ensemble OOF
    thr = threshold
    if thr is None:
        # Need OOFs and training labels
        train_csv = paths.get("train_csv")
        if not train_csv:
            log.error("train_csv not specified in config.paths; cannot compute prevalence threshold")
            raise typer.Exit(code=12)
        df_train = pd.read_csv(train_csv)
        if target not in df_train.columns:
            log.error("Target '%s' not found in training CSV", target)
            raise typer.Exit(code=13)
        # Average OOF probabilities across runs, aligning by id if present
        oofs = []
        for rd in run_dir:
            oof_path = rd / "oof.csv"
            if not oof_path.exists():
                log.error("Missing OOF file in run: %s", rd)
                raise typer.Exit(code=14)
            oof_df = pd.read_csv(oof_path)
            if id_column in df_train.columns and id_column in oof_df.columns:
                m = oof_df[[id_column, "oof"]].merge(df_train[[id_column, target]], on=id_column, how="inner")
                oofs.append(m[["oof"]].rename(columns={"oof": f"oof_{rd.name}"}))
            else:
                if len(oof_df) != len(df_train):
                    log.error("OOF length mismatch and no id to align for run: %s", rd)
                    raise typer.Exit(code=15)
                oofs.append(oof_df[["oof"]].rename(columns={"oof": f"oof_{rd.name}"}))
        # Concatenate on index (rows correspond after merge if used)
        oof_mat = pd.concat(oofs, axis=1)
        p_oof_mean = oof_mat.mean(axis=1).values
        prevalence = float(df_train[target].mean())
        k = int(round(len(p_oof_mean) * prevalence))
        if k <= 0:
            thr = 1.0
        else:
            thr = float(np.sort(p_oof_mean)[-k])
        log.info("Prevalence-matched threshold=%.4f (prevalence=%.3f)", thr, prevalence)

    y_pred = (p_test_mean >= float(thr)).astype(int)
    sub = pd.DataFrame({id_column: df_test[id_column].values, "Survived": y_pred})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    log.info("Ensemble submission written: %s (rows=%d)", out_path, len(sub))


@app.command("report-submissions")
def report_submissions(
    output_csv: Path = typer.Option(Path("artifacts/submissions_report.csv"), help="Output CSV path"),
    output_jsonl: Path = typer.Option(Path("artifacts/submissions_report.jsonl"), help="Output JSONL path"),
    config_dir: Path = typer.Option(Path("config"), help="Directory with YAML configs (for id/target)"),
) -> None:
    """Build a consolidated report of submissions with model/run details.

    Reads the ledger, augments entries with run metrics and simple stats
    (predicted positive rate, OOF metrics where available), and writes CSV/JSONL.
    """
    import json
    import numpy as np
    import pandas as pd
    import yaml

    artifacts = Path("artifacts")
    ledger_csv = artifacts / "submissions_ledger.csv"
    if not ledger_csv.exists():
        log.error("No ledger found at %s. Log submissions first.", ledger_csv)
        raise typer.Exit(code=20)

    base = yaml.safe_load((config_dir / "base.yaml").read_text()) if (config_dir / "base.yaml").exists() else {}
    id_column = base.get("id_column", "PassengerId")
    target = base.get("target", "Survived")
    paths = base.get("paths", {})
    train_csv = paths.get("train_csv", "data/train.csv")
    df_train = None
    if Path(train_csv).exists():
        df_train = pd.read_csv(train_csv)

    df = pd.read_csv(ledger_csv)
    rows = []
    for _, row in df.iterrows():
        entry = row.to_dict()
        file = Path(entry.get("file")) if isinstance(entry.get("file"), str) else None
        run_dir_val = entry.get("run_dir")
        run_dir = Path(run_dir_val) if isinstance(run_dir_val, str) and run_dir_val else None

        # Predicted positive rate from submission file
        pred_pos_rate = None
        if file and file.exists():
            try:
                sub_df = pd.read_csv(file)
                if "Survived" in sub_df.columns:
                    pred_pos_rate = float((sub_df["Survived"] == 1).mean())
            except Exception:
                pass
        entry["pred_pos_rate"] = pred_pos_rate

        # Augment with run metrics
        model = None
        metric = None
        fold_scores = None
        score_mean = None
        score_std = None
        oof_accuracy = entry.get("oof_accuracy") if pd.notna(entry.get("oof_accuracy")) else None
        if run_dir and run_dir.exists():
            cv_path = run_dir / "cv_summary.json"
            if cv_path.exists():
                try:
                    cv = json.loads(cv_path.read_text())
                    model = cv.get("model")
                    metric = cv.get("metric")
                    fold_scores = cv.get("fold_scores")
                    score_mean = cv.get("score_mean")
                    score_std = cv.get("score_std")
                except Exception:
                    pass
            # Compute OOF accuracy if missing and possible
            if oof_accuracy is None and df_train is not None and (run_dir / "oof.csv").exists():
                try:
                    oof_df = pd.read_csv(run_dir / "oof.csv")
                    thr = entry.get("threshold")
                    thr_val = float(thr) if thr is not None and pd.notna(thr) else 0.5
                    if id_column in df_train.columns and id_column in oof_df.columns:
                        merged = oof_df.merge(df_train[[id_column, target]], on=id_column, how="inner")
                        y_true = merged[target].values
                        y_proba = merged["oof"].values
                    else:
                        y_true = df_train[target].values[: len(oof_df)]
                        y_proba = oof_df["oof"].values
                    y_pred = (y_proba >= thr_val).astype(int)
                    oof_accuracy = float((y_pred == y_true).mean())
                except Exception:
                    pass
        entry.update({
            "model": model,
            "cv_metric": metric,
            "cv_fold_scores": json.dumps(fold_scores) if fold_scores is not None else None,
            "cv_score_mean": score_mean,
            "cv_score_std": score_std,
            "oof_accuracy": oof_accuracy,
        })

        rows.append(entry)

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    with output_jsonl.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")

    log.info("Wrote submissions report: %s (%d rows)", output_csv, len(out_df))


@app.command("report-markdown")
def report_markdown(
    output_md: Path = typer.Option(Path("artifacts/submissions_summary.md"), help="Output Markdown file"),
    top_k: int = typer.Option(5, help="Top-N rows to include for each ranking"),
) -> None:
    """Create a concise Markdown summary of submissions with top rankings.

    Reads the CSV report and the ledger to include config snapshots for the top entries.
    """
    import json
    import pandas as pd

    artifacts = Path("artifacts")
    report_csv = artifacts / "submissions_report.csv"
    ledger_csv = artifacts / "submissions_ledger.csv"
    if not report_csv.exists():
        log.error("Missing %s. Run `ktl report-submissions` first.", report_csv)
        raise typer.Exit(code=21)
    if not ledger_csv.exists():
        log.error("Missing %s. Log at least one submission first.", ledger_csv)
        raise typer.Exit(code=22)

    rep = pd.read_csv(report_csv)
    led = pd.read_csv(ledger_csv)

    def safe_json(s: object):
        try:
            if isinstance(s, str) and s:
                return json.loads(s)
        except Exception:
            return None
        return None

    # Prepare rankings
    rep_lb = rep.dropna(subset=["leaderboard_score"]).sort_values("leaderboard_score", ascending=False).head(top_k)
    rep_oof = rep.dropna(subset=["oof_accuracy"]).sort_values("oof_accuracy", ascending=False).head(top_k)

    def rows_to_md(rows: pd.DataFrame) -> str:
        lines = []
        for _, r in rows.iterrows():
            f = str(r.get("file"))
            # Join with ledger to access config snapshots
            snap = led[led["file"] == f].tail(1)
            features_cfg = safe_json(snap["features_cfg"].iloc[0]) if len(snap) else None
            models_cfg = safe_json(snap["models_cfg"].iloc[0]) if len(snap) else None
            cv_cfg = safe_json(snap["cv_cfg"].iloc[0]) if len(snap) else None
            feat_summary = None
            if isinstance(features_cfg, dict):
                keys = [
                    k for k in ["add_title", "add_family", "add_is_alone", "add_deck", "add_ticket_group_size", "log_fare", "bin_age"]
                    if k in features_cfg
                ]
                feat_summary = ", ".join([f"{k}={features_cfg[k]}" for k in keys])
            model_name = r.get("model") or r.get("cv_model")
            msg = r.get("message")
            if isinstance(msg, str) and len(msg) > 80:
                msg = msg[:77] + "..."
            lines.append(
                f"- {r.get('timestamp')}: model={model_name}, thr={r.get('threshold')}, pred_pos_rate={round(r.get('pred_pos_rate', 0),3)}; "
                f"OOF AUC={r.get('oof_auc_mean')}, OOF acc={r.get('oof_accuracy')}, LB={r.get('leaderboard_score')}\n"
                f"  file={f}\n"
                f"  run_dir={r.get('run_dir')}\n"
                + (f"  features: {feat_summary}\n" if feat_summary else "")
                + (f"  models_cfg_first: {models_cfg[0]['name']}\n" if isinstance(models_cfg, list) and models_cfg else "")
                + (f"  cv: {cv_cfg}\n" if isinstance(cv_cfg, dict) else "")
                + (f"  note: {msg}\n" if msg else "")
            )
        return "\n".join(lines)

    md = []
    md.append("# Submissions Summary\n")
    md.append("## Top by Leaderboard Score\n")
    md.append(rows_to_md(rep_lb))
    md.append("\n## Top by OOF Accuracy\n")
    md.append(rows_to_md(rep_oof))

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md))
    log.info("Wrote Markdown summary: %s", output_md)


@app.command()
def preprocessing(
    input_path: Path = typer.Option(..., exists=True, readable=True, help="Input CSV data file"),
    output_path: Path = typer.Option(..., help="Output CSV data file"),
    target: Optional[str] = typer.Option(None, help="Target column name"),
    select_cols: Optional[str] = typer.Option(None, help="Comma-separated list of columns to keep"),
    impute_strategy: str = typer.Option('median', help="Imputation strategy: median, mean, most_frequent, randomforest, knn, iterative"),
    scale_numeric: bool = typer.Option(True, help="Whether to scale numeric features"),
    encode_categorical: bool = typer.Option(True, help="Whether to one-hot encode categoricals"),
    add_family: bool = typer.Option(False, help="Add FamilySize feature"),
    add_is_alone: bool = typer.Option(False, help="Add IsAlone feature"),
    add_title: bool = typer.Option(False, help="Add Title feature"),
    add_deck: bool = typer.Option(False, help="Add Deck feature"),
    add_ticket_group_size: bool = typer.Option(False, help="Add TicketGroupSize feature"),
    log_fare: bool = typer.Option(False, help="Add LogFare feature"),
    bin_age: bool = typer.Option(False, help="Add AgeBin feature")
) -> None:
    """
    Generalized preprocessing CLI for Titanic dataset.
    """
    preprocess_cli(
        str(input_path), str(output_path), target, select_cols, impute_strategy,
        scale_numeric, encode_categorical, add_family, add_is_alone,
        add_title, add_deck, add_ticket_group_size, log_fare, bin_age
    )


if __name__ == "__main__":
    app()
