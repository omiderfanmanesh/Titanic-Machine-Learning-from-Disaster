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
