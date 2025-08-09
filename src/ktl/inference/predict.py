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
