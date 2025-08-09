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
