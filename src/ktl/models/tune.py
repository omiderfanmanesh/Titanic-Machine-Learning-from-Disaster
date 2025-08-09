from __future__ import annotations

from dataclasses import dataclass

from ktl.utils.logger import LoggerFactory

log = LoggerFactory.get_logger("ktl.models.tune")


@dataclass
class Tuner:
    """Parameter tuning using Optuna (stub)."""

    def optimize(self, model_name: str) -> None:
        log.warning("Tuner.optimize not implemented for %s", model_name)
