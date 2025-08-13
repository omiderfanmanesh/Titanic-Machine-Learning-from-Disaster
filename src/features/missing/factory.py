from __future__ import annotations
from typing import Dict, Any
import pandas as pd

from core.interfaces import ImputationStrategy
from .strategies.simple import SimpleImputerStrategy
from .strategies.fill import ForwardFillStrategy, BackwardFillStrategy
from .strategies.knn import KNNImputerStrategy
from .strategies.model import ModelImputerStrategy
from .estimators import DefaultEstimatorFactory
from .preprocessors import DefaultFeaturePreprocessor

class StrategyFactory:
    """OCP: add new strategies without changing orchestrator."""
    def __init__(self, debug: bool = False):
        self.est_factory = DefaultEstimatorFactory()
        self.preprocessor = DefaultFeaturePreprocessor()
        self.debug = debug

    def make(self, column: str, plan: Dict[str, Any]) -> ImputationStrategy:
        method = plan["method"]
        if method in {"mean", "median", "most_frequent", "constant"}:
            return SimpleImputerStrategy(column, plan)
        if method == "ffill":
            return ForwardFillStrategy(column, plan)
        if method == "bfill":
            return BackwardFillStrategy(column, plan)
        if method == "knn":
            return KNNImputerStrategy(column, plan)
        if method == "model":
            return ModelImputerStrategy(column, plan, self.est_factory, self.preprocessor, debug=self.debug)
        raise ValueError(f"Unknown imputation method '{method}' for column '{column}'")
