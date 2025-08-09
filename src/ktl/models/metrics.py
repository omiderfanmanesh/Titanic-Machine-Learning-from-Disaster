from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal

import numpy as np
from sklearn import metrics as skm


TaskType = Literal["regression", "binary", "multiclass"]


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Metric:
    name: str
    direction: MetricDirection
    func: Callable[[np.ndarray, np.ndarray], float]

    def __call__(self, y_true: np.ndarray, preds: np.ndarray) -> float:
        return float(self.func(y_true, preds))


def _rmse(y_true: np.ndarray, preds: np.ndarray) -> float:
    return float(np.sqrt(skm.mean_squared_error(y_true, preds)))


def _auc(y_true: np.ndarray, preds: np.ndarray) -> float:
    return float(skm.roc_auc_score(y_true, preds))


def get_metric(task: TaskType, name: str) -> Metric:
    key = (task, name.lower())
    if key == ("regression", "rmse"):
        return Metric("rmse", MetricDirection.MINIMIZE, _rmse)
    if key == ("binary", "auc"):
        return Metric("auc", MetricDirection.MAXIMIZE, _auc)
    # sensible defaults
    if task == "regression":
        return Metric("rmse", MetricDirection.MINIMIZE, _rmse)
    if task == "binary":
        return Metric("auc", MetricDirection.MAXIMIZE, _auc)
    raise ValueError(f"Unsupported metric for task={task}: {name}")
