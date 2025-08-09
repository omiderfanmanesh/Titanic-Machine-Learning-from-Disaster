from __future__ import annotations

from typing import List

import numpy as np


class Ensembler:
    """Simple ensembling utilities."""

    @staticmethod
    def simple_average(arrays: List[np.ndarray]) -> np.ndarray:
        if not arrays:
            raise ValueError("No arrays provided for ensembling")
        return np.mean(np.vstack(arrays), axis=0)

    @staticmethod
    def weighted_average(arrays: List[np.ndarray], weights: List[float]) -> np.ndarray:
        if not arrays or not weights or len(arrays) != len(weights):
            raise ValueError("Arrays and weights must be non-empty and same length")
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        stacked = np.vstack(arrays)
        return np.average(stacked, axis=0, weights=w)
