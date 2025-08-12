import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    accuracy_score,
    roc_curve
)

class ThresholdOptimizer:
    """
    Finds optimal classification thresholds for various objectives.
    """

    def __init__(self, y_true, y_proba):
        self.y_true = np.asarray(y_true).astype(int)
        self.y_proba = np.asarray(y_proba).astype(float)
        assert self.y_true.shape == self.y_proba.shape, "y_true and y_proba must have same length."

    def best_f1(self):
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        idx = np.argmax(f1_scores)
        return thresholds[idx], f1_scores[idx]

    def best_accuracy(self):
        thresholds = np.linspace(0, 1, 101)
        best_t, best_acc = 0.5, 0
        for t in thresholds:
            y_pred = (self.y_proba >= t).astype(int)
            acc = accuracy_score(self.y_true, y_pred)
            if acc > best_acc:
                best_t, best_acc = t, acc
        return best_t, best_acc

    def best_youdenj(self):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
        J = tpr - fpr
        idx = np.argmax(J)
        return thresholds[idx], J[idx]

    def best_cost(self, cost_fp=1.0, cost_fn=1.0):
        thresholds = np.linspace(0, 1, 101)
        best_t, min_cost = 0.5, float("inf")
        for t in thresholds:
            y_pred = (self.y_proba >= t).astype(int)
            fp = np.sum((self.y_true == 0) & (y_pred == 1))
            fn = np.sum((self.y_true == 1) & (y_pred == 0))
            cost = cost_fp * fp + cost_fn * fn
            if cost < min_cost:
                best_t, min_cost = t, cost
        return best_t, min_cost

    def summary(self, cost_fp=1.0, cost_fn=1.0):
        """Return all thresholds in one dict"""
        t_f1, s_f1 = self.best_f1()
        t_acc, s_acc = self.best_accuracy()
        t_j, s_j = self.best_youdenj()
        t_cost, s_cost = self.best_cost(cost_fp, cost_fn)
        return {
            "best_f1": (t_f1, s_f1),
            "best_accuracy": (t_acc, s_acc),
            "best_youdenj": (t_j, s_j),
            "min_cost": (t_cost, s_cost)
        }

