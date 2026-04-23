import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


class Evaluator:
    def __init__(self):
        self.regression_results     = {}
        self.classification_results = {}

    def nasa_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        d = np.array(y_pred) - np.array(y_true)
        return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

    def regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           name: str = "model") -> dict:
        m = {
            "rmse":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae":        float(mean_absolute_error(y_true, y_pred)),
            "r2":         float(r2_score(y_true, y_pred)),
            "nasa_score": self.nasa_score(y_true, y_pred),
        }
        self.regression_results[name] = m
        return m

    def classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: np.ndarray, name: str = "model") -> dict:
        m = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        }
        self.classification_results[name] = m
        return m

    def accuracy_within_tolerance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  tolerance: int = 10) -> float:
        return float((np.abs(np.array(y_pred) - np.array(y_true)) <= tolerance).mean())
