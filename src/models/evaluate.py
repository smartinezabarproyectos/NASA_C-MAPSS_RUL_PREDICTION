"""Métricas de evaluación: regresión, clasificación y NASA Score."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA Scoring Function — penaliza más predicciones tardías.

    s_i = exp(-d/13) - 1  si d < 0 (predicción temprana)
    s_i = exp(d/10)  - 1  si d >= 0 (predicción tardía)

    donde d = y_pred - y_true (error de predicción).
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(scores.sum())


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regresión."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score": nasa_scoring_function(y_true, y_pred),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula métricas de clasificación."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Crea tabla comparativa de resultados de múltiples modelos.

    Parameters
    ----------
    results : dict
        {'model_name': {'rmse': ..., 'mae': ..., ...}, ...}

    Returns
    -------
    pd.DataFrame con modelos como filas y métricas como columnas.
    """
    return pd.DataFrame(results).T.round(4)
