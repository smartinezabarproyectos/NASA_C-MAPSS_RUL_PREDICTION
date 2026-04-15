"""Análisis de interpretabilidad con SHAP values."""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR
from src.visualization import save_fig


def compute_shap_values(model, X: np.ndarray | pd.DataFrame, model_type: str = "tree"):
    """Calcula SHAP values según el tipo de modelo.

    Parameters
    ----------
    model : modelo entrenado
    X : datos de entrada
    model_type : 'tree' para RF/XGBoost, 'kernel' para otros

    Returns
    -------
    shap.Explanation
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))

    return explainer(X)


def plot_shap_summary(
    shap_values,
    X: pd.DataFrame,
    model_name: str,
    dataset_id: str,
) -> None:
    """SHAP summary plot — importancia global de features."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary — {model_name} — {dataset_id}")
    plt.tight_layout()
    save_fig(plt.gcf(), f"shap_summary_{model_name}_{dataset_id}")


def plot_shap_bar(
    shap_values,
    X: pd.DataFrame,
    model_name: str,
    dataset_id: str,
) -> None:
    """SHAP bar plot — importancia media absoluta."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False)
    plt.title(f"SHAP Feature Importance — {model_name} — {dataset_id}")
    plt.tight_layout()
    save_fig(plt.gcf(), f"shap_bar_{model_name}_{dataset_id}")
