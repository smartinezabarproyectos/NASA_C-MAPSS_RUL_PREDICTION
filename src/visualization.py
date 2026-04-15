"""Visualizaciones para EDA, resultados y comparaciones."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR

# Estilo global
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def save_fig(fig: plt.Figure, name: str, dpi: int = 150) -> None:
    """Guarda figura en alta resolución en la carpeta de figuras."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"✓ Figura guardada: {path}")
    plt.close(fig)


def plot_sensor_distributions(df: pd.DataFrame, sensors: list[str], dataset_id: str) -> None:
    """Histogramas de distribución de sensores."""
    n_cols = 4
    n_rows = (len(sensors) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for i, sensor in enumerate(sensors):
        axes[i].hist(df[sensor], bins=50, alpha=0.7, edgecolor="white")
        axes[i].set_title(sensor, fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distribución de sensores — {dataset_id}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"sensor_distributions_{dataset_id}")


def plot_degradation_curves(df: pd.DataFrame, sensor: str, dataset_id: str, n_units: int = 10) -> None:
    """Curvas de degradación de un sensor para N motores."""
    fig, ax = plt.subplots(figsize=(12, 5))
    units = df["unit_id"].unique()[:n_units]

    for unit in units:
        unit_data = df[df["unit_id"] == unit]
        ax.plot(unit_data["cycle"], unit_data[sensor], alpha=0.7, label=f"Motor {unit}")

    ax.set_xlabel("Ciclo")
    ax.set_ylabel(sensor)
    ax.set_title(f"Degradación — {sensor} — {dataset_id}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_fig(fig, f"degradation_{sensor}_{dataset_id}")


def plot_rul_distribution(df: pd.DataFrame, dataset_id: str) -> None:
    """Distribución del RUL en el dataset de entrenamiento."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["rul"], bins=60, alpha=0.7, edgecolor="white", color="steelblue")
    ax.set_xlabel("RUL (ciclos)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de RUL — {dataset_id}")
    fig.tight_layout()
    save_fig(fig, f"rul_distribution_{dataset_id}")


def plot_correlation_matrix(df: pd.DataFrame, sensors: list[str], dataset_id: str) -> None:
    """Heatmap de correlación entre sensores."""
    corr = df[sensors].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title(f"Correlación de sensores — {dataset_id}")
    fig.tight_layout()
    save_fig(fig, f"correlation_matrix_{dataset_id}")


def plot_predicted_vs_actual(y_true, y_pred, model_name: str, dataset_id: str) -> None:
    """Scatter plot de predicciones vs valores reales."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
    ax.plot([0, max(y_true)], [0, max(y_true)], "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("RUL Real")
    ax.set_ylabel("RUL Predicho")
    ax.set_title(f"{model_name} — Predicho vs Real — {dataset_id}")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"pred_vs_actual_{model_name}_{dataset_id}")


def plot_model_comparison(results_df: pd.DataFrame, metric: str, title: str) -> None:
    """Bar chart comparando modelos en una métrica específica."""
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df[metric].sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, f"comparison_{metric}")
