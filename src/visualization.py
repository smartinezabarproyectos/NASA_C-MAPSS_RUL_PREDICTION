import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config import DATASETS, CLASSIFICATION_W, MAX_RUL, FIGURES_DIR

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


class Visualizer:
    def __init__(self, prefix: str = "viz", figures_dir: Path = None):
        self.figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.prefix    = prefix
        self.fig_count = 0
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure, name: str = None) -> None:
        self.fig_count += 1
        fname = name or f"{self.prefix}_fig{self.fig_count:02d}.png"
        fig.savefig(self.figures_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_degradation_curves(self, df: pd.DataFrame, sensor: str,
                                n_motors: int = 10, ds_id: str = "FD001") -> "Visualizer":
        units = df["unit_id"].unique()[:n_motors]
        fig, ax = plt.subplots(figsize=(14, 6))
        for unit in units:
            ud = df[df["unit_id"] == unit].copy()
            ud["life_pct"] = ud["cycle"] / ud["cycle"].max() * 100
            ax.plot(ud["life_pct"], ud[sensor], alpha=0.6, linewidth=0.9)
        ax.set(xlabel="% vida util", ylabel=sensor,
               title=f"Curvas de degradacion — {sensor} — {ds_id}")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_mean_degradation(self, df: pd.DataFrame, sensors: list,
                              ds_id: str = "FD001") -> "Visualizer":
        n_cols = 3
        n_rows = (len(sensors) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        for i, sensor in enumerate(sensors):
            grouped = df.groupby("life_pct")[sensor]
            mean, std = grouped.mean(), grouped.std()
            axes[i].plot(mean.index, mean.values, color="#4C72B0", linewidth=2)
            axes[i].fill_between(mean.index, mean - std, mean + std,
                                 alpha=0.2, color="#4C72B0")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f"Degradacion promedio — {ds_id}", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_correlation_heatmap(self, df: pd.DataFrame, cols: list,
                                 title: str = "Correlacion") -> "Visualizer":
        corr = df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax,
                    vmin=-1, vmax=1, annot_kws={"size": 8})
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_predicted_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "Modelo") -> "Visualizer":
        error  = y_pred - y_true
        colors = np.where(error >= 0, "#C44E52", "#4C72B0")
        lim    = max(y_true.max(), y_pred.max()) * 1.05
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, c=colors, alpha=0.6, s=30)
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        ax.set(xlabel="RUL Real", ylabel="RUL Predicho",
               title=f"Predicho vs Real — {model_name}")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Modelo") -> "Visualizer":
        error = y_pred - y_true
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(error, bins=40, color="#4C72B0", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(error.mean(), color="orange", linestyle=":", linewidth=1.5,
                   label=f"Media = {error.mean():.1f}")
        ax.set(xlabel="Error (predicho - real)", ylabel="Frecuencia",
               title=f"Distribucion del error — {model_name}")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "Modelo") -> "Visualizer":
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Critico"],
                    yticklabels=["Normal", "Critico"],
                    ax=ax, linewidths=1)
        ax.set(xlabel="Predicho", ylabel="Real",
               title=f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                       model_name: str = "Modelo") -> "Visualizer":
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score   = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#4C72B0", linewidth=2,
                label=f"AUC = {auc_score:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=f"ROC — {model_name}")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_training_curves(self, train_losses: list, val_losses: list,
                             model_name: str = "Modelo") -> "Visualizer":
        epochs     = range(1, len(train_losses) + 1)
        best_epoch = int(np.argmin(val_losses)) + 1
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, color="#4C72B0", linewidth=2, label="Train")
        ax.plot(epochs, val_losses, color="#C44E52", linewidth=2,
                linestyle="--", label="Val")
        ax.axvline(best_epoch, color="green", linestyle=":", alpha=0.7,
                   label=f"Mejor epoch: {best_epoch}")
        ax.set(xlabel="Epoch", ylabel="MSE Loss",
               title=f"Curvas de entrenamiento — {model_name}")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_cross_dataset_rmse(self, results: dict) -> "Visualizer":
        models = list(results.keys())
        x, width = np.arange(len(models)), 0.2
        fig, ax = plt.subplots(figsize=(14, 7))
        for i, ds_id in enumerate(DATASETS):
            ax.bar(x + i * width, [results[m].get(ds_id, 0) for m in models],
                   width, label=ds_id, color=COLORS[i], edgecolor="white")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set(ylabel="RMSE", title="RMSE por modelo y sub-dataset")
        ax.legend(title="Dataset")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_ensemble_comparison(self, individual_preds: dict,
                                 ensemble_preds: np.ndarray,
                                 y_true: np.ndarray) -> "Visualizer":
        names  = list(individual_preds.keys()) + ["Ensemble"]
        rmses  = [float(np.sqrt(np.mean((p - y_true) ** 2)))
                  for p in individual_preds.values()]
        rmses.append(float(np.sqrt(np.mean((ensemble_preds - y_true) ** 2))))
        colors = ["#4C72B0"] * len(individual_preds) + ["#FFA15A"]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, rmses, color=colors, edgecolor="white")
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1, f"{val:.2f}",
                    ha="center", fontsize=11, fontweight="bold")
        ax.set(ylabel="RMSE", title="Modelos individuales vs Ensemble")
        plt.tight_layout()
        self._save(fig)
        return self
