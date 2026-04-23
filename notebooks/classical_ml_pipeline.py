import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.models.classical import ClassicalMLTrainer
from src.models.evaluate import Evaluator
from src.config import DATASETS, PROCESSED_DIR

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "08_classical_ml"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


class ClassicalMLPipeline:
    def __init__(self):
        self.trainer         = ClassicalMLTrainer()
        self.evaluator       = Evaluator()
        self.processed       = {}
        self.feature_cols    = {}
        self.all_reg_results = {}
        self.clf_results     = {}
        sns.set_theme(style="whitegrid")

    def load_data(self) -> "ClassicalMLPipeline":
        with open(PROCESSED_DIR / "metadata.json") as f:
            self.feature_cols = json.load(f)["feature_cols"]
        for ds_id in DATASETS:
            self.processed[ds_id] = {
                "train": pd.read_parquet(PROCESSED_DIR / f"train_{ds_id}.parquet"),
                "test":  pd.read_parquet(PROCESSED_DIR / f"test_{ds_id}.parquet"),
            }
        return self

    def _get_xy(self, ds_id: str) -> tuple:
        train = self.processed[ds_id]["train"]
        test  = self.processed[ds_id]["test"]
        avail = [c for c in self.feature_cols[ds_id] if c in test.columns]
        return (train[avail].values, train["rul"].values, train["label"].values,
                test[avail].values,  test["rul"].values,  test["label"].values)

    def train_regression(self, ds_id: str = "FD001") -> "ClassicalMLPipeline":
        X_tr, y_tr, _, X_te, y_te, _ = self._get_xy(ds_id)
        self.trainer.train_regression(X_tr, y_tr, X_te, y_te, ds_id)
        return self

    def train_classification(self, ds_id: str = "FD001") -> "ClassicalMLPipeline":
        X_tr, _, y_tr, X_te, _, y_te = self._get_xy(ds_id)
        self.trainer.train_classification(X_tr, y_tr, X_te, y_te, ds_id)
        return self

    def train_cross_dataset(self) -> "ClassicalMLPipeline":
        for ds_id in DATASETS:
            X_tr, y_tr, _, X_te, y_te, _ = self._get_xy(ds_id)
            trainer    = ClassicalMLTrainer()
            ds_results = {}
            for name, model in trainer.reg_models.items():
                model.fit(X_tr, y_tr)
                ds_results[name] = self.evaluator.regression_metrics(
                    y_te, model.predict(X_te))
            self.all_reg_results[ds_id] = ds_results
        return self

    def plot_predicted_vs_actual(self, ds_id: str = "FD001") -> "ClassicalMLPipeline":
        X_tr, y_tr, _, X_te, y_te, _ = self._get_xy(ds_id)
        trainer = ClassicalMLTrainer()
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        for i, (name, model) in enumerate(trainer.reg_models.items()):
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            rmse   = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))
            axes[i].scatter(y_te, y_pred, alpha=0.6, s=30, color=COLORS[i])
            axes[i].plot([0, max(y_te)], [0, max(y_te)], "r--", linewidth=1.5)
            axes[i].set_title(f"{name} — RMSE: {rmse:.2f}", fontweight="bold")
        plt.suptitle(f"Predicted vs Actual — {ds_id}", fontsize=14, y=1.01)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "08_classical_ml_fig01.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return self

    def plot_error_distribution(self, ds_id: str = "FD001") -> "ClassicalMLPipeline":
        X_tr, y_tr, _, X_te, y_te, _ = self._get_xy(ds_id)
        trainer = ClassicalMLTrainer()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, (name, model) in enumerate(trainer.reg_models.items()):
            model.fit(X_tr, y_tr)
            errors = model.predict(X_te) - y_te
            axes[i].hist(errors, bins=30, color=COLORS[i], alpha=0.7, edgecolor="white")
            axes[i].axvline(0, color="black", linestyle="--")
            axes[i].set_title(f"{name} — Mean: {errors.mean():.2f}", fontweight="bold")
        plt.suptitle(f"Distribucion de errores — {ds_id}", fontsize=14, y=1.01)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "08_classical_ml_fig02.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return self

    def plot_confusion_matrices(self, ds_id: str = "FD001") -> "ClassicalMLPipeline":
        X_tr, _, y_tr, X_te, _, y_te = self._get_xy(ds_id)
        trainer = ClassicalMLTrainer()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, (name, model) in enumerate(trainer.clf_models.items()):
            model.fit(X_tr, y_tr)
            cm = confusion_matrix(y_te, model.predict(X_te))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Sano", "Falla"],
                        yticklabels=["Sano", "Falla"], ax=axes[i])
            axes[i].set_title(name, fontweight="bold")
        plt.suptitle(f"Confusion Matrix — {ds_id}", fontsize=14, y=1.05)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "08_classical_ml_fig03.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return self

    def plot_roc_curves(self, ds_id: str = "FD001") -> "ClassicalMLPipeline":
        X_tr, _, y_tr, X_te, _, y_te = self._get_xy(ds_id)
        trainer = ClassicalMLTrainer()
        fig, ax = plt.subplots(figsize=(8, 7))
        for name, model in trainer.clf_models.items():
            model.fit(X_tr, y_tr)
            fpr, tpr, _ = roc_curve(y_te, model.predict_proba(X_te)[:, 1])
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC — {ds_id}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "08_classical_ml_fig04.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return self

    def plot_cross_dataset_heatmap(self) -> "ClassicalMLPipeline":
        rows = [{"dataset": ds_id, "model": name, **metrics}
                for ds_id, ds_res in self.all_reg_results.items()
                for name, metrics in ds_res.items()]
        cross_df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, metric, cmap in zip(axes, ["rmse", "r2"], ["RdYlGn_r", "RdYlGn"]):
            pivot = cross_df.pivot(index="model", columns="dataset", values=metric)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax, linewidths=0.5)
            ax.set_title(metric.upper(), fontweight="bold")
        plt.suptitle("Rendimiento cross-dataset", fontsize=15, y=1.03)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "08_classical_ml_fig05.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return self

    def run(self) -> "ClassicalMLPipeline":
        return (self
                .load_data()
                .train_regression()
                .train_classification()
                .train_cross_dataset()
                .plot_predicted_vs_actual()
                .plot_error_distribution()
                .plot_confusion_matrices()
                .plot_roc_curves()
                .plot_cross_dataset_heatmap())


if __name__ == "__main__":
    ClassicalMLPipeline().run()
