import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import seaborn as sns

from src.data_loader import DataLoader
from src.config import COLUMN_NAMES, DATASETS, USEFUL_SENSORS, DROP_SENSORS

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "02_eda_distribuciones"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
TOP_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]


class EDADistribuciones:
    def __init__(self):
        self.loader    = DataLoader()
        self.fig_count = 0
        self.loader.load_all()
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        fig.savefig(FIGURES_DIR / f"02_eda_distribuciones_fig{self.fig_count:02d}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_all_sensors_hist(self, ds_id: str = "FD001") -> "EDADistribuciones":
        train       = self.loader.data[ds_id][0]
        sensor_cols = [c for c in COLUMN_NAMES if c.startswith("sensor_")]
        fig, axes   = plt.subplots(7, 3, figsize=(18, 28))
        axes        = axes.flatten()
        for i, sensor in enumerate(sensor_cols):
            axes[i].hist(train[sensor], bins=60, alpha=0.75, edgecolor="white", color="#4C72B0")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].tick_params(labelsize=9)
        plt.suptitle(f"Distribucion de los 21 sensores — {ds_id}", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_useful_by_dataset(self) -> "EDADistribuciones":
        fig, axes = plt.subplots(len(USEFUL_SENSORS), 1, figsize=(16, 4 * len(USEFUL_SENSORS)))
        for i, sensor in enumerate(USEFUL_SENSORS):
            for ds_id in DATASETS:
                axes[i].hist(self.loader.data[ds_id][0][sensor],
                             bins=60, alpha=0.45, label=ds_id, edgecolor="white")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].legend(fontsize=9)
        plt.suptitle("Distribucion por sub-dataset — Sensores utiles", fontsize=16, y=1.001)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_boxplots(self, ds_id: str = "FD001") -> "EDADistribuciones":
        train     = self.loader.data[ds_id][0]
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes      = axes.flatten()
        for i, sensor in enumerate(USEFUL_SENSORS[:16]):
            sns.boxplot(data=train, y=sensor, ax=axes[i], color="#4C72B0", fliersize=2)
            axes[i].set_title(sensor, fontsize=11, fontweight="bold")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f"Box plots — Sensores utiles — {ds_id}", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_violin_condition(self) -> "EDADistribuciones":
        fd001    = self.loader.data["FD001"][0][TOP_SENSORS].assign(dataset="FD001")
        fd002    = self.loader.data["FD002"][0][TOP_SENSORS].assign(dataset="FD002")
        combined = pd.concat([fd001, fd002], ignore_index=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(TOP_SENSORS):
            sns.violinplot(data=combined, x="dataset", y=sensor, hue="dataset",
                           ax=axes[i], palette="Set2", inner="quartile", legend=False)
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
        plt.suptitle("Violin plots — FD001 vs FD002", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_violin_fault(self) -> "EDADistribuciones":
        fd001    = self.loader.data["FD001"][0][TOP_SENSORS].assign(dataset="FD001 (1 falla)")
        fd003    = self.loader.data["FD003"][0][TOP_SENSORS].assign(dataset="FD003 (2 fallas)")
        combined = pd.concat([fd001, fd003], ignore_index=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(TOP_SENSORS):
            sns.violinplot(data=combined, x="dataset", y=sensor, hue="dataset",
                           ax=axes[i], palette="Set1", inner="quartile", legend=False)
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
        plt.suptitle("Impacto del modo de falla en distribuciones", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_constant_sensors(self) -> "EDADistribuciones":
        all_drop  = DROP_SENSORS + ["op_setting_3"]
        train     = self.loader.data["FD001"][0]
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes      = axes.flatten()
        for i, sensor in enumerate(all_drop):
            if i < len(axes):
                axes[i].hist(train[sensor], bins=30, color="#CC4444", alpha=0.7, edgecolor="white")
                axes[i].set_title(f"{sensor} (DROP)", fontsize=11, fontweight="bold", color="red")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Sensores constantes — Se eliminan", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_ridge(self, sensor: str = "sensor_4") -> "EDADistribuciones":
        from matplotlib.gridspec import GridSpec as GS
        fig = plt.figure(figsize=(14, 8))
        gs  = GS(4, 1, hspace=-0.5)
        for i, ds_id in enumerate(DATASETS):
            ax    = fig.add_subplot(gs[i, 0])
            train = self.loader.data[ds_id][0]
            counts, edges = np.histogram(train[sensor], bins=80, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.fill_between(centers, counts, alpha=0.6, color=COLORS[i], step="mid")
            sns.kdeplot(data=train, x=sensor, ax=ax, color=COLORS[i],
                        linewidth=2, fill=True, alpha=0.3)
            ax.set_ylabel(ds_id, fontsize=12, fontweight="bold", rotation=0, labelpad=50)
            ax.set_xlim(train[sensor].quantile(0.01), train[sensor].quantile(0.99))
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i < 3:
                ax.set_xticks([])
        plt.suptitle(f"Ridge plot — {sensor} por sub-dataset", fontsize=16, y=0.95)
        self._save(fig)
        return self

    def plot_life_span(self) -> "EDADistribuciones":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            max_cycles = self.loader.data[ds_id][0].groupby("unit_id")["cycle"].max()
            axes[i].hist(max_cycles, bins=30, color=COLORS[i], alpha=0.75, edgecolor="white")
            axes[i].axvline(max_cycles.mean(), color="black", linestyle="--",
                            linewidth=1.5, label=f"Mean: {max_cycles.mean():.0f}")
            axes[i].set(title=f"{ds_id} — Vida util por motor", xlabel="Ciclos maximos")
            axes[i].legend()
        plt.suptitle("Distribucion de vida util de motores", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_std_by_dataset(self) -> "EDADistribuciones":
        rows = [{"dataset": ds_id, "sensor": s, "std": self.loader.data[ds_id][0][s].std()}
                for ds_id in DATASETS for s in USEFUL_SENSORS]
        stats_df = pd.DataFrame(rows)
        fig, ax  = plt.subplots(figsize=(16, 6))
        stats_df.pivot(index="sensor", columns="dataset", values="std").plot(
            kind="bar", ax=ax, width=0.75, edgecolor="white", color=COLORS)
        ax.set(ylabel="Desviacion estandar",
               title="Desviacion estandar por sensor y sub-dataset")
        ax.legend(title="Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save(fig)
        return self

    def run(self) -> "EDADistribuciones":
        return (self
                .plot_all_sensors_hist()
                .plot_useful_by_dataset()
                .plot_boxplots()
                .plot_violin_condition()
                .plot_violin_fault()
                .plot_constant_sensors()
                .plot_ridge()
                .plot_life_span()
                .plot_std_by_dataset())


if __name__ == "__main__":
    EDADistribuciones().run()
