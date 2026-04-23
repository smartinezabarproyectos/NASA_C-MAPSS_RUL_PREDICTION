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
import seaborn as sns

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.config import DATASETS, USEFUL_SENSORS

FIGURES_DIR  = Path(ROOT) / "paper" / "figures" / "notebooks" / "03_eda_degradacion"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
COLORS       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
SENSORS_PLOT = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_15"]


class EDADegradacion:
    def __init__(self):
        self.loader    = DataLoader()
        self.prep      = Preprocessor()
        self.fig_count = 0
        self.loader.load_all()
        self.train = self.prep.compute_rul(self.loader.data["FD001"][0])
        self.train["life_pct"] = (
            self.train.groupby("unit_id")["cycle"]
            .transform(lambda x: (x / x.max() * 100).round(0))
        )
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        fig.savefig(FIGURES_DIR / f"03_eda_degradacion_fig{self.fig_count:02d}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_raw_curves(self, n_units: int = 10) -> "EDADegradacion":
        units     = self.train["unit_id"].unique()[:n_units]
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes      = axes.flatten()
        for i, sensor in enumerate(SENSORS_PLOT):
            for unit in units:
                ud = self.train[self.train["unit_id"] == unit]
                axes[i].plot(ud["cycle"], ud[sensor], alpha=0.6, linewidth=0.8)
            axes[i].set(title=sensor, xlabel="Ciclo")
        plt.suptitle(f"Curvas crudas — {n_units} motores — FD001", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_normalized_curves(self, n_units: int = 10) -> "EDADegradacion":
        units     = self.train["unit_id"].unique()[:n_units]
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes      = axes.flatten()
        for i, sensor in enumerate(SENSORS_PLOT):
            for unit in units:
                ud = self.train[self.train["unit_id"] == unit].copy()
                ud["life_pct_u"] = ud["cycle"] / ud["cycle"].max() * 100
                axes[i].plot(ud["life_pct_u"], ud[sensor], alpha=0.6, linewidth=0.8)
            axes[i].set(title=sensor, xlabel="% vida util")
        plt.suptitle("Degradacion normalizada — FD001", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_mean_bands(self) -> "EDADegradacion":
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes      = axes.flatten()
        for i, sensor in enumerate(SENSORS_PLOT):
            grouped = self.train.groupby("life_pct")[sensor]
            mean, std = grouped.mean(), grouped.std()
            axes[i].plot(mean.index, mean.values, color="#4C72B0", linewidth=2)
            axes[i].fill_between(mean.index, mean - std, mean + std, alpha=0.2, color="#4C72B0")
            axes[i].set(title=sensor, xlabel="% vida util")
        plt.suptitle("Degradacion promedio +/- 1 STD — FD001", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_sensor11_all_datasets(self) -> "EDADegradacion":
        sensor    = "sensor_11"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes      = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            tr = self.prep.compute_rul(self.loader.data[ds_id][0])
            tr["life_pct"] = tr.groupby("unit_id")["cycle"].transform(
                lambda x: (x / x.max() * 100).round(0))
            grouped = tr.groupby("life_pct")[sensor]
            mean, std = grouped.mean(), grouped.std()
            axes[i].plot(mean.index, mean.values, color=COLORS[i], linewidth=2)
            axes[i].fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=COLORS[i])
            axes[i].set(title=ds_id, xlabel="% vida util", ylabel=sensor)
        plt.suptitle(f"Degradacion de {sensor} — 4 sub-datasets", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_motor_heatmap(self, unit_id: int = 1) -> "EDADegradacion":
        unit_data = self.train[self.train["unit_id"] == unit_id][USEFUL_SENSORS]
        fig, ax   = plt.subplots(figsize=(18, 8))
        sns.heatmap(unit_data.T, cmap="YlOrRd", xticklabels=10,
                    yticklabels=True, ax=ax, cbar_kws={"label": "Valor del sensor"})
        ax.set(xlabel="Ciclo", ylabel="Sensor",
               title=f"Heatmap de degradacion — Motor {unit_id} — FD001")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_last_cycles_heatmap(self, n_cycles: int = 50) -> "EDADegradacion":
        frames = []
        for unit in self.train["unit_id"].unique():
            ud = self.train[self.train["unit_id"] == unit].tail(n_cycles).copy()
            ud["reverse_cycle"] = range(len(ud) - 1, -1, -1)
            frames.append(ud)
        pivot  = pd.concat(frames, ignore_index=True).groupby("reverse_cycle")[USEFUL_SENSORS].mean()
        fig, ax = plt.subplots(figsize=(18, 8))
        sns.heatmap(pivot.T, cmap="YlOrRd", xticklabels=5, ax=ax,
                    cbar_kws={"label": "Valor promedio"})
        ax.set(xlabel="Ciclos antes de falla", ylabel="Sensor",
               title=f"Ultimos {n_cycles} ciclos antes de falla — FD001")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_change_rate(self) -> "EDADegradacion":
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes      = axes.flatten()
        for i, sensor in enumerate(SENSORS_PLOT):
            rates = [
                ud[sensor].diff().dropna().values[-1]
                if len(ud[sensor].diff().dropna()) > 0 else 0
                for unit in self.train["unit_id"].unique()
                for ud in [self.train[self.train["unit_id"] == unit]]
            ]
            axes[i].hist(rates, bins=50, color="#C44E52", alpha=0.7, edgecolor="white")
            axes[i].axvline(0, color="black", linestyle="--", linewidth=1)
            axes[i].set_title(f"D-{sensor}", fontsize=12, fontweight="bold")
        plt.suptitle("Tasa de cambio — ultimo ciclo — FD001", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rolling_vs_raw(self, unit_id: int = 1, window: int = 20) -> "EDADegradacion":
        ud        = self.train[self.train["unit_id"] == unit_id].copy()
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes      = axes.flatten()
        for i, sensor in enumerate(SENSORS_PLOT):
            axes[i].plot(ud["cycle"], ud[sensor], alpha=0.3, color="gray", label="Crudo")
            rolling = ud[sensor].rolling(window=window, min_periods=1).mean()
            axes[i].plot(ud["cycle"], rolling, color="#4C72B0", linewidth=2,
                         label=f"Rolling ({window})")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].legend(fontsize=9)
        plt.suptitle(f"Crudo vs Rolling — Motor {unit_id} — FD001", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_degradation_by_phase(self, sensor: str = "sensor_11") -> "EDADegradacion":
        phases    = {"Inicio (0-30%)": (0, 30), "Medio (30-70%)": (30, 70), "Final (70-100%)": (70, 100)}
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (name, (lo, hi)) in zip(axes, phases.items()):
            data = self.train[(self.train["life_pct"] >= lo) & (self.train["life_pct"] < hi)]
            ax.hist(data[sensor], bins=50, alpha=0.7, edgecolor="white", color="#4C72B0")
            ax.set(title=name, xlabel=sensor)
        plt.suptitle(f"Distribucion de {sensor} por fase — FD001", fontsize=16, y=1.05)
        plt.tight_layout()
        self._save(fig)
        return self

    def run(self) -> "EDADegradacion":
        return (self
                .plot_raw_curves()
                .plot_normalized_curves()
                .plot_mean_bands()
                .plot_sensor11_all_datasets()
                .plot_motor_heatmap()
                .plot_last_cycles_heatmap()
                .plot_change_rate()
                .plot_rolling_vs_raw()
                .plot_degradation_by_phase())


if __name__ == "__main__":
    EDADegradacion().run()
