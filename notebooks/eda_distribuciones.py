"""
EDA Distribuciones — NASA C-MAPSS
Histogramas, boxplots y violin plots de sensores por sub-dataset.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_all_datasets
from src.config import COLUMN_NAMES, DATASETS, USEFUL_SENSORS, DROP_SENSORS

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "02_eda_distribuciones"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TOP_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


class EDADistribuciones:
    """
    Analisis de distribuciones del dataset NASA C-MAPSS.

    Attributes
    ----------
    data : dict
        Diccionario con los 4 sub-datasets cargados.
    fig_count : int
        Contador de figuras generadas.
    """

    def __init__(self):
        print("Cargando datasets NASA C-MAPSS...")
        self.data = load_all_datasets()
        self.fig_count = 0
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure) -> None:
        """Guarda figura con nombre secuencial y cierra."""
        self.fig_count += 1
        path = FIGURES_DIR / f"02_eda_distribuciones_fig{self.fig_count:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardada: {path.name}")

    def plot_all_sensors_hist(self, ds_id: str = "FD001") -> "EDADistribuciones":
        """
        Histograma de los 21 sensores.
        Identifica cuales son constantes (pico unico) y cuales informativos.
        """
        train = self.data[ds_id][0]
        sensor_cols = [c for c in COLUMN_NAMES if c.startswith("sensor_")]
        fig, axes = plt.subplots(7, 3, figsize=(18, 28))
        axes = axes.flatten()
        for i, sensor in enumerate(sensor_cols):
            axes[i].hist(train[sensor], bins=60, alpha=0.75,
                         edgecolor="white", color="#4C72B0")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].tick_params(labelsize=9)
        plt.suptitle(f"Distribucion de los 21 sensores — {ds_id}",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_useful_by_dataset(self) -> "EDADistribuciones":
        """
        Histogramas superpuestos de sensores utiles para los 4 sub-datasets.
        Permite ver si la distribucion cambia entre datasets.
        """
        fig, axes = plt.subplots(len(USEFUL_SENSORS), 1,
                                 figsize=(16, 4 * len(USEFUL_SENSORS)))
        for i, sensor in enumerate(USEFUL_SENSORS):
            for ds_id in DATASETS:
                train = self.data[ds_id][0]
                axes[i].hist(train[sensor], bins=60, alpha=0.45,
                             label=ds_id, edgecolor="white")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].legend(fontsize=9)
        plt.suptitle("Distribucion por sub-dataset — Sensores utiles",
                     fontsize=16, y=1.001)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_boxplots(self, ds_id: str = "FD001") -> "EDADistribuciones":
        """
        Boxplots de sensores utiles.
        Muestra mediana, IQR y outliers de cada sensor.
        """
        train = self.data[ds_id][0]
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        for i, sensor in enumerate(USEFUL_SENSORS[:16]):
            sns.boxplot(data=train, y=sensor, ax=axes[i],
                        color="#4C72B0", fliersize=2)
            axes[i].set_title(sensor, fontsize=11, fontweight="bold")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f"Box plots — Sensores utiles — {ds_id}",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_violin_condition(self) -> "EDADistribuciones":
        """
        Violin plots FD001 vs FD002 (1 vs 6 condiciones operacionales).
        Evidencia por que se necesita normalizacion por condicion.
        """
        fd001 = self.data["FD001"][0][TOP_SENSORS].assign(dataset="FD001")
        fd002 = self.data["FD002"][0][TOP_SENSORS].assign(dataset="FD002")
        combined = pd.concat([fd001, fd002], ignore_index=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(TOP_SENSORS):
            sns.violinplot(data=combined, x="dataset", y=sensor, hue="dataset",
                           ax=axes[i], palette="Set2",
                           inner="quartile", legend=False)
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
        plt.suptitle("Violin plots — FD001 vs FD002", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_violin_fault(self) -> "EDADistribuciones":
        """
        Violin plots FD001 vs FD003 (1 vs 2 modos de falla).
        Muestra el impacto del segundo modo de falla en las distribuciones.
        """
        fd001 = self.data["FD001"][0][TOP_SENSORS].assign(dataset="FD001 (1 falla)")
        fd003 = self.data["FD003"][0][TOP_SENSORS].assign(dataset="FD003 (2 fallas)")
        combined = pd.concat([fd001, fd003], ignore_index=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(TOP_SENSORS):
            sns.violinplot(data=combined, x="dataset", y=sensor, hue="dataset",
                           ax=axes[i], palette="Set1",
                           inner="quartile", legend=False)
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
        plt.suptitle("Impacto del modo de falla en distribuciones",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_constant_sensors(self) -> "EDADistribuciones":
        """
        Histogramas de los sensores constantes que se eliminan.
        Todos muestran barra unica — varianza cero.
        """
        all_drop = DROP_SENSORS + ["op_setting_3"]
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes = axes.flatten()
        for i, sensor in enumerate(all_drop):
            if i < len(axes):
                train = self.data["FD001"][0]
                axes[i].hist(train[sensor], bins=30,
                             color="#CC4444", alpha=0.7, edgecolor="white")
                axes[i].set_title(f"{sensor} (DROP)", fontsize=11,
                                  fontweight="bold", color="red")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Sensores constantes / baja varianza — Se eliminan",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_life_span(self) -> "EDADistribuciones":
        """
        Histogramas de vida util (ciclos totales) por motor y dataset.
        Muestra la variabilidad de cuanto dura cada motor.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            train = self.data[ds_id][0]
            max_cycles = train.groupby("unit_id")["cycle"].max()
            axes[i].hist(max_cycles, bins=30, color=COLORS[i],
                         alpha=0.75, edgecolor="white")
            axes[i].axvline(max_cycles.mean(), color="black", linestyle="--",
                            linewidth=1.5, label=f"Mean: {max_cycles.mean():.0f}")
            axes[i].set_title(f"{ds_id} — Vida util por motor",
                              fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Ciclos maximos")
            axes[i].legend()
        plt.suptitle("Distribucion de vida util de motores", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_std_by_dataset(self) -> "EDADistribuciones":
        """
        Desviacion estandar de cada sensor agrupada por sub-dataset.
        Los sensores con mayor std son los mas informativos.
        """
        stats_rows = []
        for ds_id in DATASETS:
            train = self.data[ds_id][0]
            for sensor in USEFUL_SENSORS:
                stats_rows.append({
                    "dataset": ds_id,
                    "sensor":  sensor,
                    "std":     train[sensor].std(),
                })
        stats_df = pd.DataFrame(stats_rows)
        fig, ax = plt.subplots(figsize=(16, 6))
        pivot = stats_df.pivot(index="sensor", columns="dataset", values="std")
        pivot.plot(kind="bar", ax=ax, width=0.75,
                   edgecolor="white", color=COLORS)
        ax.set_ylabel("Desviacion estandar")
        ax.set_title("Desviacion estandar por sensor y sub-dataset",
                     fontsize=14, fontweight="bold")
        ax.legend(title="Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save(fig)
        return self

    def print_report(self) -> "EDADistribuciones":
        """Imprime el resumen del analisis de distribuciones."""
        print("=" * 60)
        print("  RESUMEN EDA DISTRIBUCIONES")
        print("=" * 60)
        print(f"  Figuras generadas: {self.fig_count}")
        print(f"  Guardadas en: {FIGURES_DIR}")
        print(f"  FD002 tiene rangos mas amplios (6 condiciones)")
        print(f"  FD003 muestra diferencias en top sensores (2 fallas)")
        print("=" * 60)
        return self

    def run(self) -> "EDADistribuciones":
        """Ejecuta todos los analisis en cadena."""
        return (
            self
            .plot_all_sensors_hist()
            .plot_useful_by_dataset()
            .plot_boxplots()
            .plot_violin_condition()
            .plot_violin_fault()
            .plot_constant_sensors()
            .plot_life_span()
            .plot_std_by_dataset()
            .print_report()
        )


if __name__ == "__main__":
    eda = EDADistribuciones()
    eda.run()
