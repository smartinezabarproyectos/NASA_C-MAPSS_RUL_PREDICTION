"""
EDA Degradacion — NASA C-MAPSS
Curvas de sensores a lo largo de la vida del motor, heatmaps de
degradacion y comparacion multi-dataset.
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
from src.preprocessing import add_rul_column
from src.config import DATASETS, USEFUL_SENSORS

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "03_eda_degradacion"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SENSORS_TO_PLOT = ["sensor_2", "sensor_3", "sensor_4",
                   "sensor_7", "sensor_11", "sensor_15"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


class EDADegradacion:
    """
    Analisis de patrones de degradacion del dataset NASA C-MAPSS.

    Attributes
    ----------
    data : dict
        Diccionario con los 4 sub-datasets.
    train : pd.DataFrame
        Train FD001 con columna RUL y life_pct calculadas.
    fig_count : int
        Contador de figuras generadas.
    """

    def __init__(self):
        print("Cargando datasets NASA C-MAPSS...")
        self.data  = load_all_datasets()
        self.train = add_rul_column(self.data["FD001"][0])
        self.train["life_pct"] = (
            self.train.groupby("unit_id")["cycle"]
            .transform(lambda x: (x / x.max() * 100).round(0))
        )
        self.fig_count = 0
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        path = FIGURES_DIR / f"03_eda_degradacion_fig{self.fig_count:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardada: {path.name}")

    def _add_life_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega life_pct a un DataFrame sin modificar el original."""
        df = df.copy()
        df["life_pct"] = (
            df.groupby("unit_id")["cycle"]
            .transform(lambda x: (x / x.max() * 100).round(0))
        )
        return df

    def plot_raw_curves(self, n_units: int = 10) -> "EDADegradacion":
        """
        Curvas crudas de sensores para N motores.
        Muestra variabilidad individual entre motores.
        """
        units = self.train["unit_id"].unique()[:n_units]
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(SENSORS_TO_PLOT):
            for unit in units:
                ud = self.train[self.train["unit_id"] == unit]
                axes[i].plot(ud["cycle"], ud[sensor], alpha=0.6, linewidth=0.8)
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Ciclo")
        plt.suptitle(f"Curvas crudas — {n_units} motores — FD001",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_normalized_curves(self, n_units: int = 10) -> "EDADegradacion":
        """
        Curvas normalizadas por porcentaje de vida.
        Permite comparar motores de diferente duracion.
        """
        units = self.train["unit_id"].unique()[:n_units]
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(SENSORS_TO_PLOT):
            for unit in units:
                ud = self.train[self.train["unit_id"] == unit].copy()
                ud["life_pct_u"] = ud["cycle"] / ud["cycle"].max() * 100
                axes[i].plot(ud["life_pct_u"], ud[sensor],
                             alpha=0.6, linewidth=0.8)
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].set_xlabel("% vida util")
        plt.suptitle("Degradacion normalizada — FD001", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_mean_bands(self) -> "EDADegradacion":
        """
        Curva media de degradacion con banda +/- 1 STD.
        Muestra cuando empieza la degradacion detectable (~70% de vida).
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(SENSORS_TO_PLOT):
            grouped = self.train.groupby("life_pct")[sensor]
            mean = grouped.mean()
            std  = grouped.std()
            axes[i].plot(mean.index, mean.values, color="#4C72B0", linewidth=2)
            axes[i].fill_between(mean.index, mean - std, mean + std,
                                 alpha=0.2, color="#4C72B0")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].set_xlabel("% vida util")
        plt.suptitle("Degradacion promedio +/- 1 STD — FD001",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_sensor11_all_datasets(self) -> "EDADegradacion":
        """
        Curva de sensor_11 en los 4 sub-datasets.
        Permite comparar el sensor mas importante entre datasets.
        """
        sensor = "sensor_11"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            tr = add_rul_column(self.data[ds_id][0])
            tr = self._add_life_pct(tr)
            grouped = tr.groupby("life_pct")[sensor]
            mean = grouped.mean()
            std  = grouped.std()
            axes[i].plot(mean.index, mean.values, color=COLORS[i], linewidth=2)
            axes[i].fill_between(mean.index, mean - std, mean + std,
                                 alpha=0.2, color=COLORS[i])
            axes[i].set_title(ds_id, fontsize=13, fontweight="bold")
            axes[i].set_xlabel("% vida util")
            axes[i].set_ylabel(sensor)
        plt.suptitle(f"Degradacion de {sensor} — 4 sub-datasets",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_motor_heatmap(self, unit_id: int = 1) -> "EDADegradacion":
        """
        Heatmap de sensores a lo largo del ciclo de vida de un motor.
        Filas=sensores, columnas=ciclos. Muestra evolucion conjunta.
        """
        unit_data = self.train[self.train["unit_id"] == unit_id][USEFUL_SENSORS]
        fig, ax = plt.subplots(figsize=(18, 8))
        sns.heatmap(unit_data.T, cmap="YlOrRd", xticklabels=10,
                    yticklabels=True, ax=ax,
                    cbar_kws={"label": "Valor del sensor"})
        ax.set_xlabel("Ciclo")
        ax.set_ylabel("Sensor")
        ax.set_title(f"Heatmap de degradacion — Motor {unit_id} — FD001",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_last_cycles_heatmap(self, n_cycles: int = 50) -> "EDADegradacion":
        """
        Heatmap de los ultimos N ciclos antes del fallo.
        Muestra la degradacion acelerada final de forma agregada.
        """
        last_cycles = []
        for unit in self.train["unit_id"].unique():
            ud = self.train[self.train["unit_id"] == unit].tail(n_cycles).copy()
            ud["reverse_cycle"] = range(len(ud) - 1, -1, -1)
            last_cycles.append(ud)
        last_df = pd.concat(last_cycles, ignore_index=True)
        pivot = last_df.groupby("reverse_cycle")[USEFUL_SENSORS].mean()
        fig, ax = plt.subplots(figsize=(18, 8))
        sns.heatmap(pivot.T, cmap="YlOrRd", xticklabels=5, ax=ax,
                    cbar_kws={"label": "Valor promedio"})
        ax.set_xlabel("Ciclos antes de falla")
        ax.set_ylabel("Sensor")
        ax.set_title(f"Ultimos {n_cycles} ciclos antes de falla — FD001",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_change_rate(self) -> "EDADegradacion":
        """
        Histograma de la tasa de cambio de cada sensor en el ultimo ciclo.
        Muestra que sensores cambian mas rapido al final de la vida.
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(SENSORS_TO_PLOT):
            rates = []
            for unit in self.train["unit_id"].unique():
                ud   = self.train[self.train["unit_id"] == unit]
                diff = ud[sensor].diff().dropna()
                rates.append(diff.values[-1] if len(diff) > 0 else 0)
            axes[i].hist(rates, bins=50, color="#C44E52",
                         alpha=0.7, edgecolor="white")
            axes[i].axvline(0, color="black", linestyle="--", linewidth=1)
            axes[i].set_title(f"D-{sensor} (ultimo ciclo)",
                              fontsize=12, fontweight="bold")
        plt.suptitle("Tasa de cambio de sensores — FD001", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rolling_vs_raw(self, unit_id: int = 1,
                            window: int = 20) -> "EDADegradacion":
        """
        Senal cruda vs rolling mean para un motor especifico.
        Muestra el efecto de suavizado de las rolling features.
        """
        unit_data = self.train[self.train["unit_id"] == unit_id].copy()
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        for i, sensor in enumerate(SENSORS_TO_PLOT):
            axes[i].plot(unit_data["cycle"], unit_data[sensor],
                         alpha=0.3, color="gray", label="Crudo")
            rolling = unit_data[sensor].rolling(window=window, min_periods=1).mean()
            axes[i].plot(unit_data["cycle"], rolling,
                         color="#4C72B0", linewidth=2,
                         label=f"Rolling mean ({window})")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].legend(fontsize=9)
        plt.suptitle(f"Crudo vs Rolling Mean — Motor {unit_id} — FD001",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_degradation_by_phase(self,
                                  sensor: str = "sensor_11") -> "EDADegradacion":
        """
        Distribucion de un sensor en 3 fases de vida: inicio, medio y final.
        Muestra el desplazamiento de la distribucion con la degradacion.
        """
        phases = {
            "Inicio (0-30%)":  (0, 30),
            "Medio (30-70%)":  (30, 70),
            "Final (70-100%)": (70, 100),
        }
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (phase_name, (lo, hi)) in zip(axes, phases.items()):
            phase_data = self.train[
                (self.train["life_pct"] >= lo) &
                (self.train["life_pct"] < hi)
            ]
            ax.hist(phase_data[sensor], bins=50, alpha=0.7,
                    edgecolor="white", color="#4C72B0")
            ax.set_title(phase_name, fontsize=12, fontweight="bold")
            ax.set_xlabel(sensor)
        plt.suptitle(f"Distribucion de {sensor} por fase de vida — FD001",
                     fontsize=16, y=1.05)
        plt.tight_layout()
        self._save(fig)
        return self

    def print_report(self) -> "EDADegradacion":
        """Imprime el resumen del analisis de degradacion."""
        print("=" * 60)
        print("  RESUMEN EDA DEGRADACION")
        print("=" * 60)
        print(f"  Figuras generadas: {self.fig_count}")
        print(f"  Motor sano en primero 70% de vida (degradacion no detectable)")
        print(f"  Aceleracion de degradacion en ultimos 20-30 ciclos")
        print(f"  FD002/FD004 muestran patron escalera (cambios de condicion)")
        print(f"  Rolling mean suaviza la senal y revela la tendencia")
        print("=" * 60)
        return self

    def run(self) -> "EDADegradacion":
        """Ejecuta todos los analisis en cadena."""
        return (
            self
            .plot_raw_curves()
            .plot_normalized_curves()
            .plot_mean_bands()
            .plot_sensor11_all_datasets()
            .plot_motor_heatmap()
            .plot_last_cycles_heatmap()
            .plot_change_rate()
            .plot_rolling_vs_raw()
            .plot_degradation_by_phase()
            .print_report()
        )


if __name__ == "__main__":
    eda = EDADegradacion()
    eda.run()
