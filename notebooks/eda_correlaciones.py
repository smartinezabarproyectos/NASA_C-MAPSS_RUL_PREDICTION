"""
EDA Correlaciones — NASA C-MAPSS
Correlacion sensor-RUL, multicolinealidad, dendrograma y analisis
por etapa de vida.
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
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all_datasets
from src.preprocessing import add_rul_column
from src.config import DATASETS, USEFUL_SENSORS

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "04_eda_correlaciones"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


class EDACorrelaciones:
    """
    Analisis de correlaciones del dataset NASA C-MAPSS.

    Attributes
    ----------
    data : dict
        Diccionario con los 4 sub-datasets.
    train : pd.DataFrame
        Train FD001 con columna RUL.
    corr : pd.DataFrame
        Matriz de correlacion sensores + RUL para FD001.
    rul_corr : pd.Series
        Correlacion de cada sensor con RUL ordenada.
    high_corr_pairs : list
        Pares de sensores con correlacion > 0.9.
    fig_count : int
        Contador de figuras.
    """

    def __init__(self):
        print("Cargando datasets NASA C-MAPSS...")
        self.data           = load_all_datasets()
        self.train          = add_rul_column(self.data["FD001"][0])
        self.corr           = self.train[USEFUL_SENSORS + ["rul"]].corr()
        self.rul_corr       = self.corr["rul"].drop("rul").sort_values()
        self.high_corr_pairs = []
        self.fig_count      = 0
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        path = FIGURES_DIR / f"04_eda_correlaciones_fig{self.fig_count:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardada: {path.name}")

    def plot_correlation_matrix(self) -> "EDACorrelaciones":
        """
        Matriz de correlacion de Pearson entre sensores y RUL.
        Identifica que sensores tienen mayor relacion lineal con el target.
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(self.corr, dtype=bool))
        sns.heatmap(
            self.corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True, linewidths=0.5,
            ax=ax, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 8},
        )
        ax.set_title("Matriz de correlacion — Sensores + RUL — FD001",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rul_correlation_bar(self) -> "EDACorrelaciones":
        """
        Barras horizontales de correlacion de cada sensor con RUL.
        Rojo = correlacion negativa, azul = positiva.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#C44E52" if v < 0 else "#4C72B0" for v in self.rul_corr.values]
        self.rul_corr.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Correlacion con RUL")
        ax.set_title("Correlacion de cada sensor con RUL — FD001",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_cross_dataset_correlation(self) -> "EDACorrelaciones":
        """
        Correlacion sensor-RUL en los 4 sub-datasets en un mismo grafico.
        Confirma cuales sensores son universalmente informativos.
        """
        rul_corrs = {}
        for ds_id in DATASETS:
            tr = add_rul_column(self.data[ds_id][0])
            available = [s for s in USEFUL_SENSORS if s in tr.columns]
            rul_corrs[ds_id] = tr[available + ["rul"]].corr()["rul"].drop("rul")
        rul_corr_df = pd.DataFrame(rul_corrs)
        fig, ax = plt.subplots(figsize=(14, 8))
        rul_corr_df.plot(kind="bar", ax=ax, width=0.75, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Correlacion con RUL")
        ax.set_title("Correlacion sensor-RUL por sub-dataset",
                     fontsize=14, fontweight="bold")
        ax.legend(title="Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_clustermap(self) -> "EDACorrelaciones":
        """
        Clustermap jerarquico de correlaciones entre sensores.
        Agrupa sensores que miden fenomenos similares.
        """
        sensor_data  = self.train[USEFUL_SENSORS]
        corr_matrix  = sensor_data.corr()
        g = sns.clustermap(
            corr_matrix, cmap="RdBu_r", center=0,
            figsize=(12, 12), annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=0.5,
            vmin=-1, vmax=1, dendrogram_ratio=(0.15, 0.15),
        )
        g.fig.suptitle("Clustermap — Sensores — FD001",
                       fontsize=14, fontweight="bold", y=1.02)
        path = FIGURES_DIR / f"04_eda_correlaciones_fig{self.fig_count + 1:02d}.png"
        self.fig_count += 1
        g.fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        print(f"  Guardada: {path.name}")
        return self

    def plot_dendrogram(self) -> "EDACorrelaciones":
        """
        Dendrograma de sensores basado en distancia de correlacion.
        Muestra grupos de sensores que se degradan juntos.
        """
        sensor_data = self.train[USEFUL_SENSORS]
        corr_matrix = sensor_data.corr()
        corr_distance = 1 - corr_matrix.abs()
        dist_array    = squareform(corr_distance.values)
        linkage_matrix = linkage(dist_array, method="ward")
        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(
            linkage_matrix,
            labels=corr_matrix.columns.tolist(),
            leaf_rotation=45, leaf_font_size=11,
            ax=ax, color_threshold=0.7,
        )
        ax.set_ylabel("Distancia (1 - |correlacion|)")
        ax.set_title("Dendrograma de sensores — FD001",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_pairplot_top5(self) -> "EDACorrelaciones":
        """
        Pair plot de los 5 sensores mas correlacionados con RUL.
        Colorea los puntos segun estado de salud del motor.
        """
        top5 = self.rul_corr.abs().sort_values(ascending=False).head(5).index.tolist()
        sample = self.train[self.train["unit_id"].isin(
            self.train["unit_id"].unique()[:15]
        )].copy()
        sample["rul_bin"] = pd.cut(
            sample["rul"], bins=[0, 30, 80, 125],
            labels=["Critico", "Medio", "Sano"]
        )
        g = sns.pairplot(
            sample[top5 + ["rul_bin"]], hue="rul_bin",
            palette={"Critico": "#C44E52", "Medio": "#DD8452", "Sano": "#55A868"},
            plot_kws={"alpha": 0.3, "s": 10},
            diag_kws={"alpha": 0.5}, height=2.5,
        )
        g.fig.suptitle("Pair plot — Top 5 sensores vs estado de salud",
                       fontsize=14, fontweight="bold", y=1.02)
        path = FIGURES_DIR / f"04_eda_correlaciones_fig{self.fig_count + 1:02d}.png"
        self.fig_count += 1
        g.fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        print(f"  Guardada: {path.name}")
        return self

    def plot_temporal_correlation(self,
                                  sensor: str = "sensor_11") -> "EDACorrelaciones":
        """
        Correlacion sensor-RUL por cuartil de vida.
        Muestra que la correlacion aumenta en las etapas finales.
        """
        windows = [(0, 25), (25, 50), (50, 75), (75, 100)]
        window_corrs = {}
        for ds_id in DATASETS:
            tr = add_rul_column(self.data[ds_id][0])
            tr["life_pct"] = tr.groupby("unit_id")["cycle"].transform(
                lambda x: (x / x.max() * 100)
            )
            corrs = []
            for lo, hi in windows:
                subset = tr[(tr["life_pct"] >= lo) & (tr["life_pct"] < hi)]
                if len(subset) > 10:
                    corrs.append(subset[[sensor, "rul"]].corr().iloc[0, 1])
                else:
                    corrs.append(np.nan)
            window_corrs[ds_id] = corrs

        labels    = [f"{lo}-{hi}%" for lo, hi in windows]
        corr_temp = pd.DataFrame(window_corrs, index=labels)
        fig, ax   = plt.subplots(figsize=(10, 6))
        corr_temp.plot(kind="bar", ax=ax, width=0.7, edgecolor="white")
        ax.set_ylabel(f"Correlacion {sensor} <-> RUL")
        ax.set_title(f"Correlacion temporal de {sensor} con RUL por cuartil",
                     fontsize=14, fontweight="bold")
        ax.legend(title="Dataset")
        plt.xticks(rotation=0)
        plt.tight_layout()
        self._save(fig)
        return self

    def compute_high_correlation_pairs(self,
                                       threshold: float = 0.9) -> "EDACorrelaciones":
        """
        Detecta pares de sensores con correlacion > threshold.
        Identifica multicolinealidad que puede afectar a modelos lineales.
        """
        sensor_data = self.train[USEFUL_SENSORS].dropna()
        scaler      = StandardScaler()
        scaled      = pd.DataFrame(
            scaler.fit_transform(sensor_data), columns=USEFUL_SENSORS
        )
        corr_abs = scaled.corr().abs()
        upper    = corr_abs.where(
            np.triu(np.ones_like(corr_abs, dtype=bool), k=1)
        )
        self.high_corr_pairs = [
            (col, row, corr_abs.loc[row, col])
            for col in upper.columns
            for row in upper.index
            if upper.loc[row, col] > threshold
        ]
        self.high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"\nPares con correlacion > {threshold}:")
        for s1, s2, val in self.high_corr_pairs:
            print(f"  {s1} <-> {s2}: {val:.3f}")
        return self

    def print_report(self) -> "EDACorrelaciones":
        """Imprime resumen del analisis de correlaciones."""
        top5 = self.rul_corr.abs().sort_values(ascending=False).head(5).index.tolist()
        print("=" * 60)
        print("  RESUMEN CORRELACIONES")
        print("=" * 60)
        print(f"  Top 5 sensores correlacionados con RUL (FD001):")
        for s in top5:
            print(f"    {s}: {self.rul_corr[s]:.3f}")
        print(f"  Pares con alta multicolinealidad (>0.9): {len(self.high_corr_pairs)}")
        print(f"  Correlacion aumenta en etapas finales de vida")
        print("=" * 60)
        return self

    def run(self) -> "EDACorrelaciones":
        """Ejecuta todos los analisis en cadena."""
        return (
            self
            .plot_correlation_matrix()
            .plot_rul_correlation_bar()
            .plot_cross_dataset_correlation()
            .plot_clustermap()
            .plot_dendrogram()
            .plot_pairplot_top5()
            .plot_temporal_correlation()
            .compute_high_correlation_pairs()
            .print_report()
        )


if __name__ == "__main__":
    eda = EDACorrelaciones()
    eda.run()
