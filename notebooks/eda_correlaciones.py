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
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.config import DATASETS, USEFUL_SENSORS

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "04_eda_correlaciones"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class EDACorrelaciones:
    def __init__(self):
        self.loader    = DataLoader()
        self.prep      = Preprocessor()
        self.fig_count = 0
        self.loader.load_all()
        self.train    = self.prep.compute_rul(self.loader.data["FD001"][0])
        self.corr     = self.train[USEFUL_SENSORS + ["rul"]].corr()
        self.rul_corr = self.corr["rul"].drop("rul").sort_values()
        self.high_corr_pairs = []
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        fig.savefig(FIGURES_DIR / f"04_eda_correlaciones_fig{self.fig_count:02d}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_correlation_matrix(self) -> "EDACorrelaciones":
        mask    = np.triu(np.ones_like(self.corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(self.corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax,
                    vmin=-1, vmax=1, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
        ax.set_title("Matriz de correlacion — Sensores + RUL — FD001", fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rul_correlation_bar(self) -> "EDACorrelaciones":
        colors  = ["#C44E52" if v < 0 else "#4C72B0" for v in self.rul_corr.values]
        fig, ax = plt.subplots(figsize=(10, 8))
        self.rul_corr.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Correlacion con RUL", title="Correlacion sensor-RUL — FD001")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_cross_dataset_correlation(self) -> "EDACorrelaciones":
        rul_corrs = {}
        for ds_id in DATASETS:
            tr = self.prep.compute_rul(self.loader.data[ds_id][0])
            avail = [s for s in USEFUL_SENSORS if s in tr.columns]
            rul_corrs[ds_id] = tr[avail + ["rul"]].corr()["rul"].drop("rul")
        fig, ax = plt.subplots(figsize=(14, 8))
        pd.DataFrame(rul_corrs).plot(kind="bar", ax=ax, width=0.75, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(ylabel="Correlacion con RUL", title="Correlacion sensor-RUL por sub-dataset")
        ax.legend(title="Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_clustermap(self) -> "EDACorrelaciones":
        corr_matrix = self.train[USEFUL_SENSORS].corr()
        g = sns.clustermap(corr_matrix, cmap="RdBu_r", center=0, figsize=(12, 12),
                           annot=True, fmt=".2f", annot_kws={"size": 8},
                           linewidths=0.5, vmin=-1, vmax=1, dendrogram_ratio=(0.15, 0.15))
        g.fig.suptitle("Clustermap — Sensores — FD001", fontsize=14, fontweight="bold", y=1.02)
        self.fig_count += 1
        g.fig.savefig(FIGURES_DIR / f"04_eda_correlaciones_fig{self.fig_count:02d}.png",
                      dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        return self

    def plot_dendrogram(self) -> "EDACorrelaciones":
        corr_matrix    = self.train[USEFUL_SENSORS].corr()
        dist_array     = squareform(1 - corr_matrix.abs())
        linkage_matrix = linkage(dist_array, method="ward")
        fig, ax        = plt.subplots(figsize=(14, 6))
        dendrogram(linkage_matrix, labels=corr_matrix.columns.tolist(),
                   leaf_rotation=45, leaf_font_size=11, ax=ax, color_threshold=0.7)
        ax.set(ylabel="Distancia (1 - |correlacion|)", title="Dendrograma — FD001")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_pairplot_top5(self) -> "EDACorrelaciones":
        top5   = self.rul_corr.abs().sort_values(ascending=False).head(5).index.tolist()
        sample = self.train[self.train["unit_id"].isin(
            self.train["unit_id"].unique()[:15])].copy()
        sample["rul_bin"] = pd.cut(sample["rul"], bins=[0, 30, 80, 125],
                                   labels=["Critico", "Medio", "Sano"])
        g = sns.pairplot(sample[top5 + ["rul_bin"]], hue="rul_bin",
                         palette={"Critico": "#C44E52", "Medio": "#DD8452", "Sano": "#55A868"},
                         plot_kws={"alpha": 0.3, "s": 10}, diag_kws={"alpha": 0.5}, height=2.5)
        g.fig.suptitle("Pair plot — Top 5 sensores", fontsize=14, fontweight="bold", y=1.02)
        self.fig_count += 1
        g.fig.savefig(FIGURES_DIR / f"04_eda_correlaciones_fig{self.fig_count:02d}.png",
                      dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        return self

    def plot_temporal_correlation(self, sensor: str = "sensor_11") -> "EDACorrelaciones":
        windows      = [(0, 25), (25, 50), (50, 75), (75, 100)]
        window_corrs = {}
        for ds_id in DATASETS:
            tr = self.prep.compute_rul(self.loader.data[ds_id][0])
            tr["life_pct"] = tr.groupby("unit_id")["cycle"].transform(
                lambda x: x / x.max() * 100)
            corrs = []
            for lo, hi in windows:
                subset = tr[(tr["life_pct"] >= lo) & (tr["life_pct"] < hi)]
                corrs.append(subset[[sensor, "rul"]].corr().iloc[0, 1] if len(subset) > 10 else np.nan)
            window_corrs[ds_id] = corrs
        labels    = [f"{lo}-{hi}%" for lo, hi in windows]
        corr_temp = pd.DataFrame(window_corrs, index=labels)
        fig, ax   = plt.subplots(figsize=(10, 6))
        corr_temp.plot(kind="bar", ax=ax, width=0.7, edgecolor="white")
        ax.set(ylabel=f"Correlacion {sensor} - RUL",
               title="Evolucion de correlacion sensor-RUL por fase")
        ax.legend(title="Dataset")
        plt.xticks(rotation=0)
        plt.tight_layout()
        self._save(fig)
        return self

    def compute_high_correlation_pairs(self, threshold: float = 0.9) -> "EDACorrelaciones":
        scaled   = pd.DataFrame(
            StandardScaler().fit_transform(self.train[USEFUL_SENSORS].dropna()),
            columns=USEFUL_SENSORS)
        corr_abs = scaled.corr().abs()
        upper    = corr_abs.where(np.triu(np.ones_like(corr_abs, dtype=bool), k=1))
        self.high_corr_pairs = sorted(
            [(col, row, corr_abs.loc[row, col])
             for col in upper.columns for row in upper.index
             if upper.loc[row, col] > threshold],
            key=lambda x: x[2], reverse=True)
        for s1, s2, val in self.high_corr_pairs:
            print(f"  {s1} <-> {s2}: {val:.3f}")
        return self

    def print_report(self) -> "EDACorrelaciones":
        top5 = self.rul_corr.abs().sort_values(ascending=False).head(5).index.tolist()
        print("=" * 60)
        print("  Top 5 sensores correlacionados con RUL:")
        for s in top5:
            print(f"    {s}: {self.rul_corr[s]:.3f}")
        print(f"  Pares alta multicolinealidad (>0.9): {len(self.high_corr_pairs)}")
        print("=" * 60)
        return self

    def run(self) -> "EDACorrelaciones":
        return (self
                .plot_correlation_matrix()
                .plot_rul_correlation_bar()
                .plot_cross_dataset_correlation()
                .plot_clustermap()
                .plot_dendrogram()
                .plot_pairplot_top5()
                .plot_temporal_correlation()
                .compute_high_correlation_pairs()
                .print_report())


if __name__ == "__main__":
    EDACorrelaciones().run()
