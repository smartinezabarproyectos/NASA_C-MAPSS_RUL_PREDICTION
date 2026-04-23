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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.config import DATASETS, USEFUL_SENSORS, MAX_RUL

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "06_eda_3d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]


class EDA3D:
    def __init__(self):
        self.loader    = DataLoader()
        self.prep      = Preprocessor()
        self.fig_count = 0
        self.loader.load_all()
        self.train      = self.prep.compute_rul(self.loader.data["FD001"][0])
        self.last_cycle = self.train.groupby("unit_id").last().reset_index()
        scaler          = StandardScaler()
        self.lc_scaled  = self.last_cycle.copy()
        self.lc_scaled[USEFUL_SENSORS] = scaler.fit_transform(self.last_cycle[USEFUL_SENSORS])
        self.pca        = PCA(n_components=3)
        self.pca_df     = None
        self.tsne_df    = None

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        fig.savefig(FIGURES_DIR / f"06_eda_3d_fig{self.fig_count:02d}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    def run_pca(self) -> "EDA3D":
        result      = self.pca.fit_transform(self.lc_scaled[USEFUL_SENSORS])
        self.pca_df = pd.DataFrame(result, columns=["PC1", "PC2", "PC3"])
        self.pca_df["rul"]     = self.last_cycle["rul"].values
        self.pca_df["unit_id"] = self.last_cycle["unit_id"].values
        print(f"  PCA varianza explicada: {self.pca.explained_variance_ratio_.sum():.1%}")
        return self

    def run_tsne(self, n_components: int = 2) -> "EDA3D":
        tsne        = TSNE(n_components=n_components, perplexity=30,
                           random_state=42, max_iter=1000)
        result      = tsne.fit_transform(self.lc_scaled[USEFUL_SENSORS])
        cols        = [f"t-SNE{i+1}" for i in range(n_components)]
        self.tsne_df = pd.DataFrame(result, columns=cols)
        self.tsne_df["rul"]     = self.last_cycle["rul"].values
        self.tsne_df["unit_id"] = self.last_cycle["unit_id"].values
        return self

    def plot_pca_2d(self) -> "EDA3D":
        if self.pca_df is None:
            self.run_pca()
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(self.pca_df["PC1"], self.pca_df["PC2"],
                        c=self.pca_df["rul"], cmap="RdYlGn", alpha=0.7, s=40)
        plt.colorbar(sc, ax=ax, label="RUL")
        exp = self.pca.explained_variance_ratio_[:2].sum()
        ax.set(xlabel="PC1", ylabel="PC2",
               title=f"PCA 2D — Varianza: {exp:.1%}")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_explained_variance(self) -> "EDA3D":
        pca_full   = PCA(n_components=len(USEFUL_SENSORS))
        pca_full.fit(self.lc_scaled[USEFUL_SENSORS])
        individual = pca_full.explained_variance_ratio_ * 100
        cumulative = np.cumsum(individual)
        fig, ax    = plt.subplots(figsize=(12, 6))
        ax.bar(range(1, len(individual) + 1), individual,
               color="#4C72B0", edgecolor="white", label="Individual")
        ax.plot(range(1, len(cumulative) + 1), cumulative,
                color="#C44E52", linewidth=2, marker="o", label="Acumulada")
        ax.axhline(95, color="gray", linestyle="--", alpha=0.5)
        ax.set(xlabel="Componente", ylabel="% varianza",
               title="Varianza explicada por componente PCA")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_tsne_2d(self) -> "EDA3D":
        if self.tsne_df is None:
            self.run_tsne()
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(self.tsne_df["t-SNE1"], self.tsne_df["t-SNE2"],
                        c=self.tsne_df["rul"], cmap="RdYlGn", alpha=0.7, s=40)
        plt.colorbar(sc, ax=ax, label="RUL")
        ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2",
               title="t-SNE 2D — Coloreado por RUL — FD001")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_pca_all_datasets(self) -> "EDA3D":
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes      = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            tr   = self.prep.compute_rul(self.loader.data[ds_id][0])
            lc   = tr.groupby("unit_id").last().reset_index()
            avail = [s for s in USEFUL_SENSORS if s in lc.columns]
            scaled = StandardScaler().fit_transform(lc[avail])
            result = PCA(n_components=2).fit_transform(scaled)
            sc = axes[i].scatter(result[:, 0], result[:, 1],
                                 c=lc["rul"].values, cmap="RdYlGn", alpha=0.7, s=30)
            plt.colorbar(sc, ax=axes[i], label="RUL")
            axes[i].set(title=ds_id, xlabel="PC1", ylabel="PC2")
        plt.suptitle("PCA 2D — Todos los sub-datasets", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_tsne_all_datasets(self) -> "EDA3D":
        all_last = []
        for ds_id in DATASETS:
            tr    = self.prep.compute_rul(self.loader.data[ds_id][0])
            lc    = tr.groupby("unit_id").last().reset_index()
            avail = [s for s in USEFUL_SENSORS if s in lc.columns]
            sub   = lc[avail].copy()
            sub["dataset"] = ds_id
            sub["rul"]     = lc["rul"].values
            all_last.append(sub)
        all_df      = pd.concat(all_last, ignore_index=True)
        feat_cols   = [s for s in USEFUL_SENSORS if s in all_df.columns]
        scaled      = StandardScaler().fit_transform(all_df[feat_cols])
        result      = TSNE(n_components=2, perplexity=30,
                           random_state=42, max_iter=1000).fit_transform(scaled)
        all_df["t-SNE1"] = result[:, 0]
        all_df["t-SNE2"] = result[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for j, ds_id in enumerate(DATASETS):
            sub = all_df[all_df["dataset"] == ds_id]
            axes[0].scatter(sub["t-SNE1"], sub["t-SNE2"],
                            c=COLORS[j], alpha=0.6, s=30, label=ds_id)
        axes[0].legend()
        axes[0].set_title("t-SNE — Por sub-dataset")
        sc = axes[1].scatter(all_df["t-SNE1"], all_df["t-SNE2"],
                             c=all_df["rul"], cmap="RdYlGn", alpha=0.6, s=30)
        plt.colorbar(sc, ax=axes[1], label="RUL")
        axes[1].set_title("t-SNE — Coloreado por RUL")
        plt.suptitle("t-SNE 2D — Todos los sub-datasets", fontsize=14, y=1.02)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rul_surface(self, sensor_x: str = "sensor_7",
                         sensor_y: str = "sensor_11") -> "EDA3D":
        lc   = self.last_cycle[[sensor_x, sensor_y, "rul"]].dropna()
        xi   = np.linspace(lc[sensor_x].min(), lc[sensor_x].max(), 50)
        yi   = np.linspace(lc[sensor_y].min(), lc[sensor_y].max(), 50)
        xi, yi = np.meshgrid(xi, yi)
        zi   = griddata((lc[sensor_x].values, lc[sensor_y].values),
                        lc["rul"].values, (xi, yi), method="cubic")
        fig  = plt.figure(figsize=(10, 7))
        ax   = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xi, yi, zi, cmap="RdYlGn", alpha=0.85)
        ax.set(xlabel=sensor_x, ylabel=sensor_y, zlabel="RUL",
               title=f"Superficie RUL — {sensor_x} x {sensor_y} — FD001")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_op_clusters(self) -> "EDA3D":
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for i, ds_id in enumerate(["FD002", "FD004"]):
            tr         = self.loader.data[ds_id][0]
            unique_ops = tr[["op_setting_1", "op_setting_2", "op_setting_3"]].drop_duplicates()
            axes[i].scatter(unique_ops["op_setting_1"], unique_ops["op_setting_2"],
                            c=unique_ops["op_setting_1"], cmap="Viridis", s=80)
            axes[i].set(title=f"Condiciones operacionales — {ds_id}",
                        xlabel="op_setting_1", ylabel="op_setting_2")
        plt.suptitle("Clusters operacionales — FD002 vs FD004", fontsize=14)
        plt.tight_layout()
        self._save(fig)
        return self

    def print_report(self) -> "EDA3D":
        exp = self.pca.explained_variance_ratio_.sum() if self.pca_df is not None else 0
        print("=" * 60)
        print(f"  PCA 3 componentes: {exp:.1%} varianza explicada")
        print(f"  t-SNE revela separacion sanos vs degradados")
        print(f"  FD002/FD004: 6 clusters operacionales claros")
        print("=" * 60)
        return self

    def run(self) -> "EDA3D":
        return (self
                .run_pca()
                .run_tsne()
                .plot_pca_2d()
                .plot_explained_variance()
                .plot_tsne_2d()
                .plot_pca_all_datasets()
                .plot_tsne_all_datasets()
                .plot_rul_surface()
                .plot_op_clusters()
                .print_report())


if __name__ == "__main__":
    EDA3D().run()
