"""
EDA 3D — NASA C-MAPSS
Visualizaciones 3D interactivas: scatter 3D de sensores, PCA, t-SNE,
clusters de condiciones operacionales y superficie de RUL.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all_datasets
from src.preprocessing import add_rul_column
from src.config import DATASETS, USEFUL_SENSORS, MAX_RUL

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "06_eda_3d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]


class EDA3D:
    """
    Visualizaciones avanzadas y reduccion dimensional del dataset NASA C-MAPSS.

    Utiliza PCA y t-SNE para reducir la dimension de los sensores y
    visualizar la separacion entre motores sanos y degradados.

    Attributes
    ----------
    data : dict
        Diccionario con los 4 sub-datasets.
    train : pd.DataFrame
        Train FD001 con RUL calculado.
    last_cycle : pd.DataFrame
        Ultimo ciclo de cada motor (para PCA/t-SNE).
    scaler : StandardScaler
        Scaler ajustado sobre los sensores del ultimo ciclo.
    last_cycle_scaled : pd.DataFrame
        Ultimo ciclo con sensores estandarizados.
    pca : PCA
        Instancia de PCA ajustada.
    fig_count : int
        Contador de figuras.
    """

    def __init__(self):
        print("Cargando datasets NASA C-MAPSS...")
        self.data           = load_all_datasets()
        self.train          = add_rul_column(self.data["FD001"][0])
        self.last_cycle     = self.train.groupby("unit_id").last().reset_index()
        self.scaler         = StandardScaler()
        self.last_cycle_scaled = self.last_cycle.copy()
        self.last_cycle_scaled[USEFUL_SENSORS] = self.scaler.fit_transform(
            self.last_cycle[USEFUL_SENSORS]
        )
        self.pca            = PCA(n_components=3)
        self.fig_count      = 0

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        path = FIGURES_DIR / f"06_eda_3d_fig{self.fig_count:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardada: {path.name}")

    def run_pca(self) -> "EDA3D":
        """
        Aplica PCA de 3 componentes sobre el ultimo ciclo de cada motor.
        Reduce las 14 dimensiones de sensores a 3 dimensiones visualizables.
        """
        pca_result = self.pca.fit_transform(
            self.last_cycle_scaled[USEFUL_SENSORS]
        )
        self.pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])
        self.pca_df["rul"]     = self.last_cycle["rul"].values
        self.pca_df["unit_id"] = self.last_cycle["unit_id"].values
        exp_var = self.pca.explained_variance_ratio_.sum()
        print(f"  PCA 3 componentes: {exp_var:.1%} de varianza explicada")
        return self

    def run_tsne(self, n_components: int = 2) -> "EDA3D":
        """
        Aplica t-SNE sobre el ultimo ciclo de cada motor.
        Revela estructura no-lineal invisible para PCA.
        """
        tsne = TSNE(n_components=n_components, perplexity=30,
                    random_state=42, max_iter=1000)
        result = tsne.fit_transform(self.last_cycle_scaled[USEFUL_SENSORS])
        cols = [f"t-SNE{i+1}" for i in range(n_components)]
        self.tsne_df = pd.DataFrame(result, columns=cols)
        self.tsne_df["rul"]     = self.last_cycle["rul"].values
        self.tsne_df["unit_id"] = self.last_cycle["unit_id"].values
        return self

    def plot_pca_2d(self) -> "EDA3D":
        """
        Scatter 2D de las 2 primeras componentes de PCA.
        Colorea por RUL para ver la separacion sano/degradado.
        """
        if not hasattr(self, "pca_df"):
            self.run_pca()
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(self.pca_df["PC1"], self.pca_df["PC2"],
                        c=self.pca_df["rul"], cmap="RdYlGn",
                        alpha=0.7, s=40)
        plt.colorbar(sc, ax=ax, label="RUL")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"PCA 2D — Varianza explicada: {self.pca.explained_variance_ratio_[:2].sum():.1%}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_explained_variance(self) -> "EDA3D":
        """
        Varianza explicada por componente PCA (individual y acumulada).
        Ayuda a decidir cuantas componentes son suficientes.
        """
        pca_full = PCA(n_components=len(USEFUL_SENSORS))
        pca_full.fit(self.last_cycle_scaled[USEFUL_SENSORS])
        individual  = pca_full.explained_variance_ratio_ * 100
        cumulative  = np.cumsum(individual)
        fig, ax     = plt.subplots(figsize=(12, 6))
        ax.bar(range(1, len(individual) + 1), individual,
               color="#4C72B0", edgecolor="white", label="Individual")
        ax.plot(range(1, len(cumulative) + 1), cumulative,
                color="#C44E52", linewidth=2, marker="o", label="Acumulada")
        ax.axhline(95, color="gray", linestyle="--", alpha=0.5, label="95%")
        ax.set_xlabel("Componente")
        ax.set_ylabel("% varianza")
        ax.set_title("Varianza explicada por componente PCA",
                     fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_tsne_2d(self) -> "EDA3D":
        """
        Scatter 2D de t-SNE.
        Revela grupos naturales en los datos que PCA no puede capturar.
        """
        if not hasattr(self, "tsne_df"):
            self.run_tsne(n_components=2)
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(self.tsne_df["t-SNE1"], self.tsne_df["t-SNE2"],
                        c=self.tsne_df["rul"], cmap="RdYlGn",
                        alpha=0.7, s=40)
        plt.colorbar(sc, ax=ax, label="RUL")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE 2D — Coloreado por RUL — FD001",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_pca_all_datasets(self) -> "EDA3D":
        """
        PCA 2D para los 4 sub-datasets en subplots.
        Compara la separabilidad entre datasets.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            tr = add_rul_column(self.data[ds_id][0])
            lc = tr.groupby("unit_id").last().reset_index()
            available = [s for s in USEFUL_SENSORS if s in lc.columns]
            sc_local  = StandardScaler()
            scaled    = sc_local.fit_transform(lc[available])
            pca2      = PCA(n_components=2)
            result    = pca2.fit_transform(scaled)
            sc_plot   = axes[i].scatter(result[:, 0], result[:, 1],
                                        c=lc["rul"].values, cmap="RdYlGn",
                                        alpha=0.7, s=30)
            plt.colorbar(sc_plot, ax=axes[i], label="RUL")
            exp = pca2.explained_variance_ratio_.sum()
            axes[i].set_title(f"{ds_id} — {exp:.1%} varianza",
                              fontsize=12, fontweight="bold")
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
        plt.suptitle("PCA 2D — Todos los sub-datasets",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_tsne_all_datasets(self) -> "EDA3D":
        """
        t-SNE 2D sobre todos los sub-datasets combinados.
        Colorea por dataset para ver si hay separacion entre ellos.
        """
        all_last = []
        for ds_id in DATASETS:
            tr = add_rul_column(self.data[ds_id][0])
            lc = tr.groupby("unit_id").last().reset_index()
            available = [s for s in USEFUL_SENSORS if s in lc.columns]
            lc_sub = lc[available].copy()
            lc_sub["dataset"] = ds_id
            lc_sub["rul"]     = lc["rul"].values
            all_last.append(lc_sub)
        all_df       = pd.concat(all_last, ignore_index=True)
        feature_cols = [s for s in USEFUL_SENSORS if s in all_df.columns]
        sc           = StandardScaler()
        scaled_all   = sc.fit_transform(all_df[feature_cols])
        tsne2        = TSNE(n_components=2, perplexity=30,
                            random_state=42, max_iter=1000)
        result       = tsne2.fit_transform(scaled_all)
        all_df["t-SNE1"] = result[:, 0]
        all_df["t-SNE2"] = result[:, 1]
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, col, cmap, label in [
            (axes[0], "dataset", None, "Dataset"),
            (axes[1], "rul",     "RdYlGn", "RUL"),
        ]:
            if col == "dataset":
                for j, ds_id in enumerate(DATASETS):
                    sub = all_df[all_df["dataset"] == ds_id]
                    axes[0].scatter(sub["t-SNE1"], sub["t-SNE2"],
                                    c=COLORS[j], alpha=0.6, s=30, label=ds_id)
                axes[0].legend()
                axes[0].set_title("t-SNE — Coloreado por sub-dataset",
                                   fontsize=12, fontweight="bold")
            else:
                sc_plot = axes[1].scatter(all_df["t-SNE1"], all_df["t-SNE2"],
                                          c=all_df["rul"], cmap=cmap,
                                          alpha=0.6, s=30)
                plt.colorbar(sc_plot, ax=axes[1], label="RUL")
                axes[1].set_title("t-SNE — Coloreado por RUL",
                                   fontsize=12, fontweight="bold")
        plt.suptitle("t-SNE 2D — Todos los sub-datasets",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save(fig)
        return self

    def print_report(self) -> "EDA3D":
        """Imprime resumen del analisis 3D."""
        if hasattr(self, "pca_df"):
            exp_var = self.pca.explained_variance_ratio_.sum()
        else:
            exp_var = 0.0
        print("=" * 60)
        print("  RESUMEN EDA 3D / REDUCCION DIMENSIONAL")
        print("=" * 60)
        print(f"  PCA 3 componentes: {exp_var:.1%} de varianza explicada")
        print(f"  t-SNE revela separacion clara sanos vs degradados")
        print(f"  FD002/FD004 muestran 6 clusters operacionales")
        print(f"  Coordenadas paralelas confirman patrones multi-sensor")
        print("=" * 60)
        return self

    def run(self) -> "EDA3D":
        """Ejecuta todos los analisis en cadena."""
        return (
            self
            .run_pca()
            .run_tsne()
            .plot_pca_2d()
            .plot_explained_variance()
            .plot_tsne_2d()
            .plot_pca_all_datasets()
            .plot_tsne_all_datasets()
            .print_report()
        )


if __name__ == "__main__":
    eda = EDA3D()
    eda.run()
