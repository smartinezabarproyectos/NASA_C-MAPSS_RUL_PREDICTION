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

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.config import DATASETS, MAX_RUL, CLASSIFICATION_W

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "05_eda_rul"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


class EDARUL:
    def __init__(self):
        self.loader    = DataLoader()
        self.prep      = Preprocessor()
        self.fig_count = 0
        self.loader.load_all()
        self._rul_all  = self._build_rul_all()
        sns.set_theme(style="whitegrid")

    def _build_rul_all(self) -> pd.DataFrame:
        frames = []
        for ds_id in DATASETS:
            rul = self.loader.data[ds_id][2]
            frames.append(pd.DataFrame({"rul": rul, "dataset": ds_id}))
        return pd.concat(frames, ignore_index=True)

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        path = FIGURES_DIR / f"05_eda_rul_fig{self.fig_count:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_rul_test_distributions(self) -> "EDARUL":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            rul = self.loader.data[ds_id][2]
            axes[i].hist(rul, bins=40, color=COLORS[i], alpha=0.75, edgecolor="white")
            axes[i].axvline(rul.mean(), color="black", linestyle="--", linewidth=1.5,
                            label=f"Mean: {rul.mean():.1f}")
            axes[i].axvline(rul.median(), color="gray", linestyle=":", linewidth=1.5,
                            label=f"Median: {rul.median():.1f}")
            axes[i].set(title=f"{ds_id} — RUL de test", xlabel="RUL (ciclos)")
            axes[i].legend()
        plt.suptitle("Distribucion del RUL real en test", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rul_boxplot_comparison(self) -> "EDARUL":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=self._rul_all, x="dataset", y="rul", hue="dataset",
                    palette=COLORS, legend=False, ax=ax)
        ax.set(ylabel="RUL (ciclos)", title="Distribucion de RUL de test — Comparacion")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_piecewise_linear(self, ds_id: str = "FD001", unit_id: int = 1) -> "EDARUL":
        train = self.loader.data[ds_id][0].copy()
        max_cycles        = train.groupby("unit_id")["cycle"].transform("max")
        train["rul_linear"]  = max_cycles - train["cycle"]
        train["rul_clipped"] = train["rul_linear"].clip(upper=MAX_RUL)
        ud = train[train["unit_id"] == unit_id]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ud["cycle"], ud["rul_linear"],  color="#C44E52", linewidth=2, label="RUL lineal")
        ax.plot(ud["cycle"], ud["rul_clipped"], color="#4C72B0", linewidth=2,
                linestyle="--", label=f"RUL clipped (max={MAX_RUL})")
        ax.axhline(MAX_RUL, color="gray", linestyle=":", alpha=0.5)
        ax.set(xlabel="Ciclo", ylabel="RUL",
               title=f"RUL lineal vs piece-wise — Motor {unit_id} — {ds_id}")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_multiple_rul_curves(self, ds_id: str = "FD001",
                                 units: list = None) -> "EDARUL":
        units     = units or [1, 5, 10, 20, 50, 80]
        train_rul = self.prep.compute_rul(self.loader.data[ds_id][0])
        fig, ax   = plt.subplots(figsize=(14, 6))
        for unit in units:
            ud = train_rul[train_rul["unit_id"] == unit]
            ax.plot(ud["cycle"], ud["rul"], linewidth=1.5, alpha=0.8, label=f"Motor {unit}")
        ax.axhline(MAX_RUL, color="gray", linestyle=":", alpha=0.5, label=f"Cap = {MAX_RUL}")
        ax.set(xlabel="Ciclo", ylabel="RUL (clipped)",
               title=f"Piece-wise linear RUL — {ds_id}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rul_train_distribution(self) -> "EDARUL":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            tr = self.prep.compute_rul(self.loader.data[ds_id][0])
            axes[i].hist(tr["rul"], bins=60, color=COLORS[i], alpha=0.75, edgecolor="white")
            axes[i].axvline(MAX_RUL, color="black", linestyle="--", linewidth=1.5,
                            label=f"Cap = {MAX_RUL}")
            axes[i].set(title=f"{ds_id} — RUL train (clipped)", xlabel="RUL")
            axes[i].legend()
        plt.suptitle("Distribucion del RUL en train (piece-wise)", fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_class_balance(self) -> "EDARUL":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, ds_id in enumerate(DATASETS):
            tr     = self.prep.compute_rul(self.loader.data[ds_id][0])
            tr     = self.prep.compute_binary_label(tr)
            counts = tr["label"].value_counts().sort_index()
            pct    = counts / counts.sum() * 100
            bars   = axes[i].bar(
                ["Sano (0)", f"Falla <={CLASSIFICATION_W} (1)"],
                counts.values, color=["#55A868", "#C44E52"], edgecolor="white"
            )
            for bar, p in zip(bars, pct.values):
                axes[i].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 100, f"{p:.1f}%",
                             ha="center", fontsize=11, fontweight="bold")
            axes[i].set(title=f"{ds_id} — Balance de clases", ylabel="Muestras")
        plt.suptitle(f"Clasificacion binaria — Falla en <={CLASSIFICATION_W} ciclos",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_threshold_impact(self, ds_id: str = "FD001") -> "EDARUL":
        w_values  = [10, 20, 30, 40, 50, 60]
        train_rul = self.prep.compute_rul(self.loader.data[ds_id][0])
        pct_falla = [(train_rul["rul"] <= w).mean() * 100 for w in w_values]
        fig, ax   = plt.subplots(figsize=(10, 5))
        bars = ax.bar([str(w) for w in w_values], pct_falla,
                      color="#4C72B0", edgecolor="white")
        for bar, p in zip(bars, pct_falla):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{p:.1f}%", ha="center", fontsize=11)
        ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="50% balance")
        ax.set(xlabel="Umbral W (ciclos)", ylabel="% clase positiva",
               title=f"Impacto del umbral W — {ds_id}")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rul_stem(self, ds_id: str = "FD001") -> "EDARUL":
        rul = self.loader.data[ds_id][2].reset_index()
        rul.columns = ["motor", "rul"]
        rul = rul.sort_values("rul").reset_index(drop=True)
        rul["motor"] = range(1, len(rul) + 1)
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.stem(rul["motor"], rul["rul"],
                linefmt="steelblue", markerfmt="o", basefmt=" ")
        ax.axhline(CLASSIFICATION_W, color="#C44E52", linestyle="--",
                   linewidth=1.5, label=f"W={CLASSIFICATION_W}")
        ax.set(xlabel="Motores (ordenados por RUL)", ylabel="RUL real",
               title=f"RUL real de test ordenado — {ds_id}")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_violin_by_complexity(self) -> "EDARUL":
        complexity_map = {
            "FD001": "Simple (1 cond, 1 falla)",
            "FD002": "Multi-cond (6 cond, 1 falla)",
            "FD003": "Multi-falla (1 cond, 2 fallas)",
            "FD004": "Complejo (6 cond, 2 fallas)",
        }
        df = self._rul_all.copy()
        df["complexity"] = df["dataset"].map(complexity_map)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=df, x="complexity", y="rul", hue="complexity",
                       palette="Set2", inner="quartile", legend=False, ax=ax)
        ax.set(xlabel="", ylabel="RUL (ciclos)",
               title="Distribucion de RUL por nivel de complejidad")
        plt.xticks(rotation=15)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_train_vs_test_comparison(self) -> "EDARUL":
        comparison = {}
        for ds_id in DATASETS:
            train = self.loader.data[ds_id][0]
            rul   = self.loader.data[ds_id][2]
            mc    = train.groupby("unit_id")["cycle"].max()
            comparison[ds_id] = {
                "vida_util_mean": mc.mean(), "vida_util_std": mc.std(),
                "rul_test_mean":  rul.mean(), "rul_test_std":  rul.std(),
            }
        comp_df = pd.DataFrame(comparison).T
        x, w    = np.arange(len(DATASETS)), 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - w / 2, comp_df["vida_util_mean"], w,
               yerr=comp_df["vida_util_std"], label="Vida util train",
               color="#4C72B0", edgecolor="white", capsize=4)
        ax.bar(x + w / 2, comp_df["rul_test_mean"], w,
               yerr=comp_df["rul_test_std"], label="RUL test",
               color="#DD8452", edgecolor="white", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.set(ylabel="Ciclos", title="Vida util (train) vs RUL real (test)")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_cdf(self) -> "EDARUL":
        fig, ax = plt.subplots(figsize=(12, 6))
        for ds_id, color in zip(DATASETS, COLORS):
            rul = np.sort(self.loader.data[ds_id][2].values)
            cdf = np.arange(1, len(rul) + 1) / len(rul)
            ax.plot(rul, cdf, linewidth=2, label=ds_id, color=color)
        ax.axvline(CLASSIFICATION_W, color="red", linestyle="--",
                   alpha=0.5, label=f"W={CLASSIFICATION_W}")
        ax.set(xlabel="RUL (ciclos)", ylabel="Probabilidad acumulada",
               title="CDF del RUL de test — 4 sub-datasets")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig)
        return self

    def print_report(self) -> "EDARUL":
        print("=" * 60)
        for ds_id in DATASETS:
            rul = self.loader.data[ds_id][2]
            tr  = self.prep.compute_rul(self.loader.data[ds_id][0])
            tr  = self.prep.compute_binary_label(tr)
            pct = tr["label"].mean() * 100
            print(f"  {ds_id}: mean={rul.mean():.1f} | "
                  f"median={rul.median():.1f} | falla={pct:.1f}%")
        print(f"\n  Umbral W = {CLASSIFICATION_W} | Cap = {MAX_RUL}")
        print("=" * 60)
        return self

    def run(self) -> "EDARUL":
        return (self
                .plot_rul_test_distributions()
                .plot_rul_boxplot_comparison()
                .plot_piecewise_linear()
                .plot_multiple_rul_curves()
                .plot_rul_train_distribution()
                .plot_class_balance()
                .plot_threshold_impact()
                .plot_rul_stem()
                .plot_violin_by_complexity()
                .plot_train_vs_test_comparison()
                .plot_cdf()
                .print_report())


if __name__ == "__main__":
    EDARUL().run()
