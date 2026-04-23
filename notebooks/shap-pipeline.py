import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBRegressor

from src.shap_analysis import SHAPAnalyzer
from src.config import DATASETS, PROCESSED_DIR, MODELS_DIR

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "10_shap_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class SHAPPipeline:
    def __init__(self):
        self.processed       = {}
        self.feature_cols    = {}
        self.rf_model        = None
        self.xgb_model       = None
        self.xgb_analyzer    = SHAPAnalyzer()
        self.rf_analyzer     = SHAPAnalyzer()
        self.xgb_importance  = None
        self.rf_importance   = None
        self.shap_by_dataset = {}
        self.fig_count       = 0
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure) -> None:
        self.fig_count += 1
        path = FIGURES_DIR / f"10_shap_analysis_fig{self.fig_count:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def load_data(self) -> "SHAPPipeline":
        with open(PROCESSED_DIR / "metadata.json") as f:
            self.feature_cols = json.load(f)["feature_cols"]
        for ds_id in DATASETS:
            self.processed[ds_id] = {
                "train": pd.read_parquet(PROCESSED_DIR / f"train_{ds_id}.parquet"),
                "test":  pd.read_parquet(PROCESSED_DIR / f"test_{ds_id}.parquet"),
            }
        with open(MODELS_DIR / "random_forest_reg_FD001.pkl", "rb") as f:
            self.rf_model = pickle.load(f)
        with open(MODELS_DIR / "xgboost_reg_FD001.pkl", "rb") as f:
            self.xgb_model = pickle.load(f)
        return self

    def _get_xy(self, ds_id: str = "FD001") -> tuple:
        train   = self.processed[ds_id]["train"]
        test    = self.processed[ds_id]["test"]
        exclude = {"unit_id", "cycle", "rul", "label"}
        avail   = [c for c in train.columns if c not in exclude and c in test.columns]
        return train[avail], test[avail], test["rul"].values, avail

    def compute_shap(self) -> "SHAPPipeline":
        _, X_test, _, avail = self._get_xy("FD001")
        self.xgb_analyzer.fit(self.xgb_model, X_test.values, avail)
        self.rf_analyzer.fit(self.rf_model,  X_test.values, avail)
        self.xgb_importance = pd.Series(
            np.abs(self.xgb_analyzer.shap_values).mean(axis=0),
            index=avail).sort_values(ascending=False)
        self.rf_importance = pd.Series(
            np.abs(self.rf_analyzer.shap_values).mean(axis=0),
            index=avail).sort_values(ascending=False)
        return self

    def plot_xgb_summary(self) -> "SHAPPipeline":
        _, X_test, _, avail = self._get_xy("FD001")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(self.xgb_analyzer.shap_values, X_test.values,
                          feature_names=avail, show=False)
        ax.set_title("SHAP Summary — XGBoost — FD001", fontweight="bold")
        self._save(fig)
        return self

    def plot_xgb_bar(self) -> "SHAPPipeline":
        fig, ax = plt.subplots(figsize=(10, 8))
        self.xgb_importance.sort_values().plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
        ax.set(xlabel="Mean |SHAP value|", title="SHAP Bar — XGBoost — FD001")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_rf_summary(self) -> "SHAPPipeline":
        _, X_test, _, avail = self._get_xy("FD001")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(self.rf_analyzer.shap_values, X_test.values,
                          feature_names=avail, show=False)
        ax.set_title("SHAP Summary — Random Forest — FD001", fontweight="bold")
        self._save(fig)
        return self

    def plot_rf_bar(self) -> "SHAPPipeline":
        fig, ax = plt.subplots(figsize=(10, 8))
        self.rf_importance.sort_values().plot.barh(ax=ax, color="#DD8452", edgecolor="white")
        ax.set(xlabel="Mean |SHAP value|", title="SHAP Bar — RF — FD001")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_model_comparison(self) -> "SHAPPipeline":
        imp_df = pd.DataFrame({
            "XGBoost": self.xgb_importance,
            "Random Forest": self.rf_importance,
        })
        fig, ax = plt.subplots(figsize=(12, 8))
        imp_df.plot.barh(ax=ax, width=0.75, edgecolor="white")
        ax.set(xlabel="Mean |SHAP value|", title="XGBoost vs Random Forest")
        ax.invert_yaxis()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_dependence_top3(self) -> "SHAPPipeline":
        _, X_test, _, avail = self._get_xy("FD001")
        top3 = self.xgb_importance.head(3).index.tolist()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, sensor in enumerate(top3):
            shap.dependence_plot(sensor, self.xgb_analyzer.shap_values,
                                 X_test.values, feature_names=avail,
                                 ax=axes[i], show=False)
            axes[i].set_title(sensor, fontweight="bold")
        plt.suptitle("SHAP Dependence — Top 3 — XGBoost", fontsize=14, y=1.03)
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_waterfall(self, motor_idx: int = 0) -> "SHAPPipeline":
        _, X_test, y_test, avail = self._get_xy("FD001")
        xgb_exp  = shap.TreeExplainer(self.xgb_model)
        shap_obj = xgb_exp(X_test)
        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_obj[motor_idx], show=False)
        plt.title(f"Motor {motor_idx} — RUL real: {y_test[motor_idx]}", fontsize=12)
        self._save(fig)
        return self

    def plot_heatmap(self) -> "SHAPPipeline":
        _, X_test, _, avail = self._get_xy("FD001")
        shap_df = pd.DataFrame(self.xgb_analyzer.shap_values, columns=avail)
        sorted_ = shap_df.iloc[shap_df.sum(axis=1).argsort()]
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(sorted_.T, cmap="RdBu_r", center=0,
                    xticklabels=False, yticklabels=True, ax=ax)
        ax.set(xlabel="Motores", title="SHAP Heatmap — XGBoost")
        plt.tight_layout()
        self._save(fig)
        return self

    def compute_cross_dataset_shap(self) -> "SHAPPipeline":
        for ds_id in DATASETS:
            X_train, X_test, _, avail = self._get_xy(ds_id)
            model = XGBRegressor(n_estimators=300, max_depth=6,
                                 learning_rate=0.05, random_state=42, n_jobs=-1)
            model.fit(X_train.values,
                      self.processed[ds_id]["train"]["rul"].values)
            sv = shap.TreeExplainer(model)(X_test)
            self.shap_by_dataset[ds_id] = pd.Series(
                np.abs(sv.values).mean(axis=0), index=avail)
        return self

    def plot_cross_dataset_importance(self) -> "SHAPPipeline":
        shap_cross = pd.DataFrame(self.shap_by_dataset)
        fig, ax    = plt.subplots(figsize=(14, 8))
        shap_cross.plot.barh(ax=ax, width=0.75, edgecolor="white")
        ax.set(xlabel="Mean |SHAP value|", title="SHAP cross-dataset")
        ax.invert_yaxis()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_consistency(self) -> "SHAPPipeline":
        shap_cross = pd.DataFrame(self.shap_by_dataset)
        top5_per   = {ds: set(shap_cross[ds].nlargest(5).index) for ds in DATASETS}
        all_top5   = set().union(*top5_per.values())
        cons       = {s: sum(1 for ds in DATASETS if s in top5_per[ds]) for s in all_top5}
        cons_df    = pd.Series(cons).sort_values(ascending=False)
        cmap       = {4: "#2ca02c", 3: "#ff7f0e", 2: "#1f77b4", 1: "#d62728"}
        fig, ax    = plt.subplots(figsize=(10, 6))
        cons_df.plot.barh(ax=ax,
                          color=[cmap[v] for v in cons_df.values],
                          edgecolor="white")
        ax.set(xlabel="Aparece en top 5 de N datasets", title="Consistencia")
        ax.invert_yaxis()
        plt.tight_layout()
        self._save(fig)
        return self

    def run(self) -> "SHAPPipeline":
        return (self
                .load_data()
                .compute_shap()
                .plot_xgb_summary()
                .plot_xgb_bar()
                .plot_rf_summary()
                .plot_rf_bar()
                .plot_model_comparison()
                .plot_dependence_top3()
                .plot_waterfall()
                .plot_heatmap()
                .compute_cross_dataset_shap()
                .plot_cross_dataset_importance()
                .plot_consistency())


if __name__ == "__main__":
    SHAPPipeline().run()
