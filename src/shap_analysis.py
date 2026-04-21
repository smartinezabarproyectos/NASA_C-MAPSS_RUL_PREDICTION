"""
SHAPAnalyzer — NASA C-MAPSS
Calcula e interpreta valores SHAP para modelos Random Forest y XGBoost.
Responde que sensores fisicos son los mas importantes para la prediccion.
"""

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
import shap

from src.config import DATASETS, USEFUL_SENSORS

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "10_shap_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class SHAPAnalyzer:
    """
    Calcula e interpreta valores SHAP para modelos basados en arboles.

    Usa shap.TreeExplainer que es especifico y eficiente para
    Random Forest y XGBoost. Genera summary plots, bar plots,
    dependence plots y heatmaps de interpretabilidad.

    Attributes
    ----------
    model : object
        Modelo sklearn o XGBoost ya entrenado.
    explainer : shap.TreeExplainer
        Explainer ajustado sobre el modelo.
    shap_values : np.ndarray
        Matriz de valores SHAP (n_samples, n_features).
    feature_names : list
        Nombres de las features.
    fig_count : int
        Contador de figuras generadas.
    """

    def __init__(self, model=None, feature_names: list = None):
        self.model         = model
        self.explainer     = None
        self.shap_values   = None
        self.feature_names = feature_names or USEFUL_SENSORS
        self.fig_count     = 0

    def _save(self, fig: plt.Figure) -> None:
        """Guarda figura con nombre secuencial."""
        self.fig_count += 1
        path = FIGURES_DIR / f"10_shap_analysis_fig{self.fig_count:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardada: {path.name}")

    def fit(self, model, X: np.ndarray,
            feature_names: list = None) -> "SHAPAnalyzer":
        """
        Ajusta el TreeExplainer y calcula los valores SHAP.

        Parameters
        ----------
        model : object
            Modelo Random Forest o XGBoost entrenado.
        X : np.ndarray
            Datos de test sobre los que calcular SHAP.
        feature_names : list, optional
            Nombres de las features.

        Returns
        -------
        SHAPAnalyzer
            La misma instancia para encadenamiento.
        """
        self.model = model
        if feature_names:
            self.feature_names = feature_names
        self.explainer   = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)
        print(f"  SHAP calculado: {self.shap_values.shape}")
        return self

    def plot_summary(self, X: np.ndarray,
                     title: str = "SHAP Summary") -> "SHAPAnalyzer":
        """
        Genera el summary plot de SHAP.

        Cada punto es una muestra. La posicion horizontal es el valor SHAP,
        el color indica si el valor del feature era alto (rojo) o bajo (azul).
        El ancho del enjambre refleja la importancia global del feature.

        Parameters
        ----------
        X : np.ndarray
            Datos de test (mismos usados en fit).
        title : str
            Titulo del grafico.

        Returns
        -------
        SHAPAnalyzer
            La misma instancia para encadenamiento.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, X,
            feature_names=self.feature_names,
            show=False, plot_size=None,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        self._save(fig)
        return self

    def plot_bar(self, title: str = "SHAP Feature Importance") -> "SHAPAnalyzer":
        """
        Genera el bar plot de importancia media absoluta SHAP.

        Promedio del valor absoluto de SHAP por feature.
        Da una vision rapida y clara del ranking de importancia global.

        Parameters
        ----------
        title : str
            Titulo del grafico.

        Returns
        -------
        SHAPAnalyzer
            La misma instancia para encadenamiento.
        """
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)
        sorted_names  = [self.feature_names[i] for i in sorted_idx]
        sorted_values = mean_abs[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(sorted_names, sorted_values,
                color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_dependence(self, feature: str,
                        X: np.ndarray) -> "SHAPAnalyzer":
        """
        Genera el dependence plot para un feature especifico.

        Muestra la relacion entre el valor del feature y su SHAP value.
        Revela si la relacion es lineal o no lineal (threshold, saturacion).

        Parameters
        ----------
        feature : str
            Nombre del feature a analizar.
        X : np.ndarray
            Datos de test.

        Returns
        -------
        SHAPAnalyzer
            La misma instancia para encadenamiento.
        """
        if feature not in self.feature_names:
            print(f"  Feature '{feature}' no encontrado.")
            return self

        feat_idx = self.feature_names.index(feature)
        fig, ax  = plt.subplots(figsize=(10, 6))
        ax.scatter(X[:, feat_idx], self.shap_values[:, feat_idx],
                   alpha=0.5, c=X[:, feat_idx], cmap="RdYlGn", s=20)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel(f"SHAP value ({feature})")
        ax.set_title(f"SHAP Dependence — {feature}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_heatmap(self, title: str = "SHAP Heatmap") -> "SHAPAnalyzer":
        """
        Genera un heatmap de valores SHAP para todas las muestras.

        Filas = muestras ordenadas por prediccion, columnas = features.
        Permite ver patrones de que features dominan en motores criticos vs sanos.

        Parameters
        ----------
        title : str
            Titulo del grafico.

        Returns
        -------
        SHAPAnalyzer
            La misma instancia para encadenamiento.
        """
        import seaborn as sns
        shap_df = pd.DataFrame(
            self.shap_values, columns=self.feature_names
        )
        row_order = shap_df.sum(axis=1).argsort()
        shap_sorted = shap_df.iloc[row_order]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            shap_sorted.T, cmap="RdBu_r", center=0,
            xticklabels=False, yticklabels=True, ax=ax,
            cbar_kws={"label": "SHAP value"},
        )
        ax.set_xlabel("Muestras (ordenadas por prediccion)")
        ax.set_ylabel("Feature")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def get_top_features(self, n: int = 5) -> list:
        """
        Retorna los N features mas importantes segun SHAP.

        Parameters
        ----------
        n : int
            Numero de features a retornar.

        Returns
        -------
        list
            Lista de tuplas (feature_name, mean_abs_shap) ordenadas
            de mayor a menor importancia.
        """
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]
        return [
            (self.feature_names[i], float(mean_abs[i]))
            for i in sorted_idx[:n]
        ]

    def cross_dataset_comparison(self, models: dict,
                                 X_tests: dict) -> "SHAPAnalyzer":
        """
        Compara importancia SHAP de sensor_11 y sensor_4 en los 4 datasets.

        Parameters
        ----------
        models : dict
            Diccionario {ds_id: modelo_entrenado}.
        X_tests : dict
            Diccionario {ds_id: X_test}.

        Returns
        -------
        SHAPAnalyzer
            La misma instancia para encadenamiento.
        """
        import seaborn as sns
        all_importances = []
        for ds_id in DATASETS:
            if ds_id not in models or ds_id not in X_tests:
                continue
            exp   = shap.TreeExplainer(models[ds_id])
            sv    = exp.shap_values(X_tests[ds_id])
            ma    = np.abs(sv).mean(axis=0)
            for i, feat in enumerate(self.feature_names):
                all_importances.append({
                    "dataset": ds_id,
                    "feature": feat,
                    "importance": float(ma[i]),
                })
        imp_df = pd.DataFrame(all_importances)
        top_feats = (
            imp_df.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .head(8)
            .index.tolist()
        )
        pivot = imp_df[imp_df["feature"].isin(top_feats)].pivot(
            index="feature", columns="dataset", values="importance"
        )
        fig, ax = plt.subplots(figsize=(12, 7))
        pivot.plot(kind="bar", ax=ax, width=0.75, edgecolor="white")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title("Importancia SHAP — Top features — 4 sub-datasets",
                     fontsize=14, fontweight="bold")
        ax.legend(title="Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save(fig)
        return self

    def print_report(self) -> "SHAPAnalyzer":
        """Imprime el ranking de los top 5 features por importancia SHAP."""
        top5 = self.get_top_features(n=5)
        print("=" * 60)
        print("  RESUMEN SHAP ANALYSIS")
        print("=" * 60)
        print("  Top 5 features por importancia:")
        for i, (feat, val) in enumerate(top5, 1):
            print(f"    {i}. {feat}: {val:.4f}")
        print(f"  Figuras generadas: {self.fig_count}")
        print(f"  Guardadas en: {FIGURES_DIR}")
        print("=" * 60)
        return self
