"""
Visualizer — NASA C-MAPSS
Funciones reutilizables para generar graficos con estilo consistente
en todos los notebooks y el pipeline de resultados.
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
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config import DATASETS, USEFUL_SENSORS, CLASSIFICATION_W, MAX_RUL

FIGURES_DIR = Path(ROOT) / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


class Visualizer:
    """
    Genera graficos de resultados con estilo consistente para el proyecto.

    Centraliza toda la logica de visualizacion para evitar duplicacion
    entre notebooks. Todos los metodos guardan las figuras en disco
    con nombre descriptivo y devuelven la instancia para encadenamiento.

    Attributes
    ----------
    figures_dir : Path
        Carpeta raiz donde se guardan las figuras.
    fig_count : int
        Contador de figuras generadas.
    prefix : str
        Prefijo para los nombres de archivo (ej: '08_classical_ml').
    """

    def __init__(self, prefix: str = "viz",
                 figures_dir: Path = None):
        self.figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.prefix    = prefix
        self.fig_count = 0
        sns.set_theme(style="whitegrid")

    def _save(self, fig: plt.Figure,
              name: str = None) -> None:
        """Guarda figura con nombre secuencial o nombre especifico."""
        self.fig_count += 1
        filename = name if name else f"{self.prefix}_fig{self.fig_count:02d}.png"
        path = self.figures_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardada: {path.name}")

    # ------------------------------------------------------------------
    # DEGRADACION
    # ------------------------------------------------------------------

    def plot_degradation_curves(self, df: pd.DataFrame,
                                sensor: str,
                                n_motors: int = 10,
                                ds_id: str = "FD001") -> "Visualizer":
        """
        Curvas de degradacion de un sensor para N motores.

        Usa tiempo normalizado (% vida) para comparar motores
        de diferente duracion en el mismo eje X.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de train con columna RUL calculada.
        sensor : str
            Nombre del sensor a graficar.
        n_motors : int
            Numero de motores a mostrar.
        ds_id : str
            ID del sub-dataset para el titulo.
        """
        units = df["unit_id"].unique()[:n_motors]
        fig, ax = plt.subplots(figsize=(14, 6))
        for unit in units:
            ud = df[df["unit_id"] == unit].copy()
            ud["life_pct"] = ud["cycle"] / ud["cycle"].max() * 100
            ax.plot(ud["life_pct"], ud[sensor],
                    alpha=0.6, linewidth=0.9)
        ax.set_xlabel("% vida util")
        ax.set_ylabel(sensor)
        ax.set_title(f"Curvas de degradacion — {sensor} — {ds_id}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_mean_degradation(self, df: pd.DataFrame,
                              sensors: list,
                              ds_id: str = "FD001") -> "Visualizer":
        """
        Curva media de degradacion con banda +/- 1 STD por sensor.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columna life_pct ya calculada.
        sensors : list
            Lista de sensores a graficar.
        ds_id : str
            ID del sub-dataset para el titulo.
        """
        n_cols = 3
        n_rows = (len(sensors) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        for i, sensor in enumerate(sensors):
            grouped = df.groupby("life_pct")[sensor]
            mean    = grouped.mean()
            std     = grouped.std()
            axes[i].plot(mean.index, mean.values,
                         color="#4C72B0", linewidth=2)
            axes[i].fill_between(mean.index, mean - std, mean + std,
                                 alpha=0.2, color="#4C72B0")
            axes[i].set_title(sensor, fontsize=12, fontweight="bold")
            axes[i].set_xlabel("% vida util")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f"Degradacion promedio +/- 1 STD — {ds_id}",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        self._save(fig)
        return self

    # ------------------------------------------------------------------
    # CORRELACIONES
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                 cols: list,
                                 title: str = "Matriz de correlacion") -> "Visualizer":
        """
        Heatmap de correlaciones de Pearson entre las columnas indicadas.

        Muestra solo el triangulo inferior para evitar redundancia.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con los datos.
        cols : list
            Columnas a incluir en la matriz.
        title : str
            Titulo del grafico.
        """
        corr = df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True,
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    # ------------------------------------------------------------------
    # RESULTADOS DE MODELOS
    # ------------------------------------------------------------------

    def plot_predicted_vs_actual(self, y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str = "Modelo") -> "Visualizer":
        """
        Scatter plot de RUL predicho vs real.

        Colorea los puntos segun si el error es positivo (prediccion tardia,
        peligrosa) o negativo (prediccion temprana, conservadora).
        La linea diagonal representa prediccion perfecta.

        Parameters
        ----------
        y_true : np.ndarray
            RUL real en ciclos.
        y_pred : np.ndarray
            RUL predicho en ciclos.
        model_name : str
            Nombre del modelo para el titulo.
        """
        error  = y_pred - y_true
        colors = np.where(error >= 0, "#C44E52", "#4C72B0")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, c=colors, alpha=0.6, s=30)
        lim = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfecto")
        ax.set_xlabel("RUL Real (ciclos)")
        ax.set_ylabel("RUL Predicho (ciclos)")
        ax.set_title(f"Predicho vs Real — {model_name}",
                     fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_error_distribution(self, y_true: np.ndarray,
                                y_pred: np.ndarray,
                                model_name: str = "Modelo") -> "Visualizer":
        """
        Histograma de la distribucion del error de prediccion.

        La distribucion ideal es una campana centrada en 0.
        Desviaciones positivas son mas peligrosas (predicciones tardias).

        Parameters
        ----------
        y_true : np.ndarray
            RUL real.
        y_pred : np.ndarray
            RUL predicho.
        model_name : str
            Nombre del modelo.
        """
        error = y_pred - y_true
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(error, bins=40, color="#4C72B0",
                alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--",
                   linewidth=1.5, label="Error = 0")
        ax.axvline(error.mean(), color="orange", linestyle=":",
                   linewidth=1.5, label=f"Media = {error.mean():.1f}")
        ax.set_xlabel("Error (predicho - real)")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribucion del error — {model_name}",
                     fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_confusion_matrix(self, y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model_name: str = "Modelo") -> "Visualizer":
        """
        Matriz de confusion para el problema de clasificacion binaria.

        Muestra TP, TN, FP, FN con colores intuitivos.
        Los falsos negativos (motor critico clasificado como normal)
        son los mas peligrosos operacionalmente.

        Parameters
        ----------
        y_true : np.ndarray
            Labels reales (0/1).
        y_pred : np.ndarray
            Predicciones binarias (0/1).
        model_name : str
            Nombre del modelo.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Critico"],
            yticklabels=["Normal", "Critico"],
            ax=ax, linewidths=1,
        )
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusion — {model_name}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_roc_curve(self, y_true: np.ndarray,
                       y_prob: np.ndarray,
                       model_name: str = "Modelo") -> "Visualizer":
        """
        Curva ROC con area bajo la curva (AUC).

        La curva ROC grafica la tasa de deteccion (recall) vs
        la tasa de falsas alarmas para todos los thresholds posibles.
        AUC = 1.0 es perfecto, AUC = 0.5 equivale a adivinar al azar.

        Parameters
        ----------
        y_true : np.ndarray
            Labels reales (0/1).
        y_prob : np.ndarray
            Probabilidades de la clase positiva.
        model_name : str
            Nombre del modelo.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score   = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#4C72B0", linewidth=2,
                label=f"AUC = {auc_score:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("Tasa de Falsos Positivos")
        ax.set_ylabel("Tasa de Verdaderos Positivos (Recall)")
        ax.set_title(f"Curva ROC — {model_name}",
                     fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_training_curves(self, train_losses: list,
                             val_losses: list,
                             model_name: str = "Modelo") -> "Visualizer":
        """
        Curvas de loss de entrenamiento y validacion por epoch.

        Permite detectar overfitting (val_loss sube mientras train_loss baja)
        y verificar que el early stopping funciono correctamente.

        Parameters
        ----------
        train_losses : list
            Loss de entrenamiento por epoch.
        val_losses : list
            Loss de validacion por epoch.
        model_name : str
            Nombre del modelo.
        """
        epochs = range(1, len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, color="#4C72B0",
                linewidth=2, label="Train Loss")
        ax.plot(epochs, val_losses, color="#C44E52",
                linewidth=2, linestyle="--", label="Val Loss")
        best_epoch = int(np.argmin(val_losses)) + 1
        ax.axvline(best_epoch, color="green", linestyle=":",
                   alpha=0.7, label=f"Mejor epoch: {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"Curvas de entrenamiento — {model_name}",
                     fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_cross_dataset_rmse(self, results: dict) -> "Visualizer":
        """
        Barras agrupadas de RMSE por modelo y sub-dataset.

        Permite comparar visualmente como se degrada el rendimiento
        de cada modelo al aumentar la complejidad del dataset.

        Parameters
        ----------
        results : dict
            Diccionario {modelo: {ds_id: rmse}}.
        """
        models   = list(results.keys())
        datasets = DATASETS
        x        = np.arange(len(models))
        width    = 0.2
        fig, ax  = plt.subplots(figsize=(14, 7))
        for i, ds_id in enumerate(datasets):
            rmse_vals = [results[m].get(ds_id, 0) for m in models]
            ax.bar(x + i * width, rmse_vals, width,
                   label=ds_id, color=COLORS[i], edgecolor="white")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE por modelo y sub-dataset",
                     fontsize=14, fontweight="bold")
        ax.legend(title="Dataset")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_feature_importance(self, model,
                                feature_names: list,
                                model_name: str = "Modelo") -> "Visualizer":
        """
        Importancia de features nativa del modelo (RF o XGBoost).

        Alternativa rapida a SHAP cuando solo se necesita un ranking
        aproximado sin el costo computacional de SHAP.

        Parameters
        ----------
        model : object
            Modelo Random Forest o XGBoost entrenado.
        feature_names : list
            Nombres de las features.
        model_name : str
            Nombre del modelo para el titulo.
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(feature_names))

        sorted_idx    = np.argsort(importances)
        sorted_names  = [feature_names[i] for i in sorted_idx]
        sorted_values = importances[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(sorted_names, sorted_values,
                color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Importancia")
        ax.set_title(f"Feature Importance — {model_name}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def plot_ensemble_comparison(self, individual_preds: dict,
                                 ensemble_preds: np.ndarray,
                                 y_true: np.ndarray) -> "Visualizer":
        """
        Compara predicciones de modelos individuales vs ensemble.

        Grafica el RMSE de cada modelo junto al del ensemble para
        mostrar la mejora obtenida por el promediado.

        Parameters
        ----------
        individual_preds : dict
            Diccionario {model_name: y_pred}.
        ensemble_preds : np.ndarray
            Predicciones del ensemble.
        y_true : np.ndarray
            RUL real.
        """
        names  = list(individual_preds.keys()) + ["Ensemble"]
        rmses  = [
            float(np.sqrt(np.mean((p - y_true) ** 2)))
            for p in individual_preds.values()
        ]
        rmses.append(float(np.sqrt(np.mean((ensemble_preds - y_true) ** 2))))
        colors = ["#4C72B0"] * len(individual_preds) + ["#FFA15A"]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, rmses, color=colors, edgecolor="white")
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center",
                    fontsize=11, fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.set_title("Modelos individuales vs Ensemble",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig)
        return self

    def print_report(self) -> "Visualizer":
        """Imprime resumen de figuras generadas."""
        print("=" * 60)
        print(f"  Visualizer — {self.prefix}")
        print(f"  Figuras generadas: {self.fig_count}")
        print(f"  Guardadas en: {self.figures_dir}")
        print("=" * 60)
        return self
