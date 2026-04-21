"""
Evaluator — NASA C-MAPSS
Centraliza el calculo de todas las metricas de evaluacion
para modelos de regresion y clasificacion.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)


class Evaluator:
    """
    Calcula y almacena metricas de evaluacion de modelos.

    Attributes
    ----------
    regression_results : dict
        Resultados de metricas de regresion por modelo.
    classification_results : dict
        Resultados de metricas de clasificacion por modelo.
    """

    def __init__(self):
        self.regression_results     = {}
        self.classification_results = {}

    def nasa_score(self, y_true: np.ndarray,
                   y_pred: np.ndarray) -> float:
        """
        Calcula la NASA Scoring Function asimetrica.

        Para errores negativos (prediccion temprana, conservadora):
            s = exp(-d/13) - 1
        Para errores positivos (prediccion tardia, peligrosa):
            s = exp(d/10) - 1

        La funcion penaliza 12x mas las predicciones tardias porque
        en la realidad significan no hacer mantenimiento a tiempo.

        Parameters
        ----------
        y_true : np.ndarray
            RUL real en ciclos.
        y_pred : np.ndarray
            RUL predicho en ciclos.

        Returns
        -------
        float
            NASA Score total (0 es perfecto, mayor es peor).
        """
        d = np.array(y_pred) - np.array(y_true)
        scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
        return float(np.sum(scores))

    def regression_metrics(self, y_true: np.ndarray,
                           y_pred: np.ndarray,
                           name: str = "model") -> dict:
        """
        Calcula RMSE, MAE, R2 y NASA Score para un modelo de regresion.

        Parameters
        ----------
        y_true : np.ndarray
            RUL real en ciclos.
        y_pred : np.ndarray
            RUL predicho en ciclos.
        name : str
            Nombre del modelo para almacenar resultados.

        Returns
        -------
        dict
            Diccionario con las 4 metricas calculadas.
        """
        metrics = {
            "rmse":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae":        float(mean_absolute_error(y_true, y_pred)),
            "r2":         float(r2_score(y_true, y_pred)),
            "nasa_score": self.nasa_score(y_true, y_pred),
        }
        self.regression_results[name] = metrics
        return metrics

    def classification_metrics(self, y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_prob: np.ndarray,
                               name: str = "model") -> dict:
        """
        Calcula Accuracy, Precision, Recall, F1 y AUC-ROC.

        Parameters
        ----------
        y_true : np.ndarray
            Labels reales (0/1).
        y_pred : np.ndarray
            Predicciones binarias (0/1).
        y_prob : np.ndarray
            Probabilidades de la clase positiva.
        name : str
            Nombre del modelo.

        Returns
        -------
        dict
            Diccionario con las 5 metricas de clasificacion.
        """
        metrics = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred,
                                               zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred,
                                            zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred,
                                        zero_division=0)),
            "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        }
        self.classification_results[name] = metrics
        return metrics

    def accuracy_within_tolerance(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   tolerance: int = 10) -> float:
        """
        Calcula el porcentaje de predicciones dentro de +/- tolerance ciclos.

        Metrica intuitiva para presentar en la app:
        'el modelo acerto en 73 de 100 motores dentro de +/-10 ciclos'.

        Parameters
        ----------
        y_true : np.ndarray
            RUL real.
        y_pred : np.ndarray
            RUL predicho.
        tolerance : int
            Tolerancia en ciclos.

        Returns
        -------
        float
            Porcentaje de predicciones dentro de la tolerancia [0, 1].
        """
        abs_error = np.abs(np.array(y_pred) - np.array(y_true))
        return float((abs_error <= tolerance).mean())

    def print_regression_results(self) -> "Evaluator":
        """Imprime resumen de resultados de regresion."""
        print("=" * 60)
        print("  METRICAS DE REGRESION")
        print("=" * 60)
        for name, metrics in self.regression_results.items():
            print(f"  {name}:")
            print(f"    RMSE:       {metrics['rmse']:.4f}")
            print(f"    MAE:        {metrics['mae']:.4f}")
            print(f"    R2:         {metrics['r2']:.4f}")
            print(f"    NASA Score: {metrics['nasa_score']:.2f}")
        print("=" * 60)
        return self

    def print_classification_results(self) -> "Evaluator":
        """Imprime resumen de resultados de clasificacion."""
        print("=" * 60)
        print("  METRICAS DE CLASIFICACION")
        print("=" * 60)
        for name, metrics in self.classification_results.items():
            print(f"  {name}:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")
        print("=" * 60)
        return self
