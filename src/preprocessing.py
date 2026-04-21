"""
Preprocessor — NASA C-MAPSS
Limpieza, normalizacion por condicion operacional, calculo de RUL
y scaling de features.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.config import (DROP_SENSORS, USEFUL_SENSORS,
                        MAX_RUL, CLASSIFICATION_W)


class Preprocessor:
    """
    Transforma los datos crudos de NASA C-MAPSS en datos limpios
    y normalizados listos para el modelado.

    Attributes
    ----------
    max_rul : int
        Techo del RUL para el modelo piece-wise lineal.
    classification_w : int
        Umbral en ciclos para el label binario de clasificacion.
    scaler : MinMaxScaler
        Scaler ajustado sobre los datos de entrenamiento.
    feature_cols : list
        Columnas de features usadas para el scaling.
    """

    def __init__(self, max_rul: int = MAX_RUL,
                 classification_w: int = CLASSIFICATION_W):
        self.max_rul          = max_rul
        self.classification_w = classification_w
        self.scaler           = MinMaxScaler()
        self.feature_cols     = []

    def drop_useless_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina los sensores constantes y op_setting_3.

        Los sensores en DROP_SENSORS tienen varianza < 0.001 en todos
        los sub-datasets y no aportan informacion para detectar degradacion.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con las 26 columnas originales.

        Returns
        -------
        pd.DataFrame
            DataFrame con 19 columnas utiles.
        """
        cols_to_drop = DROP_SENSORS + ["op_setting_3"]
        return df.drop(
            columns=[c for c in cols_to_drop if c in df.columns]
        )

    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el target de regresion RUL con cap de max_rul ciclos.

        Para cada motor busca su ciclo maximo, resta el ciclo actual
        y aplica el techo (piece-wise linear model).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrenamiento con columnas unit_id y cycle.

        Returns
        -------
        pd.DataFrame
            DataFrame con columna 'rul' agregada.
        """
        df = df.copy()
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        df["rul"] = (max_cycle - df["cycle"]).clip(upper=self.max_rul)
        return df

    def compute_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea el target de clasificacion binaria.

        Asigna 1 si RUL <= classification_w (motor critico),
        0 si no (motor normal).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columna 'rul' ya calculada.

        Returns
        -------
        pd.DataFrame
            DataFrame con columna 'label' agregada.
        """
        df = df.copy()
        df["label"] = (df["rul"] <= self.classification_w).astype(int)
        return df

    def normalize_by_operating_condition(self,
                                          df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza sensores dentro de cada cluster de condicion operacional.

        Aplica z-score normalization por grupo de (op_setting_1, op_setting_2).
        Critico para FD002 y FD004 donde hay 6 condiciones distintas.
        Sin esto el modelo confunde cambios de condicion con degradacion.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas de sensores y op_settings.

        Returns
        -------
        pd.DataFrame
            DataFrame con sensores normalizados por condicion.
        """
        df = df.copy()
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        condition_cols = ["op_setting_1", "op_setting_2"]
        conditions = df[condition_cols].drop_duplicates()

        for _, cond in conditions.iterrows():
            mask = (
                (df["op_setting_1"] == cond["op_setting_1"]) &
                (df["op_setting_2"] == cond["op_setting_2"])
            )
            subset = df[mask]
            for col in sensor_cols:
                mean = subset[col].mean()
                std  = subset[col].std()
                if std > 0:
                    df.loc[mask, col] = (df.loc[mask, col] - mean) / std

        return df

    def fit_scaler(self, train_df: pd.DataFrame,
                   feature_cols: list) -> "Preprocessor":
        """
        Ajusta el MinMaxScaler sobre los datos de entrenamiento.

        Solo debe llamarse con datos de train para evitar data leakage.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame de entrenamiento.
        feature_cols : list
            Columnas a escalar.

        Returns
        -------
        Preprocessor
            La misma instancia para encadenamiento.
        """
        self.feature_cols = feature_cols
        self.scaler.fit(train_df[feature_cols])
        return self

    def apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el scaler ya ajustado a un DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a escalar (train o test).

        Returns
        -------
        pd.DataFrame
            DataFrame con features escaladas a [0, 1].
        """
        df = df.copy()
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df

    def process_train(self, df: pd.DataFrame,
                      ds_id: str = "FD001") -> pd.DataFrame:
        """
        Pipeline completo de preprocesamiento para datos de entrenamiento.

        Aplica en orden: drop columnas, RUL, label binario,
        normalizacion por condicion (si aplica) y scaling.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrenamiento crudo.
        ds_id : str
            ID del sub-dataset para decidir si normalizar por condicion.

        Returns
        -------
        pd.DataFrame
            DataFrame procesado listo para feature engineering.
        """
        df = self.drop_useless_columns(df)
        df = self.compute_rul(df)
        df = self.compute_binary_label(df)
        if ds_id in ["FD002", "FD004"]:
            df = self.normalize_by_operating_condition(df)
        return df

    def process_test(self, df: pd.DataFrame,
                     ds_id: str = "FD001") -> pd.DataFrame:
        """
        Pipeline de preprocesamiento para datos de test.

        No calcula RUL (viene del archivo RUL_FDxxx.txt).
        Aplica las mismas transformaciones que train excepto RUL/label.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de test crudo.
        ds_id : str
            ID del sub-dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame de test procesado.
        """
        df = self.drop_useless_columns(df)
        if ds_id in ["FD002", "FD004"]:
            df = self.normalize_by_operating_condition(df)
        return df


# Funciones de compatibilidad para codigo existente
def drop_useless_columns(df):
    return Preprocessor().drop_useless_columns(df)

def add_rul_column(df, max_rul=MAX_RUL):
    return Preprocessor(max_rul=max_rul).compute_rul(df)

def add_binary_label(df, w=CLASSIFICATION_W):
    return Preprocessor(classification_w=w).compute_binary_label(df)

def normalize_by_operating_condition(df):
    return Preprocessor().normalize_by_operating_condition(df)


if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.config import DATASETS

    loader = DataLoader()
    loader.load_all()
    prep   = Preprocessor()

    for ds_id in DATASETS:
        train, test, rul = loader.data[ds_id]
        train_proc = prep.process_train(train, ds_id)
        test_proc  = prep.process_test(test, ds_id)
        print(f"{ds_id} — train: {train_proc.shape}, test: {test_proc.shape}")
