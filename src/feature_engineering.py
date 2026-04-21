import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

from src.config import USEFUL_SENSORS


class FeatureEngineer:
    """
    Crea rolling features (media, std, tendencia) para cada sensor.

    Expande el dataset de 16 features base a 58 features totales.
    Cada sensor genera 3 features adicionales calculadas sobre una
    ventana deslizante de 'window' ciclos por motor.

    Attributes
    ----------
    window : int
        Tamano de la ventana deslizante en ciclos.
    sensors : list
        Lista de sensores sobre los que calcular rolling features.
    feature_cols : list
        Lista de todas las columnas de features disponibles despues
        del feature engineering (base + rolling).
    """

    def __init__(self, window: int = 30, sensors: list = None):
        self.window      = window
        self.sensors     = sensors if sensors else USEFUL_SENSORS
        self.feature_cols = []

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega rolling mean, rolling std y trend para cada sensor.

        Procesa cada motor por separado con groupby('unit_id') para
        no mezclar historiales entre motores. Los primeros (window-1)
        ciclos de cada motor reciben backward fill.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas de sensores y unit_id.

        Returns
        -------
        pd.DataFrame
            DataFrame con 3*len(sensors) columnas adicionales.
        """
        df = df.copy()
        for sensor in self.sensors:
            if sensor not in df.columns:
                continue
            roll = df.groupby("unit_id")[sensor].rolling(
                self.window, min_periods=1
            )
            mean_col = f"{sensor}_roll_mean"
            std_col  = f"{sensor}_roll_std"
            trnd_col = f"{sensor}_trend"

            df[mean_col] = (
                roll.mean()
                .reset_index(level=0, drop=True)
            )
            df[std_col] = (
                roll.std()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )
            df[trnd_col] = (
                df.groupby("unit_id")[mean_col]
                .diff()
                .fillna(0)
            )
        return df

    def get_feature_cols(self, df: pd.DataFrame,
                         exclude: set = None) -> list:
        """
        Retorna la lista de columnas de features disponibles en df.

        Excluye columnas no-feature como unit_id, cycle, rul, label.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame procesado con rolling features.
        exclude : set, optional
            Columnas adicionales a excluir.

        Returns
        -------
        list
            Lista de columnas de features.
        """
        base_exclude = {"unit_id", "cycle", "rul", "label"}
        if exclude:
            base_exclude = base_exclude | set(exclude)
        self.feature_cols = [c for c in df.columns if c not in base_exclude]
        return self.feature_cols

    def get_ml_feature_cols(self, df: pd.DataFrame) -> list:
        """
        Retorna solo las features base (sin rolling) disponibles en test.

        Los modelos clasicos usan solo estas 16 features porque el test
        de NASA solo tiene el ultimo ciclo sin historial.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrenamiento procesado.

        Returns
        -------
        list
            Lista de 16 features base.
        """
        exclude = {"unit_id", "cycle", "rul", "label"}
        return [
            c for c in df.columns
            if c not in exclude
            and "roll_mean" not in c
            and "roll_std" not in c
            and "trend" not in c
        ]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el pipeline completo de feature engineering.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame preprocesado de entrenamiento.

        Returns
        -------
        pd.DataFrame
            DataFrame con todas las features.
        """
        df = self.add_rolling_features(df)
        self.feature_cols = self.get_feature_cols(df)
        return df


# Funcion de compatibilidad para codigo existente
def add_rolling_features(df, window=30):
    return FeatureEngineer(window=window).add_rolling_features(df)


if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor

    loader = DataLoader()
    loader.load_all()
    prep   = Preprocessor()
    fe     = FeatureEngineer(window=30)

    train, _, _ = loader.data["FD001"]
    train_proc  = prep.process_train(train, "FD001")
    train_fe    = fe.fit_transform(train_proc)

    print(f"Features base: {len(fe.get_ml_feature_cols(train_proc))}")
    print(f"Features totales: {len(fe.feature_cols)}")
    print(f"Shape final: {train_fe.shape}")
