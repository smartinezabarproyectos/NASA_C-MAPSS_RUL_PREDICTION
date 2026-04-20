import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    MAX_RUL,
    CLASSIFICATION_W,
    DROP_SENSORS,
    DROP_SETTINGS,
    USEFUL_SENSORS,
)


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = (max_cycles - df["cycle"]).clip(upper=MAX_RUL)
    return df


def add_binary_label(df: pd.DataFrame, w: int = CLASSIFICATION_W) -> pd.DataFrame:

    df = df.copy()
    df["label"] = (df["rul"] <= w).astype(int)
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:

    cols_to_drop = [c for c in DROP_SENSORS + DROP_SETTINGS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def normalize_sensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, MinMaxScaler]:
    """Normaliza sensores con MinMaxScaler ajustado en train.

    Returns
    -------
    train_norm, test_norm (o None), scaler
    """
    scaler = MinMaxScaler()
    sensor_cols = [c for c in USEFUL_SENSORS if c in train_df.columns]

    train_out = train_df.copy()
    train_out[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])

    test_out = None
    if test_df is not None:
        test_out = test_df.copy()
        test_out[sensor_cols] = scaler.transform(test_df[sensor_cols])

    return train_out, test_out, scaler


def preprocess_train(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de preprocesamiento para train."""
    df = add_rul_column(df)
    df = add_binary_label(df)
    df = drop_useless_columns(df)
    return df


def preprocess_test(
    test_df: pd.DataFrame,
    rul_series: pd.Series,
) -> pd.DataFrame:
    """Pipeline de preprocesamiento para test.

    Usa solo el último ciclo de cada motor y agrega el RUL real.
    """
    test_df = drop_useless_columns(test_df)

    # Último ciclo de cada motor
    last_cycle = test_df.groupby("unit_id").last().reset_index()
    last_cycle["rul"] = rul_series.values
    last_cycle["label"] = (last_cycle["rul"] <= CLASSIFICATION_W).astype(int)

    return last_cycle


def normalize_by_operating_condition(df: pd.DataFrame, op_cols=None) -> pd.DataFrame:
    if op_cols is None:
        op_cols = ["op_setting_1", "op_setting_2"]

    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    df["op_cluster"] = df[op_cols].round(2).astype(str).agg("_".join, axis=1)

    for col in sensor_cols:
        group_mean = df.groupby("op_cluster")[col].transform("mean")
        group_std = df.groupby("op_cluster")[col].transform("std").replace(0, 1)
        df[col] = (df[col] - group_mean) / group_std

    df = df.drop(columns=["op_cluster"])
    return df
