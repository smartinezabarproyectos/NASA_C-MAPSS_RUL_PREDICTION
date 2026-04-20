

import pandas as pd

from src.config import WINDOW_SIZE, USEFUL_SENSORS


def add_rolling_features(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    sensors: list[str] | None = None,
) -> pd.DataFrame:

    df = df.copy()
    sensors = sensors or [s for s in USEFUL_SENSORS if s in df.columns]

    for sensor in sensors:
        grouped = df.groupby("unit_id")[sensor]
        df[f"{sensor}_roll_mean"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{sensor}_roll_std"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )

    return df


def add_trend_features(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    sensors: list[str] | None = None,
) -> pd.DataFrame:

    df = df.copy()
    sensors = sensors or [s for s in USEFUL_SENSORS if s in df.columns]

    for sensor in sensors:
        roll_col = f"{sensor}_roll_mean"
        if roll_col not in df.columns:
            grouped = df.groupby("unit_id")[sensor]
            df[roll_col] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        df[f"{sensor}_trend"] = df.groupby("unit_id")[roll_col].diff().fillna(0)

    return df


def build_features(df: pd.DataFrame, window: int = WINDOW_SIZE) -> pd.DataFrame:
    df = add_rolling_features(df, window=window)
    df = add_trend_features(df, window=window)
    return df
