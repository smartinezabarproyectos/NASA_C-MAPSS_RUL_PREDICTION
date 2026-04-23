import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from src.config import USEFUL_SENSORS


class FeatureEngineer:
    def __init__(self, window: int = 30, sensors: list = None):
        self.window       = window
        self.sensors      = sensors or USEFUL_SENSORS
        self.feature_cols = []

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for sensor in self.sensors:
            if sensor not in df.columns:
                continue
            roll = df.groupby("unit_id")[sensor].rolling(self.window, min_periods=1)
            df[f"{sensor}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{sensor}_roll_std"]  = roll.std().reset_index(level=0, drop=True).fillna(0)
            df[f"{sensor}_trend"]     = df.groupby("unit_id")[f"{sensor}_roll_mean"].diff().fillna(0)
        return df

    def get_feature_cols(self, df: pd.DataFrame, exclude: set = None) -> list:
        base = {"unit_id", "cycle", "rul", "label"} | (exclude or set())
        self.feature_cols = [c for c in df.columns if c not in base]
        return self.feature_cols

    def get_ml_feature_cols(self, df: pd.DataFrame) -> list:
        exclude = {"unit_id", "cycle", "rul", "label"}
        return [c for c in df.columns if c not in exclude
                and "roll_mean" not in c
                and "roll_std" not in c
                and "trend" not in c]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_rolling_features(df)
        self.feature_cols = self.get_feature_cols(df)
        return df


def add_rolling_features(df, window=30):
    return FeatureEngineer(window=window).add_rolling_features(df)
