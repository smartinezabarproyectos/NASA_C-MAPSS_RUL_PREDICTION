import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.config import DROP_SENSORS, MAX_RUL, CLASSIFICATION_W


class Preprocessor:
    def __init__(self, max_rul: int = MAX_RUL, classification_w: int = CLASSIFICATION_W):
        self.max_rul          = max_rul
        self.classification_w = classification_w
        self.scaler           = MinMaxScaler()
        self.feature_cols     = []

    def drop_useless_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = DROP_SENSORS + ["op_setting_3"]
        return df.drop(columns=[c for c in cols if c in df.columns])

    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        df["rul"] = (max_cycle - df["cycle"]).clip(upper=self.max_rul)
        return df

    def compute_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label"] = (df["rul"] <= self.classification_w).astype(int)
        return df


    def normalize_by_operating_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        df          = df.copy()
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        df[sensor_cols] = df[sensor_cols].astype(float)
        cond_cols   = ["op_setting_1", "op_setting_2"]
        df[sensor_cols] = (
            df.groupby(cond_cols)[sensor_cols]
            .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
        )
        return df
    def fit_scaler(self, train_df: pd.DataFrame, feature_cols: list) -> "Preprocessor":
        self.feature_cols = feature_cols
        self.scaler.fit(train_df[feature_cols])
        return self

    def apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df

    def process_train(self, df: pd.DataFrame, ds_id: str = "FD001") -> pd.DataFrame:
        df = self.drop_useless_columns(df)
        df = self.compute_rul(df)
        df = self.compute_binary_label(df)
        if ds_id in ("FD002", "FD004"):
            df = self.normalize_by_operating_condition(df)
        return df

    def process_test(self, df: pd.DataFrame, ds_id: str = "FD001") -> pd.DataFrame:
        df = self.drop_useless_columns(df)
        if ds_id in ("FD002", "FD004"):
            df = self.normalize_by_operating_condition(df)
        return df


def drop_useless_columns(df):
    return Preprocessor().drop_useless_columns(df)

def add_rul_column(df, max_rul=MAX_RUL):
    return Preprocessor(max_rul=max_rul).compute_rul(df)

def add_binary_label(df, w=CLASSIFICATION_W):
    return Preprocessor(classification_w=w).compute_binary_label(df)

def normalize_by_operating_condition(df):
    return Preprocessor().normalize_by_operating_condition(df)
