import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.config import COLUMN_NAMES, DATASETS, USEFUL_SENSORS, DROP_SENSORS

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "01_eda_basico"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class EDABasico:
    def __init__(self):
        self.loader          = DataLoader()
        self.summary_df      = None
        self.null_df         = None
        self.life_df         = None
        self.var_df          = None
        self.constant_sensors = None
        self.rul_df          = None
        self.loader.load_all()

    def compute_summary(self) -> "EDABasico":
        rows = []
        for ds_id in DATASETS:
            train, test, rul = self.loader.data[ds_id]
            rows.append({
                "dataset":       ds_id,
                "train_rows":    train.shape[0],
                "train_engines": train["unit_id"].nunique(),
                "test_rows":     test.shape[0],
                "test_engines":  test["unit_id"].nunique(),
                "rul_count":     len(rul),
                "train_cols":    train.shape[1],
            })
        self.summary_df = pd.DataFrame(rows).set_index("dataset")
        return self

    def compute_nulls(self) -> "EDABasico":
        report = {}
        for ds_id in DATASETS:
            train, test, _ = self.loader.data[ds_id]
            report[f"{ds_id}_train"] = train.isnull().sum().sum()
            report[f"{ds_id}_test"]  = test.isnull().sum().sum()
        self.null_df = pd.DataFrame.from_dict(report, orient="index", columns=["total_nulls"])
        return self

    def compute_life_spans(self) -> "EDABasico":
        spans = {}
        for ds_id in DATASETS:
            cycles = self.loader.data[ds_id][0].groupby("unit_id")["cycle"].max()
            spans[ds_id] = {
                "min": cycles.min(), "max": cycles.max(),
                "mean": cycles.mean(), "median": cycles.median(), "std": cycles.std(),
            }
        self.life_df = pd.DataFrame(spans).T
        return self

    def compute_variance(self) -> "EDABasico":
        sensor_cols = [c for c in COLUMN_NAMES if c.startswith("sensor_") or c.startswith("op_setting_")]
        report = {ds_id: self.loader.data[ds_id][0][sensor_cols].var() for ds_id in DATASETS}
        self.var_df = pd.DataFrame(report)
        self.var_df["is_constant"] = (self.var_df < 0.001).all(axis=1)
        self.constant_sensors = self.var_df[self.var_df["is_constant"]].index.tolist()
        return self

    def compute_op_conditions(self) -> "EDABasico":
        op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
        for ds_id in DATASETS:
            train = self.loader.data[ds_id][0]
            unique = train[op_cols].drop_duplicates().shape[0]
            print(f"  {ds_id}: {unique} condiciones operacionales unicas")
            if unique <= 6:
                print(train[op_cols].drop_duplicates().sort_values(op_cols).reset_index(drop=True).to_string())
        return self

    def compute_rul_summary(self) -> "EDABasico":
        summary = {}
        for ds_id in DATASETS:
            rul = self.loader.data[ds_id][2]
            summary[ds_id] = {
                "min": rul.min(), "max": rul.max(), "mean": rul.mean(),
                "median": rul.median(), "std": rul.std(),
                "q25": rul.quantile(0.25), "q75": rul.quantile(0.75),
            }
        self.rul_df = pd.DataFrame(summary).T
        return self

    def inspect_engine(self, ds_id: str = "FD001", unit_id: int = 1) -> "EDABasico":
        engine = self.loader.data[ds_id][0]
        engine = engine[engine["unit_id"] == unit_id]
        print(f"\nMotor {unit_id} — {len(engine)} ciclos — {ds_id}")
        print(engine.head().to_string())
        print(engine.tail().to_string())
        return self

    def print_report(self) -> "EDABasico":
        n = len(self.constant_sensors) if self.constant_sensors else "?"
        print("=" * 60)
        print(f"  Sub-datasets:        {len(DATASETS)}")
        print(f"  Columnas por archivo: {len(COLUMN_NAMES)}")
        print(f"  Valores nulos:        0 en todos")
        print(f"  Sensores constantes:  {n} -> se eliminan")
        print(f"  Sensores utiles:      {len(USEFUL_SENSORS)}")
        print(f"  FD001/FD003:          1 condicion operacional")
        print(f"  FD002/FD004:          6 condiciones operacionales")
        print(f"  FD001/FD002:          1 modo de falla (HPC)")
        print(f"  FD003/FD004:          2 modos de falla (HPC + Fan)")
        print("=" * 60)
        if self.summary_df is not None:
            print(self.summary_df.to_string())
        if self.life_df is not None:
            print(self.life_df.to_string())
        if self.rul_df is not None:
            print(self.rul_df.to_string())
        return self

    def run(self) -> "EDABasico":
        return (self
                .compute_summary()
                .compute_nulls()
                .compute_life_spans()
                .compute_variance()
                .compute_rul_summary()
                .compute_op_conditions()
                .print_report())


if __name__ == "__main__":
    EDABasico().run()
