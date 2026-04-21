"""
EDA Basico — NASA C-MAPSS
Analisis exploratorio inicial: estructura, estadisticas descriptivas,
sensores constantes y condiciones operacionales.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

from src.data_loader import load_all_datasets
from src.config import COLUMN_NAMES, DATASETS, USEFUL_SENSORS, DROP_SENSORS


class EDABasico:

    def __init__(self):
        print("Cargando datasets NASA C-MAPSS...")
        self.data = load_all_datasets()
        self.summary_df = None
        self.null_df = None
        self.life_df = None
        self.var_df = None
        self.constant_sensors = None
        self.rul_df = None

    def compute_summary(self) -> "EDABasico":
        """Calcula resumen de dimensiones y motores por sub-dataset."""
        summary = []
        for ds_id in DATASETS:
            train, test, rul = self.data[ds_id]
            summary.append({
                "dataset":       ds_id,
                "train_rows":    train.shape[0],
                "train_engines": train["unit_id"].nunique(),
                "test_rows":     test.shape[0],
                "test_engines":  test["unit_id"].nunique(),
                "rul_count":     len(rul),
                "train_cols":    train.shape[1],
            })
        self.summary_df = pd.DataFrame(summary).set_index("dataset")
        return self

    def compute_nulls(self) -> "EDABasico":
        """Verifica valores nulos en train y test de cada sub-dataset."""
        null_report = {}
        for ds_id in DATASETS:
            train, test, _ = self.data[ds_id]
            null_report[f"{ds_id}_train"] = train.isnull().sum().sum()
            null_report[f"{ds_id}_test"]  = test.isnull().sum().sum()
        self.null_df = pd.DataFrame.from_dict(
            null_report, orient="index", columns=["total_nulls"]
        )
        return self

    def compute_life_spans(self) -> "EDABasico":
        """Calcula estadisticas de vida util (ciclos) por motor y dataset."""
        life_spans = {}
        for ds_id in DATASETS:
            train = self.data[ds_id][0]
            cycles = train.groupby("unit_id")["cycle"].max()
            life_spans[ds_id] = {
                "min":    cycles.min(),
                "max":    cycles.max(),
                "mean":   cycles.mean(),
                "median": cycles.median(),
                "std":    cycles.std(),
            }
        self.life_df = pd.DataFrame(life_spans).T
        return self

    def compute_variance(self) -> "EDABasico":
        """
        Calcula varianza de cada sensor en los 4 sub-datasets.
        Identifica sensores constantes (varianza < 0.001 en todos).
        """
        sensor_cols = [
            c for c in COLUMN_NAMES
            if c.startswith("sensor_") or c.startswith("op_setting_")
        ]
        variance_report = {}
        for ds_id in DATASETS:
            train = self.data[ds_id][0]
            variance_report[ds_id] = train[sensor_cols].var()

        self.var_df = pd.DataFrame(variance_report)
        self.var_df["is_constant"] = (self.var_df < 0.001).all(axis=1)
        self.constant_sensors = (
            self.var_df[self.var_df["is_constant"]].index.tolist()
        )
        return self

    def compute_op_conditions(self) -> "EDABasico":
        """Imprime las combinaciones unicas de condiciones operacionales."""
        op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
        print("\nCondiciones operacionales unicas por sub-dataset:")
        for ds_id in DATASETS:
            train = self.data[ds_id][0]
            unique_combos = train[op_cols].drop_duplicates().shape[0]
            print(f"  {ds_id}: {unique_combos} combinaciones")
            if unique_combos <= 6:
                print(
                    train[op_cols]
                    .drop_duplicates()
                    .sort_values(op_cols)
                    .reset_index(drop=True)
                    .to_string()
                )
        return self

    def compute_rul_summary(self) -> "EDABasico":
        """Calcula estadisticas descriptivas del RUL real en test."""
        rul_summary = {}
        for ds_id in DATASETS:
            rul = self.data[ds_id][2]
            rul_summary[ds_id] = {
                "min":    rul.min(),
                "max":    rul.max(),
                "mean":   rul.mean(),
                "median": rul.median(),
                "std":    rul.std(),
                "q25":    rul.quantile(0.25),
                "q75":    rul.quantile(0.75),
            }
        self.rul_df = pd.DataFrame(rul_summary).T
        return self

    def inspect_engine(self, ds_id: str = "FD001", unit_id: int = 1) -> None:
        """Muestra primeras y ultimas filas de un motor especifico."""
        train = self.data[ds_id][0]
        engine = train[train["unit_id"] == unit_id]
        print(f"\nMotor {unit_id} — {len(engine)} ciclos — {ds_id}")
        print(engine.head().to_string())
        print(engine.tail().to_string())

    def print_report(self) -> "EDABasico":
        """Imprime el resumen ejecutivo del EDA basico."""
        n_const = len(self.constant_sensors) if self.constant_sensors else "?"
        print("=" * 60)
        print("  RESUMEN EDA BASICO")
        print("=" * 60)
        print(f"  Sub-datasets:        {len(DATASETS)}")
        print(f"  Columnas por archivo: {len(COLUMN_NAMES)}")
        print(f"  Valores nulos:        0 en todos")
        print(f"  Sensores constantes:  {n_const} -> se eliminan")
        print(f"  Sensores utiles:      {len(USEFUL_SENSORS)}")
        print(f"  FD001/FD003:          1 condicion operacional")
        print(f"  FD002/FD004:          6 condiciones operacionales")
        print(f"  FD001/FD002:          1 modo de falla (HPC)")
        print(f"  FD003/FD004:          2 modos de falla (HPC + Fan)")
        print("=" * 60)
        if self.summary_df is not None:
            print("\nRESUMEN POR SUB-DATASET:")
            print(self.summary_df.to_string())
        if self.life_df is not None:
            print("\nVIDA UTIL (ciclos):")
            print(self.life_df.to_string())
        if self.rul_df is not None:
            print("\nRUL REAL EN TEST:")
            print(self.rul_df.to_string())
        return self

    def run(self) -> "EDABasico":
        """Ejecuta el analisis completo en cadena."""
        return (
            self
            .compute_summary()
            .compute_nulls()
            .compute_life_spans()
            .compute_variance()
            .compute_rul_summary()
            .compute_op_conditions()
            .print_report()
        )


if __name__ == "__main__":
    eda = EDABasico()
    eda.run()
