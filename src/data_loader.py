"""
DataLoader — NASA C-MAPSS
Responsable de leer los archivos crudos del disco y devolverlos
como DataFrames de pandas listos para procesar.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from src.config import COLUMN_NAMES, DATASETS, RAW_DIR


class DataLoader:
    """
    Carga los archivos crudos de NASA C-MAPSS desde disco.

    Attributes
    ----------
    raw_dir : Path
        Ruta a la carpeta con los archivos .txt de NASA.
    data : dict
        Diccionario con los 4 sub-datasets cargados.
        Cada valor es una tupla (train_df, test_df, rul_series).
    """

    def __init__(self, raw_dir: Path = None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
        self.data    = {}

    def load_dataset(self, ds_id: str) -> tuple:
        """
        Carga train, test y RUL de un sub-dataset especifico.

        Parameters
        ----------
        ds_id : str
            Identificador del sub-dataset, ej: 'FD001'.

        Returns
        -------
        tuple
            (train_df, test_df, rul_series)
        """
        def _read(filename):
            return pd.read_csv(
                self.raw_dir / filename,
                sep=r"\s+", header=None, names=COLUMN_NAMES
            )

        train = _read(f"train_{ds_id}.txt")
        test  = _read(f"test_{ds_id}.txt")
        rul   = pd.read_csv(
            self.raw_dir / f"RUL_{ds_id}.txt",
            sep=r"\s+", header=None, names=["rul"]
        )["rul"]

        return train, test, rul

    def load_all(self) -> "DataLoader":
        """
        Carga los 4 sub-datasets y los almacena en self.data.

        Returns
        -------
        DataLoader
            La misma instancia para encadenamiento.
        """
        for ds_id in DATASETS:
            self.data[ds_id] = self.load_dataset(ds_id)
            print(f"  Cargado {ds_id}: "
                  f"train={self.data[ds_id][0].shape}, "
                  f"test={self.data[ds_id][1].shape}")
        return self

    def get(self, ds_id: str) -> tuple:
        """
        Retorna la tupla (train, test, rul) de un sub-dataset.

        Parameters
        ----------
        ds_id : str
            Identificador del sub-dataset.
        """
        if ds_id not in self.data:
            self.data[ds_id] = self.load_dataset(ds_id)
        return self.data[ds_id]


# Funcion de compatibilidad para codigo que usa la API antigua
def load_all_datasets() -> dict:
    loader = DataLoader()
    loader.load_all()
    return loader.data


def load_dataset(ds_id: str) -> tuple:
    return DataLoader().load_dataset(ds_id)


if __name__ == "__main__":
    loader = DataLoader()
    loader.load_all()
    for ds_id, (train, test, rul) in loader.data.items():
        print(f"{ds_id} — train: {train.shape}, test: {test.shape}, rul: {len(rul)}")
