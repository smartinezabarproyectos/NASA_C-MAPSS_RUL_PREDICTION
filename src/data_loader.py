import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from src.config import COLUMN_NAMES, DATASETS, RAW_DIR


class DataLoader:
    def __init__(self, raw_dir: Path = None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
        self.data    = {}

    def load_dataset(self, ds_id: str) -> tuple:
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
        for ds_id in DATASETS:
            self.data[ds_id] = self.load_dataset(ds_id)
        return self

    def get(self, ds_id: str) -> tuple:
        if ds_id not in self.data:
            self.data[ds_id] = self.load_dataset(ds_id)
        return self.data[ds_id]


def load_all_datasets() -> dict:
    return DataLoader().load_all().data


def load_dataset(ds_id: str) -> tuple:
    return DataLoader().load_dataset(ds_id)
