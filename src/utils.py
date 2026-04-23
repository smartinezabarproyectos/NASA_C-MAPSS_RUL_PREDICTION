import random
import time
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


class Utils:
    @staticmethod
    def set_seeds(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

    @staticmethod
    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def save_pickle(obj, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path: Path):
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    @staticmethod
    @contextmanager
    def timer(name: str = ""):
        start = time.time()
        try:
            yield
        finally:
            print(f"  [{name}] {time.time() - start:.2f}s")


def set_seeds(seed=42):
    Utils.set_seeds(seed)

def get_device():
    return Utils.get_device()

def print_dataset_info(train, test, rul, ds_id):
    print(f"\n{ds_id}: train={train.shape}, test={test.shape}, rul={len(rul)}")
