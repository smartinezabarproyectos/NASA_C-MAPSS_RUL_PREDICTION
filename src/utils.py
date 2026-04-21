"""
Utils — NASA C-MAPSS
Funciones auxiliares de reproducibilidad, serializacion y medicion de tiempo.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import random
import time
import pickle
from contextlib import contextmanager

import numpy as np
import torch


class Utils:
    """
    Utilidades generales del proyecto NASA C-MAPSS.

    Agrupa funciones de reproducibilidad, serializacion,
    medicion de tiempo y deteccion de device.
    """

    @staticmethod
    def set_seeds(seed: int = 42) -> None:
        """
        Fija semillas de random, numpy y torch para reproducibilidad.

        Tambien configura cudnn.deterministic para operaciones en GPU.

        Parameters
        ----------
        seed : int
            Semilla a usar en todos los generadores.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

    @staticmethod
    def get_device() -> torch.device:
        """
        Detecta y retorna el device disponible (GPU o CPU).

        Returns
        -------
        torch.device
            'cuda' si hay GPU disponible, 'cpu' si no.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  Device: CPU")
        return device

    @staticmethod
    def save_pickle(obj, path: Path) -> None:
        """
        Guarda un objeto como archivo pickle.

        Parameters
        ----------
        obj : any
            Objeto a serializar.
        path : Path
            Ruta de destino del archivo .pkl.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path: Path):
        """
        Carga un objeto desde archivo pickle.

        Parameters
        ----------
        path : Path
            Ruta del archivo .pkl a cargar.

        Returns
        -------
        any
            Objeto deserializado.
        """
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    @staticmethod
    @contextmanager
    def timer(name: str = ""):
        """
        Context manager que mide el tiempo de ejecucion de un bloque.

        Uso:
            with Utils.timer("Entrenamiento XGBoost"):
                model.fit(X, y)

        Parameters
        ----------
        name : str
            Nombre del bloque para mostrar en el output.
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            print(f"  [{name}] {elapsed:.2f}s")


def print_dataset_info(train, test, rul, ds_id):
    """Funcion de compatibilidad para codigo existente."""
    print(f"\n{ds_id}:")
    print(f"  Train: {train.shape[0]:,} filas, "
          f"{train['unit_id'].nunique()} motores")
    print(f"  Test:  {test.shape[0]:,} filas, "
          f"{test['unit_id'].nunique()} motores")
    print(f"  RUL:   {len(rul)} valores")


def set_seeds(seed=42):
    Utils.set_seeds(seed)


def get_device():
    return Utils.get_device()
