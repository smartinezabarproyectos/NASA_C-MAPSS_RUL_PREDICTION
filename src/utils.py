"""Utilidades generales del proyecto."""

import random
import numpy as np
import tensorflow as tf

from src.config import RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Fija semillas para reproducibilidad total."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_dataset_info(train_df, test_df, rul_series, dataset_id: str) -> None:
    """Imprime info básica de un sub-dataset."""
    print(f"\n{'='*50}")
    print(f"  Dataset: {dataset_id}")
    print(f"{'='*50}")
    print(f"  Train: {train_df.shape[0]:,} filas | {train_df['unit_id'].nunique()} motores")
    print(f"  Test:  {test_df.shape[0]:,} filas | {test_df['unit_id'].nunique()} motores")
    print(f"  RUL:   {len(rul_series)} valores")
    print(f"  Rango RUL test: [{rul_series.min()}, {rul_series.max()}]")
