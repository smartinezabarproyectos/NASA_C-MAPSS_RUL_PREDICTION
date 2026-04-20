import pandas as pd

from src.config import RAW_DIR, COLUMN_NAMES, DATASETS


def load_dataset(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    if dataset_id not in DATASETS:
        raise ValueError(f"{dataset_id} no es válido. Opciones: {DATASETS}")

    train_df = pd.read_csv(
        RAW_DIR / f"train_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    test_df = pd.read_csv(
        RAW_DIR / f"test_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    rul_series = pd.read_csv(
        RAW_DIR / f"RUL_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=["rul"],
    ).squeeze()

    return train_df, test_df, rul_series


def load_all_datasets() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    return {ds: load_dataset(ds) for ds in DATASETS}
