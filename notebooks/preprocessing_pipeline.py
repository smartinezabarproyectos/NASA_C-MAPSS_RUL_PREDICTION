import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.config import DATASETS, PROCESSED_DIR, MAX_RUL, CLASSIFICATION_W

FIGURES_DIR = Path(ROOT) / "paper" / "figures" / "notebooks" / "07_preprocessing"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class PreprocessingPipeline:
    def __init__(self):
        self.loader       = DataLoader()
        self.preprocessor = Preprocessor()
        self.engineer     = FeatureEngineer()
        self.processed    = {}
        self.feature_cols = {}

    def load_data(self) -> "PreprocessingPipeline":
        self.loader.load_all()
        return self

    def process_datasets(self) -> "PreprocessingPipeline":
        for ds_id in DATASETS:
            train_raw, test_raw, rul = self.loader.data[ds_id]
            train = self.preprocessor.process_train(train_raw, ds_id)
            test  = self.preprocessor.process_test(test_raw, ds_id)
            test["rul"] = rul.values
            self.processed[ds_id] = {"train": train, "test": test, "rul": rul}
        return self

    def build_features(self) -> "PreprocessingPipeline":
        for ds_id in DATASETS:
            self.processed[ds_id]["train"] = self.engineer.fit_transform(
                self.processed[ds_id]["train"]
            )
        return self

    def normalize(self) -> "PreprocessingPipeline":
        exclude = {"unit_id", "cycle", "rul", "label"}
        for ds_id in DATASETS:
            train = self.processed[ds_id]["train"]
            test  = self.processed[ds_id]["test"]
            all_cols  = [c for c in train.columns if c not in exclude]
            avail     = [c for c in all_cols if c in test.columns]
            self.feature_cols[ds_id] = avail
            self.preprocessor.fit_scaler(train, avail)
            self.processed[ds_id]["train_norm"] = self.preprocessor.apply_scaler(train)
            self.processed[ds_id]["test_norm"]  = self.preprocessor.apply_scaler(test)
        return self

    def plot_normalization_effect(self, ds_id: str = "FD001") -> "PreprocessingPipeline":
        raw  = self.processed[ds_id]["train"]
        norm = self.processed[ds_id]["train_norm"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(raw["sensor_11"],  bins=50, alpha=0.7, color="#C44E52", edgecolor="white")
        axes[0].set_title("sensor_11 — Antes")
        axes[1].hist(norm["sensor_11"], bins=50, alpha=0.7, color="#4C72B0", edgecolor="white")
        axes[1].set_title("sensor_11 — Despues")
        plt.suptitle(f"Efecto MinMaxScaler — {ds_id}", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "07_preprocessing_fig01.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return self

    def save_parquets(self) -> "PreprocessingPipeline":
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for ds_id in DATASETS:
            self.processed[ds_id]["train_norm"].to_parquet(
                PROCESSED_DIR / f"train_{ds_id}.parquet", index=False)
            self.processed[ds_id]["test_norm"].to_parquet(
                PROCESSED_DIR / f"test_{ds_id}.parquet", index=False)
        return self

    def save_metadata(self) -> "PreprocessingPipeline":
        meta = {
            "feature_cols":     self.feature_cols,
            "max_rul":          MAX_RUL,
            "classification_w": CLASSIFICATION_W,
            "window_size":      self.engineer.window,
        }
        with open(PROCESSED_DIR / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        return self

    def run(self) -> "PreprocessingPipeline":
        return (self
                .load_data()
                .process_datasets()
                .build_features()
                .normalize()
                .plot_normalization_effect()
                .save_parquets()
                .save_metadata())


if __name__ == "__main__":
    PreprocessingPipeline().run()
