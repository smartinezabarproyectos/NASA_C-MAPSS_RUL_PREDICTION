import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.models.deep_learning import FlexibleModel, DEVICE
from src.models.train import SequenceBuilder, DLTrainer
from src.models.evaluate import Evaluator
from src.preprocessing import Preprocessor
from src.utils import Utils
from src.config import DATASETS, PROCESSED_DIR, SEQUENCE_LENGTH, MAX_RUL


class DeepLearningPipeline:
    MODEL_TYPES = ["lstm", "gru", "tcn", "transformer"]
    N_TRIALS    = 50
    RUL_MAX     = float(MAX_RUL)

    def __init__(self):
        Utils.set_seeds(42)
        self.trainer        = DLTrainer(epochs=50, patience=10)
        self.seq_builder    = SequenceBuilder(seq_len=SEQUENCE_LENGTH)
        self.evaluator      = Evaluator()
        self.processed      = {}
        self.feature_cols   = {}
        self.studies        = {}
        self.best_models    = {}
        self.optuna_results = {}

    def load_data(self) -> "DeepLearningPipeline":
        with open(PROCESSED_DIR / "metadata.json") as f:
            self.feature_cols = json.load(f)["feature_cols"]
        for ds_id in DATASETS:
            self.processed[ds_id] = {
                "train": pd.read_parquet(PROCESSED_DIR / f"train_{ds_id}.parquet"),
                "test":  pd.read_parquet(PROCESSED_DIR / f"test_{ds_id}.parquet"),
            }
        return self

    def _prepare_splits(self, ds_id: str) -> tuple:
        train_raw = self.processed[ds_id]["train"]
        exclude   = {"unit_id", "cycle", "rul", "label"}
        cols      = [c for c in train_raw.columns if c not in exclude]
        tr_norm   = Preprocessor().normalize_by_operating_condition(train_raw)
        tr_ids, val_ids = train_test_split(
            tr_norm["unit_id"].unique(), test_size=0.2, random_state=42)
        tr  = tr_norm[tr_norm["unit_id"].isin(tr_ids)].copy()
        val = tr_norm[tr_norm["unit_id"].isin(val_ids)].copy()
        scaler   = MinMaxScaler()
        tr[cols]  = scaler.fit_transform(tr[cols])
        val[cols] = scaler.transform(val[cols])
        X_tr, y_tr_r  = self.seq_builder.build(tr,  cols, "rul")
        X_val, y_val_r = self.seq_builder.build(val, cols, "rul")
        return (X_tr, y_tr_r / self.RUL_MAX,
                X_val, y_val_r / self.RUL_MAX,
                X_tr.shape[2], cols)

    def _objective(self, trial, model_type, X_tr, y_tr, X_val, y_val, n_feat) -> float:
        hidden   = trial.suggest_int("hidden_size", 64, 256, step=64)
        n_layers = trial.suggest_int("num_layers", 1, 3)
        dropout  = trial.suggest_float("dropout", 0.2, 0.5, step=0.05)
        lr       = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd       = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_sz = trial.suggest_categorical("batch_size", [32, 64, 128])
        d_model  = trial.suggest_categorical("d_model", [32, 64, 128]) if model_type == "transformer" else 64
        nhead    = trial.suggest_categorical("nhead", [2, 4]) if model_type == "transformer" else 4
        if model_type == "transformer" and d_model % nhead != 0:
            nhead = 2

        model = FlexibleModel(n_feat, model_type, hidden, n_layers,
                              dropout, d_model, nhead).to(DEVICE)
        tl, vl = DLTrainer(epochs=30, patience=5).make_loaders(
            X_tr, y_tr, X_val, y_val, batch_size=batch_sz)
        trainer = DLTrainer(epochs=30, patience=5)
        model   = trainer.train(model, tl, vl, lr=lr, weight_decay=wd)
        preds   = trainer.predict(model, X_val)
        return float(np.sqrt(np.mean((preds - y_val * self.RUL_MAX) ** 2)))

    def run_optuna(self, n_trials: int = N_TRIALS) -> "DeepLearningPipeline":
        for ds_id in DATASETS:
            X_tr, y_tr, X_val, y_val, n_feat, _ = self._prepare_splits(ds_id)
            for model_type in self.MODEL_TYPES:
                key   = f"{model_type}_{ds_id}"
                study = optuna.create_study(direction="minimize",
                                            pruner=optuna.pruners.MedianPruner())
                study.optimize(
                    lambda t, mt=model_type: self._objective(
                        t, mt, X_tr, y_tr, X_val, y_val, n_feat),
                    n_trials=n_trials, show_progress_bar=False)
                self.studies[key]        = study
                self.optuna_results[key] = {
                    "best_rmse":   study.best_value,
                    "best_params": study.best_params,
                }
        return self

    def retrain_best(self) -> "DeepLearningPipeline":
        for ds_id in DATASETS:
            X_tr, y_tr, X_val, y_val, n_feat, _ = self._prepare_splits(ds_id)
            self.best_models[ds_id] = {}
            for model_type in self.MODEL_TYPES:
                p  = self.studies[f"{model_type}_{ds_id}"].best_params
                model = FlexibleModel(
                    n_feat, model_type,
                    p["hidden_size"], p["num_layers"], p["dropout"],
                    p.get("d_model", 64), p.get("nhead", 4)
                ).to(DEVICE)
                tl, vl = self.trainer.make_loaders(
                    X_tr, y_tr, X_val, y_val, batch_size=p["batch_size"])
                model = self.trainer.train(model, tl, vl,
                                           lr=p["lr"], weight_decay=p["weight_decay"])
                self.trainer.save(model, model_type, ds_id)
                self.best_models[ds_id][model_type] = model
        return self

    def evaluate_ensemble(self) -> "DeepLearningPipeline":
        for ds_id in DATASETS:
            _, _, X_val, y_val, _, _ = self._prepare_splits(ds_id)
            y_real    = y_val * self.RUL_MAX
            all_preds = []
            for model_type in self.MODEL_TYPES:
                preds = self.trainer.predict(self.best_models[ds_id][model_type], X_val)
                all_preds.append(preds)
                m = self.evaluator.regression_metrics(y_real, preds)
                self.optuna_results[f"result_{model_type}_{ds_id}"] = m
            ens = np.mean(all_preds, axis=0)
            self.optuna_results[f"result_ensemble_{ds_id}"] = (
                self.evaluator.regression_metrics(y_real, ens))
        return self

    def save_results(self) -> "DeepLearningPipeline":
        with open(PROCESSED_DIR / "optuna_results.json", "w") as f:
            json.dump(self.optuna_results, f, indent=2, default=str)
        return self

    def run(self, n_trials: int = N_TRIALS) -> "DeepLearningPipeline":
        return (self
                .load_data()
                .run_optuna(n_trials=n_trials)
                .retrain_best()
                .evaluate_ensemble()
                .save_results())


if __name__ == "__main__":
    DeepLearningPipeline().run(n_trials=50)
