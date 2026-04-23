import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import SEQUENCE_LENGTH, MAX_RUL, MODELS_DIR
from src.models.deep_learning import DEVICE


class SequenceBuilder:
    def __init__(self, seq_len: int = SEQUENCE_LENGTH):
        self.seq_len = seq_len

    def build(self, df, feature_cols: list, target_col: str = "rul") -> tuple:
        X_list, y_list = [], []
        for unit_id in df["unit_id"].unique():
            unit = df[df["unit_id"] == unit_id]
            vals = unit[feature_cols].values
            tgts = unit[target_col].values / MAX_RUL
            for i in range(len(vals)):
                start = max(0, i - self.seq_len + 1)
                seq   = vals[start:i + 1]
                if len(seq) < self.seq_len:
                    seq = np.vstack([np.zeros((self.seq_len - len(seq), seq.shape[1])), seq])
                X_list.append(seq)
                y_list.append(tgts[i])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


class DLTrainer:
    def __init__(self, epochs: int = 50, patience: int = 10, models_dir: Path = None):
        self.epochs       = epochs
        self.patience     = patience
        self.models_dir   = Path(models_dir) if models_dir else MODELS_DIR
        self.train_losses = []
        self.val_losses   = []

    def make_loaders(self, X_train, y_train, X_val, y_val,
                     batch_size: int = 64) -> tuple:
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds,   batch_size=batch_size, shuffle=False))

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, lr: float = 1e-3,
              weight_decay: float = 1e-4) -> nn.Module:
        model     = model.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

        best_val, best_weights, no_improve = float("inf"), None, 0
        self.train_losses, self.val_losses = [], []

        for epoch in range(self.epochs):
            model.train()
            t_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * len(Xb)
            t_loss /= len(train_loader.dataset)

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    v_loss += criterion(model(Xb), yb).item() * len(Xb)
            v_loss /= len(val_loader.dataset)

            self.train_losses.append(t_loss)
            self.val_losses.append(v_loss)
            scheduler.step(v_loss)

            if v_loss < best_val:
                best_val     = v_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")

        if best_weights:
            model.load_state_dict(best_weights)
        return model

    def save(self, model: nn.Module, model_type: str, ds_id: str) -> "DLTrainer":
        torch.save(model.state_dict(), self.models_dir / f"{model_type}_optuna_{ds_id}.pt")
        return self

    def predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()
        return preds * MAX_RUL


def create_sequences(df, cols, target_col, seq_len=SEQUENCE_LENGTH):
    return SequenceBuilder(seq_len).build(df, cols, target_col)
