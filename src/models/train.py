"""
DLTrainer — NASA C-MAPSS
Creacion de secuencias temporales y loop de entrenamiento
para los modelos de Deep Learning con PyTorch.
"""

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
    """
    Convierte datos tabulares en secuencias 3D para modelos DL.

    Attributes
    ----------
    seq_len : int
        Longitud de cada secuencia (ciclos consecutivos).
    """

    def __init__(self, seq_len: int = SEQUENCE_LENGTH):
        self.seq_len = seq_len

    def build(self, df, feature_cols: list,
              target_col: str = "rul") -> tuple:
        """
        Genera ventanas deslizantes de longitud seq_len por motor.

        Para un motor con 150 ciclos genera 70 secuencias.
        Cada secuencia tiene forma (seq_len, n_features).
        El target es el RUL del ultimo ciclo de la ventana.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame procesado con rolling features.
        feature_cols : list
            Columnas de features a incluir en la secuencia.
        target_col : str
            Columna target (por defecto 'rul').

        Returns
        -------
        tuple
            (X, y) donde X.shape = (n_seq, seq_len, n_features)
            y y.shape = (n_seq,)
        """
        X_list, y_list = [], []
        for unit_id in df["unit_id"].unique():
            unit = df[df["unit_id"] == unit_id]
            vals = unit[feature_cols].values
            tgts = unit[target_col].values / MAX_RUL

            for i in range(len(vals)):
                start = max(0, i - self.seq_len + 1)
                seq   = vals[start:i + 1]
                if len(seq) < self.seq_len:
                    pad = np.zeros((self.seq_len - len(seq), seq.shape[1]))
                    seq = np.vstack([pad, seq])
                X_list.append(seq)
                y_list.append(tgts[i])

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


class DLTrainer:
    """
    Entrena modelos de Deep Learning con PyTorch.

    Implementa el loop de entrenamiento con early stopping,
    ReduceLROnPlateau y guardado del mejor modelo.

    Attributes
    ----------
    epochs : int
        Numero maximo de epochs de entrenamiento.
    patience : int
        Epochs sin mejora antes de detener el entrenamiento.
    models_dir : Path
        Carpeta donde se guardan los modelos .pt.
    train_losses : list
        Historico de loss de entrenamiento por epoch.
    val_losses : list
        Historico de loss de validacion por epoch.
    """

    def __init__(self, epochs: int = 50, patience: int = 10,
                 models_dir: Path = None):
        self.epochs      = epochs
        self.patience    = patience
        self.models_dir  = Path(models_dir) if models_dir else MODELS_DIR
        self.train_losses = []
        self.val_losses   = []

    def make_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     batch_size: int = 64) -> tuple:
        """
        Convierte arrays numpy en DataLoaders de PyTorch.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Datos de entrenamiento.
        X_val, y_val : np.ndarray
            Datos de validacion.
        batch_size : int
            Tamano del batch.

        Returns
        -------
        tuple
            (train_loader, val_loader)
        """
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, lr: float = 1e-3,
              weight_decay: float = 1e-4) -> nn.Module:
        """
        Ejecuta el loop de entrenamiento con early stopping.

        Guarda los mejores pesos cuando el val_loss mejora.
        Restaura los mejores pesos al finalizar.

        Parameters
        ----------
        model : nn.Module
            Modelo de Deep Learning a entrenar.
        train_loader : DataLoader
            DataLoader de entrenamiento.
        val_loader : DataLoader
            DataLoader de validacion.
        lr : float
            Learning rate inicial.
        weight_decay : float
            Regularizacion L2.

        Returns
        -------
        nn.Module
            Modelo con los mejores pesos restaurados.
        """
        model = model.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=lr,
                          weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=5,
                                      factor=0.5, verbose=False)

        best_val_loss = float("inf")
        best_weights  = None
        no_improve    = 0
        self.train_losses = []
        self.val_losses   = []

        for epoch in range(self.epochs):
            # Entrenamiento
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(train_loader.dataset)

            # Validacion
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    preds   = model(X_batch)
                    loss    = criterion(preds, y_batch)
                    val_loss += loss.item() * len(X_batch)
            val_loss /= len(val_loader.dataset)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"    Early stopping en epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if best_weights:
            model.load_state_dict(best_weights)

        return model

    def save(self, model: nn.Module, model_type: str,
             ds_id: str) -> "DLTrainer":
        """
        Guarda los pesos del modelo como archivo .pt.

        Parameters
        ----------
        model : nn.Module
            Modelo entrenado.
        model_type : str
            Tipo de modelo: 'lstm', 'gru', 'tcn', 'transformer'.
        ds_id : str
            ID del sub-dataset.

        Returns
        -------
        DLTrainer
            La misma instancia para encadenamiento.
        """
        path = self.models_dir / f"{model_type}_optuna_{ds_id}.pt"
        torch.save(model.state_dict(), path)
        print(f"    Guardado: {path.name}")
        return self

    def predict(self, model: nn.Module,
                X: np.ndarray) -> np.ndarray:
        """
        Hace inferencia con el modelo y devuelve predicciones en ciclos.

        Convierte el array numpy a tensor, pasa por el modelo en modo
        eval y multiplica por MAX_RUL para recuperar la escala original.

        Parameters
        ----------
        model : nn.Module
            Modelo entrenado.
        X : np.ndarray
            Secuencias de entrada.

        Returns
        -------
        np.ndarray
            Predicciones de RUL en ciclos.
        """
        model.eval()
        with torch.no_grad():
            X_t   = torch.FloatTensor(X).to(DEVICE)
            preds = model(X_t).cpu().numpy()
        return preds * MAX_RUL


# Funcion de compatibilidad
def create_sequences(df, cols, target_col, seq_len=SEQUENCE_LENGTH):
    return SequenceBuilder(seq_len).build(df, cols, target_col)
