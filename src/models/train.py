import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
)
from src.models.deep_learning import DEVICE


def train_classical_model(model, X_train, y_train, model_name, dataset_id):
    model.fit(X_train, y_train)
    save_path = MODELS_DIR / f"{model_name}_{dataset_id}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ {model_name} guardado en {save_path}")


def create_sequences(df, feature_cols, target_col="rul", seq_length=SEQUENCE_LENGTH):
    X_sequences, y_targets = [], []

    for unit_id in df["unit_id"].unique():
        unit_data = df[df["unit_id"] == unit_id]
        features = unit_data[feature_cols].values
        targets = unit_data[target_col].values

        if len(features) < seq_length:
            pad_length = seq_length - len(features)
            features = np.vstack([np.tile(features[0], (pad_length, 1)), features])
            targets = np.concatenate([np.full(pad_length, targets[0]), targets])

        for i in range(len(features) - seq_length + 1):
            X_sequences.append(features[i : i + seq_length])
            y_targets.append(targets[i + seq_length - 1])

    return np.array(X_sequences), np.array(y_targets)


def train_dl_model(model, X_train, y_train, model_name, dataset_id):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        train_maes = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_maes.append(torch.mean(torch.abs(pred - y_batch)).item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_mae = torch.mean(torch.abs(val_pred - y_val_t)).item()

        train_loss = np.mean(train_losses)
        train_mae = np.mean(train_maes)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {train_loss:.4f} — val_loss: {val_loss:.4f} — val_mae: {val_mae:.4f}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping en epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    save_path = MODELS_DIR / f"{model_name}_{dataset_id}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✓ {model_name} guardado en {save_path}")

    class History:
        def __init__(self, h):
            self.history = h

    return History(history)


def predict_dl(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        return model(X_t).cpu().numpy()
