import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBRegressor, XGBClassifier
from src.config import MODELS_DIR


class ClassicalMLTrainer:
    def __init__(self, models_dir: Path = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reg_models  = self._build_regression_models()
        self.clf_models  = self._build_classification_models()
        self.results     = {}

    def _build_regression_models(self) -> dict:
        return {
            "linear_regression": LinearRegression(),
            "svr":               SVR(kernel="rbf", C=100, epsilon=0.1),
            "random_forest_reg": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "xgboost_reg":       XGBRegressor(n_estimators=300, max_depth=6,
                                              learning_rate=0.05, random_state=42),
        }

    def _build_classification_models(self) -> dict:
        return {
            "random_forest_clf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "xgboost_clf":       XGBClassifier(n_estimators=300, max_depth=6,
                                               learning_rate=0.05, random_state=42),
        }

    def _nasa_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        d = y_pred - y_true
        return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

    def train_regression(self, X_train, y_train, X_test, y_test, ds_id) -> "ClassicalMLTrainer":
        for name, model in self.reg_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            self.results[f"{name}_{ds_id}"] = {
                "rmse":       float(np.sqrt(mean_squared_error(y_test, preds))),
                "mae":        float(mean_absolute_error(y_test, preds)),
                "r2":         float(r2_score(y_test, preds)),
                "nasa_score": self._nasa_score(y_test, preds),
            }
            with open(self.models_dir / f"{name}_{ds_id}.pkl", "wb") as f:
                pickle.dump(model, f)
        return self

    def train_classification(self, X_train, y_train, X_test, y_test, ds_id) -> "ClassicalMLTrainer":
        for name, model in self.clf_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            self.results[f"{name}_{ds_id}"] = {
                "accuracy":  float(accuracy_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall":    float(recall_score(y_test, preds, zero_division=0)),
                "f1":        float(f1_score(y_test, preds, zero_division=0)),
                "auc_roc":   float(roc_auc_score(y_test, proba)),
            }
            with open(self.models_dir / f"{name}_{ds_id}.pkl", "wb") as f:
                pickle.dump(model, f)
        return self

    def load_model(self, name: str, ds_id: str):
        with open(self.models_dir / f"{name}_{ds_id}.pkl", "rb") as f:
            return pickle.load(f)
