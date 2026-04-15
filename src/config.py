from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "paper" / "figures"

# Crear directorios si no existen
for _dir in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


DATASETS = ("FD001", "FD002", "FD003", "FD004")

COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]


DROP_SETTINGS = ["op_setting_3"]


USEFUL_SENSORS = [s for s in COLUMN_NAMES if s.startswith("sensor_") and s not in DROP_SENSORS]

# ── Hiperparámetros generales ───────────────────────────────────────
MAX_RUL = 125          # Clip del RUL máximo (piece-wise linear)
WINDOW_SIZE = 30       # Ventana para rolling features
CLASSIFICATION_W = 30  # Umbral para clasificación binaria (falla en W ciclos?)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Sequence length para modelos Deep Learning ──────────────────────
SEQUENCE_LENGTH = 80
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
