import os
from pathlib import Path

# Rutas del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Directorios
RAW_DATA_DIR = DATA_DIR / "raw"  # Solo el directorio
RAW_DATA_FILE = RAW_DATA_DIR / "startup data.csv"  # Ruta completa al archivo

PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Asegurar que existen las carpetas necesarias
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, 
                MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Parámetros del modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Parámetros para preprocesamiento de datos
TARGET_COLUMN = 'status'