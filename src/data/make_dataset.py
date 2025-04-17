import pandas as pd
import sys
from pathlib import Path
import os

# Añadir directorio raíz al path para poder importar desde src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROCESSED_DATA_DIR, TARGET_COLUMN, RAW_DATA_FILE

def load_data(file_path=None):
    """
    Carga el dataset desde la ubicación especificada
    """
    if file_path is None:
        file_path = RAW_DATA_FILE 
    
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset cargado exitosamente con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

def validate_data(df):
    """
    Realiza validaciones básicas sobre el dataset
    """
    # Verificar que el target existe
    if TARGET_COLUMN not in df.columns:
        print(f"Error: La columna target '{TARGET_COLUMN}' no existe en el dataset")
        return False
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    print(f"Columnas con valores nulos:\n{null_counts[null_counts > 0]}")
    
    # Verificar distribución de la variable objetivo
    target_dist = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    print(f"Distribución del target:\n{target_dist}")
    
    return True

def main():
    """
    Proceso principal para cargar, validar y guardar el dataset inicial
    """
    print("Cargando dataset...")
    df = load_data()
    
    if df is not None and validate_data(df):
        print("Guardando dataset validado...")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        df.to_csv(PROCESSED_DATA_DIR / "startup_data_validated.csv", index=False)
        print("Dataset guardado exitosamente")
    else:
        print("No se pudo procesar el dataset")

if __name__ == "__main__":
    main()