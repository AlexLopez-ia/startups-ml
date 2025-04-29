"""
Script para cargar modelos y hacer predicciones.
"""
import joblib
from pathlib import Path
from typing import Union

def load_model(model_path: Union[str, Path]):
    """Carga un modelo serializado desde disco.

    Args:
        model_path (str | Path): Ruta al archivo del modelo.

    Returns:
        Estimador cargado.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path)

def predict(model, X):
    """Realiza predicciones con un modelo entrenado.

    Args:
        model: Estimador con método `predict`.
        X: DataFrame o array de features.

    Returns:
        np.ndarray: Predicciones.
    """
    return model.predict(X)