"""
Script para entrenar modelos de machine learning y guardarlos.
Uso: Importar funciones desde notebooks o ejecutar como script.
"""
from typing import Any

def train_model(X_train, y_train, model: Any) -> Any:
    """Entrena un estimador de machine learning.

    Args:
        X_train: Matriz o DataFrame de características de entrenamiento.
        y_train: Vector o Serie de etiquetas de entrenamiento.
        model: Estimador con método `fit`.

    Returns:
        model: Estimador entrenado.
    """
    model.fit(X_train, y_train)
    return model