"""
Funciones para evaluar modelos de clasificación.
"""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evalúa un modelo de clasificación.

    Args:
        model: Estimador con método `predict`.
        X_test: Features de prueba.
        y_test: Labels de prueba.

    Returns:
        Tuple con (acc, report, cm):
            - acc (float): Precisión.
            - report (dict): Métricas por clase (`output_dict=True`).
            - cm (np.ndarray): Matriz de confusión.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm