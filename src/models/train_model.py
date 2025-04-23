"""
Script para entrenar modelos de machine learning y guardarlos.
Uso: Importar funciones desde notebooks o ejecutar como script.
"""
def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model