import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import seaborn as sns

# Añadir directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import FIGURES_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, TARGET_COLUMN, TEST_SIZE

def perform_bias_variance_analysis(X_train, X_test, y_train, y_test, 
                                  model_class=DecisionTreeClassifier,
                                  param_name='max_depth',
                                  param_range=range(1, 20),
                                  random_state=RANDOM_STATE):
    """
    Realiza análisis de sesgo-varianza para un tipo de modelo y un rango de parámetros
    
    Args:
        X_train, X_test, y_train, y_test: Conjuntos de datos para entrenamiento y prueba
        model_class: Clase del modelo a analizar (por defecto: DecisionTreeClassifier)
        param_name: Nombre del parámetro a variar (por defecto: 'max_depth')
        param_range: Rango de valores para el parámetro
        random_state: Semilla aleatoria para reproducibilidad
    
    Returns:
        Tuple con (parámetros, scores_train, scores_test, mejor_parámetro)
    """
    train_scores = []
    test_scores = []
    
    for param_value in param_range:
        # Crear modelo con el parámetro específico
        model_params = {param_name: param_value, 'random_state': random_state}
        model = model_class(**model_params)
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Evaluar en train y test
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calcular y guardar métricas
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
    
    # Encontrar el mejor valor del parámetro
    best_idx = np.argmax(test_scores)
    best_param = param_range[best_idx]
    
    return param_range, train_scores, test_scores, best_param

def plot_learning_curve(estimator, X, y, cv=5, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Genera curvas de aprendizaje para diagnosticar el sesgo-varianza de un modelo.
    
    Args:
        estimator: Modelo de scikit-learn
        X: Features
        y: Variable objetivo
        cv: Número de folds para validación cruzada
        n_jobs: Número de trabajos paralelos
        train_sizes: Tamaños relativos de los conjuntos de entrenamiento
        
    Returns:
        Muestra la gráfica de curvas de aprendizaje
    """
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring="accuracy", shuffle=True
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, 
             label="Precisión en entrenamiento")
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, 
                     alpha=0.15, color="blue")
    
    plt.plot(train_sizes, test_mean, color="green", marker="s", markersize=5, 
             linestyle="--", label="Precisión en validación")
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, 
                     alpha=0.15, color="green")
    
    plt.title("Curva de Aprendizaje")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Precisión")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    return plt

def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5, n_jobs=None):
    """
    Genera curvas de validación para un parámetro específico del modelo.
    
    Args:
        estimator: Modelo de scikit-learn
        X: Features
        y: Variable objetivo
        param_name: Nombre del parámetro a evaluar
        param_range: Rango de valores para el parámetro
        cv: Número de folds para validación cruzada
        n_jobs: Número de trabajos paralelos
        
    Returns:
        Muestra la gráfica de curvas de validación
    """
    plt.figure(figsize=(10, 6))
    
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring="accuracy", n_jobs=n_jobs
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(param_range, train_mean, color="blue", marker="o", markersize=5, 
             label="Precisión en entrenamiento")
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, 
                     alpha=0.15, color="blue")
    
    plt.plot(param_range, test_mean, color="green", marker="s", markersize=5, 
             linestyle="--", label="Precisión en validación")
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, 
                     alpha=0.15, color="green")
    
    plt.title(f"Curva de Validación ({param_name})")
    plt.xlabel(f"Valor de {param_name}")
    plt.ylabel("Precisión")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    return plt

def diagnose_bias_variance(estimator, X_train, y_train, X_test, y_test):
    """
    Realiza un diagnóstico completo de sesgo-varianza para un modelo.
    
    Args:
        estimator: Modelo de scikit-learn ya entrenado
        X_train: Features de entrenamiento
        y_train: Variable objetivo de entrenamiento
        X_test: Features de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        Un diccionario con métricas de diagnóstico y conclusiones
    """
    # Predicciones
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    
    # Cálculo de precisiones
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Diagnóstico
    diagnosis = {}
    diagnosis["train_accuracy"] = train_accuracy
    diagnosis["test_accuracy"] = test_accuracy
    diagnosis["accuracy_difference"] = train_accuracy - test_accuracy
    
    # Interpretación
    if train_accuracy > 0.95 and test_accuracy > 0.95:
        if abs(train_accuracy - test_accuracy) < 0.03:
            diagnosis["conclusion"] = "El modelo tiene buen rendimiento en ambos conjuntos, pero podría estar sobreajustado si el problema es complejo."
        else:
            diagnosis["conclusion"] = "El modelo muestra signos de sobreajuste."
    elif train_accuracy < 0.7:
        diagnosis["conclusion"] = "El modelo tiene alto sesgo (subajuste)."
    elif train_accuracy - test_accuracy > 0.1:
        diagnosis["conclusion"] = "El modelo tiene alta varianza (sobreajuste)."
    else:
        diagnosis["conclusion"] = "El modelo muestra un equilibrio razonable entre sesgo y varianza."
    
    return diagnosis

def plot_complexity_analysis(model_class, X_train, X_test, y_train, y_test, 
                            param_name, param_range, random_state=42):
    """
    Analiza cómo cambia el rendimiento del modelo con diferentes niveles de complejidad.
    
    Args:
        model_class: Clase del modelo (no instanciado)
        X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba
        param_name: Nombre del parámetro de complejidad (ej: 'max_depth')
        param_range: Lista de valores para el parámetro
        random_state: Semilla aleatoria
        
    Returns:
        Gráfica de análisis de complejidad
    """
    train_scores = []
    test_scores = []
    
    for param_value in param_range:
        # Crear y entrenar modelo con el parámetro específico
        params = {param_name: param_value, 'random_state': random_state}
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        # Evaluar en train y test
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, 'o-', color='blue', label='Train')
    plt.plot(param_range, test_scores, 'o-', color='green', label='Test')
    plt.xlabel(f'Valor de {param_name}')
    plt.ylabel('Accuracy')
    plt.title(f'Análisis de complejidad: Efecto de {param_name}')
    plt.legend()
    plt.grid(True)
    
    return plt

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    
    # Cargar datos procesados
    df = pd.read_csv(PROCESSED_DATA_DIR / "startup_data_processed.csv")
    
    # Identificar columnas que podrían causar fuga de datos
    future_info_cols = [col for col in df.columns if 'closed_at_' in col]
    id_cols = [col for col in df.columns if 'id_c:' in col]
    
    print(f"Eliminando {len(future_info_cols)} columnas con información futura (closed_at_)")
    print(f"Eliminando {len(id_cols)} columnas de identificadores específicos")
    
    # Separar features y target
    X = df.drop([TARGET_COLUMN, 'labels'] + future_info_cols + id_cols, axis=1)
    y = df[TARGET_COLUMN]
    
    print(f"Dimensiones del conjunto de datos después de la limpieza: {X.shape}")
    
    # División en train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    print("\nAnalizando sesgo-varianza para el modelo de árbol de decisión...")
    
    # Entrenar modelo con parámetros optimizados para reducir sobreajuste
    dt_classifier = DecisionTreeClassifier(
        max_depth=5,  # Limitar la profundidad
        min_samples_leaf=15,  # Requerir más muestras por nodo hoja
        random_state=RANDOM_STATE
    )
    dt_classifier.fit(X_train, y_train)
    
    # Diagnóstico
    diagnosis = diagnose_bias_variance(dt_classifier, X_train, y_train, X_test, y_test)
    print("\nDiagnóstico de sesgo-varianza:")
    for key, value in diagnosis.items():
        print(f"{key}: {value}")
    
    # Evaluación con validación cruzada para una estimación más robusta
    cv_scores = cross_val_score(dt_classifier, X, y, cv=5, scoring='accuracy')
    print(f"\nPrecisión con validación cruzada (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Análisis de complejidad
    print("\nRealizando análisis de complejidad...")
    max_depths = [1, 2, 3, 5, 10, None]
    plt = plot_complexity_analysis(
        DecisionTreeClassifier,
        X_train, X_test, y_train, y_test,
        param_name="max_depth",
        param_range=max_depths,
        random_state=RANDOM_STATE
    )
    
    # Guardar gráfico
    plt.savefig(FIGURES_DIR / "complexity_analysis.png", dpi=300, bbox_inches="tight")
    print(f"Análisis de complejidad guardado como 'complexity_analysis.png'")
    
    # Mostrar las 20 características más importantes
    if hasattr(dt_classifier, 'feature_importances_'):
        importances = dt_classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(20, len(X.columns))
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Características más Importantes")
        plt.barh(range(top_n), importances[indices[:top_n]], align="center")
        plt.yticks(range(top_n), [X.columns[i] for i in indices[:top_n]])
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
        print(f"Importancia de características guardada como 'feature_importance.png'")