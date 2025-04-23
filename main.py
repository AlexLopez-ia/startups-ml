"""
main.py - Punto de entrada para ejecutar tareas principales del proyecto startups-ml.

Uso:
    python main.py --train        # Entrena y guarda el modelo
    python main.py --evaluate    # Evalúa el modelo en test
    python main.py --predict     # Predice sobre nuevos datos
"""
import argparse
from pathlib import Path
from src.models import train_model, predict_model, evaluate_model
import joblib
import pandas as pd

MODELS_DIR = Path('models')
DATA_DIR = Path('data')

# Ejemplo de rutas y nombres (ajusta según tu estructura real)
MODEL_PATH = MODELS_DIR / 'final_rf_pipeline.pkl'
X_TEST_PATH = DATA_DIR / 'X_test.csv'
Y_TEST_PATH = DATA_DIR / 'y_test.csv'


def main():
    parser = argparse.ArgumentParser(description='Gestor de tareas para startups-ml')
    parser.add_argument('--train', action='store_true', help='Entrena y guarda el modelo')
    parser.add_argument('--evaluate', action='store_true', help='Evalúa el modelo en test')
    parser.add_argument('--predict', type=str, help='Ruta a CSV de datos para predecir')
    args = parser.parse_args()

    if args.train:
        print('Funcionalidad de entrenamiento: implementa aquí la llamada a tu pipeline de entrenamiento')
        # Ejemplo: train_model.train_model(...)
        pass
    elif args.evaluate:
        print('Cargando modelo y datos de test...')
        model = joblib.load(MODEL_PATH)
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH)
        acc, report, cm = evaluate_model.evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {acc}\n")
        print(report)
        print(cm)
    elif args.predict:
        print(f'Prediciendo sobre {args.predict}...')
        model = joblib.load(MODEL_PATH)
        X_new = pd.read_csv(args.predict)
        preds = predict_model.predict(model, X_new)
        print('Predicciones:')
        print(preds)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()