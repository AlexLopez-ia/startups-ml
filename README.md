# Startups ML Project

## Tabla de contenidos
- [¿De qué trata este repositorio?](#de-qu%C3%A9-trata-este-repositorio)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación y requisitos](#instalaci%C3%B3n-y-requisitos)
- [Uso](#uso)
  - [Preprocesamiento](#preprocesamiento)
  - [Ingeniería de características](#ingenier%C3%ADa-de-caracter%C3%ADsticas)
  - [Entrenamiento y evaluación](#entrenamiento-y-evaluaci%C3%B3n)
  - [Visualización](#visualizaci%C3%B3n)
- [Tests](#tests)
- [Buenas prácticas aplicadas](#buenas-pr%C3%A1cticas-aplicadas)
- [Resultados](#resultados)

---

## ¿De qué trata este repositorio?
Este proyecto muestra el ciclo completo de un case study de Machine Learning para predecir el éxito o fracaso de startups según sus características y su historial de financiación.

## Estructura del proyecto
```text
startups-ml/
├── data/                  # Datos originales y procesados
│   ├── raw/               # Datos crudos
│   ├── interim/           # Datos limpiados (sin escalado)
│   └── processed/         # Datos finales (preprocesados)
├── notebooks/             # Notebooks de EDA y feature engineering
├── src/                   # Código fuente
│   ├── data/              # Preprocesamiento y gestión de datos
│   ├── features/          # Funciones de ingeniería de características
│   ├── models/            # Entrenamiento, predicción y evaluación
│   └── visualization/     # Visualización reutilizable
├── tests/                 # Tests unitarios con pytest
├── models/                # Modelos exportados (.pkl)
├── images/                # Gráficos y visualizaciones
├── requirements.txt       # Dependencias Python
└── README.md              # Documentación principal
```

## Instalación y requisitos
- Python 3.9+
- Crear y activar un entorno virtual:
  ```bash
  python -m venv venv
  source venv/bin/activate   # o venv\Scripts\activate en Windows
  ```
- Instalar dependencias:
  ```bash
  pip install -r requirements.txt
  ```

## Uso

### Preprocesamiento
Genera datos limpios e intermedios:
```bash
python -m src.data.preprocess
```

### Ingeniería de características
Las funciones están en `src.features.feature_engineering`:
```python
from src.features.feature_engineering import create_features

df_feat = create_features(df)
```

### Entrenamiento y evaluación
```python
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model

model = train_model(X_train, y_train, my_model)
acc, report, cm = evaluate_model(model, X_test, y_test)
```

### Visualización
Funciones en `src.visualization.visualize` para generar gráficos y SHAP.

## Tests
Ejecuta todos los tests:
```bash
pytest -v
```

## Buenas prácticas aplicadas
- Estructura modular y separada por responsabilidad
- Tipado, docstrings y copias seguras de DataFrame
- Cobertura de tests alta (>90%)
- Manejo de errores y warnings controlados
- Serialización con joblib y pipelines de sklearn

## Resultados
- **Accuracy en test:** ~0.97
- Gráficos en `/images`
- Modelo final: `/models/final_pipeline.pkl`
