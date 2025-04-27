# Startups ML Project

## Tabla de contenidos
- [¿De qué trata este repositorio?](#de-qu%C3%A9-trata-este-repositorio)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Pasos](#pasos)
    - [1. Preprocesamiento y análisis exploratorio](#1-preprocesamiento-y-an%C3%A1lisis-exploratorio)
    - [2. Entrenamiento y evaluación de modelos](#2-entrenamiento-y-evaluaci%C3%B3n-de-modelos)
    - [3. Guardar y reutilizar el pipeline](#3-guardar-y-reutilizar-el-pipeline)
    - [4. Uso del CLI (main.py)](#4-uso-del-cli-mainpy)
    - [5. Servir el modelo con una API (FastAPI)](#5-servir-el-modelo-con-una-api-fastapi)
    - [6. (Opcional) Interfaz visual con Streamlit](#6-opcional-interfaz-visual-con-streamlit)
- [Buenas prácticas aplicadas](#buenas-pr%C3%A1cticas-aplicadas)
- [Resultados y visualizaciones](#resultados-y-visualizaciones)

---

## ¿De qué trata este repositorio?
Este proyecto muestra el ciclo completo de un caso de machine learning en el mundo real: predecir el éxito o fracaso de startups a partir de sus características iniciales y su historial de financiación.

Incluye:
- Análisis exploratorio, modelado, tuning y evaluación en notebooks
- Exportación de pipelines y modelos finales
- Un CLI profesional para entrenar, evaluar y predecir desde terminal
- Una API REST con FastAPI para servir el modelo y aceptar archivos CSV de entrada

---

## Estructura del proyecto
```text
startups-ml/
│
├── data/                    # Datasets originales y procesados
├── notebooks/               # Notebooks por fase del proyecto
├── models/                  # Modelos entrenados (.pkl)
├── images/                  # Visualizaciones y gráficos SHAP
├── src/
│   ├── data/                # Scripts de gestión de datos
│   ├── models/              # Scripts de entrenamiento, predicción y evaluación
│   └── visualization/       # Funciones de visualización reutilizables
├── main.py                  # CLI para entrenamiento, evaluación y predicción
├── app.py                   # API REST (FastAPI)
├── requirements.txt         # Dependencias Python
└── README.md                # Este archivo
```

---

## Requisitos
- Python 3.9+
- Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## Pasos

### 1. Preprocesamiento y análisis exploratorio
- Explora y limpia los datos en los notebooks de `/notebooks`.
- Realiza análisis exploratorio, visualizaciones y preprocesamiento (encoding, imputación, etc).

### 2. Entrenamiento y evaluación de modelos
- Entrena y ajusta modelos (Random Forest, XGBoost, etc.) en los notebooks.
- Evalúa con métricas como accuracy, precision, recall, f1-score y matriz de confusión.
- Analiza la importancia de las variables con SHAP.

### 3. Guardar y reutilizar el pipeline
- Exporta el pipeline entrenado (preprocesado + modelo) como `.pkl` en `/models`.
- Exporta conjuntos de datos de test y ejemplos en `/data`.

### 4. Uso del CLI (`main.py`)
Entrena, evalúa y predice desde terminal:

```bash
# Entrenar el modelo
python main.py --train

# Evaluar el modelo en test
python main.py --evaluate

# Predecir sobre nuevos datos
python main.py --predict data/nuevos_datos.csv

# Ver ayuda
python main.py --help
```

### 5. Servir el modelo con una API (FastAPI)
Lanza la API con:

```bash
uvicorn app:app --reload
```

- Accede a la documentación interactiva en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Sube un archivo CSV en `/predict` y obtén predicciones al instante.

### 6. (Opcional) Interfaz visual con Streamlit
Puedes crear una interfaz visual sencilla para cargar archivos y mostrar resultados de forma amigable para usuarios no técnicos.

---

## Buenas prácticas aplicadas
- Estructura modular y profesional de carpetas
- Separación de lógica de datos, modelos y visualización
- Uso de pipelines para reproducibilidad
- Validación y manejo de errores en la API
- Documentación clara en notebooks y código
- Ejemplo de integración ML + API listo para producción/demostración

---

## Resultados y visualizaciones
- **Accuracy en test:** ~0.97
- **Matriz de confusión, SHAP y gráficos clave** en `/images`
- **Modelo final** guardado en `/models/final_rf_pipeline.pkl`

> **Pendiente:**
> - Subir alguna imagen relevante del modelo (por ejemplo, matriz de confusión o gráfico SHAP) a la carpeta `/images` y enlazarla aquí en el README.
> - Construir una interfaz visual sencilla (por ejemplo, con Streamlit) para probar el modelo de forma interactiva.
