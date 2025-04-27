from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List
import io

# Cargar el modelo solo una vez al inicio
MODEL_PATH = "models/final_rf_pipeline.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error cargando el modelo: {e}")

app = FastAPI(title="API de predicción para startups-ml")

# Esquema para predicción por JSON (opcional, para entrada directa)
class StartupFeatures(BaseModel):
    features: dict

@app.get("/")
def read_root():
    return {"mensaje": "API de predicción de éxito de startups. Usa /predict para obtener predicciones."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe un archivo CSV, predice y devuelve los resultados.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible.")
    # Leer el CSV recibido
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo el archivo: {e}")

    # Validación mínima: comprobar columnas
    expected_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df.columns
    if not all(col in df.columns for col in expected_cols):
        raise HTTPException(
            status_code=400,
            detail=f"El archivo debe contener las columnas: {expected_cols}"
        )

    # Hacer predicción
    try:
        preds = model.predict(df)
        # Si quieres probabilidades: probs = model.predict_proba(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    # Devolver resultados
    return {
        "predicciones": preds.tolist(),
        "n_muestras": len(preds)
    }