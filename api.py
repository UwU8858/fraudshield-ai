"""
API de FraudShield AI — FastAPI
Expone el modelo entrenado como un servicio HTTP. Recibe los datos crudos
de una transacción, aplica el mismo pipeline de Feature Engineering usado
en entrenamiento, y devuelve la predicción de fraude.

Uso: uvicorn api:app --reload
Documentación interactiva: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib
import sys
import os

import logging
from datetime import datetime

logging.basicConfig(
    filename='logs/predicciones.log',
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import transform_feature_engineering, limpiar_nombre_columna, GRUPOS_V

app = FastAPI(title="FraudShield AI API", version="1.0")


class TransaccionInput(BaseModel):
    """Datos crudos de una transacción, como campos flexibles clave-valor."""
    datos: Dict[str, Any]


@app.on_event("startup")
def cargar_artefactos():
    """Carga el modelo y las estadísticas de Feature Engineering UNA SOLA VEZ al iniciar."""
    global modelo, stats, feature_cols
    modelo = joblib.load('models/modelo_final.pkl')
    stats = joblib.load('models/feature_engineering_stats.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    print("Modelo y artefactos cargados correctamente.")


@app.get("/")
def raiz():
    return {"mensaje": "FraudShield AI API activa. Ve a /docs para la documentación interactiva."}


@app.post("/predecir")
def predecir(transaccion: TransaccionInput):
    """
    Recibe los datos crudos de una transacción y devuelve la probabilidad
    de fraude según el modelo entrenado.
    """
    try:
        df_input = pd.DataFrame([transaccion.datos])

        # Forzamos conversión numérica en columnas que puedan serlo, sin
        # romper columnas genuinamente de texto (ProductCD, card4, etc.).
        # errors='ignore' fue removido en pandas 3.0, así que replicamos
        # su comportamiento manualmente con errors='coerce' + verificación.
        for col in df_input.columns:
            convertido = pd.to_numeric(df_input[col], errors='coerce')
            if convertido.notna().sum() > 0 or df_input[col].isna().all():
                df_input[col] = convertido

        df_input['hour_of_day'] = (df_input['TransactionDT'] // 3600) % 24

        df_procesado = transform_feature_engineering(df_input, stats, GRUPOS_V)

        cols_categoricas = ['ProductCD', 'card4', 'card6', 'P_emaildomain_grouped',
                             'R_emaildomain_grouped', 'DeviceInfo_grouped', 'id_30_grouped',
                             'id_31_grouped', 'id_33_grouped', 'M1', 'M2', 'M3', 'M4', 'M5',
                             'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23',
                             'id_27', 'id_28', 'id_29', 'id_34', 'id_35', 'id_36', 'id_37',
                             'id_38', 'DeviceType']
        cols_presentes = [c for c in cols_categoricas if c in df_procesado.columns]
        df_encoded = pd.get_dummies(df_procesado, columns=cols_presentes, dummy_na=True)

        df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)
        df_encoded.columns = [limpiar_nombre_columna(c) for c in df_encoded.columns]

        probabilidad = modelo.predict_proba(df_encoded)[0, 1]

        logging.info(
            f"TransactionID={transaccion.datos.get('TransactionID', 'N/A')} | "
            f"probabilidad={round(float(probabilidad), 4)} | "
            f"prediccion={'fraude' if probabilidad >= 0.5 else 'legitima'}"
        )

        return {
            "probabilidad_fraude": round(float(probabilidad), 4),
            "prediccion": "fraude" if probabilidad >= 0.5 else "legitima"
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Falta el campo requerido: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la transacción: {e}")