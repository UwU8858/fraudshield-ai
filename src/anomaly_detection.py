"""
Script de detección de anomalías complementaria (FraudShield).
Genera una capa adicional de priorización basada en Isolation Forest,
identificando el segmento de transacciones más atípicas para revisión manual.

Este script es COMPLEMENTARIO al modelo principal (LightGBM, ver train.py) —
no reemplaza la clasificación de fraude, sirve como señal adicional de
priorización para el 1% de casos más extremos.

Uso: python src/anomaly_detection.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

from feature_engineering import GRUPOS_V


# Variables seleccionadas en la Etapa 9, mediante importancia de SHAP
# filtrada a numéricas continuas, excluyendo identificadores (ej. card1)
FEATURES_ANOMALIAS = ['TransactionAmt', 'V70', 'C14', 'C13', 'C1', 'D2',
                       'C11', 'C5', 'V91', 'D4', 'V294', 'V308']

# Basado en el hallazgo de la Etapa 9: solo el segmento más extremo (1%)
# mostró señal real de asociación con fraude
CONTAMINATION_DEFAULT = 0.01


def entrenar_detector_anomalias(X_train, features=FEATURES_ANOMALIAS,
                                  contamination=CONTAMINATION_DEFAULT):
    """
    Entrena Isolation Forest sobre un subconjunto de variables numéricas
    ya validadas (Etapa 9) como relevantes para el problema de fraude.
    """
    X_subset = X_train[features]

    modelo_anomalias = IsolationForest(
        n_estimators=200,
        max_samples=0.5,
        contamination=contamination,
        random_state=42,
        n_jobs=1
    )
    modelo_anomalias.fit(X_subset)

    return modelo_anomalias


def evaluar_anomalias(modelo_anomalias, X, y, features=FEATURES_ANOMALIAS):
    """
    Evalúa la relación entre las anomalías detectadas y el fraude real,
    solo con fines de diagnóstico (no se usa isFraud para entrenar).
    """
    X_subset = X[features]
    anomaly_pred = modelo_anomalias.predict(X_subset)

    comparacion = pd.DataFrame({'anomaly_pred': anomaly_pred, 'isFraud': y.values})
    tasa_anomalo = comparacion[comparacion['anomaly_pred'] == -1]['isFraud'].mean()
    tasa_normal = comparacion[comparacion['anomaly_pred'] == 1]['isFraud'].mean()

    print(f"Tasa de fraude en transacciones anómalas: {tasa_anomalo:.2%}")
    print(f"Tasa de fraude en transacciones normales: {tasa_normal:.2%}")

    return anomaly_pred


def main():
    print("Cargando datos procesados (generados por train.py)...")
    train_encoded = pd.read_parquet('data/processed/train_encoded_final.parquet')
    feature_cols = joblib.load('models/feature_cols.pkl')

    from feature_engineering import limpiar_nombre_columna
    X_train = train_encoded[feature_cols].copy()
    X_train.columns = [limpiar_nombre_columna(col) for col in X_train.columns]
    y_train = train_encoded['isFraud']

    print("Entrenando detector de anomalías...")
    modelo_anomalias = entrenar_detector_anomalias(X_train)

    print("Evaluando relación con fraude real...")
    evaluar_anomalias(modelo_anomalias, X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo_anomalias, 'models/modelo_anomalias.pkl')
    print("Modelo de anomalías guardado en models/modelo_anomalias.pkl")


if __name__ == "__main__":
    main()