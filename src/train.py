"""
Script de entrenamiento de FraudShield.
Ejecuta el pipeline completo: carga → split temporal → feature engineering →
entrenamiento del modelo final → guardado a disco.

Uso: python src/train.py
"""

import pandas as pd
import joblib
import os
from lightgbm import LGBMClassifier

from feature_engineering import (
    fit_feature_engineering,
    transform_feature_engineering,
    limpiar_nombre_columna,
    GRUPOS_V
)


def cargar_y_unir_datos(ruta_transaction, ruta_identity):
    """Carga los CSV originales y hace el merge (LEFT JOIN)."""
    train_transaction = pd.read_csv(ruta_transaction)
    train_identity = pd.read_csv(ruta_identity)
    df = train_transaction.merge(train_identity, on='TransactionID', how='left')
    df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24
    return df


def hacer_split_temporal(df, proporcion_train=0.8):
    """Split temporal 80/20, respetando el orden cronológico."""
    df_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
    split_point = int(len(df_sorted) * proporcion_train)
    train = df_sorted.iloc[:split_point]
    test = df_sorted.iloc[split_point:]
    return train, test


def aplicar_feature_engineering_completo(train_df, test_df, stats):
    """Aplica transform_feature_engineering + One-Hot Encoding a train y test."""
    train_fe = transform_feature_engineering(train_df, stats, GRUPOS_V)
    test_fe = transform_feature_engineering(test_df, stats, GRUPOS_V)

    cols_baja_cardinalidad = ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29',
                              'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType']
    cols_agrupadas = ['P_emaildomain_grouped', 'R_emaildomain_grouped', 'DeviceInfo_grouped',
                       'id_30_grouped', 'id_31_grouped', 'id_33_grouped']
    cols_negocio = ['ProductCD', 'card4', 'card6', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
    cols_para_encoding = cols_baja_cardinalidad + cols_agrupadas + cols_negocio

    train_encoded = pd.get_dummies(train_fe, columns=cols_para_encoding, dummy_na=True)
    test_encoded = pd.get_dummies(test_fe, columns=cols_para_encoding, dummy_na=True)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    cols_a_eliminar = ['P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'id_30', 'id_31', 'id_33']
    train_encoded = train_encoded.drop(columns=[c for c in cols_a_eliminar if c in train_encoded.columns])
    test_encoded = test_encoded.drop(columns=[c for c in cols_a_eliminar if c in test_encoded.columns])

    return train_encoded, test_encoded


def main():
    print("1. Cargando datos...")
    df = cargar_y_unir_datos(
        "data/raw/train_transaction.csv",
        "data/raw/train_identity.csv"
    )

    print("2. Haciendo split temporal...")
    train, test = hacer_split_temporal(df)

    print("3. Calculando estadísticas de Feature Engineering (fit)...")
    stats = fit_feature_engineering(train, GRUPOS_V)

    print("4. Aplicando Feature Engineering (transform)...")
    train_encoded, test_encoded = aplicar_feature_engineering_completo(train, test, stats)

    cols_excluir = ['isFraud', 'TransactionID', 'TransactionDT']
    feature_cols = [col for col in train_encoded.columns if col not in cols_excluir]

    X_train = train_encoded[feature_cols].copy()
    y_train = train_encoded['isFraud']
    X_test = test_encoded[feature_cols].copy()
    y_test = test_encoded['isFraud']

    X_train.columns = [limpiar_nombre_columna(col) for col in X_train.columns]
    X_test.columns = [limpiar_nombre_columna(col) for col in X_test.columns]

    print("5. Entrenando modelo LightGBM (configuración default)...")
    modelo = LGBMClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    print("6. Guardando modelo y estadísticas a disco...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo_final.pkl')
    joblib.dump(stats, 'models/feature_engineering_stats.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')

    print("Entrenamiento completo. Modelo guardado en models/modelo_final.pkl")

    print("Guardando datasets procesados para reutilización...")
    os.makedirs('data/processed', exist_ok=True)
    train_encoded.to_parquet('data/processed/train_encoded_final.parquet')
    test_encoded.to_parquet('data/processed/test_encoded_final.parquet')


if __name__ == "__main__":
    main()