"""
Dashboard de FraudShield AI — Streamlit
Permite seleccionar una transacción de test y ver la predicción del
modelo junto con su explicación SHAP, además de consultar el historial
completo de una tarjeta específica (card1).

Uso: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudShield AI", layout="wide")

st.title("FraudShield AI — Detección de Fraude")
st.markdown("Sistema de detección de fraude financiero con explicabilidad (SHAP)")


def limpiar_nombres(columnas):
    return [c.replace(':', '_').replace('/', '_').replace('.', '_').replace(' ', '_') for c in columnas]


@st.cache_resource
def cargar_modelo_y_datos():
    """Carga el modelo, stats, y datos de test una sola vez (cacheado)."""
    modelo = joblib.load('models/modelo_final.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    test_encoded = pd.read_parquet('data/processed/test_encoded_final.parquet')
    explainer = shap.TreeExplainer(modelo)
    return modelo, feature_cols, test_encoded, explainer


modelo, feature_cols, test_encoded, explainer = cargar_modelo_y_datos()

# ---------------------------------------------------------------------
# Sección 1: selección de una transacción individual
# ---------------------------------------------------------------------

st.sidebar.header("1. Selecciona una transacción")

indice_seleccionado = st.sidebar.selectbox(
    "TransactionID",
    options=test_encoded['TransactionID'].tolist()
)

fila = test_encoded[test_encoded['TransactionID'] == indice_seleccionado]
X_fila = fila[feature_cols]

X_fila_clean = X_fila.copy()
X_fila_clean.columns = limpiar_nombres(X_fila_clean.columns)

prediccion_proba = modelo.predict_proba(X_fila_clean)[0, 1]

col1, col2 = st.columns(2)

with col1:
    st.metric("Probabilidad de Fraude", f"{prediccion_proba:.1%}")

with col2:
    umbral = 0.5
    es_fraude = prediccion_proba >= umbral
    if es_fraude:
        st.error("⚠️ Transacción registrada como SOSPECHOSA")
    else:
        st.success("✅ Transacción registrada como LEGÍTIMA")

isFraud_real = fila['isFraud'].values[0]
st.caption(f"Etiqueta real (isFraud): {'Fraude' if isFraud_real == 1 else 'No fraude'}")

st.subheader("Detalles de la transacción")


def obtener_categoria_activa(fila, prefijo):
    """Busca cuál columna dummy de un prefijo dado está activa (=True/1) en esta fila."""
    columnas_prefijo = [c for c in fila.columns if c.startswith(prefijo)]
    for col in columnas_prefijo:
        if fila[col].values[0] == 1 or fila[col].values[0] == True:
            return col.replace(prefijo, '').strip('_')
    return "Desconocido"


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Monto", f"${fila['TransactionAmt'].values[0]:,.2f}")

with col2:
    producto = obtener_categoria_activa(fila, 'ProductCD_')
    st.metric("Producto", producto)

with col3:
    marca_tarjeta = obtener_categoria_activa(fila, 'card4_')
    st.metric("Marca de tarjeta", marca_tarjeta)

with col4:
    tipo_tarjeta = obtener_categoria_activa(fila, 'card6_')
    st.metric("Tipo", tipo_tarjeta)

with col5:
    st.metric("Número de tarjeta (card1)", int(fila['card1'].values[0]))

st.subheader("Explicación de la predicción (SHAP)")

shap_values_fila = explainer.shap_values(X_fila_clean)

fig, ax = plt.subplots(figsize=(7, 4))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_fila[0],
        base_values=explainer.expected_value,
        data=X_fila_clean.iloc[0],
        feature_names=X_fila_clean.columns.tolist()
    ),
    show=False
)
st.pyplot(fig)

# ---------------------------------------------------------------------
# Sección 2: historial completo de la tarjeta de la transacción actual
# ---------------------------------------------------------------------

card1_actual = int(fila['card1'].values[0])

st.header(f"Historial de la tarjeta {card1_actual}")

transacciones_tarjeta = test_encoded[test_encoded['card1'] == card1_actual]

st.caption(f"Se encontraron {len(transacciones_tarjeta)} transacciones para esta tarjeta en el conjunto de test.")

X_tarjeta_clean = transacciones_tarjeta[feature_cols].copy()
X_tarjeta_clean.columns = limpiar_nombres(X_tarjeta_clean.columns)

predicciones_tarjeta = modelo.predict_proba(X_tarjeta_clean)[:, 1]

resumen_tarjeta = pd.DataFrame({
    'TransactionID': transacciones_tarjeta['TransactionID'].values,
    'Monto': transacciones_tarjeta['TransactionAmt'].values,
    'Probabilidad de Fraude (modelo)': predicciones_tarjeta,
    'Fraude Real': transacciones_tarjeta['isFraud'].values
}).sort_values('TransactionID')

st.dataframe(resumen_tarjeta, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(10, 4))
colores = ['red' if f == 1 else 'steelblue' for f in resumen_tarjeta['Fraude Real']]
ax2.bar(range(len(resumen_tarjeta)), resumen_tarjeta['Probabilidad de Fraude (modelo)'], color=colores)
ax2.set_xlabel("Transacción (orden cronológico)")
ax2.set_ylabel("Probabilidad de Fraude")
ax2.set_title(f"Historial de predicciones — Tarjeta {card1_actual}")
ax2.axhline(y=0.5, color='gray', linestyle='--', label='Umbral 0.5')
ax2.legend()
st.pyplot(fig2)

st.caption("🔴 Rojo = fraude real | 🔵 Azul = transacción legítima real")