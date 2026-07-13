# Dashboard (Streamlit) — FraudShield AI

## Objetivo

Construir una interfaz visual e interactiva para que un usuario (analista
de fraude, o para fines de demo) pueda evaluar transacciones individuales
del modelo entrenado, con su explicación SHAP, y consultar el historial de
transacciones de una tarjeta específica.

## Por qué Streamlit

Se eligió Streamlit sobre construir un frontend completo desde cero por
ser la herramienta estándar de la industria para prototipos rápidos de
dashboards de ML — permite iterar sobre la funcionalidad sin invertir en
ingeniería de frontend, reutilizando directamente los artefactos ya
guardados en la Etapa 10 (`modelo_final.pkl`, `feature_cols.pkl`,
`test_encoded_final.parquet`).

## Estructura del dashboard (`app.py`, raíz del proyecto)

### Sección 1: análisis individual de una transacción

- Selector de `TransactionID` en la barra lateral.
- Predicción del modelo (probabilidad de fraude) junto con un veredicto
  visual claro (rojo = sospechosa, verde = legítima), comparado contra la
  etiqueta real (`isFraud`) para contexto durante la demo.
- **Detalles de negocio de la transacción:** monto, producto, marca y tipo
  de tarjeta, y número de tarjeta (`card1`) — reconstruidos a partir de
  las columnas dummy de One-Hot Encoding mediante una función
  (`obtener_categoria_activa`) que busca cuál columna de un prefijo dado
  está activa en la fila.
- **Explicación SHAP (waterfall plot)** de la predicción específica,
  reutilizando `shap.TreeExplainer` sobre el modelo ya cargado.

### Sección 2: historial de la tarjeta actual

Muestra automáticamente todas las transacciones de test asociadas al
mismo `card1` que la transacción seleccionada en la Sección 1 — sin
necesidad de un campo de búsqueda manual adicional, ya que `card1_actual`
se deriva directamente de la transacción activa en cada ejecución del
script. Incluye una tabla (TransactionID, monto, probabilidad predicha,
etiqueta real) y un gráfico de barras donde el color indica la verdad
real (rojo = fraude, azul = legítimo) y la altura la confianza del modelo,
permitiendo evaluar de un vistazo si el modelo identificó correctamente
los fraudes históricos de esa tarjeta.

## Decisiones de diseño e iteración

- Se descartó una versión con búsqueda manual de tarjeta vía
  `session_state`, tras encontrar un comportamiento no obvio de Streamlit
  (un widget con `key` prioriza el valor guardado en `session_state` sobre
  el parámetro `value=` en ejecuciones posteriores, impidiendo la
  actualización automática esperada). Se simplificó a que el historial de
  tarjeta se calcule siempre a partir de la transacción activa — más
  simple, sin necesidad de sincronización manual de estado, y con mejor
  experiencia de usuario (un flujo, sin pasos adicionales).
- Todo el código de transformación (limpieza de nombres de columna) se
  extrajo a una función `limpiar_nombres` reutilizable, evitando duplicar
  la misma lógica en tres lugares del archivo.
- `@st.cache_resource` evita recargar el modelo y los datos en cada
  interacción del usuario, cargándolos una sola vez por sesión.

## Próximo paso

Etapa 12: crear la API (FastAPI), para exponer el modelo como un servicio
programático, más allá de la interfaz visual de este dashboard.
