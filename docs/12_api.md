# API (FastAPI) — FraudShield AI

## Objetivo

Exponer el modelo entrenado como un servicio HTTP programático, permitiendo
que otros sistemas (no solo humanos vía dashboard) consulten una predicción
de fraude, coherente con la arquitectura en cascada definida en la Etapa 1.

## Decisión de diseño: entrada flexible con datos crudos

Se decidió que la API reciba los **datos crudos** de una transacción (los
campos originales del dataset, antes de Feature Engineering) en vez de
requerir que el consumidor externo ya mande el vector de ~550 columnas
procesadas — mucho más realista para un sistema externo real. El esquema
de entrada usa un diccionario flexible (`Dict[str, Any]` vía Pydantic) en
vez de listar explícitamente cada uno de los ~400 campos originales,
priorizando velocidad de desarrollo para este proyecto de portafolio;
se documenta como mejora futura definir un esquema estricto y
completamente tipado.

## Estructura (`api.py`, raíz del proyecto)

- **Carga de artefactos en el evento `startup`:** el modelo, las
  estadísticas de Feature Engineering (`stats`) y la lista de columnas
  finales se cargan **una sola vez** al iniciar el servidor, no en cada
  solicitud — mismo principio que `@st.cache_resource` en el dashboard
  (Etapa 11).
- **Endpoint `POST /predecir`:** reutiliza directamente
  `transform_feature_engineering` de `src/feature_engineering.py` (Etapa
  10) para aplicar el mismo pipeline de transformación usado en
  entrenamiento, seguido de One-Hot Encoding y alineación de columnas
  (`reindex`) contra `feature_cols.pkl`.
- **Manejo de errores explícito:** `KeyError` (campo faltante) devuelve
  HTTP 400 con detalle; cualquier otro error, HTTP 500.
- **Documentación automática:** disponible en `/docs` (Swagger UI),
  generada por FastAPI sin código adicional.

## Incidencia técnica: `pd.to_numeric(errors='ignore')` removido en pandas 3.0

Al recibir datos vía JSON, los valores nulos llegan como `None` de Python
(no como `NaN` de pandas), lo que hace que pandas infiera esas columnas
como tipo `object` en vez de numérico — LightGBM rechaza columnas
`object`. La solución estándar (`pd.to_numeric(col, errors='ignore')`,
para convertir solo lo convertible sin romper columnas de texto genuino)
ya no existe en pandas 3.0.3 (el parámetro fue removido). Se reemplazó con
`errors='coerce'` combinado con una verificación manual: solo se aplica la
conversión si al menos un valor de la columna sí pudo convertirse a
número, preservando intactas las columnas genuinamente categóricas
(`ProductCD`, `card4`, etc.).

## Verificación de extremo a extremo

Se generó una transacción de prueba a partir de una fila real de
`train_transaction.csv` + `train_identity.csv` (sin la etiqueta `isFraud`,
que no debe enviarse a la API), convertida a JSON. La API respondió
correctamente con una predicción coherente
(`probabilidad_fraude: 0.0118`, `"legitima"`), confirmando que el pipeline
completo (recepción de datos crudos → Feature Engineering → One-Hot
Encoding → predicción) funciona de extremo a extremo fuera del entorno de
notebook.

## Próximo paso

Etapa 13: Dockerizar — empaquetar la API (y opcionalmente el dashboard) en
un contenedor para despliegue consistente en cualquier entorno.
