# Comparación de Modelos — FraudShield AI

## Objetivo

Comparar empíricamente XGBoost, LightGBM y CatBoost (configuración base,
sin optimizar) sobre el dataset ya transformado en la Etapa 5, y elegir un
modelo candidato para la Etapa 7 (optimización de hiperparámetros) con
evidencia, no por preferencia de portafolio.

## Resultados

| Modelo | PR-AUC | F1 (umbral 50/50) | Precision | Recall | Tiempo entrenamiento |
|---|---|---|---|---|---|
| Regresión Logística (baseline, Etapa 4) | 0.114 | 0.204 | 0.187 | 0.224 | segundos |
| XGBoost | 0.506 | 0.506 | 0.602 | 0.437 | 50.5s |
| LightGBM | 0.514 | 0.496 | 0.580 | 0.433 | **8.4s** |
| CatBoost | **0.556** | **0.544** | 0.606 | 0.493 | 106.1s |

Los tres modelos de gradient boosting superan ampliamente al baseline de
regresión logística (~4.5x en PR-AUC), confirmando la presencia de señal no
lineal importante en el dataset (bloques de columnas V) que un modelo
lineal no podía capturar.

## Decisión: LightGBM

CatBoost obtiene la mejor calidad (PR-AUC 0.556 vs 0.514 de LightGBM), pero
a un costo de **~12.6x más tiempo de entrenamiento** (106.1s vs 8.4s) por
una mejora de calidad relativamente modesta (~4 puntos de PR-AUC, ~5 puntos
de F1).

**Criterio de decisión:** la Etapa 7 (optimización de hiperparámetros)
requiere entrenar el modelo elegido docenas o cientos de veces con
distintas combinaciones de parámetros (búsqueda de grid/random/bayesiana).
En ese contexto, el costo de tiempo de CatBoost se multiplica
significativamente, mientras que la ventaja de calidad marginal no
justifica ese costo acumulado. **Se elige LightGBM** como modelo candidato
principal a llevar a la Etapa 7, priorizando eficiencia de iteración sobre
una ganancia de calidad pequeña — bajo el supuesto explícito de que el
proyecto necesitará reentrenar/experimentar frecuentemente, no solo
entrenar una vez.

**Nota de contexto:** esta decisión no es universal — en un escenario donde
el modelo se entrena una sola vez y solo se usa para inferencia por un
periodo largo, o donde el tiempo de entrenamiento no fuera una restricción
real, CatBoost sería la elección defendible por su mejor calidad bruta.

## Incidencias técnicas resueltas durante esta etapa

- **Columnas de texto sin procesar:** `train_identity` incluye columnas
  categóricas (`id_12`-`id_38`, `DeviceType`, `DeviceInfo`) que no se
  habían codificado en la Etapa 5 (el foco había sido `train_transaction`).
  Se resolvió con el mismo criterio de cardinalidad ya usado: One-Hot
  directo para baja cardinalidad, y una función genérica
  (`agrupar_por_top_optimo`, registrada con un decorador `@registrar` en un
  diccionario de funciones del proyecto) para cardinalidad alta
  (`DeviceInfo`, `id_30`, `id_31`, `id_33`), que detecta automáticamente el
  punto de corte "top N" según ganancia marginal de cobertura.
- **Caracteres especiales en nombres de columnas:** LightGBM rechaza
  columnas con caracteres no alfanuméricos (generadas por `get_dummies` a
  partir de valores como `IP_PROXY:ANONYMOUS` o `Trident/7.0`). Se creó una
  copia de `X_train`/`X_test` con nombres limpiados (`re.sub` reemplazando
  cualquier carácter no alfanumérico por `_`), dejando intactos los
  DataFrames originales para XGBoost/CatBoost.
- **Pérdida de trabajo por reinicio de WSL:** el notebook de trabajo nunca
  se guardó exitosamente en disco durante la sesión (quedó en 0 bytes) y
  se perdió al reiniciarse WSL por agotamiento de memoria (demasiados
  datasets y modelos grandes cargados simultáneamente). Se recuperó
  exitosamente el código completo desde el historial de IPython
  (`~/.ipython/profile_default/history.sqlite`). **Lección crítica:
  guardar el notebook frecuentemente de forma manual, y guardar el dataset
  procesado a disco (`.parquet`) para no depender de mantener todo en
  memoria.**

## Próximo paso

Etapa 7: optimización de hiperparámetros de LightGBM.
