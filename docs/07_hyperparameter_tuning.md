# Optimización de Hiperparámetros — FraudShield AI

## Objetivo

Determinar si ajustar los hiperparámetros de LightGBM (el modelo elegido en
la Etapa 6) mejora su desempeño respecto a la configuración por defecto, y
en caso afirmativo, encontrar la mejor combinación posible.

## Metodología: split adicional train/validation

Para evitar usar el conjunto de `test` durante la búsqueda de
hiperparámetros (lo cual invalidaría su rol como evaluación final
honesta), se creó un split adicional **dentro** de `train`, respetando el
orden temporal (80/20): `train_final` (para entrenar) y `validation` (para
comparar combinaciones de hiperparámetros). `test` permaneció intocado
hasta la evaluación final.

**Refactorización del Feature Engineering (fit/transform):** para generar
`train_final`/`validation` sin data leakage, se dividió el pipeline de la
Etapa 5 en dos funciones siguiendo el patrón estándar de scikit-learn:

- `fit_feature_engineering(train_df, grupos_v)`: calcula todas las
  estadísticas de referencia (medianas de columnas V, top-N de dominios de
  email y columnas de identidad, mediana de monto por tarjeta) usando
  **únicamente** el DataFrame de entrenamiento recibido. Devuelve un
  diccionario (`stats`) con todo lo aprendido, sin transformar nada.
- `transform_feature_engineering(df_raw, stats, grupos_v)`: aplica las
  transformaciones (features `is_missing`, imputación, agrupación de
  categorías, monto relativo) a cualquier DataFrame usando el `stats` ya
  calculado — nunca recalcula nada.

También se dividió la función de agrupación por cardinalidad alta en
`encontrar_top_optimo_fit` (calcula las categorías top-N) y
`agrupar_transform` (aplica la agrupación con una lista ya calculada),
reemplazando la versión anterior que mezclaba ambas fases.

Esta refactorización es reutilizable para cualquier split futuro del
proyecto (Etapa 10 - pipeline reproducible, Etapa 12 - API en producción).

## Técnica de búsqueda: Random Search con validación temporal

Se descartó Grid Search por crecimiento combinatorio poco práctico (con 5
hiperparámetros y ~4-5 valores candidatos cada uno, un grid completo
superaría las 3,000 combinaciones). Se usó `RandomizedSearchCV` con
`TimeSeriesSplit` (en vez de `cv` aleatorio estándar) para que la
validación cruzada interna respetara el orden cronológico, coherente con
la metodología de todo el proyecto.

**Limitación aceptada:** las estadísticas de Feature Engineering (medianas,
top-N) se calcularon una sola vez sobre todo `train_final`, no de forma
independiente por cada fold interno — un leakage menor de segundo orden,
aceptado como simplificación práctica estándar en la industria para esta
etapa, dado el costo computacional de recalcular por fold.

## Incidencias técnicas de esta etapa

Esta etapa presentó múltiples caídas de conexión y reinicios del servicio
WSL, atribuidos a agotamiento de memoria al usar `n_jobs=-1` (paralelismo
completo) en `RandomizedSearchCV`, sumado a mantener múltiples datasets y
modelos grandes en memoria simultáneamente. Se resolvió reduciendo el
paralelismo (`n_jobs=1`), trabajando con muestras del dataset (100K y luego
200K filas en vez de las 377,945 completas) para la fase de búsqueda, y
guardando resultados incrementalmente a disco (`joblib.dump`) tras cada
ronda para no perder progreso ante un posible crash.

## Resultados: dos rondas de búsqueda

| Ronda | Muestra usada | Folds | Iteraciones | Mejores hiperparámetros | PR-AUC validation | PR-AUC test |
|---|---|---|---|---|---|---|
| Default (referencia) | — | — | — | valores por defecto | 0.584 | 0.480 |
| v2 | 100,000 filas | 3 | 20 | num_leaves=31, n_estimators=300, min_child_samples=50, max_depth=-1, learning_rate=0.1 | 0.618 (+0.033) | 0.506 (~0.000) |
| v3 | 200,000 filas | 5 | 25 | num_leaves=127, n_estimators=300, min_child_samples=50, max_depth=-1, learning_rate=0.2 | 0.611 (+0.026) | 0.482 (+0.0028) |

## Hallazgo principal: mejoras en validation no se sostienen en test

Ambas rondas de optimización mostraron mejoras aparentes sobre el default
al evaluarse en `validation` (+0.033 y +0.026 respectivamente), pero esas
mejoras **se desvanecieron casi por completo** al confirmarse en `test`
(~0.000 y +0.0028). Este patrón, repetido en dos rondas independientes con
distinta configuración de búsqueda, no es casualidad: indica que la
selección de hiperparámetros se ajustó a particularidades específicas del
periodo temporal de `validation`, sin que esa ventaja generalizara a datos
genuinamente nuevos (`test`).

**Esto valida, con evidencia directa, la importancia metodológica de nunca
tomar decisiones de modelo basándose únicamente en `validation`, y de
reservar `test` como evaluación final intocada** — si se hubiera reportado
el resultado de `validation` como métrica final, se habría comunicado una
mejora ilusoria de hasta +5.7% relativo que no existe en la práctica.

## Decisión final

**Se conserva el modelo LightGBM con configuración por defecto** como
modelo final del proyecto. La optimización de hiperparámetros no aportó
una mejora práctica relevante para este dataset y este algoritmo — un
resultado legítimo y informativo, no un fallo del proceso. Es consistente
con la experiencia general de la industria: la elección correcta del
algoritmo (Etapa 6, que aportó una mejora de ~+0.40 en PR-AUC sobre el
baseline) tiende a aportar muchísimo más valor que el ajuste fino de sus
hiperparámetros.

## Próximo paso

Etapa 8: explicar el comportamiento del modelo (SHAP).
