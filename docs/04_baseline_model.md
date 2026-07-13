# Modelo Base (Baseline) — FraudShield AI

## Objetivo

Establecer una línea base de desempeño simple e interpretable, contra la
cual se compararán los modelos más complejos de etapas posteriores
(Feature Engineering, comparación XGBoost/LightGBM/CatBoost, optimización
de hiperparámetros).

## Split de datos: temporal, no aleatorio

Se ordenó el dataset por `TransactionDT` y se partió 80% (más antiguo) para
entrenamiento, 20% (más reciente) para prueba — sin mezclar aleatoriamente.

**Justificación:** un sistema de detección de fraude en producción siempre
predice sobre transacciones futuras respecto a los datos con los que fue
entrenado. Un split aleatorio mezclaría pasado y futuro entre train y test,
lo cual no representa las condiciones reales de uso (data leakage temporal).
Esta decisión es independiente de si el tiempo mostró o no señal predictiva
en el EDA (Etapa 3) — son dos preguntas distintas: "¿el tiempo predice
fraude?" vs. "¿debo simular condiciones reales de producción al evaluar?".

- Train: 472,432 filas (hasta `TransactionDT` = 12,192,842)
- Test: 118,108 filas (desde `TransactionDT` = 12,192,900)
- Tasa de fraude: 3.51% (train) vs 3.44% (test) — consistente, sin sesgo de
  proporción introducido por el split.

## Features utilizadas (deliberadamente simples)

`TransactionAmt`, `ProductCD`, `card4`, `card6`, `hour_of_day` — variables
con evidencia de relación con `isFraud` ya confirmada en el EDA (Etapa 3).

Las variables categóricas (`ProductCD`, `card4`, `card6`) se codificaron con
**One-Hot Encoding** (`pd.get_dummies`), apropiado para variables nominales
sin orden natural entre categorías (a diferencia de Label Encoding, que
introduciría una relación de distancia/orden inexistente).

**Incidencia resuelta:** al codificar train y test por separado, dos
categorías de `card6` con muestra mínima (`charge card`, `debit or credit`)
no aparecieron en test, generando un desalineamiento de columnas. Se
resolvió con `X_test.reindex(columns=X_train.columns, fill_value=0)`.

## Modelo: Regresión Logística

`LogisticRegression(max_iter=1000, random_state=42)`, sin Feature Scaling
aplicado todavía (pendiente para una futura iteración de comparación).

## Resultados

| Umbral | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| 0.5 (default) | 0.000 | 0.000 | 0.000 | 0.1136 |
| 0.124 (óptimo F1) | 0.187 | 0.224 | 0.204 | 0.1136 |

**Hallazgo clave:** con el umbral por defecto (0.5), el modelo predice
"no fraude" para el 100% de las transacciones de test (accuracy ≈ 96.6%,
pero Precision/Recall/F1 = 0). Esto confirma en la práctica, con un
experimento propio, por qué accuracy es una métrica inválida en este
problema.

Sin embargo, el **PR-AUC (0.1136) es más de 3 veces la tasa base de fraude
(~0.034)** — evidencia de que el modelo sí capturó señal real, y que el
problema era el umbral de decisión, no la ausencia de aprendizaje. El
umbral óptimo según F1 (que balancea Precision y Recall al 50%) es 0.124.

## Limitación reconocida

El umbral que maximiza F1 no es necesariamente el umbral óptimo para el
negocio: F1 pondera Precision y Recall por igual, pero la Etapa 1 estableció
que un falso negativo (fraude no detectado) cuesta más que un falso positivo
(fricción). Elegir el umbral final de producción requeriría una función de
costo de negocio explícita (costo estimado por FN y por FP), pendiente para
una etapa posterior del proyecto. Por ahora se documentan ambos umbrales
(0.5 y 0.124) como referencia.

## Próximo paso

Etapa 5: Feature Engineering — tratamiento formal de los 15 bloques de
nulidad en columnas V, imputación, posible Feature Scaling, y variables
derivadas (ej. monto relativo por tarjeta/producto).
