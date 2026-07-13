# Explicabilidad del Modelo (SHAP) — FraudShield AI

## Objetivo

Entender no solo que el modelo (LightGBM, configuración default) funciona,
sino **por qué** toma cada decisión — tanto a nivel global (qué variables
importan en general) como a nivel individual (por qué predijo fraude en
una transacción específica).

## Metodología

Se usó `shap.TreeExplainer`, optimizado para modelos basados en árboles,
calculando los valores SHAP sobre el conjunto `train_final` completo. Para
la visualización (summary plot), se usó una submuestra aleatoria de 50,000
transacciones — el cálculo pesado ya se realizó sobre el modelo completo;
la submuestra solo limita cuántos puntos se dibujan, sin afectar la validez
de las conclusiones.

## Hallazgos del Summary Plot (importancia global)

- **`TransactionAmt` es la variable más importante del modelo**, a pesar de
  haber sido clasificada como señal moderada/débil de forma aislada en el
  EDA (Etapa 3). Esto no es una contradicción: sugiere que la variable se
  vuelve informativa en combinación con otras (interacciones que un modelo
  de árboles captura y que la correlación de Pearson, lineal, no detecta).
  Es consistente con la hipótesis que motivó la feature derivada
  `amt_relative_to_card`.
- **Confirmación directa de hallazgos del EDA:** `card6_debit` empuja hacia
  menos fraude y `card6_credit` hacia más fraude (coincide con las tasas
  2.4% vs. 6.7% ya calculadas); `ProductCD_R` también aparece entre las
  variables influyentes, coherente con las diferencias de tasa de fraude
  por producto.
- **Variables nunca analizadas individualmente en el EDA resultan clave:**
  varias columnas `C`, `D` y `V` (`C1`, `C13`, `C14`, `D2`, `D4`, `V70`,
  `V258`, `V294`, `V308`) aparecen entre las más importantes del modelo,
  validando la decisión de no descartar variables solo por correlación de
  Pearson baja.

## Caso individual 1: fraude verdadero detectado con alta confianza

Predicción final `f(x) = 3.465` (escala log-odds), partiendo de un valor
base `E[f(X)] = -4.506` (fuertemente sesgado hacia "no fraude" por el
desbalance de clases, 96.5% no-fraude). La variable dominante fue `V262`
(+2.38), seguida de `V163` (+0.68) e `id_30_grouped_otros` (+0.41,
dispositivo poco común). Una sola variable relevante empujó en dirección
contraria (`R_emaildomain_grouped_anonymous_com`, -0.23), sin ser
suficiente para cambiar la conclusión. Las 570 variables restantes
aportaron en conjunto +3.44, indicando que la predicción se sostiene en
una combinación amplia de señales, no en una o dos variables aisladas.

## Caso individual 2: falso positivo (transacción legítima marcada como fraude)

Predicción final `f(x) = 1.23` — **notablemente menor** que la confianza
del fraude verdadero (3.465), a pesar de que ambos casos cruzan el umbral
de clasificación. La variable dominante fue `V258` (+2.31, misma familia
de columnas anónimas que en el caso 1). Se observaron **señales
contradictorias entre las propias columnas C**: `C1` (+1.01) y `C4`
(+0.36) empujaron hacia fraude, mientras `C13` (-0.58) y `C14` (-0.28)
empujaron hacia no-fraude; la suma neta se inclinó incorrectamente hacia
fraude. `TransactionAmt` (monto bajo, 23.2) aportó -0.25, correctamente en
la dirección de "no fraude", pero sin peso suficiente.

## Patrón identificado: consistencia de la evidencia vs. confianza del modelo

Comparando ambos casos, se observa un patrón: **cuando las variables
apuntan de forma consistente en una sola dirección, el modelo acierta con
alta confianza; cuando hay señales contradictorias entre variables
(como las columnas C en el caso 2), la confianza final es menor y el
riesgo de error aumenta.**

**Implicación práctica para el umbral de decisión:** dado que los falsos
positivos observados tienden a tener confianza más baja que los verdaderos
positivos, el umbral de decisión (pendiente de calibrar con una función de
costo de negocio explícita, según quedó documentado en la Etapa 4) es una
palanca real para filtrar varios falsos positivos de baja confianza sin
sacrificar tantos verdaderos positivos, que en general muestran confianza
mucho más alta.

## Próximo paso

Etapa 9: detección de anomalías.
