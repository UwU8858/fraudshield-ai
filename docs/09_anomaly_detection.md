# Detección de Anomalías — FraudShield AI

## Objetivo

Explorar un enfoque complementario al modelo supervisado (LightGBM): usar
detección de anomalías no supervisada (Isolation Forest) para identificar
transacciones estadísticamente atípicas, evaluando si aporta señal
adicional relevante para fraude, independientemente de las etiquetas
usadas para entrenar el modelo principal.

## Diferencia conceptual: supervisado vs. no supervisado

El modelo supervisado (LightGBM) aprende a reconocer patrones de fraude
**ya vistos** en los datos de entrenamiento etiquetados. Un detector de
anomalías no supervisado no usa la etiqueta `isFraud` para aprender —
identifica puntos estadísticamente diferentes del comportamiento general.
Su valor potencial es detectar fraude de **tipo nuevo**, que un modelo
supervisado no reconocería por no haberlo visto antes. Las etiquetas reales
sí se usan, pero únicamente para **evaluar** qué tan bien se alinean las
anomalías detectadas con el fraude conocido — nunca para entrenar el
detector.

## Selección de variables: proceso iterativo con correcciones

1. **Primer intento (descartado):** selección arbitraria basada solo en
   haber aparecido en dos gráficos de SHAP (Etapa 8).
2. **Segundo intento (descartado):** ranking por dispersión estadística
   pura (distancia entre máximo y mediana, en desviaciones estándar) sin
   relación al problema de fraude. Se identificó un error metodológico:
   dividir por una mediana cercana a cero producía ratios artificialmente
   enormes sin significado real; corregido usando distancia relativa a la
   desviación estándar, con un filtro adicional (`std > 1.0`) para evitar
   que columnas casi constantes distorsionaran el ranking.
3. **Enfoque final (usado):** combinación de ambos criterios — se calculó
   la **importancia global exacta de SHAP** (promedio del valor absoluto
   de `shap_values` por las 579 columnas), filtrando después solo
   variables numéricas continuas (excluyendo dummies binarias de One-Hot
   Encoding, que no aportan gradiente de "distancia" útil para el
   algoritmo).

**Auditoría de identificadores:** se verificó la cardinalidad de cada
variable candidata antes de aceptarla. `card1` (11,945 valores únicos,
funcionalmente un identificador de tarjeta sin magnitud real) se excluyó a
pesar de su alta importancia en SHAP — el mismo problema conceptual de
Label Encoding en variables nominales, aplicado a un contexto distinto:
"tipo numérico" no equivale a "magnitud significativa". Se estableció un
criterio para distinguir identificadores de cantidades reales: un
identificador tiene valores que casi no se repiten (cada código es único);
una cantidad real (monto, conteo) tiene valores que se repiten
naturalmente muchas veces. `V308`, con alta cardinalidad aparente pero
72% de sus valores concentrados en 0 y el resto repitiéndose miles de
veces, se confirmó como cantidad real, no identificador.

**Variables finales:** `TransactionAmt`, `V70`, `C14`, `C13`, `C1`, `D2`,
`C11`, `C5`, `V91`, `D4`, `V294`, `V308`.

## Experimentos y resultados

| Configuración | Tasa fraude en anómalas | Tasa fraude en normales | Diferencia |
|---|---|---|---|
| contamination=3.5%, max_samples default | 2.74% | 3.44% | -20% (peor que el promedio) |
| contamination=3.5%, max_samples=0.5 | 3.70% | 3.40% | +9% (señal débil) |
| **contamination=1.0%, max_samples=0.5** | **5.93%** | **3.39%** | **+75% (señal real)** |

## Hallazgo principal

Con `contamination` calcado de la tasa de fraude real (3.5%), Isolation
Forest **no mostró relación confiable** con el fraude verdadero —
resultado inicialmente contraintuitivo, pero explicable: las variables
usadas son importantes para el modelo supervisado en **combinación
compleja** con cientos de otras variables, no como indicadores aislados de
"rareza = fraude". Una transacción con valores extremos en estas variables
puede ser, con la misma probabilidad, una compra legítima atípica.

Al reducir el umbral a **contamination=1%** (aislando solo el segmento
verdaderamente más extremo, no un 3.5% amplio), sí emergió una señal real:
las transacciones más atípicas concentran casi el doble de tasa de fraude
que el resto. Esto es coherente con el propósito real de la detección de
anomalías en un sistema de fraude: **una capa adicional para priorizar
revisión de los casos más extremos**, no un sustituto o proxy directo del
modelo de clasificación supervisada.

## Conclusión de la etapa

La detección de anomalías no reemplaza al modelo supervisado (LightGBM
sigue siendo muy superior en capacidad de discriminación, con PR-AUC de
~0.48-0.58 según el conjunto evaluado, contra una señal mucho más modesta
de Isolation Forest). Su valor complementario real está en identificar,
dentro del universo ya marcado por el sistema, el subconjunto de
transacciones estadísticamente más extremas, útil como criterio adicional
de priorización para revisión manual — no como mecanismo de detección
independiente y de amplio alcance.

## Próximo paso

Etapa 10: pipeline reproducible.
