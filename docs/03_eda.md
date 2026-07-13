# Análisis Exploratorio de Datos (EDA) — FraudShield AI

## Preparación

Se realizó el merge de `train_transaction` y `train_identity` mediante
**LEFT JOIN** sobre `TransactionID`, para conservar todas las transacciones
(incluidas las que no tienen identidad asociada):

```python
df = train_transaction.merge(train_identity, on='TransactionID', how='left')
```

Resultado: `(590540, 434)` filas/columnas.

## Metodología de selección de variables

El EDA no cubrió las 434 columnas una por una. Se priorizó en este orden:

1. Variables de negocio con significado claro (`TransactionAmt`, `TransactionDT`).
2. Variables categóricas de negocio y parcialmente ofuscadas (`ProductCD`, `card4`, `card6`, `M1`-`M9`).
3. Variables anónimas (`V1`-`V339`), atacadas por **grupos automáticos**, no una por una.

## Hallazgos por variable

### TransactionAmt (monto)

- La media sugiere diferencia entre clases (149.2 fraude vs 134.5 legítima),
  pero es engañosa: existen outliers extremos en transacciones legítimas
  (máximo $31,937) que distorsionan el promedio.
- La mediana (75 fraude vs 68.5 legítima) y la forma general de ambas
  distribuciones (histogramas en escala log) son muy similares.
- **Conclusión: variable moderada/débil de forma aislada.** Candidata a
  combinarse con otras variables en Feature Engineering (ej. monto relativo
  al histórico de la tarjeta/producto).

### TransactionDT (tiempo) — hora del día

- `TransactionDT` es un contador de segundos desde un origen arbitrario, no
  una fecha real (Vesta no reveló la fecha de inicio).
- Al descomponer en hora del día (`(TransactionDT // 3600) % 24`), ambas
  clases muestran el **mismo patrón cíclico** (valle ~8-10h, pico ~17-23h
  del ciclo relativo).
- **Conclusión: no es una señal de fraude.** El fraude sigue el mismo ritmo
  de actividad general; no hay una hora donde el fraude se destaque
  proporcionalmente más.
- Día de la semana: análisis descartado en detalle por limitación de tiempo;
  se documenta que sería posible pero no se puede nombrar el día real (no se
  conoce a qué día corresponde el "día 0"). Mes: descartado — el dataset
  cubre solo ~6 meses y ya se confirmó (min/max de TransactionDT por clase)
  que no hay concentración temporal del fraude.

### ProductCD

- Señal fuerte y confiable. Tasa de fraude: C = 11.7% (n=68,519) vs
  W = 2.0% (n=439,670) — ~6x de diferencia entre la categoría más y menos
  riesgosa, ambas con muestra grande.

### card4 / card6

- `card4`: Discover tiene la tasa más alta (7.7%, n=6,651), Amex la más baja
  (2.9%, n=8,328); Visa/Mastercard (~3.4%) concentran el grueso del volumen.
- `card6`: crédito (6.7%, n=148,986) tiene ~2.7x más tasa de fraude que
  débito (2.4%, n=439,938) — ambas con muestra confiable.
- Categorías `charge card` (n=15) y `debit or credit` (n=30) **no tienen
  muestra suficiente** para ninguna conclusión estadística, ni alta ni baja.

### Columnas V — estructura de nulidad

- Se identificaron automáticamente **15 bloques funcionales** entre las 339
  columnas V, agrupando por % de nulos idéntico y confirmando con matriz de
  correlación de nulidad (todas = 1.0, sin falsos positivos).
- Estrategia futura: una feature `is_missing` por bloque (15 en total) en
  vez de 339 columnas individuales.

### Correlación de variables numéricas con isFraud

- Correlación de Pearson global: la más alta es `V257` (0.38) — relación
  **moderada**, no fuerte. La más negativa ronda -0.14 (`D8`).
- Las columnas con mayor correlación positiva (`V257`, `V246`, `V244`,
  `V242`, `V258`, `V228`...) pertenecen mayormente al bloque de nulidad de
  77.9% (`V217`-`V278`, 46 columnas).
- Dentro de ese mismo bloque, la correlación decae fuertemente: solo ~6-8
  columnas cargan señal real: el resto tiene correlación cercana a 0.
  **Compartir nulidad (patrón MNAR) no implica compartir poder predictivo
  en el valor numérico** — son propiedades distintas del mismo bloque.
- Limitación reconocida: la correlación de Pearson solo mide relaciones
  lineales. Columnas con correlación baja no se descartan en esta etapa;
  la decisión final de qué variables mantener se hará en Feature
  Engineering / selección de modelos, usando feature importance de los
  modelos entrenados (capaces de capturar relaciones no lineales).

## Conclusiones generales de la etapa

- Las variables categóricas de negocio (`ProductCD`, `card6`) muestran
  señales de fraude mucho más fuertes y claras que las variables numéricas
  continuas exploradas (`TransactionAmt`) o temporales (`hour_of_day`).
- La correlación lineal más alta encontrada en todo el dataset (V257, 0.38)
  se considera moderada, no fuerte — ninguna variable aislada explica el
  fraude por sí sola. Esto refuerza la necesidad de Feature Engineering y
  modelos capaces de combinar múltiples señales.
- Toda conclusión de tasa de fraude por categoría se valida con el tamaño
  de muestra (`count`) antes de aceptarse — una tasa extrema sobre pocos
  casos no es evidencia confiable.

## Próximo paso

Etapa 4: construir un modelo base sencillo (regresión logística).
