# Feature Engineering — FraudShield AI

## Objetivo

Transformar el dataset crudo (post-EDA) en un conjunto de variables listo
para modelar, resolviendo formalmente la nulidad, la codificación de
categóricas y agregando al menos una variable derivada con hipótesis de
negocio explícita.

## Principio transversal: evitar data leakage

Toda estadística usada para transformar los datos (medianas, categorías
frecuentes, medianas por grupo) se calculó **únicamente con `train`**, y se
aplicó igual a `test` sin recalcular nada — simulando las condiciones
reales de producción, donde el modelo nunca tiene acceso a datos futuros al
momento de definir sus reglas de transformación.

## 1. Nulidad de columnas V — features `is_missing` por bloque

A partir de los 15 bloques funcionales identificados en la Etapa 3
(columnas que faltan siempre en las mismas filas, correlación de nulidad =
1.0), se crearon **15 columnas binarias** (`v_grp1_is_missing` ...
`v_grp15_is_missing`), una por bloque — no una por columna individual.

**Justificación:** la nulidad es una propiedad compartida por todo el
bloque (correlación de nulidad = 1.0), mientras que el poder predictivo del
valor numérico varía columna por columna dentro del mismo bloque (ej. en
el bloque V217-V278, la correlación con `isFraud` va de 0.38 a ~0). Son dos
propiedades distintas: se separan en dos decisiones distintas.

## 2. Imputación de columnas V — mediana calculada solo en train

Las 339 columnas V numéricas se imputaron con la **mediana** (no la media,
por ser más robusta a outliers) calculada exclusivamente sobre `train`, y
aplicada igual a `test`. Verificado: 0 nulos restantes en ambos conjuntos.

## 3. Cardinalidad media — agrupación de dominios de email

`P_emaildomain` y `R_emaildomain` (~60 valores únicos cada una) se
agruparon en categorías manejables mediante análisis de cobertura marginal:

- **P_emaildomain**: top 5 dominios (gmail, yahoo, hotmail, anonymous, aol)
  cubren 88.6% de los casos con dominio conocido. Se eligió 5 en vez de 6+
  porque el análisis de cobertura marginal mostró un "codo" claro (+5.7 pts
  al pasar de 4→5 categorías, solo +1.6 pts al pasar de 5→6).
- **R_emaildomain**: mismo proceso, análisis de cobertura marginal
  calculado de forma independiente (no se reutilizó la lista de
  P_emaildomain). Nulidad de ~76% (vs ~15.6% en P_emaildomain) — hipótesis:
  R_emaildomain (destinatario) probablemente solo se llena en envíos a
  terceros, a diferencia de P_emaildomain (comprador), casi siempre
  presente.

En ambos casos, los nulos se etiquetaron como categoría propia
(`sin_dominio`), separada de `otros` (dominios poco frecuentes pero
existentes) — para no perder la posible señal MNAR de ausencia de email.

## 4. One-Hot Encoding

Se aplicó a las 14 columnas categóricas del dataset (`ProductCD`, `card4`,
`card6`, `P_emaildomain_grouped`, `R_emaildomain_grouped`, `M1`-`M9`) con
`dummy_na=True`, para preservar la distinción entre "nulo" (no sabemos el
valor) y "valor negativo explícito" (ej. M1 = 'F'), que de otra forma
quedarían representados igual (todo ceros).

Se verificó alineación de columnas entre train y test con
`reindex(columns=train.columns, fill_value=0)`. Resultado final: 483
columnas en ambos conjuntos.

## 5. Feature derivada: monto relativo por tarjeta

```
amt_relative_to_card = TransactionAmt / mediana histórica de TransactionAmt para esa card1
```

La mediana histórica se calculó agrupando por `card1` (identificador de
tarjeta específico, no la marca `card4`) **solo con datos de train**.
Un valor de 1.0 indica un monto típico para esa tarjeta; valores mucho
mayores o menores a 1.0 indican montos atípicos respecto al historial de
esa tarjeta específica.

**Manejo de tarjetas nuevas en test:** debido al split temporal, 1,293
transacciones de test corresponden a tarjetas (`card1`) sin historial en
train. Se imputaron con el valor neutral **1.0** ("monto típico"), en lugar
de dejar `NaN` o usar la media general, por no tener evidencia para asumir
un sesgo hacia arriba o abajo en esos casos.

## Próximo paso

Etapa 6: comparación empírica de XGBoost, LightGBM y CatBoost usando este
dataset ya transformado, seleccionando el modelo final con base en
evidencia (no en preferencia de portafolio).
