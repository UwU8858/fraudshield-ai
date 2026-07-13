# Comprensión del Dataset — FraudShield AI

## Fuente

**IEEE-CIS Fraud Detection** (competencia de Kaggle, datos reales de transacciones
provistos por Vesta Corporation). Se utilizan únicamente los archivos de
entrenamiento, ya que son los únicos que incluyen la variable objetivo:

- `train_transaction.csv`
- `train_identity.csv`

Los archivos `test_transaction.csv` y `test_identity.csv` no se utilizan (no
tienen `isFraud`, son para la competencia original de Kaggle). El proyecto
realiza su propio split de train/test a partir de los datos de `train`.

## Estructura general

| Tabla | Filas | Columnas | Contenido |
|---|---|---|---|
| `train_transaction` | 590,540 | 394 | Datos de la transacción: montos, tarjeta, distancias, deltas de tiempo, variables anónimas `V1`-`V339`, variable objetivo `isFraud` |
| `train_identity` | 144,233 | 41 | Datos de dispositivo/identidad asociados a la transacción |

**Relación entre tablas:** solo ~24% de las transacciones (144,233 de 590,540)
tienen identidad asociada. El join entre ambas tablas deberá manejar
explícitamente las transacciones sin identidad (no es una relación 1:1).

## Variable objetivo: `isFraud`

Distribución fuertemente desbalanceada:

- No fraude (0): 96.5%
- Fraude (1): 3.5%

**Implicación:** accuracy no es una métrica válida para este problema (un
modelo trivial que siempre prediga "no fraude" obtendría ~96.5% de accuracy
sin ningún valor práctico). Se usarán métricas apropiadas para desbalance
(Precision, Recall, F1, PR-AUC) — se definirán en detalle en la etapa de
evaluación de modelos.

## Composición de tipos de columnas

| Tabla | float64 | int64 | str (categóricas) |
|---|---|---|---|
| `train_transaction` | 376 | 4 | 14 |
| `train_identity` | 23 | 1 | 17 |

### Columnas categóricas de `train_transaction`

**Grupo A — baja cardinalidad (categorías de negocio):**
`ProductCD` (5), `card4` (4), `card6` (4) — codificación directa viable.

**Grupo B — cardinalidad media:**
`P_emaildomain` (~59), `R_emaildomain` (~60) — requieren estrategia de
agrupación (ej. agrupar dominios poco frecuentes) antes de codificar, para
evitar explosión dimensional con One-Hot Encoding directo.

**Grupo C — flags de "match" (`M1`-`M9`):**
Binarias `T`/`F`/nulo, excepto `M4`, que tiene 3 categorías (`M0`, `M1`,
`M2`) y requiere tratamiento distinto al resto del grupo.

## Patrones de nulidad (missingness)

### En `train_identity`

Nulidad muy desigual entre columnas de fingerprinting de dispositivo
(`id_07`, `id_08`: ~96% nulos; otras columnas `id_*` con nulidad variable).

**Hipótesis de trabajo (MNAR — Missing Not At Random):** la ausencia de
estas señales puede correlacionar con intentos de ocultar la identidad del
dispositivo, y por tanto con mayor probabilidad de fraude. Estrategia
futura: crear features binarias `is_missing` en lugar de solo imputar.

### En `train_transaction`

Columnas con nulidad muy alta: `dist2` (93.6%), `D7` (93.4%), `D12`-`D14`
(~89-90%).

**Hallazgo confirmado:** un bloque de 12 columnas (`V138`, `V139`, `V141`,
`V142`, `V146`, `V149`, `V153`, `V154`, `V157`, `V158`, `V161`, `V162`)
presenta una matriz de correlación de nulidad de **1.0** entre todas ellas
— faltan siempre en las mismas filas exactas. Esto indica que forman un
bloque funcional (probablemente de una misma fuente de datos o cálculo
interno de Vesta), no nulos independientes.

**Estrategia futura:** crear una sola feature `is_missing` a nivel de grupo
en lugar de una por columna, para evitar redundancia. Se explorará en la
etapa de EDA si existen más bloques de este tipo entre las ~339 columnas
`V`.

## Consideraciones técnicas

- **Memoria:** `train_transaction` ocupa ~1.7 GB en memoria. El join con
  `train_identity` y el entrenamiento de modelos incrementarán este uso —
  a vigilar conforme el proyecto avance.
- **pandas 3.0.3:** el proyecto usa el nuevo backend de strings (dtype
  `str` en vez de `object`). Código o documentación de referencia escrita
  para pandas <2.x puede asumir `object`.

## Próximo paso

Etapa 3: Análisis Exploratorio de Datos (EDA) completo.