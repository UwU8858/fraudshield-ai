# Pipeline Reproducible — FraudShield AI

## Objetivo

Consolidar el trabajo disperso en notebooks (carga de datos, Feature
Engineering, entrenamiento, detección de anomalías) en un flujo
automatizado y reproducible, ejecutable con un solo comando, independiente
del orden de ejecución de celdas de un notebook.

## Estructura de código añadida

```
src/
├── __init__.py
├── feature_engineering.py   # funciones fit/transform reutilizables
├── train.py                  # pipeline del modelo principal (LightGBM)
└── anomaly_detection.py      # capa complementaria (Isolation Forest)
```

## `feature_engineering.py`

Consolida las funciones ya construidas y validadas en la Etapa 7:
`fit_feature_engineering`, `transform_feature_engineering`,
`encontrar_top_optimo_fit`, `agrupar_transform`, `limpiar_nombre_columna`,
y la constante `GRUPOS_V` (los 15 bloques de nulidad de columnas V
identificados en la Etapa 3). Sin gráficas ni exploración — solo lógica
de transformación ya probada.

## `train.py`

Ejecuta el flujo completo del modelo principal:
1. Carga y merge de `train_transaction`/`train_identity`.
2. Split temporal 80/20.
3. `fit_feature_engineering` sobre train, `transform_feature_engineering`
   sobre train y test, One-Hot Encoding con alineación de columnas.
4. Entrenamiento de LightGBM (configuración default — decisión de la
   Etapa 7, tras confirmar que la optimización de hiperparámetros no
   aportaba mejora real y consistente).
5. Guarda a disco: `modelo_final.pkl`, `feature_engineering_stats.pkl`,
   `feature_cols.pkl`, y los datasets ya procesados en `.parquet`
   (`train_encoded_final.parquet`, `test_encoded_final.parquet`), para que
   scripts posteriores no necesiten recalcular el Feature Engineering.

Se ejecuta con: `python src/train.py` desde la raíz del proyecto.

## `anomaly_detection.py`

Script **complementario y opcional**, separado deliberadamente del
pipeline principal — no se ejecuta automáticamente como parte de
`train.py`. Decisión de diseño basada en el hallazgo de la Etapa 9: la
detección de anomalías no reemplaza ni mejora la clasificación
supervisada, su valor es priorizar el 1% de transacciones más extremas
para revisión manual adicional. Lee los datos ya procesados por
`train.py` (`train_encoded_final.parquet`), entrena Isolation Forest con
la configuración validada (`contamination=0.01`, `max_samples=0.5`, las 12
variables numéricas seleccionadas y auditadas en la Etapa 9), y guarda
`modelo_anomalias.pkl`.

Se ejecuta con: `python src/anomaly_detection.py`, después de haber
corrido `train.py` al menos una vez.

## Incidencia técnica relevante: límite de memoria de WSL2

Al ejecutar `train.py` como proceso independiente, se produjo un crash por
memoria (`Out of memory: Killed process`), a pesar de que la máquina
cuenta con 32GB de RAM física. Diagnóstico: WSL2 limita por defecto su uso
de memoria a un techo interno (no configurado explícitamente), sin
aprovechar toda la RAM disponible del sistema host. Se resolvió creando
`C:\Users\<usuario>\.wslconfig` con:

```ini
[wsl2]
memory=20GB
processors=8
swap=4GB
```

seguido de `wsl --shutdown` y reinicio. Esto resolvió el problema de raíz,
explicando también varios de los crashes de memoria ocurridos en etapas
anteriores del proyecto (Etapas 6 y 7).

## Verificación de reproducibilidad

Se ejecutó `anomaly_detection.py` sobre los datos generados por `train.py`
(usando `train` completo, no la submuestra `train_final` de la Etapa 9), y
el resultado (tasa de fraude en anómalas 5.90% vs. 3.49% en normales)
replicó casi exactamente el hallazgo original de la Etapa 9 (5.93% vs.
3.39%), confirmando que el pipeline es consistente y reproducible más allá
del entorno específico del notebook.

## Próximo paso

Etapa 11: dashboard (Streamlit).
