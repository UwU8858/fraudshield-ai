# FraudShield AI

Sistema de detección de fraude financiero con Machine Learning — proyecto de portafolio que cubre el ciclo completo de un producto de ciencia de datos: desde la definición del problema de negocio hasta el despliegue de un modelo en producción, con explicabilidad, monitoreo y una interfaz interactiva.

## Tabla de contenido

- [Resumen del proyecto](#resumen-del-proyecto)
- [Arquitectura](#arquitectura)
- [Dataset](#dataset)
- [Resultados principales](#resultados-principales)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo correr el proyecto](#cómo-correr-el-proyecto)
- [Documentación por etapa](#documentación-por-etapa)

## Resumen del proyecto

FraudShield AI busca minimizar la **pérdida neta del negocio** — la suma del costo por fraude no detectado más el costo por fricción de falsos positivos — en lugar de simplemente maximizar la detección de fraude a cualquier costo. El proyecto está diseñado conceptualmente como un sistema de **detección en cascada** de dos etapas (filtro rápido en tiempo real + análisis profundo asíncrono), inspirado en cómo operan sistemas reales de la industria (Stripe Radar, PayPal). Esta implementación se enfoca en la **Etapa 2** de esa arquitectura: el modelo de análisis profundo con feature engineering avanzado y explicabilidad.

## Arquitectura

```
Datos crudos (Kaggle IEEE-CIS)
        │
        ▼
  Feature Engineering (src/feature_engineering.py)
        │
        ▼
  Modelo LightGBM (models/modelo_final.pkl)
        │
        ├──► API REST (api.py, FastAPI)
        ├──► Dashboard interactivo (app.py, Streamlit)
        └──► Capa complementaria de anomalías (src/anomaly_detection.py)
```

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle, datos reales de Vesta Corporation): ~590,000 transacciones, ~434 columnas originales, con un desbalance de clases de 96.5% legítimas / 3.5% fraude.

Se eligió sobre el dataset clásico de tarjetas de crédito (ULB) porque este último viene con features ya transformadas por PCA (anónimas), lo que elimina la posibilidad de hacer feature engineering real — una de las habilidades centrales que este proyecto busca demostrar.

## Resultados principales

- **Modelo final:** LightGBM (configuración por defecto). Se comparó contra XGBoost, CatBoost y una regresión logística baseline. LightGBM se eligió por su balance entre calidad (PR-AUC ≈ 0.48 en test) y velocidad de entrenamiento, relevante para la etapa de optimización de hiperparámetros.
- **Feature Engineering:** 15 features de nulidad por bloque en las columnas anónimas `V*` (detectadas automáticamente por correlación de nulidad), imputación con mediana, agrupación de variables categóricas de alta cardinalidad, y una feature derivada de monto relativo por tarjeta.
- **Optimización de hiperparámetros:** se probó extensivamente, mostrando mejoras aparentes en validación que no se sostenían en el conjunto de test — evidencia real de por qué nunca se debe optimizar sobre el conjunto de evaluación final. Se conservó la configuración por defecto.
- **Explicabilidad (SHAP):** confirma que las variables más importantes coinciden con los hallazgos del análisis exploratorio (tipo de tarjeta, tipo de producto), y revela que columnas nunca analizadas individualmente en el EDA (bloques `V`, `C`, `D`) también son relevantes para el modelo.
- **Función de costo de negocio:** se construyó con datos reales de la industria (costo de revisión manual ≈ $3.47 USD, multiplicador de costo total de fraude ≈ 4.41x), encontrando que el umbral de decisión que minimiza el costo real de negocio es sustancialmente más bajo que el umbral óptimo según métricas puramente estadísticas (F1).

## Estructura del repositorio

```
├── docs/               # Documentación detallada de cada una de las 15 etapas del proyecto
├── models/             # Modelo entrenado y artefactos de Feature Engineering (.pkl)
├── notebooks/          # Notebooks de exploración, EDA y bitácora del proyecto
├── src/
│   ├── feature_engineering.py   # Funciones fit/transform reutilizables
│   ├── train.py                 # Pipeline de entrenamiento del modelo principal
│   └── anomaly_detection.py     # Capa complementaria (Isolation Forest)
├── api.py              # API REST (FastAPI)
├── app.py              # Dashboard interactivo (Streamlit)
├── Dockerfile          # Contenedor para desplegar la API
└── requirements.txt
```

## Cómo correr el proyecto

### 1. Clonar el repositorio y preparar el entorno

```bash
git clone https://github.com/UwU8858/fraudshield-ai.git
cd fraudshield-ai
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Descargar el dataset

Este repositorio no incluye los datos crudos (por tamaño). Descarga el dataset [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) de Kaggle y coloca los archivos CSV en `data/raw/`.

### 3. Entrenar el modelo

```bash
python src/train.py
```

Esto genera `models/modelo_final.pkl` y los datasets procesados en `data/processed/`.

### 4. Correr la API

```bash
uvicorn api:app --reload
```

Documentación interactiva disponible en `http://localhost:8000/docs`.

### 5. Correr el dashboard

```bash
streamlit run app.py
```

### 6. Correr con Docker (alternativa a los pasos 4-5 para la API)

```bash
docker build -t fraudshield-api .
docker run -d -p 8000:8000 --name fraudshield-container fraudshield-api
```

## Documentación por etapa

El proyecto se desarrolló en 15 etapas, cada una documentada en detalle en `docs/`:

1. [Definición del problema de negocio](docs/01_problem_definition.md)
2. [Comprensión del dataset](docs/02_dataset_overview.md)
3. [Análisis exploratorio de datos](docs/03_eda.md)
4. [Modelo base (baseline)](docs/04_baseline_model.md)
5. [Feature Engineering](docs/05_feature_engineering.md)
6. [Comparación de modelos](docs/06_model_comparison.md)
7. [Optimización de hiperparámetros](docs/07_hyperparameter_tuning.md)
8. [Explicabilidad (SHAP)](docs/08_explainability.md)
9. [Detección de anomalías](docs/09_anomaly_detection.md)
10. [Pipeline reproducible](docs/10_pipeline.md)
11. [Dashboard](docs/11_dashboard.md)
12. [API](docs/12_api.md)
13. [Dockerización](docs/13_docker.md)
14. [Monitoreo](docs/14_monitoring.md)
