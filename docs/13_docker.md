# Dockerización — FraudShield AI

## Objetivo

Empaquetar la API en un contenedor Docker, para que pueda ejecutarse de
forma idéntica en cualquier máquina con Docker instalado, sin depender de
replicar manualmente el entorno virtual de Python ni la estructura de
carpetas del proyecto original.

## Instalación y configuración de Docker

Se instaló Docker Desktop para Windows (arquitectura AMD64, verificada
con `echo $env:PROCESSOR_ARCHITECTURE` en PowerShell), con integración
WSL2 habilitada explícitamente en Settings → Resources → WSL Integration
(no viene activada por defecto). Se resolvió un error de permisos inicial
(`permission denied... docker.sock`) agregando el usuario al grupo
`docker` (`sudo usermod -aG docker $USER`), reabriendo la sesión de WSL
para que el cambio tomara efecto.

## `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Se usa `python:3.11-slim` como imagen base (misma versión de Python que el
proyecto), copiando primero `requirements.txt` e instalando dependencias
antes que el resto del código — aprovecha el cacheo de capas de Docker:
si solo cambia el código pero no las dependencias, no se reinstala nada.

## `requirements.txt` específico para producción

Se construyó una lista de dependencias mínima y específica para lo que la
API realmente necesita (`fastapi`, `uvicorn`, `pandas`, `lightgbm`,
`joblib`, `pydantic`, `pyarrow`, `scikit-learn`), en vez de usar
`pip freeze` completo del entorno de desarrollo — evita incluir
dependencias de exploración (`matplotlib`, `jupyter`, `xgboost`,
`catboost`, `shap`, `streamlit`) que la API no usa, manteniendo la imagen
más ligera.

## Incidencias técnicas resueltas

1. **`libgomp.so.1: cannot open shared object file`:** LightGBM requiere
   la librería de sistema OpenMP (`libgomp`) para su paralelización
   interna, ausente en la imagen `python:3.11-slim` por ser minimalista.
   Se resolvió instalándola explícitamente vía `apt-get install libgomp1`
   dentro del Dockerfile, antes de instalar las dependencias de Python.

2. **`ModuleNotFoundError: No module named 'sklearn'`:** el objeto
   guardado del modelo (`modelo_final.pkl`) tiene una dependencia interna
   de scikit-learn (el wrapper `LGBMClassifier` usa esa API por debajo),
   que no se había incluido en el `requirements.txt` inicial al construirlo
   manualmente en vez de usar `pip freeze`. Se agregó `scikit-learn` a la
   lista de dependencias.

Ambos errores se detectaron mediante `docker logs <nombre_contenedor>`,
que expone el traceback completo de Python del proceso fallido dentro del
contenedor — herramienta esencial de diagnóstico cuando un contenedor
arranca y se detiene inmediatamente (`docker ps` lo muestra vacío,
`docker ps -a` revela el estado "Exited" con el código de salida).

## Verificación de extremo a extremo

Tras construir la imagen (`docker build -t fraudshield-api .`) y correr el
contenedor (`docker run -d -p 8000:8000 --name fraudshield-container
fraudshield-api`), se confirmó que la API responde de forma idéntica a la
ejecución directa con `uvicorn` fuera de Docker: mismo resultado exacto
(`probabilidad_fraude: 0.0118`, `"legitima"`) para la misma transacción de
prueba, confirmando el empaquetado exitoso.

## Próximo paso

Etapa 14: monitoreo.
