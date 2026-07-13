# Despliegue — FraudShield AI

## Objetivo

Llevar la API a un entorno accesible públicamente, más allá de `localhost`,
para que el proyecto sea demostrable como parte de un portafolio sin
necesitar que quien lo revise replique el entorno local.

## Control de versiones: GitHub

Se creó una cuenta de GitHub y se inicializó Git en el proyecto local. Se
configuró un `.gitignore` explícito para excluir: el entorno virtual
(`venv/`), los datos crudos y procesados (`data/raw/`, `data/processed/`,
por tamaño), archivos temporales de Python y Jupyter, logs de ejecución, y
resultados de búsquedas de hiperparámetros — manteniendo el repositorio
enfocado en código y artefactos esenciales (modelo entrenado, stats de
Feature Engineering, documentación).

**Incidencias resueltas durante la subida:**
- Se detectó que varios archivos de documentación (`docs/*.md`) y el
  cuaderno de notas habían sido descargados a Windows pero nunca movidos
  al proyecto real en WSL — solo existían como descargas sueltas. Se
  recuperaron y copiaron a las rutas correctas (`docs/`, `notebooks/`)
  antes del commit inicial.
- Una carpeta con nombre corrupto (`docs/02_dataset_overview.mddocs/`,
  generada por un error de concatenación en una sesión anterior) contenía
  en realidad los 13 documentos de las etapas — se movió su contenido a
  `docs/` directamente y se eliminó la carpeta vacía.
- Autenticación con GitHub requirió generar un Personal Access Token
  (GitHub ya no acepta contraseña de cuenta para operaciones Git).
- Al agregar el README después del commit inicial, un conflicto de
  historiales divergentes (`fatal: Need to specify how to reconcile
  divergent branches`) se resolvió configurando la estrategia de merge
  explícitamente (`git config pull.rebase false`) y hacendo
  `git pull --allow-unrelated-histories`.

## Plataforma de despliegue: Render

Se eligió Render por su capa gratuita, integración directa con GitHub
(despliegue automático desde la rama `main`), y detección automática del
`Dockerfile` ya construido en la Etapa 13 — sin necesitar configuración
adicional de build.

**Configuración usada:**
- Source: repositorio de GitHub `fraudshield-ai`, rama `main`.
- Language/Environment: Docker (autodetectado).
- Instance Type: Free.

**Limitación conocida del plan gratuito:** la instancia se "duerme" tras
un periodo de inactividad, y la primera solicitud tras ese estado puede
tardar 50+ segundos en responder mientras el servicio "despierta" — una
limitación esperada y documentada del nivel gratuito, no un error del
despliegue.

## Verificación

Tras el despliegue exitoso, se confirmó que la API responde correctamente
desde la URL pública (`https://fraudshield-ai-h0va.onrender.com`), tanto
en el endpoint raíz como en `/predecir`, con el mismo comportamiento
verificado localmente y en Docker en las etapas anteriores.

## Cierre del proyecto

Con esta etapa se completa el ciclo de las 15 etapas planeadas desde el
inicio del proyecto: desde la definición del problema de negocio hasta un
sistema desplegado, accesible públicamente, con documentación completa de
cada decisión técnica y metodológica tomada en el camino.
