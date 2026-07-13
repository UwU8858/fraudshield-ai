# Definición del Problema de Negocio — FraudShield AI

## Contexto

FraudShield AI es un sistema de detección de fraude financiero diseñado para
identificar transacciones fraudulentas y minimizar el impacto económico del
fraude sobre el negocio, sin introducir fricción innecesaria sobre clientes
legítimos.

## Declaración del problema

> FraudShield AI busca minimizar la pérdida neta del negocio, definida como
> la suma del costo por fraude no detectado más el costo por fricción de
> falsos positivos — no maximizar la detección de fraude a cualquier costo.

## Arquitectura conceptual: detección en cascada

Los sistemas de detección de fraude en producción (Stripe Radar, PayPal,
procesadores de tarjetas) no dependen de un único modelo. Utilizan un
pipeline de dos etapas:

1. **Etapa 1 — Filtro rápido (tiempo real, milisegundos).**
   Reglas de negocio + modelo ligero que aprueba la mayoría de las
   transacciones y marca las sospechosas o en zona gris para análisis
   posterior. Prioriza velocidad sobre profundidad.

2. **Etapa 2 — Análisis profundo (asíncrono, segundos).**
   Modelo de ensemble (comparación entre XGBoost, LightGBM y CatBoost)
   con feature engineering avanzado y explicabilidad (SHAP), que examina
   a fondo las transacciones marcadas por la Etapa 1 y alimenta una cola
   de revisión o bloqueo.

### Alcance de este proyecto

Este proyecto implementa completamente la **Etapa 2**. La Etapa 1 se
documenta a nivel de diseño conceptual, pero no se construye infraestructura
de streaming en tiempo real (p. ej. Kafka), ya que el valor central del
proyecto está en ciencia de datos aplicada (feature engineering, selección
de modelos, explicabilidad, evaluación), no en ingeniería de datos en
tiempo real. La API final (FastAPI) sí expone el modelo de la Etapa 2 como
un servicio que responde en tiempo real.

## Costo asimétrico de errores

En detección de fraude, los dos tipos de error de clasificación no cuestan
lo mismo:

| Tipo de error | Significado | Costo relativo |
|---|---|---|
| Falso negativo | Fraude real clasificado como legítimo | Alto (pérdida monetaria directa, disputas, reputación) |
| Falso positivo | Transacción legítima bloqueada/marcada | Bajo (fricción, costo de verificación) |

Esta asimetría implica que **accuracy no es una métrica válida** para este
problema (los datasets de fraude son altamente desbalanceados: un modelo
que nunca predice fraude puede tener accuracy >99% y ser completamente
inútil). La métrica y el umbral de decisión se definirán más adelante en
función de esta asimetría de costos, no de forma arbitraria.

## Fuera de alcance (por ahora)

- Infraestructura de streaming en tiempo real (Kafka, feature store online).
- Implementación productiva de la Etapa 1 (filtro rápido).
- Optimización de latencia sub-100ms.

Estas decisiones podrán revisarse si el proyecto avanza hacia una fase de
MLOps más madura.

## Próximo paso

Etapa 2 del proyecto: comprensión del dataset.
