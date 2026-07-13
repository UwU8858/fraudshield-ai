# Monitoreo — FraudShield AI

## Objetivo

Diseñar una capa de monitoreo para el modelo en producción, cerrando una
tarea pendiente desde la Etapa 4/7: definir una función de costo de
negocio real (no solo métricas técnicas como PR-AUC o F1), y usarla para
determinar el umbral de decisión óptimo.

## Función de costo de negocio, respaldada con datos reales de la industria

Se investigaron cifras reales de la industria de prevención de fraude en
vez de usar estimaciones arbitrarias:

- **Costo de Falso Positivo = $3.47 USD**, basado en datos de Signifyd
  sobre el costo promedio de una revisión manual de fraude.
- **Costo de Falso Negativo = monto de la transacción × 4.41**, aplicando
  el "multiplicador de costo total del fraude" del estudio *True Cost of
  Fraud* de LexisNexis, que captura el costo total del fraude (no solo el
  monto directo perdido, sino también multas, comisiones y esfuerzo de
  investigación).

```python
def calcular_costo_total(y_true, y_pred, montos, costo_fp=3.47, multiplicador_fn=4.41):
    """
    Costo de Falso Positivo: $3.47 USD (Signifyd, revisión manual)
    Costo de Falso Negativo: monto de la transacción x 4.41 (multiplicador
    'True Cost of Fraud' de LexisNexis)
    """
    falsos_positivos = (y_true == 0) & (y_pred == 1)
    falsos_negativos = (y_true == 1) & (y_pred == 0)
    costo_fp_total = falsos_positivos.sum() * costo_fp
    costo_fn_total = (montos[falsos_negativos] * multiplicador_fn).sum()
    return costo_fp_total + costo_fn_total
```

## Hallazgo: el umbral óptimo de negocio es mucho más bajo que el óptimo estadístico

Se calculó el costo total en el conjunto de test para un rango extenso de
umbrales (0.0001 a 0.15), encontrando el mínimo en:

| Umbral | Costo total |
|---|---|
| 0.5 (default) | $2,033,121 |
| 0.124 (óptimo F1, Etapa 4) | ~similar al rango 0.10-0.15 |
| 0.05 | $776,081 |
| **0.0058 (óptimo de negocio)** | **$313,817** |
| 0 (marcar todo como sospechoso) | $395,733 |

El umbral óptimo real (0.0058) es drásticamente más bajo que cualquier
umbral usado hasta ahora en el proyecto. Se verificó que no es un
artefacto de rango de búsqueda: el costo del caso extremo ("marcar todo
como sospechoso", umbral=0) es peor ($395,733) que el óptimo encontrado
($313,817), confirmando que es un mínimo genuino, no un límite de borde.

**Interpretación:** dado que un fraude no detectado puede costar cientos
de dólares (monto × 4.41) mientras una falsa alarma cuesta solo $3.47, el
modelo necesita operar de forma mucho más "alarmista" de lo que sugerían
las métricas puramente estadísticas (F1, PR-AUC) para minimizar el costo
real de negocio — confirmando cuantitativamente la asimetría de costos
identificada cualitativamente desde la Etapa 1.

## Logging de predicciones — implementado

A diferencia del resto de esta sección (diseñada pero no implementada por
alcance del proyecto), el logging básico de predicciones **sí se
implementó y verificó funcionando**, tanto en ejecución directa como
dentro del contenedor Docker (Etapa 13). Se usó el módulo estándar
`logging` de Python en `api.py`:

```python
import logging

logging.basicConfig(
    filename='logs/predicciones.log',
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)
```

Cada predicción exitosa registra `TransactionID`, probabilidad y
veredicto, con timestamp automático. Esto es el ingrediente mínimo real
para cualquier análisis futuro de drift, auditoría, o cálculo de costo
real acumulado en producción.

**Incidencia resuelta:** al reconstruir la imagen de Docker tras agregar
el logging, el contenedor arrancaba y se detenía inmediatamente
(`FileNotFoundError: '/app/logs/predicciones.log'`) — la carpeta `logs/`
no existía dentro de la imagen, ya que el `Dockerfile` solo copiaba
`api.py`, `src/` y `models/`. Se resolvió agregando `RUN mkdir -p logs` en
el `Dockerfile`, antes del `CMD` final, para crear la carpeta durante la
construcción de la imagen. Se verificó el log accediendo al contenedor en
ejecución con `docker exec fraudshield-container cat logs/predicciones.log`.

## Diseño de monitoreo (piezas no implementadas, documentadas para producción futura)

Las siguientes piezas se documentan como diseño, sin implementación de
código: requieren infraestructura que corre de forma continua en el
tiempo (servidor persistente, base de datos de logs, scheduler para jobs
periódicos), fuera del alcance razonable de este proyecto de portafolio.

1. **Detección de data drift:** comparar periódicamente las estadísticas
   de las transacciones entrantes en producción contra las de `train`
   original (por ejemplo, con pruebas estadísticas como Kolmogorov-Smirnov
   sobre las variables más importantes de SHAP), para detectar cambios en
   el comportamiento de los datos que puedan indicar necesidad de
   reentrenamiento.
3. **Monitoreo de costo de negocio real:** en producción, la métrica más
   relevante no es PR-AUC sino el costo total calculado con la función de
   esta etapa — un aumento sostenido en el costo por transacción podría
   indicar degradación del modelo o cambio en los patrones de fraude.
4. **Reentrenamiento periódico:** dado que el fraude evoluciona (motivo
   original de explorar detección de anomalías en la Etapa 9), se
   recomienda reentrenar el modelo con datos recientes de forma periódica
   (por ejemplo, mensual), y no depender de un modelo estático
   indefinidamente.

## Próximo paso

Etapa 15: despliegue.
