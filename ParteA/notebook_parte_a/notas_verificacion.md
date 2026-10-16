# Notas de verificación — Parte A: Transfer Learning

Auditoría final del notebook `TransferLearningWorkshop.ipynb` (y su copia entregable
`grupoSamuelHerrera_CristianCabarcas_EmmanuelAlvarez_LAB#2A.ipynb`). Ambos archivos son
idénticos (74 celdas: 49 markdown + 25 código).

> Convención de escritura/ejecución: el notebook se **escribe en local** (sin GPU, sin datos,
> sin pesos) y se **ejecuta con "Run all" en Google Colab**. Por eso las respuestas empíricas
> quedan como plantillas y hay valores marcados como `SUPUESTO` a confirmar en Colab.

---

## 1. Validación nbformat

- `nbformat.validate()` → **OK** en ambos archivos (74 celdas cada uno).
- Único aviso: `MissingIDFieldWarning` preexistente en celdas base (no es error; no bloquea).

**Resultado: PASA.**

---

## 2. TODO y `= None` pendientes

No queda **ningún stub ejecutable** sin completar. El escaneo por texto arroja coincidencias,
pero todas son **falsos positivos** verificados celda por celda:

### Comentarios `# TODO n` (ya implementados debajo)
| Celda | TODO | Estado real |
|------|------|-------------|
| 14 | TODO 1 (transforms) | Implementado: `trans = transforms.Compose([Resize(256), CenterCrop(224), ToTensor(), std_normalize])` |
| 29 | TODO 2 (extracción features) | Implementado: `feature_tensor = torch.zeros(num, 512*7*7)`, `label_tensor = torch.zeros(num)`, loop con `.flatten(1)` y batch parcial |
| 31 | TODO 3 (clasificador lineal) | Implementado: `net = nn.Linear(512*7*7, 2)` |
| 34 | TODO 4a/4b/4c (M2 congelado) | Implementado: carga DEFAULT, `classifier = nn.Linear(25088,2)`, congela `features`, cuenta entrenables por código |
| 41 | TODO 5 (fine-tuning) | Implementado: recarga checkpoint, descongela `features`, entrena `lr=1e-4` |
| 51 | TODO 6a/6b/6c (ResNet-18) | Implementado: congela todo, `fc.in_features` por código → `Linear(.,2)`, entrena 2 épocas |

Los comentarios `# TODO` se dejaron **a propósito** como rótulo didáctico; el código real está
inmediatamente debajo de cada uno.

### Coincidencias `= None` (todas legítimas)
| Celda | Línea | Por qué es correcto |
|------|------|----------------------|
| 5 | `loss_fn=None, optimizer=None, writer=None` | Valores por defecto en la firma de `train(...)` |
| 41 | `vgg = torchvision.models.vgg16(weights=None)` | Reconstrucción de arquitectura antes de `load_state_dict` (no se descargan pesos) |
| 43 | `vgg_sched = ...vgg16(weights=None)` | Igual, para el experimento de scheduler |
| 61 | `tiempo_eff = None` | Inicializador previo a la medición con `time.time()`; se reasigna si hay datos |
| 73 | `weights=None` (M2 y M3) | Smoke_Test: construcción sin descargar pesos, con tensores aleatorios |

**Resultado: PASA — 0 stubs reales pendientes.**

---

## 3. Celdas con `<!-- COMPLETAR tras ejecución -->` (10 en total)

Son respuestas **empíricas** que exigen números de la ejecución en Colab. Hay que rellenarlas
tras "Run all":

| Celda | Sección / Pregunta |
|------|---------------------|
| 44 | Conclusión del experimento de **scheduler** (preguntas 3 y 7) |
| 47 | Conclusión del experimento de **data augmentation** (pregunta 2) |
| 54 | Respuesta empírica Actividad 6: **¿ResNet-18 más rápido que VGG?** (tiempos por época) |
| 60 | (OPCIONAL) Comparación accuracy/tiempo **EfficientNet-B0 vs ResNet-18 vs VGG-16** |
| 62 | (OPCIONAL) Conclusión de la comparación — pregunta 8 |
| 63 | Encabezado Sec. 10 (nota general sobre respuestas empíricas) |
| 65 | Pregunta 2 (data augmentation) — tabla de conclusión |
| 66 | Pregunta 3 (scheduler) — conclusión |
| 70 | Pregunta 7 (referencia a pregunta 3) — conclusión |
| 71 | Pregunta 8 (OPCIONAL, EfficientNet) — tabla comparativa |

Además, en la celda de verificación de estructura (Sec. 2) hay un `IMAGEFOLDER_ROOT` que puede
quedar como **`SUPUESTO: confirmar en Colab`** si en local no se detectan clases (comportamiento
esperado). Confírmalo tras la descarga de Kaggle.

---

## 4. Confianza por respuesta teórica

Escala: **Alto** = respuesta estándar, bien asentada y correctamente expresada; **Medio** =
correcta pero con matices/dependencias de contexto que conviene revisar; **Bajo** = requiere
verificación adicional. (Las preguntas empíricas no se puntúan aquí: dependen de la ejecución.)

| # | Pregunta teórica | Dónde | Confianza | Observación |
|---|------------------|-------|-----------|-------------|
| A1 | Origen de media/std de ImageNet y por qué importan con pesos preentrenados | Sec. 2 + Preg. 5 | **Alto** | Explicación canónica (estadísticas del set de entrenamiento; `(x-mean)/std`); correcta. |
| A2a | ¿Por qué la última capa de VGG tiene 1000 salidas? | Sec. 3 | **Alto** | 1000 clases de ImageNet; trivial y correcto. |
| A2b | ¿Qué hace Dropout y por qué regulariza? | Sec. 3 | **Alto** | Descripción estándar de desactivación aleatoria; correcta. |
| A2c | Nº total de parámetros de VGG-16 | Sec. 3 | **Alto** | No se afirma en prosa; se remite a la salida de `summary` (anti-invención correcta). |
| A4 | ¿Por qué congelar las convolucionales antes de entrenar la cabeza? | Sec. 6 | **Alto** | Argumento correcto: gradientes grandes de la cabeza aleatoria dañarían pesos preentrenados. |
| A3 / P6 | `CrossEntropyLoss` vs `NLLLoss` + `LogSoftmax`; por qué no duplicar LogSoftmax | Sec. 5 + Preg. 6 | **Alto** | Correcto: `CrossEntropyLoss == NLLLoss(LogSoftmax(logits))`; añadir LogSoftmax lo aplica dos veces. |
| A6 | ¿Qué es un residual block y qué problema resuelve? | Sec. 8 | **Alto** | Skip connections, aprender el residuo, mitiga vanishing gradients / degradación. Correcto. |
| P1 | ¿Cuándo NO conviene transfer learning? (≥2 escenarios) | Sec. 10 | **Alto** | Escenarios válidos: dominio distinto (médico/satelital/multiespectral) y dataset grande. Sólido. |
| P4 | ¿Por qué ResNet usa BatchNorm y VGG no? Efecto en entrenamiento | Sec. 10 | **Medio** | Correcto en lo esencial (contexto histórico 2014 vs 2015; BN estabiliza activaciones, permite redes profundas y lr mayores). Matiz a revisar: la afirmación de BN como "regularizador por ruido de batch" es cierta pero discutida en la literatura reciente; mencionar que existe `vgg16_bn` ya está incluido y es un buen matiz. |
| P5 | Origen de media/std y qué pasaría con otros valores | Sec. 10 | **Alto** | Amplía A1 correctamente; buen matiz sobre usar stats propias solo al entrenar desde cero. |

**Promedio de confianza teórica: Alto.** Solo P4 queda en **Medio** por un matiz (el efecto
regularizador de BatchNorm conviene presentarlo como secundario/discutido, no como su función
principal).

---

## 5. Veredicto final

- nbformat: **OK** (ambos archivos).
- Stubs TODO / `= None` reales pendientes: **0**.
- Plantillas empíricas por completar en Colab: **10** (listadas en §3) + confirmar `IMAGEFOLDER_ROOT`.
- Respuestas teóricas: todas correctas; **9/10 Alto, 1/10 Medio** (P4, matiz menor).

El notebook está **listo para el "Run all" en Colab**. Tras ejecutarlo, rellenar las 10 celdas
con el marcador `<!-- COMPLETAR tras ejecución -->` usando los valores reales (accuracy, tiempos,
comparación de curvas) y confirmar los valores `SUPUESTO` de la verificación de estructura.
