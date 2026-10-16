# Producto

Laboratorio académico de Deep Learning (Núcleo II, 10%) del curso de Redes Neuronales y Aprendizaje Profundo de la Universidad EAFIT. El entregable de cada parte es un notebook Jupyter completo y **ejecutado**, autocontenido, con el código resuelto y las respuestas a las preguntas en celdas Markdown.

El laboratorio tiene dos partes, cada una con peso del 50%:

- **Parte A — Transfer Learning con PyTorch**: adaptar modelos preentrenados en ImageNet (VGG-16, ResNet-18) al problema binario Cats vs. Dogs. Cubre extracción de features con backbone congelado, reemplazo del clasificador y fine-tuning por etapas, comparando arquitecturas con TensorBoard.
- **Parte B — Segmentación Semántica con U-Net**: segmentación binaria pixel a pixel sobre el Oxford-IIIT Pet Dataset. Cubre la construcción de U-Net desde cero, el uso de un encoder ResNet-18 preentrenado (ResNetUNet), pérdidas robustas al desbalance (Dice, BCE+Dice) y evaluación con IoU y Dice.

## Fuente de verdad

El documento de requisitos `Instrucciones_Lab2_ParteAyB/Laboratorio 2-DL (1).md` es la fuente de verdad. Si el notebook y el `.md` difieren, **manda el `.md`**: reporta la diferencia explícitamente, no la resuelvas en silencio.

## Convenciones del dominio

- Los notebooks están estructurados como workshops guiados: contienen celdas con marcadores `TODO N` que son los puntos a completar. La solución va justo donde está el TODO. **Cambios mínimos**: conserva las celdas y funciones base y no reescribas lo que no está marcado como TODO, salvo que sea un error demostrable (en cuyo caso explícalo).
- El contenido explicativo, los títulos y las respuestas se escriben en **español**.
- El reporte debe ser legible sin ejecutar: explica cada decisión en Markdown.

## Reglas anti-invención (obligatorias)

1. Afirma solo lo que esté en el `.md`, en los notebooks o en documentación oficial conocida con certeza. Si no, escribe "No sé / no verificable aquí".
2. Nunca inventes resultados, métricas, tiempos, estructura de carpetas de un dataset, firmas de API ni referencias bibliográficas.
3. Lo que no puedas verificar sin ejecutar márcalo como SUPUESTO y acompáñalo de una celda de verificación que lo compruebe en Colab.
4. Las respuestas que dependan de resultados experimentales se dejan como plantilla con `<!-- COMPLETAR tras ejecución -->`, junto con el código que imprime los números necesarios.
5. En respuestas teóricas, registra en `notas_verificacion.md` (fuera del notebook) un nivel de confianza (alto/medio/bajo) por respuesta y qué habría que comprobar en documentación.
