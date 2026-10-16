# Contexto: Laboratorio 2 de Deep Learning (EAFIT)

## Proyecto
Dos partes, un notebook entregable por parte (se completan los TODO de los notebooks existentes):
- ParteA/notebook_parte_a/TransferLearningWorkshop.ipynb (Transfer Learning con VGG-16 / ResNet-18, Cats vs Dogs)
- ParteB/notebook_parte_b/UNet_Workshop.ipynb (U-Net y ResNetUNet, Oxford-IIIT Pet)
Fuente de verdad de los requisitos: Instrucciones_Lab2_ParteAyB/Laboratorio 2 DL 1.md
Si el notebook y el .md difieren, manda el .md; reporta la diferencia, no la resuelvas en silencio.

## Entorno
- El código se ESCRIBE aquí (sin GPU y sin datos) y se EJECUTA en Google Colab (GPU).
- Las carpetas data_parte_a y data_parte_b están vacías a propósito. No intentes descargar datasets.
- Usa rutas relativas ('data/...') y variables configurables (DATA_ROOT, CKPT_DIR) para que el mismo notebook corra en local y en Colab.
- No instales paquetes ni ejecutes entrenamientos sin preguntar.
- Verificación permitida localmente: validar el .ipynb con nbformat y pruebas de shapes con tensores aleatorios y weights=None (sin descargar pesos).

## Reglas anti-invención (obligatorias)
1. Solo afirma lo que esté en el .md, en los notebooks o en documentación oficial que conozcas con certeza. Si no sabes, escribe "No sé / no verificable aquí".
2. Nunca inventes resultados, métricas, tiempos, estructura de carpetas de un dataset, firmas de API ni referencias bibliográficas.
3. Lo que no puedas verificar sin ejecutar, márcalo como SUPUESTO y crea una celda de verificación que lo compruebe en Colab.
4. Las respuestas que dependan de resultados experimentales se dejan como plantilla con <!-- COMPLETAR tras ejecución -->, junto con el código que imprime los números necesarios.
5. En las respuestas teóricas, antes de escribirlas haz una revisión interna de cada afirmación; en `notas_verificacion.md` (fuera del notebook) anota para cada respuesta un nivel de confianza (alto/medio/bajo) y qué habría que comprobar en documentación.
6. Cambios mínimos: conserva las celdas y funciones base, y no reescribas lo que no está marcado como TODO salvo que sea un error demostrable (explícalo).
7. Todo el texto del notebook va en español.