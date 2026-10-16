# Requirements Document

_Parte A — Transfer Learning_

## Introduction

Esta funcionalidad consiste en **completar el notebook `TransferLearningWorkshop.ipynb` (Parte A del Laboratorio 2)** para resolver, en español y de forma autocontenida, las Actividades 1 a 6, todas las preguntas de cada actividad y las 8 preguntas de investigación del enunciado (`Instrucciones_Lab2_ParteAyB/Laboratorio 2-DL (1).md`), que es la **fuente de verdad**. El tema es *transfer learning* con PyTorch sobre el problema binario Cats vs. Dogs, usando tres modelos:

- **M1** = VGG-16 con `features` fijas (congeladas) + una capa lineal.
- **M2** = VGG-16 con el clasificador reemplazado + *fine-tuning*.
- **M3** = ResNet-18 adaptado.

El código se **escribe en local** (sin GPU y sin datos) y se **ejecuta en Google Colab** (con GPU). Las carpetas `ParteA/data_parte_a/` están vacías a propósito y no se descargan datasets en local. Todo el contenido del notebook (títulos, explicaciones y respuestas) debe estar en **español**. El entregable final es un único notebook renombrado a `grupoSamuelHerrera_CristianCabarcas_EmmanuelAlvarez_LAB#2A.ipynb`.

Este documento define los requisitos. El diseño (`design.md`) y las tareas (`tasks.md`) se elaborarán en fases posteriores, tras la aprobación del usuario.

### Discrepancias detectadas entre el `.md` (fuente de verdad) y el notebook base

Durante la lectura se detectaron diferencias que, según las reglas del proyecto, **ganan a favor del `.md`** y deben documentarse explícitamente en el notebook (no resolverse en silencio):

- **D1 — Dataset [DERIVADO - fuera del .md]:** el `.md` (Actividad 1) exige el dataset de Kaggle `kunalgupta2616/dog-vs-cat-images-data`. El notebook base descarga el ZIP de Microsoft (`kagglecatsanddogs_5340.zip`) con `wget` y asume la estructura `data/PetImages/Cat`, `data/PetImages/Dog`. La mecánica concreta de resolución (CLI de Kaggle, Secrets de Colab, verificación por código de la estructura) es un derivado de los steering/encargo, no una exigencia literal del `.md`. Resolución: usar la fuente de Kaggle; verificar por código la estructura real de carpetas antes de construir `ImageFolder`.
- **D2 — Preguntas de investigación [DERIVADO - fuera del .md]:** el `.md` lista **8** preguntas de investigación (la 8, EfficientNet-B0, es un reto opcional). La sección 10 del notebook base lista solo **6** y en orden distinto. La forma de reconciliar la numeración del notebook base con la del `.md` es un derivado de análisis, no una exigencia literal del `.md`. Resolución: cubrir las 8 del `.md`.
- **D3 — `summary()` con tamaño 244:** el `.md` (Actividad 4, nota) indica corregir un `summary()` que usa 244 en lugar de 224. El notebook base, en su estado actual, ya invoca `summary(..., input_size=(1, 3, 224, 224))`. Resolución: dejar constancia de la corrección y asegurar por verificación que **todas** las llamadas a `summary` usan 224.

## Glossary

- **Notebook:** el archivo `ParteA/notebook_parte_a/TransferLearningWorkshop.ipynb` que se completa y, como entregable, su copia renombrada.
- **Autor (del notebook):** la persona o proceso que edita las celdas del Notebook para resolver los TODO y responder preguntas.
- **Pipeline_Transforms:** la composición de transformaciones de preprocesamiento (`Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize`).
- **VGG16:** el modelo `torchvision.models.vgg16` cargado con la API moderna de pesos (`VGG16_Weights.DEFAULT`), salvo cuando se indique `weights=None`.
- **ResNet18:** el modelo `torchvision.models.resnet18` cargado con la API moderna de pesos (`ResNet18_Weights.DEFAULT`), salvo cuando se indique `weights=None`.
- **M1:** VGG-16 como extractor de features fijo (`vgg.features` congelado) + una capa `nn.Linear`.
- **M2:** VGG-16 con `vgg.classifier` reemplazado por `nn.Linear(25088, 2)`, entrenado por etapas (congelado y luego fine-tuned).
- **M3:** ResNet-18 con `fc` reemplazado por una capa lineal hacia 2 clases.
- **Celda_Verificacion:** celda de código pensada para ejecutarse en Colab que comprueba por código un SUPUESTO (p. ej. shapes, nombres de carpetas, conteo de parámetros).
- **Smoke_Test:** celda de verificación local que, con `weights=None` y tensores aleatorios (sin descargar pesos ni datos), comprueba shapes y conteos de parámetros entrenables.
- **DATA_ROOT:** variable configurable con la ruta raíz de datos, para que el Notebook corra igual en local y en Colab.
- **CKPT_DIR:** variable configurable con la ruta donde se guardan y cargan los checkpoints (`.pth`).
- **Marcador_Pendiente:** el comentario literal `<!-- COMPLETAR tras ejecución -->` usado en plantillas de respuestas empíricas.
- **TensorBoard:** `torch.utils.tensorboard.SummaryWriter`, con logs escritos en un directorio `runs/` relativo al Notebook.
- **Pregunta_Teorica:** pregunta cuya respuesta no depende de ejecutar el Notebook (conceptual/documental).
- **Pregunta_Empirica:** pregunta cuya respuesta depende de resultados de ejecución (accuracy, tiempos, comparación de curvas).

## Requirements

### Requisito 1: [DERIVADO - fuera del .md] Configuración, rutas y entorno local/Colab

**Historia de usuario:** Como estudiante que escribe en local y ejecuta en Colab, quiero que el Notebook use rutas relativas y variables configurables, para que corra igual en ambos entornos sin descargar datos en local.

#### Criterios de aceptación

1. THE Notebook SHALL definir `device` como `'cuda' if torch.cuda.is_available() else 'cpu'`.
2. THE Notebook SHALL definir las variables `DATA_ROOT` y `CKPT_DIR` como rutas relativas configurables antes de su primer uso.
3. THE Notebook SHALL fijar la semilla de reproducibilidad con `torch.manual_seed(42)` y usar `torch.Generator().manual_seed(42)` en las divisiones aleatorias del dataset.
4. WHERE una ruta de datos o de checkpoint es necesaria, THE Notebook SHALL construir dicha ruta a partir de `DATA_ROOT` o `CKPT_DIR` en lugar de literales absolutos.
5. THE Notebook SHALL conservar las celdas y funciones base no marcadas como TODO, modificando únicamente los puntos marcados o los errores demostrables documentados.

### Requisito 2: [DERIVADO - fuera del .md] Dataset de Kaggle y verificación de estructura (Actividad 1, D1)

**Historia de usuario:** [DERIVADO - fuera del .md] Como estudiante, quiero descargar el dataset exacto del `.md` desde Kaggle y verificar su estructura real por código, para construir `ImageFolder` sobre la raíz correcta sin inventar nombres de carpetas. Nota: la fuente de datos (Kaggle `kunalgupta2616/...`) proviene del `.md`, pero la mecánica concreta (CLI de Kaggle, Secrets de Colab e inspección de estructura por código) es un derivado de los steering/encargo.

#### Criterios de aceptación

1. [DERIVADO - fuera del .md] THE Notebook SHALL documentar en una celda Markdown la discrepancia D1 (el `.md` exige Kaggle `kunalgupta2616/dog-vs-cat-images-data` y el notebook base usaba el ZIP de Microsoft con `wget`), indicando que prevalece el `.md`.
2. THE Notebook SHALL reemplazar la celda de descarga con `wget` por una celda que use la CLI de Kaggle para descargar `kunalgupta2616/dog-vs-cat-images-data`.
3. WHERE se ejecuta en Colab, THE Notebook SHALL obtener las credenciales de Kaggle desde los Secrets de Colab, sin incrustar credenciales en el código.
4. THE Notebook SHALL incluir una Celda_Verificacion que inspeccione por código la estructura real de carpetas descomprimida y determine la raíz adecuada para `ImageFolder`.
5. THE Notebook SHALL construir `ImageFolder` usando la raíz determinada por código en la Celda_Verificacion, sin asumir literales de nombres de carpetas de clase.
6. THE Notebook SHALL conservar la invocación de `check_image_dir` para eliminar imágenes corruptas, ajustando los patrones glob a la estructura real verificada.
7. IF la estructura de carpetas descargada no puede determinarse por código, THEN THE Notebook SHALL imprimir el árbol de directorios inspeccionado y marcar la raíz de `ImageFolder` como SUPUESTO a confirmar en Colab.

### Requisito 3: Pipeline de preprocesamiento (Actividad 1 — TODO 1)

**Historia de usuario:** Como estudiante, quiero el pipeline de transforms correcto para ImageNet, para que los pesos preentrenados produzcan representaciones válidas.

#### Criterios de aceptación

1. THE Notebook SHALL definir `Pipeline_Transforms` como `transforms.Compose` en el orden `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize`.
2. THE Pipeline_Transforms SHALL usar `mean=[0.485, 0.456, 0.406]` y `std=[0.229, 0.224, 0.225]` en la normalización.
3. THE Notebook SHALL responder la Pregunta_Teorica de la Actividad 1 (origen de los valores de media/std de ImageNet y por qué son importantes al usar pesos preentrenados) de forma completa en una celda Markdown en español.

### Requisito 4: Modelo preentrenado y verificación de salida (Actividad 2 — inferencia)

**Historia de usuario:** Como estudiante, quiero cargar VGG-16 preentrenado con la API moderna y verificar su salida, para confirmar que el modelo clasifica en las 1000 clases de ImageNet.

#### Criterios de aceptación

1. THE Notebook SHALL cargar VGG16 con la API moderna de pesos (`torchvision.models.VGG16_Weights.DEFAULT`), sin usar el argumento deprecado `pretrained=True`.
2. WHEN se infiere sobre una imagen de muestra, THE Notebook SHALL verificar por código que la salida tiene shape `[1, 1000]`.
3. THE Notebook SHALL mostrar el nombre de la clase de ImageNet predicha usando el mapeo de clases.
4. THE Notebook SHALL responder la Pregunta_Teorica de la Actividad 2 (número total de parámetros de VGG-16 obtenido por código o `summary`, qué hace Dropout y por qué mejora la generalización, y por qué la última capa tiene 1000 salidas) de forma completa en celdas Markdown en español.

### Requisito 5: Extracción de features de 800 imágenes (Actividad 2 — TODO 2)

**Historia de usuario:** Como estudiante, quiero precomputar los vectores de features de 800 imágenes con el backbone fijo, para entrenar luego un clasificador sin recomputar features.

#### Criterios de aceptación

1. THE Notebook SHALL inicializar `feature_tensor` con shape `[800, 512*7*7]` y `label_tensor` con shape `[800]`.
2. WHILE se extraen features, THE Notebook SHALL usar `torch.no_grad()` para no calcular gradientes.
3. THE Notebook SHALL aplanar las activaciones de `vgg.features` a `512*7*7` por imagen y almacenarlas junto con sus etiquetas en los tensores.
4. THE Notebook SHALL imprimir los shapes finales de `feature_tensor` y `label_tensor` para verificación.

### Requisito 6: Clasificador lineal sobre features (Actividad 3 — TODO 3, M1)

**Historia de usuario:** Como estudiante, quiero entrenar una sola capa lineal sobre las features precomputadas, para obtener M1 en su primera etapa sin backprop a través de la CNN.

#### Criterios de aceptación

1. THE Notebook SHALL definir `net` como una única `nn.Linear(512*7*7, 2)` sin ninguna activación final (sin `LogSoftmax`).
2. THE Notebook SHALL entrenar `net` por 5 épocas con `nn.CrossEntropyLoss` registrando las curvas en TensorBoard.
3. THE Notebook SHALL responder la Pregunta_Teorica de la Actividad 3 (diferencia entre `CrossEntropyLoss` y `NLLLoss + LogSoftmax`, y por qué combinarlos es un error) de forma completa en una celda Markdown en español.

### Requisito 7: Reemplazar clasificador y congelar features (Actividad 4 — TODO 4, M2 etapa congelada, D3)

**Historia de usuario:** Como estudiante, quiero recargar VGG-16 limpio, reemplazar el clasificador y congelar las convolucionales, para entrenar solo la cabeza y verificar por código que los parámetros entrenables son ~50k.

#### Criterios de aceptación

1. THE Notebook SHALL cargar VGG16 limpio con pesos preentrenados y reemplazar `vgg.classifier` por `nn.Linear(25088, 2)`.
2. THE Notebook SHALL congelar todos los parámetros de `vgg.features` con `requires_grad = False`.
3. THE Notebook SHALL calcular por código el número de parámetros entrenables (p. ej. sumando `p.numel()` para `p` con `p.requires_grad`) e imprimirlo, en lugar de afirmar el valor en texto.
4. THE Notebook SHALL invocar `summary` con `input_size=(1, 3, 224, 224)` y documentar la corrección D3 (244 → 224).
5. THE Notebook SHALL entrenar M2 por 1 época sobre el dataset completo registrando las curvas en TensorBoard con un tag de etapa congelada.
6. THE Notebook SHALL guardar el modelo entrenado en un archivo `.pth` cuya ruta se construye con `CKPT_DIR`.
7. THE Notebook SHALL responder la Pregunta_Teorica de la Actividad 4 (por qué es importante congelar las capas convolucionales antes de entrenar el nuevo clasificador) de forma completa en una celda Markdown en español.

### Requisito 8: Fine-tuning completo (Actividad 5 — TODO 5, M2 etapa fine-tuned)

**Historia de usuario:** Como estudiante, quiero cargar el modelo guardado, descongelar las convolucionales y entrenar con lr pequeño, para especializar el extractor sin destruir los pesos preentrenados.

#### Criterios de aceptación

1. THE Notebook SHALL cargar el modelo guardado en la Actividad 4 desde la ruta construida con `CKPT_DIR`.
2. THE Notebook SHALL descongelar los parámetros de `vgg.features` con `requires_grad = True`.
3. THE Notebook SHALL entrenar 1 época adicional con `lr=1e-4` registrando las curvas en TensorBoard con un tag de etapa fine-tuned.
4. THE Notebook SHALL incluir la comparación de curvas congelado vs. fine-tuned en TensorBoard (tags separados bajo `runs/`).

### Requisito 9: Adaptar ResNet-18 y medir tiempos (Actividad 6 — TODO 6, M3)

**Historia de usuario:** Como estudiante, quiero adaptar ResNet-18 determinando `in_features` por código y medir el tiempo por época, para comparar M3 con VGG en velocidad y accuracy.

#### Criterios de aceptación

1. THE Notebook SHALL cargar ResNet18 con la API moderna de pesos (`ResNet18_Weights.DEFAULT`).
2. THE Notebook SHALL congelar todos los parámetros de ResNet18 con `requires_grad = False`.
3. THE Notebook SHALL determinar las entradas del clasificador por código mediante `resnet18.fc.in_features` y reemplazar `resnet18.fc` por una `nn.Linear` hacia 2 clases usando ese valor.
4. THE Notebook SHALL entrenar M3 por 2 épocas registrando las curvas en TensorBoard.
5. [DERIVADO - fuera del .md] THE Notebook SHALL medir el tiempo por época con `time.time()` para los entrenamientos de VGG y de ResNet, e imprimir los tiempos para su comparación.
6. THE Notebook SHALL responder la parte teórica de la pregunta de la Actividad 6 (qué es un residual block y qué problema resuelve en redes profundas) de forma completa en una celda Markdown en español.
7. [DERIVADO - fuera del .md] THE Notebook SHALL dejar la parte empírica de la pregunta de la Actividad 6 (si ResNet-18 entrena más rápido que VGG y por qué, según los tiempos medidos) como plantilla con el Marcador_Pendiente más el código que imprime los tiempos necesarios.

### Requisito 10: Preguntas de investigación 1–8 (trazabilidad 1:1 con el `.md`)

**Historia de usuario:** Como estudiante, quiero que cada una de las 8 preguntas de investigación del `.md` tenga un criterio de aceptación propio con su mismo número, para que la cobertura sea rastreable 1:1 y ninguna pregunta quede sin responder.

> Nota de trazabilidad: la numeración 1–8 de esta sección corresponde **exactamente** a la numeración de las preguntas de investigación del `.md` (fuente de verdad). Cada criterio indica si la pregunta es teórica (respuesta completa en Markdown), empírica (experimento + plantilla con Marcador_Pendiente) u opcional. Las preguntas 3 y 7 del `.md` son casi idénticas (schedulers): la respuesta sustantiva se da **una sola vez** en la pregunta 3 y la pregunta 7 la referencia explícitamente.

#### Criterios de aceptación

1. **Pregunta 1 (teórica):** THE Notebook SHALL responder de forma completa en una celda Markdown en español cuándo NO conviene usar transfer learning, incluyendo al menos dos escenarios concretos.
2. **Pregunta 2 (empírica):** THE Notebook SHALL implementar y experimentar transforms de data augmentation (ver R11.1) y SHALL dejar la conclusión empírica como plantilla con el Marcador_Pendiente (ver R11.3).
3. **Pregunta 3 (empírica):** THE Notebook SHALL agregar un learning rate scheduler (`StepLR` o `CosineAnnealingLR`) al fine-tuning de la Actividad 5 (ver R11.2), comparar las curvas y SHALL dejar la conclusión empírica como plantilla con el Marcador_Pendiente (ver R11.3). Esta es la **única** respuesta sustantiva sobre schedulers.
4. **Pregunta 4 (teórica):** THE Notebook SHALL responder de forma completa en una celda Markdown en español por qué ResNet usa BatchNorm y VGG no, y qué efecto tiene en el entrenamiento.
5. **Pregunta 5 (teórica):** THE Notebook SHALL responder de forma completa en una celda Markdown en español el origen de los valores de media/std de normalización de ImageNet y qué pasaría al usar otros valores.
6. **Pregunta 6 (teórica):** THE Notebook SHALL responder de forma completa en una celda Markdown en español la diferencia entre `CrossEntropyLoss`, `NLLLoss` y `LogSoftmax`, y por qué no se debe añadir `LogSoftmax`.
7. **Pregunta 7 (empírica, referencia a la 3):** WHERE la pregunta 7 del `.md` (learning rate scheduling: agregar `StepLR`/`CosineAnnealingLR` al fine-tuning y comparar curvas) es casi idéntica a la pregunta 3, THE Notebook SHALL responderla referenciando explícitamente la respuesta y el experimento de la pregunta 3, sin duplicar la respuesta sustantiva.
8. **Pregunta 8 (opcional, empírica):** WHERE se aborda el reto opcional de EfficientNet-B0, THE Notebook SHALL compararlo en accuracy y tiempo contra ResNet-18 y VGG-16 (ver R12) y SHALL dejar la comparación como plantilla con el Marcador_Pendiente.
9. THE Notebook SHALL responder de forma completa en celdas Markdown las partes teóricas de las preguntas de actividad: qué hace Dropout, por qué 1000 salidas, por qué congelar las convolucionales, y qué es un residual block.
10. WHERE una respuesta teórica contiene una afirmación no verificable sin documentación, THE Autor SHALL registrar en `notas_verificacion.md` (fuera del Notebook) el nivel de confianza (alto/medio/bajo) y qué comprobar.

### Requisito 11: Experimentos adicionales de investigación (data augmentation y schedulers)

**Historia de usuario:** Como estudiante, quiero implementar los experimentos de data augmentation y de learning rate scheduling con logs en TensorBoard, para responder con evidencia las preguntas de investigación 2/3 y 7.

#### Criterios de aceptación

1. THE Notebook SHALL implementar como experimento adicional transforms de data augmentation en el pipeline de entrenamiento, registrando sus curvas en TensorBoard bajo `runs/`.
2. THE Notebook SHALL implementar como experimento adicional un learning rate scheduler (`torch.optim.lr_scheduler.StepLR` o `CosineAnnealingLR`) en el loop de fine-tuning de la Actividad 5, registrando sus curvas en TensorBoard bajo `runs/`.
3. THE Notebook SHALL dejar las conclusiones empíricas de ambos experimentos (impacto en generalización, comparación de curvas) como plantilla con el Marcador_Pendiente más el código que produce las curvas/tablas necesarias.

### Requisito 12: Reto opcional EfficientNet-B0 (Pregunta 8)

**Historia de usuario:** Como estudiante, quiero incluir el reto opcional de EfficientNet-B0 como sección claramente marcada como opcional, para comparar accuracy y tiempo contra ResNet-18 y VGG-16 sin que sea obligatorio.

#### Criterios de aceptación

1. THE Notebook SHALL incluir una sección rotulada como **opcional** que use `EfficientNet-B0` de `torchvision.models` con la API moderna de pesos.
2. THE Notebook SHALL dejar la comparación de accuracy final y tiempo de entrenamiento de EfficientNet-B0 contra ResNet-18 y VGG-16 como plantilla con el Marcador_Pendiente más el código que mide esos valores.

### Requisito 13: [DERIVADO - fuera del .md] Reglas anti-invención y respuestas empíricas

**Historia de usuario:** [DERIVADO - fuera del .md] Como estudiante que no ejecuta en local, quiero que todo lo no verificable quede marcado como SUPUESTO o plantilla, para no inventar resultados.

#### Criterios de aceptación

1. THE Notebook SHALL dejar como plantilla con el Marcador_Pendiente toda respuesta a una Pregunta_Empirica (accuracy, tiempos, comparación de curvas, "¿ResNet entrena más rápido?").
2. WHERE una respuesta empírica se deja como plantilla, THE Notebook SHALL incluir el código que imprime las tablas o produce las curvas necesarias para completarla.
3. WHERE una afirmación no es verificable sin ejecutar, THE Notebook SHALL marcarla como SUPUESTO y acompañarla de una Celda_Verificacion pensada para Colab.
4. THE Notebook SHALL evitar afirmar métricas, tiempos, estructura de carpetas de dataset, firmas de API o referencias que no estén en el `.md`, en el notebook o en documentación oficial conocida con certeza.

### Requisito 14: Autocontención y español

**Historia de usuario:** Como lector que no ejecuta el notebook, quiero una explicación en Markdown antes de cada bloque de código y todo en español, para entender la solución sin ejecutarla.

#### Criterios de aceptación

1. THE Notebook SHALL incluir una celda Markdown explicativa en español antes de cada bloque de código relevante.
2. THE Notebook SHALL escribir todos los títulos, explicaciones y respuestas en español.

### Requisito 15: [DERIVADO - fuera del .md] Verificación local sin datos ni GPU (Smoke Test y nbformat)

**Historia de usuario:** [DERIVADO - fuera del .md] Como autor en local sin GPU ni datos, quiero un smoke test con `weights=None` y validación con nbformat, para verificar shapes y conteos de parámetros entrenables de M2 y M3 sin descargar nada.

#### Criterios de aceptación

1. THE Notebook SHALL incluir un Smoke_Test que, con `weights=None` y tensores aleatorios, construya M2 (VGG-16 con `nn.Linear(25088, 2)` y `features` congeladas) y verifique por código el shape de salida y el conteo de parámetros entrenables.
2. THE Notebook SHALL incluir en el Smoke_Test la construcción de M3 (ResNet-18 con `fc` reemplazado determinando `in_features` por código y backbone congelado) y la verificación por código del shape de salida y el conteo de parámetros entrenables.
3. THE Smoke_Test SHALL ejecutarse sin descargar pesos preentrenados ni datasets (usando únicamente `weights=None` y tensores aleatorios).
4. THE entregable SHALL ser un `.ipynb` bien formado, validable con `nbformat`.

### Requisito 16: Entregable renombrado

**Historia de usuario:** Como estudiante, quiero el notebook final con el nombre exacto del grupo, para cumplir el formato de entrega.

#### Criterios de aceptación

1. THE entregable final SHALL nombrarse `grupoSamuelHerrera_CristianCabarcas_EmmanuelAlvarez_LAB#2A.ipynb`.
2. THE entregable final SHALL contener todo el código completo y las respuestas a todas las preguntas en celdas Markdown, de forma autocontenida.

### Requisito 17: Comparación de experimentos con TensorBoard y curvas matplotlib (Actividad 9)

**Historia de usuario:** Como estudiante, quiero comparar mis experimentos en TensorBoard y, además, dejar las curvas dibujadas con matplotlib dentro del Notebook, para que el reporte sea autocontenido aunque el panel de TensorBoard no se conserve.

#### Criterios de aceptación

1. THE Notebook SHALL incluir una celda de TensorBoard con las líneas mágicas `%load_ext tensorboard` y `%tensorboard --logdir runs`.
2. THE Notebook SHALL usar un tag de logging DISTINTO bajo `runs/` para cada experimento, de modo que las curvas puedan compararse en TensorBoard.
3. THE Notebook SHALL dejar las curvas de cada experimento dibujadas con matplotlib dentro del propio Notebook (por ejemplo, vía la función `plot_results` ya existente), de modo que el reporte sea autocontenido aunque el panel de TensorBoard no se conserve.

### Requisito 18: Entrega del notebook ejecutado

**Historia de usuario:** Como estudiante, quiero entregar el notebook ya EJECUTADO con todas las salidas visibles, para que el evaluador vea los resultados sin tener que ejecutarlo.

#### Criterios de aceptación

1. THE entregable final SHALL entregarse ejecutado, con las salidas (outputs) de TODAS las celdas visibles.
2. THE Notebook ejecutado SHALL generarse en Google Colab mediante "Run all" (ejecutar todas las celdas).
3. WHERE la ejecución del Notebook es necesaria, THE usuario SHALL ejecutarla en Colab, dado que Kiro NO ejecuta el notebook; este paso SHALL aparecer como ítem en la lista de verificación final de entrega.

## Tabla de trazabilidad (requisito del `.md` → celda(s) del Notebook)

> Nota: los identificadores de celda son lógicos (según las secciones del notebook base). El mapeo exacto a índices de celda se afinará en el diseño; aquí se referencian por sección/TODO.

| Requisito del `.md` | Requisito spec | Celda(s) del Notebook |
|---|---|---|
| Entorno local/Colab, rutas `DATA_ROOT`/`CKPT_DIR`, semilla | R1 | Sec. 0 (imports y configuración); nuevas celdas de configuración |
| Act. 1 — Dataset Kaggle `kunalgupta2616/...` (D1) y verificación de estructura | R2 | Sec. 2 (celda de descarga reemplazada; nueva Celda_Verificacion de estructura; `check_image_dir`) |
| Act. 1 — Pipeline Resize256→CenterCrop224→ToTensor→Normalize (TODO 1) | R3 | Sec. 2 (TODO 1) + celda Markdown de respuesta |
| Act. 1 — Pregunta media/std ImageNet | R3.3, R10.5 | Celda Markdown tras TODO 1 |
| Act. 2 — VGG-16 moderno, salida [1,1000], clase ImageNet | R4 | Sec. 3 (carga VGG, inferencia, `class_map`) |
| Act. 2 — Pregunta params totales / Dropout / 1000 salidas | R4.4, R10.9 | Celdas Markdown de Sec. 3 |
| Act. 2 — Extracción features [800, 512*7*7] con `no_grad` (TODO 2) | R5 | Sec. 4 (TODO 2) |
| Act. 3 — `net` = `Linear(512*7*7, 2)` sin LogSoftmax, 5 épocas, TensorBoard (TODO 3) | R6 | Sec. 5 (TODO 3) |
| Act. 3 — Pregunta CrossEntropy vs NLLLoss+LogSoftmax | R6.3, R10.6 | Celda Markdown de Sec. 5 |
| Act. 4 — Reemplazar `classifier`, congelar `features`, ~50k entrenables por código, `summary` 224 (D3), 1 época, guardar `.pth` (TODO 4) | R7 | Sec. 6 (TODO 4a/4b/4c, `summary`, train frozen, `torch.save`) |
| Act. 4 — Pregunta congelar convolucionales | R7.7, R10.9 | Celda Markdown de Sec. 6 |
| Act. 5 — Cargar modelo, descongelar `features`, 1 época lr=1e-4, comparar curvas (TODO 5) | R8 | Sec. 7 (TODO 5, train finetuned) |
| Act. 6 — ResNet-18 moderno, congelar, `fc.in_features` por código, 2 épocas, tiempos (TODO 6) | R9 | Sec. 8 (TODO 6a/6b/6c) + medición de tiempos |
| Act. 6 — Pregunta residual block + ¿ResNet más rápido? | R9.6, R9.7, R10.9 | Celdas Markdown de Sec. 8 |
| Investigación 1 — cuándo NO transfer learning (teórica) | R10.1 | Sec. 10 (Markdown) |
| Investigación 2 — data augmentation (empírica, experimento) | R10.2, R11.1, R11.3 | Nueva celda de experimento + Markdown |
| Investigación 3 — StepLR/CosineAnnealingLR en Act. 5 (empírica, respuesta única) | R10.3, R11.2, R11.3 | Nueva celda de experimento en loop de Act. 5 + Markdown |
| Investigación 4 — BatchNorm en ResNet vs VGG (teórica) | R10.4 | Sec. 10 (Markdown) |
| Investigación 5 — origen media/std ImageNet (teórica) | R10.5 | Sec. 10 (Markdown) |
| Investigación 6 — CrossEntropyLoss vs NLLLoss vs LogSoftmax (teórica) | R10.6 | Sec. 10 (Markdown) |
| Investigación 7 — learning rate scheduling (empírica, referencia a la 3) | R10.7 | Sec. 10 (Markdown, referencia a R10.3/R11.2) |
| Investigación 8 — EfficientNet-B0 (opcional, empírica) | R10.8, R12 | Nueva sección opcional |
| Autocontención y español | R14 | Todas las celdas Markdown |
| Comparación de experimentos — TensorBoard (`%load_ext`/`%tensorboard --logdir runs`) + curvas matplotlib (Act. 9) | R17 | Celda de TensorBoard + celdas `plot_results` |
| Smoke test M2/M3 y nbformat [DERIVADO - fuera del .md] | R15 | Nueva celda de verificación local |
| Entregable renombrado | R16 | Nombre de archivo del entregable |
| Entregable ejecutado (Run all en Colab, outputs visibles) | R18 | Entregable final + lista de verificación de entrega |

## Qué NO se pudo verificar (requiere ejecución en Colab)

Los siguientes puntos no son verificables sin ejecutar en Colab (con GPU, credenciales de Kaggle y datos) y quedan como SUPUESTO/plantilla:

- **Nombres y estructura reales de carpetas del dataset de Kaggle** `kunalgupta2616/dog-vs-cat-images-data` (raíz de `ImageFolder`, nombres de clases, patrones glob para `check_image_dir`). Se determinarán por código en la Celda_Verificacion al ejecutar.
- **Número total de imágenes del dataset** y validez de la división train/test base (el notebook base asume 20000/resto, que depende del tamaño real del dataset de Kaggle).
- **Valores numéricos de accuracy** de M1, M2 (congelado y fine-tuned), M3 y del reto opcional.
- **Tiempos por época** de VGG, ResNet y EfficientNet, y por tanto la respuesta empírica a "¿ResNet entrena más rápido que VGG?".
- **Conteo exacto de parámetros entrenables de M2 (~50k)** antes de ejecutar; se calcula por código al construir el modelo (el Smoke_Test local lo verifica con `weights=None`, pero el valor definitivo se confirma en Colab).
- **Número total de parámetros de VGG-16** reportado por `summary` (se obtiene por código al ejecutar).
- **Comparación de curvas** congelado vs. fine-tuned, con/sin data augmentation y con/sin scheduler en TensorBoard.
- **Índice y nombre de la clase ImageNet predicha** para la imagen de muestra (depende del dato real cargado).
- **Credenciales de Kaggle desde Secrets de Colab**: su configuración ocurre en el entorno Colab, fuera del alcance local.
- **Notebook ejecutado con todas las salidas visibles (R18)**: Kiro NO ejecuta el notebook. La ejecución "Run all" debe realizarla el usuario en Google Colab.

### Lista de verificación final de entrega

- [ ] El notebook está renombrado a `grupoSamuelHerrera_CristianCabarcas_EmmanuelAlvarez_LAB#2A.ipynb` (R16).
- [ ] El notebook se ejecutó en Colab con "Run all" y las salidas de TODAS las celdas están visibles (R18).
- [ ] Las curvas de cada experimento aparecen dibujadas con matplotlib dentro del notebook, además de en TensorBoard (R17).
- [ ] Todas las plantillas con el Marcador_Pendiente se completaron con los resultados reales de la ejecución (R13).
