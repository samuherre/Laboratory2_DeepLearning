# Implementation Plan: Parte A — Transfer Learning

## Overview

Plan de implementación accionable para **completar el notebook** `ParteA/notebook_parte_a/TransferLearningWorkshop.ipynb` según `design.md`, con enfoque de **cambios mínimos**: se resuelven los TODO 1–6 y se añaden únicamente las celdas nuevas que exigen los requisitos derivados (configuración, descarga Kaggle, verificación de estructura, tiempos, experimentos de augmentation/scheduler, EfficientNet opcional, respuestas 1–8, Smoke_Test). Se conservan las celdas y funciones base (`check_image_dir`, `display_dataset`, `train_epoch`, `eval_epoch`, `train`, `plot_results`).

El código se **escribe en local** (sin GPU ni datos). Las tareas NO descargan datasets, NO instalan paquetes y NO ejecutan entrenamientos en local. La ejecución "Run all" en Colab es responsabilidad del usuario y figura como ítem final de verificación/entrega: **Kiro no ejecuta el notebook**.

Reglas anti-invención aplicadas en todas las tareas: nombres de carpetas del dataset, conteos de parámetros, accuracies, tiempos por época y número de parámetros de VGG-16 se **determinan por código al ejecutar** o se dejan como SUPUESTO con `Celda_Verificacion`; toda respuesta empírica se deja como plantilla con el marcador `<!-- COMPLETAR tras ejecución -->` acompañada del código que imprime los números/curvas.

## Tasks

- [x] 1. Configuración de entorno dual local/Colab (Sec. 0)
  - Editar la celda de configuración (code cell de la Sec. 0) conservando `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
  - Añadir `DATA_ROOT` y `CKPT_DIR` como rutas relativas configurables (p. ej. `'data'`) definidas antes de su primer uso, con `os.makedirs(..., exist_ok=True)`
  - Añadir `torch.manual_seed(42)`
  - Añadir una celda Markdown explicativa en español antes del bloque de configuración
  - No modificar las funciones auxiliares base de la Sec. 1
  - _Requirements: R1.1, R1.2, R1.3, R1.4, R1.5, R14.1, R14.2_

- [x] 2. Dataset de Kaggle y verificación de estructura (Sec. 2, D1)
  - [x] 2.1 Documentar la discrepancia D1 y reemplazar la descarga `wget` por Kaggle CLI
    - Añadir celda Markdown en español que documente D1 (el `.md` exige Kaggle `kunalgupta2616/dog-vs-cat-images-data`; el notebook base usaba el ZIP de Microsoft con `wget`) indicando que prevalece el `.md`
    - Reemplazar la celda de descarga `wget` por una celda que use la CLI de Kaggle para descargar `kunalgupta2616/dog-vs-cat-images-data` hacia `DATA_ROOT`
    - Obtener credenciales de Kaggle desde Secrets de Colab (`userdata.get(...)`), sin incrustar credenciales en el código; imprimir mensaje claro si faltan
    - _Requirements: R2.1, R2.2, R2.3, R14.1_
  - [x] 2.2 Añadir Celda_Verificacion de estructura de carpetas
    - Añadir celda que inspeccione por código el árbol descomprimido bajo `DATA_ROOT` (p. ej. `os.walk`) e imprima el árbol
    - Implementar `encontrar_raiz_imagefolder(base)` que determine por código `IMAGEFOLDER_ROOT` (carpeta cuyos hijos directos son subdirectorios de clase)
    - Si la estructura no es determinable por código, imprimir el árbol y marcar `IMAGEFOLDER_ROOT` como SUPUESTO con comentario `# SUPUESTO: confirmar en Colab`
    - _Requirements: R2.4, R2.7, R13.3_
  - [x] 2.3 Construir ImageFolder y limpieza de corruptas sobre la raíz verificada
    - Ajustar la invocación de `check_image_dir` para iterar sobre las clases reales descubiertas bajo `IMAGEFOLDER_ROOT`, sin literales `data/PetImages/Cat` ni `.../Dog`
    - Construir `ImageFolder(IMAGEFOLDER_ROOT, transform=trans)` usando la raíz verificada, sin asumir literales de nombres de clase
    - Añadir split defensivo con `random_split` que use `n_train = 20000 if n > 20000 else int(0.8*n)` y `torch.Generator().manual_seed(42)`
    - Añadir celda Markdown explicativa en español
    - _Requirements: R2.5, R2.6, R1.3, R14.1_

- [x] 3. TODO 1 — Pipeline de transforms (Sec. 2)
  - Completar `trans` como `transforms.Compose([Resize(256), CenterCrop(224), ToTensor(), std_normalize])` en ese orden exacto, conservando `std_normalize` con `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`
  - Añadir celda Markdown en español que responda la Pregunta_Teorica de la Actividad 1 (origen de media/std de ImageNet y por qué importan con pesos preentrenados)
  - _Requirements: R3.1, R3.2, R3.3, R14.1, R14.2_

- [x] 4. VGG-16 preentrenado e inferencia de muestra (Sec. 3)
  - Verificar que la carga usa la API moderna `VGG16_Weights.DEFAULT` (sin `pretrained=True`); conservar celda base
  - Añadir verificación por código de que la salida de inferencia tiene shape `[1, 1000]` (p. ej. `assert res.shape == torch.Size([1, 1000])`)
  - Conservar la celda que muestra el nombre de la clase ImageNet predicha vía `class_map`
  - Asegurar que la llamada `summary(vgg, input_size=(1, 3, 224, 224))` usa 224 (D3) y añadir Markdown que deje constancia de la corrección D3 (244 → 224)
  - Añadir celdas Markdown en español que respondan la Pregunta_Teorica de la Actividad 2: nº total de parámetros de VGG-16 (obtenido por código/`summary`, no afirmado en texto), qué hace Dropout y por qué 1000 salidas
  - _Requirements: R4.1, R4.2, R4.3, R4.4, R10.9, R14.1, R14.2_

- [x] 5. TODO 2 — Extracción de features de 800 imágenes (Sec. 4)
  - Inicializar `feature_tensor` con shape `[800, 512*7*7]` y `label_tensor` con shape `[800]`
  - Dentro del loop, bajo `torch.no_grad()`, aplanar las activaciones de `vgg.features` a `512*7*7` por imagen y almacenarlas junto con sus etiquetas
  - Imprimir los shapes finales de `feature_tensor` y `label_tensor` para verificación
  - Añadir celda Markdown explicativa en español
  - _Requirements: R5.1, R5.2, R5.3, R5.4, R14.1_

- [x] 6. TODO 3 — Clasificador lineal sobre features (Sec. 5, M1)
  - Definir `net = nn.Linear(512*7*7, 2)` (una sola capa, sin `LogSoftmax`)
  - Conservar el entrenamiento de 5 épocas con `nn.CrossEntropyLoss` y logging en TensorBoard con tag `linear` bajo `runs/`
  - Añadir celda Markdown en español que responda la Pregunta_Teorica de la Actividad 3 (`CrossEntropyLoss` vs `NLLLoss + LogSoftmax` y por qué combinarlos es un error)
  - _Requirements: R6.1, R6.2, R6.3, R14.1, R14.2_

- [x] 7. Checkpoint — Verificar coherencia de Sec. 0–5
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. TODO 4 — Reemplazar clasificador y congelar features (Sec. 6, M2 congelado, D3)
  - [x] 8.1 Completar TODO 4a/4b/4c y conteo de entrenables por código
    - TODO 4a: cargar VGG-16 limpio con `VGG16_Weights.DEFAULT`
    - TODO 4b: reemplazar `vgg.classifier` por `nn.Linear(25088, 2)`
    - TODO 4c: congelar `vgg.features` con `requires_grad = False`
    - Añadir cálculo e impresión por código de los parámetros entrenables (`sum(p.numel() for p in vgg.parameters() if p.requires_grad)`), sin afirmar el valor "~50k" en texto
    - Asegurar `summary(vgg, input_size=(1, 3, 224, 224))` (D3)
    - Añadir celda Markdown en español que responda la Pregunta_Teorica de la Actividad 4 (por qué congelar las convolucionales antes de entrenar la cabeza)
    - _Requirements: R7.1, R7.2, R7.3, R7.4, R7.7, R10.9, R13.4, R14.1, R14.2_
  - [x] 8.2 Entrenar M2 congelado 1 época y guardar checkpoint
    - Conservar el entrenamiento de 1 época sobre el dataset completo con tag `frozen` en TensorBoard
    - Guardar el modelo en `os.path.join(CKPT_DIR, 'cats_dogs_weights.pth')` (sin literal absoluto)
    - _Requirements: R7.5, R7.6_

- [x] 9. TODO 5 — Fine-tuning y experimento de scheduler (Sec. 7, M2 fine-tuned)
  - [x] 9.1 Completar TODO 5 (descongelar y entrenar lr=1e-4)
    - Cargar el checkpoint desde `os.path.join(CKPT_DIR, 'cats_dogs_weights.pth')` reconstruyendo la arquitectura (`vgg16(weights=None)` + `classifier = nn.Linear(25088, 2)`) antes de `load_state_dict`
    - Descongelar `vgg.features` con `requires_grad = True`
    - Entrenar 1 época con `lr=1e-4` y tag `finetuned` en TensorBoard
    - Dejar las curvas congelado vs fine-tuned comparables con tags separados bajo `runs/`
    - _Requirements: R8.1, R8.2, R8.3, R8.4_
  - [x] 9.2 Añadir experimento de learning rate scheduler (preguntas 3 y 7)
    - Añadir celda con un loop de fine-tuning que use `torch.optim.lr_scheduler.StepLR` o `CosineAnnealingLR` (reutilizando `train_epoch`/`eval_epoch`), con tag distinto bajo `runs/` (p. ej. `finetuned_sched`)
    - Dibujar las curvas con `plot_results` dentro del notebook
    - Dejar la conclusión empírica (impacto del scheduler, comparación de curvas) como plantilla con `<!-- COMPLETAR tras ejecución -->` más el código que produce las curvas
    - Añadir celda Markdown explicativa en español
    - _Requirements: R10.3, R11.2, R11.3, R13.1, R13.2, R17.2, R17.3, R14.1_

- [x] 10. TODO 6 — Adaptar ResNet-18 y medir tiempos (Sec. 8, M3)
  - [x] 10.1 Completar TODO 6a/6b/6c (congelar, fc por código, entrenar)
    - Verificar carga con `ResNet18_Weights.DEFAULT`
    - TODO 6a: congelar todos los parámetros con `requires_grad = False`
    - TODO 6b: determinar `num_features = resnet18.fc.in_features` por código y reemplazar `resnet18.fc = nn.Linear(num_features, 2)`
    - TODO 6c: entrenar 2 épocas con tag `resnet18` en TensorBoard
    - Añadir celda Markdown en español que responda la parte teórica de la Actividad 6 (qué es un residual block y qué problema resuelve)
    - _Requirements: R9.1, R9.2, R9.3, R9.4, R9.6, R10.9, R13.4, R14.1, R14.2_
  - [x] 10.2 Medir tiempos por época de VGG y ResNet y plantilla empírica
    - Añadir celda que mida el tiempo por época con `time.time()` para los entrenamientos de VGG y ResNet e imprima los tiempos para comparación
    - Dejar la parte empírica ("¿ResNet-18 entrena más rápido que VGG y por qué?") como plantilla con `<!-- COMPLETAR tras ejecución -->` más el código que imprime los tiempos
    - _Requirements: R9.5, R9.7, R13.1, R13.2_

- [x] 11. Experimento de data augmentation (pregunta 2)
  - Añadir celda que defina un pipeline de entrenamiento con augmentations (p. ej. `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`), manteniendo `Resize(256)→CenterCrop(224)→ToTensor→Normalize` en validación
  - Registrar las curvas en TensorBoard con tag distinto bajo `runs/` (p. ej. `aug`) y dibujarlas con `plot_results` dentro del notebook
  - Dejar la conclusión empírica (impacto en generalización) como plantilla con `<!-- COMPLETAR tras ejecución -->` más el código que produce las curvas
  - Añadir celda Markdown explicativa en español
  - _Requirements: R10.2, R11.1, R11.3, R13.1, R13.2, R17.2, R17.3, R14.1_

- [x] 12. Comparación de experimentos con TensorBoard (Sec. 9)
  - Conservar la celda de TensorBoard con `%load_ext tensorboard` y `%tensorboard --logdir runs`
  - Verificar que cada experimento usa un tag distinto bajo `runs/` (`linear`, `frozen`, `finetuned`, `resnet18`, `aug`, `finetuned_sched`)
  - Asegurar que cada experimento dibuja sus curvas con `plot_results` dentro del notebook (reporte autocontenido)
  - _Requirements: R17.1, R17.2, R17.3_

- [x] 13. Checkpoint — Verificar secciones 6–9 y experimentos
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. (OPCIONAL) EfficientNet-B0 (pregunta 8)
  - [x] 14.1 Añadir sección opcional EfficientNet-B0
    - Añadir sección rotulada como **OPCIONAL** que cargue `efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)` (API moderna), congele el backbone y adapte el clasificador final determinando sus entradas por código (p. ej. `model.classifier[-1].in_features`)
    - Añadir celda Markdown explicativa en español
    - _Requirements: R12.1, R10.8, R14.1_
  - [x] 14.2 Plantilla de comparación de accuracy y tiempo (OPCIONAL)
    - Dejar la comparación de accuracy final y tiempo de entrenamiento de EfficientNet-B0 contra ResNet-18 y VGG-16 como plantilla con `<!-- COMPLETAR tras ejecución -->` más el código que mide esos valores
    - _Requirements: R12.2, R10.8, R13.1, R13.2_

- [x] 15. Preguntas de investigación 1–8 (Sec. 10)
  - Reescribir la Sec. 10 para cubrir las 8 preguntas del `.md` (D2) en su numeración, todo en español
  - Pregunta 1 (teórica): cuándo NO conviene usar transfer learning, con al menos dos escenarios concretos
  - Pregunta 2 (empírica): referenciar el experimento de data augmentation (Tarea 11) y dejar la conclusión como plantilla con `<!-- COMPLETAR tras ejecución -->`
  - Pregunta 3 (empírica): referenciar el experimento de scheduler (Tarea 9.2) con la respuesta sustantiva única sobre schedulers y plantilla `<!-- COMPLETAR tras ejecución -->`
  - Pregunta 4 (teórica): por qué ResNet usa BatchNorm y VGG no, y su efecto en el entrenamiento
  - Pregunta 5 (teórica): origen de media/std de ImageNet y qué pasaría con otros valores
  - Pregunta 6 (teórica): diferencia entre `CrossEntropyLoss`, `NLLLoss` y `LogSoftmax` y por qué no añadir `LogSoftmax`
  - Pregunta 7 (empírica): referenciar explícitamente la respuesta y el experimento de la pregunta 3, sin duplicar contenido
  - Pregunta 8 (OPCIONAL, empírica): referenciar la sección EfficientNet-B0 (Tarea 14) con plantilla `<!-- COMPLETAR tras ejecución -->`
  - Incluir las partes teóricas de actividad (Dropout, 1000 salidas, congelar convolucionales, residual block) en sus secciones y/o en la Sec. 10
  - _Requirements: R10.1, R10.2, R10.3, R10.4, R10.5, R10.6, R10.7, R10.8, R10.9, R13.1, R13.2, R14.1, R14.2_

- [x] 16. Smoke_Test local con weights=None (M2 y M3)
  - Añadir celda nueva al final del notebook que, con `weights=None` y tensores aleatorios `[B, 3, 224, 224]` (sin descargar pesos ni datos), construya:
    - M2: VGG-16 con `classifier = nn.Linear(25088, 2)` y `features` congeladas; verificar `out.shape == (B, 2)` e imprimir el conteo de parámetros entrenables
    - M3: ResNet-18 con `fc` reemplazado determinando `in_features` por código y backbone congelado; verificar `out.shape == (B, 2)` e imprimir el conteo de parámetros entrenables
  - Añadir celda Markdown explicativa en español
  - _Requirements: R15.1, R15.2, R15.3, R14.1_

- [x] 17. Validar el notebook con nbformat
  - Validar que el `.ipynb` está bien formado con `nbformat` (en el entorno de escritura, sin ejecutar celdas)
  - _Requirements: R15.4_

- [x] 18. Renombrar el entregable final
  - Crear la copia del notebook renombrada a `grupoSamuelHerrera_CristianCabarcas_EmmanuelAlvarez_LAB#2A.ipynb`, autocontenida con todo el código y las respuestas en Markdown
  - _Requirements: R16.1, R16.2_

- [~] 19. (Usuario, en Colab — Kiro NO ejecuta) Ejecutar "Run all" y verificar entrega
  - El usuario ejecuta "Run all" en Google Colab con GPU y Secrets de Kaggle configurados, dejando visibles las salidas de TODAS las celdas
  - Completar todas las plantillas con el marcador `<!-- COMPLETAR tras ejecución -->` con los resultados reales (accuracies, tiempos, comparación de curvas) y confirmar los SUPUESTOS de las `Celda_Verificacion`
  - Confirmar que las curvas de cada experimento aparecen dibujadas con matplotlib dentro del notebook
  - _Requirements: R18.1, R18.2, R18.3, R13.1, R17.3_

## Notes

- Las tareas marcadas como **OPCIONAL** (Tarea 14 y la pregunta 8 de la Tarea 15) corresponden al reto EfficientNet-B0 y pueden omitirse sin afectar la entrega obligatoria.
- Las sub-tareas marcadas con `*` no existen en este plan: no hay suite de tests de código ni Correctness Properties (el `design.md` omite PBT de forma justificada), por lo que no se añaden sub-tareas de property/unit tests. La única verificación local permitida es el Smoke_Test (Tarea 16) y la validación `nbformat` (Tarea 17).
- Cada tarea referencia requisitos concretos (R1–R18) para trazabilidad; ningún requisito queda huérfano.
- Reglas anti-invención: ningún conteo de parámetros, nombre de carpeta del dataset, accuracy ni tiempo se afirma en texto; se determina por código o se deja como plantilla/SUPUESTO.
- Las partes teóricas registran su nivel de confianza en `notas_verificacion.md` (fuera del notebook), según R10.10.
- La Tarea 19 es responsabilidad del usuario en Colab: **Kiro no ejecuta el notebook**.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2"] },
    { "id": 3, "tasks": ["2.3", "3"] },
    { "id": 4, "tasks": ["4"] },
    { "id": 5, "tasks": ["5"] },
    { "id": 6, "tasks": ["6"] },
    { "id": 7, "tasks": ["8.1"] },
    { "id": 8, "tasks": ["8.2"] },
    { "id": 9, "tasks": ["9.1"] },
    { "id": 10, "tasks": ["9.2", "11"] },
    { "id": 11, "tasks": ["10.1"] },
    { "id": 12, "tasks": ["10.2"] },
    { "id": 13, "tasks": ["12", "14.1"] },
    { "id": 14, "tasks": ["14.2", "16"] },
    { "id": 15, "tasks": ["15", "17"] },
    { "id": 16, "tasks": ["18"] }
  ]
}
```
