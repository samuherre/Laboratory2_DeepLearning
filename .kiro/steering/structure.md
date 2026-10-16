# Estructura del proyecto

```
Lab2_DL_ECS/
├── Instrucciones_Lab2_ParteAyB/
│   └── Laboratorio 2-DL (1).md        # Enunciado completo: objetivos, actividades, preguntas, entregables
├── ParteA/                            # Transfer Learning (Cats vs. Dogs)
│   ├── data_parte_a/                  # Dataset de la Parte A (vacío; se descarga/extrae aquí)
│   └── notebook_parte_a/
│       └── TransferLearningWorkshop.ipynb
└── ParteB/                            # Segmentación U-Net (Oxford-IIIT Pet)
    ├── data_parte_b/                  # Dataset de la Parte B (vacío; se descarga aquí)
    └── notebook_parte_b/
        └── UNet_Workshop.ipynb
```

## Organización

- Cada parte del laboratorio vive en su propia carpeta (`ParteA`, `ParteB`) y es independiente.
- El notebook de cada parte está en `notebook_parte_x/` y el dataset correspondiente va en `data_parte_x/`. Las carpetas de datos existen vacías y se llenan al descargar/extraer los datasets.
- El enunciado de referencia está en `Instrucciones_Lab2_ParteAyB/`; consúltalo para los requisitos exactos de cada actividad y las preguntas a responder.

## Convenciones de archivos

- Los notebooks de trabajo son `TransferLearningWorkshop.ipynb` (A) y `UNet_Workshop.ipynb` (B).
- Los **entregables finales** se nombran `grupoX_LAB#2A.ipynb` y `grupoX_LAB#2B.ipynb` (un único notebook por parte, con todo ejecutado).
- Los logs de TensorBoard se escriben en un directorio `runs/` relativo al notebook en ejecución, con un subdirectorio por experimento (p. ej. `runs/transfer_frozen`, `runs/unet_scratch`).
- Los pesos de modelos guardados usan la extensión `.pth` (p. ej. `data/cats_dogs_weights.pth`).
- `notas_verificacion.md` (fuera de los notebooks) registra el nivel de confianza de las respuestas teóricas y qué queda por comprobar en documentación.
- Las rutas de datos y checkpoints se parametrizan con `DATA_ROOT` y `CKPT_DIR` para que el notebook corra igual en local y en Colab.

## Flujo típico dentro de un notebook

1. Imports y configuración (device, semilla).
2. Funciones auxiliares (descarga/limpieza de datos, visualización, loops de entrenamiento/evaluación, métricas).
3. Dataset y pipeline de transforms.
4. Definición/carga de modelos.
5. Entrenamiento por etapas con logging a TensorBoard.
6. Comparación de resultados y respuestas a las preguntas de investigación en celdas Markdown.
