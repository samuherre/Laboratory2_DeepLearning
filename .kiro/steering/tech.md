# Stack técnico

El proyecto es un conjunto de notebooks Jupyter de PyTorch. No hay sistema de build, empaquetado ni gestor de dependencias declarado (no existe `requirements.txt`, `pyproject.toml` ni `environment.yml`).

## Entorno de trabajo

- El código se **escribe en local** (sin GPU y sin datasets) y se **ejecuta en Google Colab** (con GPU). No asumas un entorno local con PyTorch, CUDA o datos disponibles.
- Las carpetas `data_parte_a/` y `data_parte_b/` están vacías a propósito. **No descargues datasets** ni intentes poblarlas.
- **No instales paquetes ni ejecutes entrenamientos** sin preguntar antes.
- Usa rutas relativas (`data/...`) y variables configurables (`DATA_ROOT`, `CKPT_DIR`) para que el mismo notebook corra en local y en Colab.

## Librerías principales

- **PyTorch** (`torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`) — modelos, pérdidas y entrenamiento.
- **torchvision** (`torchvision.models`, `torchvision.transforms`, `torchvision.transforms.functional`, `torchvision.datasets`) — modelos preentrenados (VGG-16, ResNet-18), transforms y datasets (OxfordIIITPet, ImageFolder).
- **torchinfo** (`summary`) — inspección de parámetros y shapes del modelo.
- **TensorBoard** (`torch.utils.tensorboard.SummaryWriter`) — logging de curvas de loss y métricas.
- **matplotlib**, **numpy**, **PIL (Pillow)** — visualización y manejo de imágenes.

## Convenciones de código

- Usa siempre la **API moderna de pesos** de torchvision: `models.resnet18(weights=models.ResNet18_Weights.DEFAULT)`, no el argumento `pretrained=True` deprecado.
- Device: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`; mueve modelos y tensores con `.to(device)`.
- Reproducibilidad: fija semillas con `torch.manual_seed(42)` / `torch.Generator().manual_seed(42)`.
- Normalización ImageNet obligatoria para modelos preentrenados: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- Pipeline de preprocesamiento de Parte A: `Resize(256) → CenterCrop(224) → ToTensor → Normalize`.
- La red devuelve **logits crudos**. Con `nn.CrossEntropyLoss` (clasificación) o `nn.BCEWithLogitsLoss` (segmentación) la activación final (softmax/sigmoid) se aplica dentro de la pérdida; nunca añadas `LogSoftmax`/`Sigmoid` como última capa del modelo.
- En segmentación, imagen y máscara deben recibir las **mismas** transformaciones geométricas; usa interpolación `BILINEAR` para la imagen y `NEAREST` para la máscara.
- Al extraer features con backbone congelado usa `torch.no_grad()`; para congelar capas, `param.requires_grad = False`.

## Verificación

No hay suite de tests ni entrenamientos en local. La verificación permitida aquí es:

- Validar el `.ipynb` con **nbformat** (que el notebook esté bien formado).
- Pruebas de shapes con tensores aleatorios y `weights=None` (sin descargar pesos preentrenados), p. ej. `assert out.shape == (2, 1, 256, 256)`.

El entrenamiento, la descarga de datos/pesos y la revisión de curvas en **TensorBoard** se hacen en Colab. Los logs se escriben en `runs/` y se visualizan allí:

```python
# En Colab / Jupyter
%load_ext tensorboard
%tensorboard --logdir runs
```

Lo que no se pueda verificar sin ejecutar debe marcarse como SUPUESTO y comprobarse en una celda de verificación pensada para Colab.
