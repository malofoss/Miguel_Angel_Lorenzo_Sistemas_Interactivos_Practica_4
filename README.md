# Práctica 4: pre-CLIP, CLIP y buscador multimodal

Miguel Ángel Lorenzo Fossati, Sistemas Interactivos Inteligentes (UIE)

## Qué hay en la entrega

| Archivo | Para qué sirve |
|---|---|
| `dataset/` | Las 20 imágenes, en 4 carpetas (food, cars, motorcycles, dog), y `dataset.json` con las captions |
| `src/parte2_preclip.py` | Parte 2: ResNet18 + modelo de texto, entrenados por separado |
| `src/parte3_clip.py` | Parte 3: CLIP |
| `src/parte4_buscador.py` | Parte 4: buscador por texto |
| `src/utils.py` | Funciones comunes: leer el dataset y comparar embeddings |
| `src/clip_utils.py` | Funciones para usar CLIP (las comparten las partes 3 y 4) |
| `src/entorno_test.py`, `Dockerfile`, `Makefile`, `requirements.txt` | Archivos del profesor, sin cambios |

## Cómo se ejecuta

### Con Docker

```
make build
make run                                 # prueba del entorno
make run SCRIPT=src/parte2_preclip.py    # parte 2
make run SCRIPT=src/parte3_clip.py       # parte 3
make run SCRIPT=src/parte4_buscador.py   # parte 4 (buscador)
```

### Sin Docker, con uv (Windows)

```
uv venv --python 3.12
uv pip install --torch-backend cpu -r requirements.txt
uv run python src/entorno_test.py
uv run python src/parte2_preclip.py
uv run python src/parte3_clip.py
uv run python src/parte4_buscador.py
```

La primera vez se descargan los modelos automáticamente. No hace falta GPU.

## Dataset

20 imágenes en 4 categorías de 5: comida (`food`), coches (`cars`), motos (`motorcycles`) y perro (`dog`).
Cada entrada de `dataset/dataset.json` tiene el `id`, la `category`, la ruta de la `image` y la `caption` en inglés.

## Parte 2: pre-CLIP

- Imágenes: **ResNet18** preentrenada de torchvision. Le quito la última capa (la de clasificación) para quedarme con las características: 512 números por imagen.
- Captions: **sentence-transformers/distiluse-base-multilingual-cased**.

Los dos modelos se entrenaron por separado y no comparten espacio, así que no comparo una imagen con su caption directamente. Lo que hago es comparar dentro de cada modalidad (imagen con imagen y caption con caption) y ver si las dos se organizan igual:

- Similitud media entre elementos de la misma categoría y de categorías distintas.
- Correlación de Spearman entre las similitudes de imagen y las de texto (compara el orden, no los valores).
- El elemento más parecido a cada uno según las imágenes y según las captions.

## Parte 3: CLIP

Uso **openai/clip-vit-base-patch32** de la librería Transformers. Como CLIP pone imagen y texto en el mismo espacio, aquí sí comparo cada imagen con las 20 captions:

- Similitud de cada imagen con su caption y en qué puesto queda su caption.
- En cuántas imágenes la caption más parecida es la suya, y en cuántas es al menos de su categoría.
- Lo mismo al revés (de caption a imagen).
- Al final repito el análisis de la parte 2 con los embeddings de CLIP, para poder comparar.

## Parte 4: buscador

```
uv run python src/parte4_buscador.py --k 5
```

- Escribo una consulta en inglés, por ejemplo `a dog on a boat`.
- Devuelve las k imágenes más parecidas, con su puntuación (similitud coseno).
- `k` se elige con `--k` al arrancar o escribiendo `k=3` dentro del programa.
- Los embeddings de las imágenes se calculan una sola vez al arrancar. En cada consulta solo se calcula el embedding del texto.
- Para salir, se escribe `salir`.
