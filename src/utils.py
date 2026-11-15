"""Funciones comunes que uso en las tres partes: leer el dataset y comparar embeddings."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# La carpeta dataset está al lado de la carpeta src
CARPETA_DATASET = Path(__file__).resolve().parent.parent / "dataset"


def cargar_dataset():
    """Leo dataset.json: una lista con el id, la categoría, la imagen y la caption de cada foto."""
    with open(CARPETA_DATASET / "dataset.json", encoding="utf-8") as f:
        return json.load(f)


def abrir_imagen(elemento):
    """Abro la foto de un elemento del dataset."""
    imagen = Image.open(CARPETA_DATASET / elemento["image"])
    imagen = ImageOps.exif_transpose(imagen)  # las fotos del móvil a veces vienen giradas
    return imagen.convert("RGB")              # los modelos esperan 3 canales (RGB)


def similitud_coseno(a, b):
    """Similitud coseno entre cada fila de a y cada fila de b.

    Primero hago que todos los vectores midan 1 (normalizar). Así el producto
    escalar es directamente el coseno del ángulo: 1 = muy parecidos, 0 = nada que ver.
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def pares(matriz):
    """Devuelvo la similitud de cada pareja de elementos distintos (sin repetir parejas)."""
    filas, columnas = np.triu_indices(len(matriz), k=1)
    return matriz[filas, columnas]


def spearman(x, y):
    """Correlación de Spearman: comparo el ORDEN de los valores, no los valores.

    Cambio cada valor por su posición en el ranking y calculo la correlación.
    1 = las dos listas ordenan igual, 0 = no tienen relación.
    """
    rango_x = np.argsort(np.argsort(x))
    rango_y = np.argsort(np.argsort(y))
    return np.corrcoef(rango_x, rango_y)[0, 1]


def vecino_mas_parecido(matriz):
    """Para cada elemento, busco el otro elemento que más se le parece (sin contarse a sí mismo)."""
    copia = matriz.copy()
    np.fill_diagonal(copia, -np.inf)
    return copia.argmax(axis=1)


def media_dentro_y_fuera(matriz, categorias):
    """Similitud media entre elementos de la misma categoría y entre categorías distintas."""
    categorias = np.array(categorias)
    misma = categorias[:, None] == categorias[None, :]
    filas, columnas = np.triu_indices(len(matriz), k=1)
    valores = matriz[filas, columnas]
    misma = misma[filas, columnas]
    return valores[misma].mean(), valores[~misma].mean()


def comparar_modalidades(sim_img, sim_txt, datos):
    """Comparo cómo se organizan las imágenes entre sí con cómo se organizan las captions.

    Lo uso igual en la parte 2 y en la parte 3 para poder comparar los resultados.
    """
    categorias = [d["category"] for d in datos]
    ids = [d["id"] for d in datos]

    dentro, fuera = media_dentro_y_fuera(sim_img, categorias)
    print(f"Imagenes: similitud media misma categoria = {dentro:.3f} | distinta categoria = {fuera:.3f}")
    dentro, fuera = media_dentro_y_fuera(sim_txt, categorias)
    print(f"Captions: similitud media misma categoria = {dentro:.3f} | distinta categoria = {fuera:.3f}")

    correlacion = spearman(pares(sim_img), pares(sim_txt))
    print(f"Correlacion de Spearman entre las similitudes de imagen y las de texto: {correlacion:.3f}")

    vecino_img = vecino_mas_parecido(sim_img)
    vecino_txt = vecino_mas_parecido(sim_txt)
    print("\nElemento     | mas parecido por imagen | mas parecido por caption")
    for i in range(len(datos)):
        print(f"{ids[i]:<12} | {ids[vecino_img[i]]:<23} | {ids[vecino_txt[i]]}")

    total = len(datos)
    mismo = sum(vecino_img[i] == vecino_txt[i] for i in range(total))
    cat_img = sum(categorias[i] == categorias[vecino_img[i]] for i in range(total))
    cat_txt = sum(categorias[i] == categorias[vecino_txt[i]] for i in range(total))
    print(f"\nEl vecino por imagen y por caption es el mismo en {mismo} de {total}")
    print(f"El vecino por imagen es de la misma categoria en {cat_img} de {total}")
    print(f"El vecino por caption es de la misma categoria en {cat_txt} de {total}")
