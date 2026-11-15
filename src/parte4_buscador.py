"""Parte 4: buscador de imágenes con CLIP.

Escribo una frase en inglés y el programa me devuelve las k imágenes del
dataset que más se parecen, cada una con su puntuación.
"""
import argparse

from clip_utils import cargar_clip, embeddings_imagenes, embeddings_textos
from utils import abrir_imagen, cargar_dataset, similitud_coseno


def buscar(consulta, k, modelo, procesador, emb_img, datos):
    """Devuelvo los k elementos más parecidos a la consulta, con su puntuación."""
    emb_consulta = embeddings_textos(modelo, procesador, [consulta])
    puntuaciones = similitud_coseno(emb_consulta, emb_img)[0]
    mejores = puntuaciones.argsort()[::-1][:k]  # ordeno de mayor a menor y me quedo con k
    return [(datos[i], puntuaciones[i]) for i in mejores]


def main():
    parser = argparse.ArgumentParser(description="Buscador de imagenes con CLIP")
    parser.add_argument("--k", type=int, default=5, help="numero de resultados a mostrar")
    k = parser.parse_args().k

    datos = cargar_dataset()
    modelo, procesador = cargar_clip()

    # Calculo los embeddings de las imágenes UNA sola vez, aquí al empezar.
    # Después, en cada consulta, solo calculo el embedding del texto.
    print("Calculando los embeddings de las imagenes...")
    emb_img = embeddings_imagenes(modelo, procesador, [abrir_imagen(d) for d in datos])

    print(f"Listo: {len(datos)} imagenes. Escribe una consulta en ingles.")
    print("Para cambiar el numero de resultados escribe k=3 (por ejemplo). Para salir, escribe salir.")
    while True:
        consulta = input(f"\nConsulta (k={k}): ").strip()
        if consulta.lower() in ("salir", "exit", "q"):
            break
        if consulta == "":
            continue
        if consulta.lower().startswith("k="):
            valor = consulta[2:].strip()
            if valor.isdigit() and int(valor) > 0:
                k = min(int(valor), len(datos))
            else:
                print("k tiene que ser un numero mayor que 0")
            continue

        resultados = buscar(consulta, k, modelo, procesador, emb_img, datos)
        for puesto, (elemento, puntuacion) in enumerate(resultados, start=1):
            print(f"{puesto}. {puntuacion:.3f}  {elemento['image']:<28} {elemento['caption']}")


if __name__ == "__main__":
    main()
