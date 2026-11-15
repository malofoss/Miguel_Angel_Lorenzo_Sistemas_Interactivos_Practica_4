"""Parte 3 (CLIP): imagen y texto en el mismo espacio.

Ahora sí puedo comparar cada imagen con cada caption directamente y comprobar
si la caption más parecida a cada imagen es la suya.
"""
from clip_utils import cargar_clip, embeddings_imagenes, embeddings_textos
from utils import abrir_imagen, cargar_dataset, comparar_modalidades, similitud_coseno


def main():
    datos = cargar_dataset()
    ids = [d["id"] for d in datos]
    categorias = [d["category"] for d in datos]

    modelo, procesador = cargar_clip()
    emb_img = embeddings_imagenes(modelo, procesador, [abrir_imagen(d) for d in datos])
    emb_txt = embeddings_textos(modelo, procesador, [d["caption"] for d in datos])
    print("PARTE 3: CLIP")
    print(f"Embeddings de imagen: {emb_img.shape} | embeddings de texto: {emb_txt.shape}\n")

    # Fila = imagen, columna = caption
    sim = similitud_coseno(emb_img, emb_txt)

    print("Imagen       | sim. con su caption | puesto de su caption | caption mas parecida")
    aciertos = 0
    aciertos_categoria = 0
    for i in range(len(datos)):
        orden = sim[i].argsort()[::-1]            # captions de más a menos parecida
        puesto = list(orden).index(i) + 1         # en qué puesto queda la caption correcta
        mejor = orden[0]
        aciertos += mejor == i
        aciertos_categoria += categorias[mejor] == categorias[i]
        print(f"{ids[i]:<12} | {sim[i, i]:<19.3f} | {puesto:<20} | {ids[mejor]}")

    total = len(datos)
    aciertos_texto = sum(sim[:, j].argmax() == j for j in range(total))
    print(f"\nImagen -> caption: la mas parecida es la suya en {aciertos} de {total}")
    print(f"Imagen -> caption: la mas parecida es de su categoria en {aciertos_categoria} de {total}")
    print(f"Caption -> imagen: la mas parecida es la suya en {aciertos_texto} de {total}")
    print(f"Similitud media con su caption: {sim.diagonal().mean():.3f} | con las demas: "
          f"{(sim.sum() - sim.diagonal().sum()) / (total * total - total):.3f}")

    # El mismo análisis de la parte 2, para poder comparar los dos mundos
    print("\nMismo analisis que en la parte 2, pero con los embeddings de CLIP:")
    comparar_modalidades(similitud_coseno(emb_img, emb_img), similitud_coseno(emb_txt, emb_txt), datos)


if __name__ == "__main__":
    main()
