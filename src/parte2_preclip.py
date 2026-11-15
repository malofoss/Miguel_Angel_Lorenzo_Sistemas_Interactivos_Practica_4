"""Parte 2 (pre-CLIP): una red de imagen y un modelo de texto entrenados por separado.

Como los dos modelos no comparten espacio, no puedo comparar una imagen con su
caption directamente. Lo que comparo es si las imágenes se parecen entre sí de
la misma forma en que se parecen entre sí sus captions.
"""
import torch
from sentence_transformers import SentenceTransformer
from torchvision.models import ResNet18_Weights, resnet18

from utils import abrir_imagen, cargar_dataset, comparar_modalidades, similitud_coseno

NOMBRE_MODELO_TEXTO = "sentence-transformers/distiluse-base-multilingual-cased"


def embeddings_resnet(datos):
    """Saco un embedding de cada imagen con ResNet18 preentrenada."""
    pesos = ResNet18_Weights.DEFAULT
    red = resnet18(weights=pesos)
    # Quito la cabeza de clasificación (la última capa, que da las 1000 clases de ImageNet)
    # para quedarme con las características de la red (512 números por imagen)
    red.fc = torch.nn.Identity()
    red.eval()

    preparar = pesos.transforms()  # el mismo redimensionado y normalizado con que se entrenó la red
    lote = torch.stack([preparar(abrir_imagen(d)) for d in datos])
    with torch.no_grad():
        return red(lote).numpy()


def embeddings_texto(datos):
    """Saco un embedding de cada caption con el modelo de texto."""
    modelo = SentenceTransformer(NOMBRE_MODELO_TEXTO)
    return modelo.encode([d["caption"] for d in datos])


def main():
    datos = cargar_dataset()
    emb_img = embeddings_resnet(datos)
    emb_txt = embeddings_texto(datos)
    print("PARTE 2: pre-CLIP (ResNet18 + modelo de texto)")
    print(f"Embeddings de imagen: {emb_img.shape} | embeddings de texto: {emb_txt.shape}\n")

    # Solo comparo dentro de cada modalidad: imagen con imagen y texto con texto
    sim_img = similitud_coseno(emb_img, emb_img)
    sim_txt = similitud_coseno(emb_txt, emb_txt)
    comparar_modalidades(sim_img, sim_txt, datos)


if __name__ == "__main__":
    main()
