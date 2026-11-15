"""Funciones para usar CLIP. Las comparten la parte 3 y el buscador (parte 4)."""
import torch
from transformers import CLIPModel, CLIPProcessor

NOMBRE_CLIP = "openai/clip-vit-base-patch32"


def cargar_clip():
    """Cargo el modelo CLIP preentrenado y su procesador (prepara imágenes y textos)."""
    modelo = CLIPModel.from_pretrained(NOMBRE_CLIP)
    procesador = CLIPProcessor.from_pretrained(NOMBRE_CLIP)
    modelo.eval()  # modo evaluación: no voy a entrenar nada
    return modelo, procesador


def embeddings_imagenes(modelo, procesador, imagenes):
    """Paso las imágenes por el encoder de imagen de CLIP y las proyecto al espacio compartido."""
    entradas = procesador(images=imagenes, return_tensors="pt")
    with torch.no_grad():  # no necesito gradientes porque no entreno
        salida = modelo.vision_model(pixel_values=entradas["pixel_values"])
        embeddings = modelo.visual_projection(salida.pooler_output)
    return embeddings.numpy()


def embeddings_textos(modelo, procesador, textos):
    """Paso los textos por el encoder de texto de CLIP y los proyecto al espacio compartido."""
    entradas = procesador(text=textos, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        salida = modelo.text_model(input_ids=entradas["input_ids"],
                                   attention_mask=entradas["attention_mask"])
        embeddings = modelo.text_projection(salida.pooler_output)
    return embeddings.numpy()
