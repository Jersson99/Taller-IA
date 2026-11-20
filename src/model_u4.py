"""Inferencia local del modelo de Red Neuronal (Unidad 4 — PLN con Embeddings)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "modelo_triaje_u4.keras"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata_u4.json"

UMBRAL_CONFIANZA_ALTA = 0.80
UMBRAL_CONFIANZA_MEDIA = 0.65


def limpiar_texto(texto: str) -> str:
    """Normaliza un ticket de texto preservando palabras y números relevantes."""
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9áéíóúñü\s']", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto


def texto_a_secuencia(texto: str, word_to_index: dict[str, int], max_len: int) -> np.ndarray:
    """Convierte un texto crudo en una secuencia de enteros rellenada (padding='post')."""
    tokens = limpiar_texto(texto).split()
    oov_idx = word_to_index.get("<OOV>", 1)
    seq = [word_to_index.get(tok, oov_idx) for tok in tokens[:max_len]]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    return np.array([seq], dtype="int32")


@lru_cache(maxsize=1)
def cargar_modelo_y_metadatos(
    model_path: str = str(DEFAULT_MODEL_PATH),
    metadata_path: str = str(DEFAULT_METADATA_PATH),
) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Carga una sola vez el modelo Keras (.keras) y su metadata JSON."""
    m_path = Path(model_path)
    meta_path = Path(metadata_path)
    if not m_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {m_path}. "
            "Ejecuta primero notebooks/entrenamiento_u4.ipynb."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de metadatos en {meta_path}. "
            "Ejecuta primero notebooks/entrenamiento_u4.ipynb."
        )

    modelo = tf.keras.models.load_model(m_path)
    with meta_path.open("r", encoding="utf-8") as f:
        metadatos = json.load(f)
    return modelo, metadatos


def predecir_ticket(texto_ticket: str) -> dict[str, Any]:
    """Ejecuta la inferencia local de la red neuronal sobre un ticket crudo."""
    modelo, metadatos = cargar_modelo_y_metadatos()
    word_to_index: dict[str, int] = metadatos["word_to_index"]
    class_names: list[str] = metadatos["class_names"]
    max_len: int = int(metadatos["max_len"])

    x_input = texto_a_secuencia(texto_ticket, word_to_index, max_len)
    probs = modelo.predict(x_input, verbose=0)[0]

    idx_pred = int(np.argmax(probs))
    categoria = class_names[idx_pred]
    confianza = float(probs[idx_pred])

    if confianza >= UMBRAL_CONFIANZA_ALTA:
        nivel_certeza = "Alta"
    elif confianza >= UMBRAL_CONFIANZA_MEDIA:
        nivel_certeza = "Media"
    else:
        nivel_certeza = "Baja"

    distribucion = {
        class_names[i]: round(float(probs[i]), 4)
        for i in np.argsort(probs)[::-1]
    }

    return {
        "texto_entrada": texto_ticket,
        "categoria": categoria,
        "confianza": round(confianza, 4),
        "confianza_pct": f"{confianza * 100:.1f}%",
        "nivel_certeza": nivel_certeza,
        "requiere_escalamiento_baja_confianza": confianza < UMBRAL_CONFIANZA_MEDIA,
        "distribucion_probabilidades": distribucion,
        "arquitectura_modelo": metadatos.get("best_model_name", "Embedding + LSTM"),
    }
