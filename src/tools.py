"""Definición de las dos (2) herramientas (@tool) obligatorias del Agente (Unidad 5)."""

from __future__ import annotations

import json
from langchain.tools import tool

from src.knowledge_base import buscar_en_manual_rag, evaluar_reglas_escalamiento
from src.model_u4 import predecir_ticket


@tool
def clasificar_ticket_neuronal(texto_ticket: str) -> str:
    """Clasifica un ticket o queja bancaria (PQRS) usando la red neuronal entrenada en la Unidad 4.

    Usa esta herramienta SIEMPRE que el usuario proporcione el texto de un ticket, queja o
    solicitud de un cliente bancario que deba ser clasificado en una categoría operativa.
    NO la uses para saludos, preguntas generales ni consultas puras sobre políticas internas.

    Args:
        texto_ticket: Texto crudo del ticket o queja del cliente (en inglés o texto tal cual llega).

    Returns:
        Resumen compacto con la categoría predicha, porcentaje de confianza, nivel de certeza
        y bandera de escalamiento por baja confianza.
    """
    resultado = predecir_ticket(texto_ticket)
    top3 = list(resultado["distribucion_probabilidades"].items())[:3]
    top3_str = ", ".join(f"{cat}: {prob * 100:.1f}%" for cat, prob in top3)
    return (
        f"Categoría predicha: {resultado['categoria']} | "
        f"Confianza: {resultado['confianza']:.4f} ({resultado['confianza_pct']}) | "
        f"Nivel de certeza: {resultado['nivel_certeza']} | "
        f"Escalar por baja confianza (<65%): "
        f"{'SÍ' if resultado['requiere_escalamiento_baja_confianza'] else 'NO'} | "
        f"Top-3 probabilidades: [{top3_str}]"
    )


@tool
def consultar_politica_y_reglas_triaje(
    categoria: str,
    confianza: float = 1.0,
    consulta_normativa: str = "",
) -> str:
    """Consulta el manual interno de políticas bancarias (RAG) y evalúa reglas de SLA y escalamiento.

    Usa esta herramienta en dos escenarios:
    1) Después de ejecutar `clasificar_ticket_neuronal`, pasando la categoría y la confianza (0.0 a 1.0)
       para obtener el SLA oficial, la prioridad, las reglas de escalamiento y el protocolo interno.
    2) Cuando el usuario pregunte directamente por políticas, tiempos SLA, reembolsos, exenciones
       o protocolos del banco sin entregar un ticket de cliente para clasificar.

    Args:
        categoria: Categoría bancaria ('Fraude y Seguridad', 'Transferencias y Pagos',
            'Tarjetas y Activación', 'Comisiones y Reembolsos' o 'Cuenta y Verificación').
        confianza: Probabilidad entregada por la red neuronal entre 0.0 y 1.0 (usa 1.0 si es consulta
            pura de normativa sin clasificación previa).
        consulta_normativa: Pregunta o tema puntual a buscar en el manual de políticas (RAG).

    Returns:
        Dictamen estructurado con reglas deterministas de SLA/escalamiento y fragmentos relevantes del manual.
    """
    reglas = evaluar_reglas_escalamiento(categoria=categoria, confianza=confianza)
    query_rag = consulta_normativa.strip() or f"política SLA protocolo {reglas['categoria_normalizada']}"
    fragmentos, motor = buscar_en_manual_rag(query_rag, k=2)

    salida = {
        "reglas_negocio": reglas,
        "motor_rag": motor,
        "fragmentos_politica": fragmentos,
    }
    return json.dumps(salida, ensure_ascii=False, indent=2)


HERRAMIENTAS_AGENTE = [
    clasificar_ticket_neuronal,
    consultar_politica_y_reglas_triaje,
]
