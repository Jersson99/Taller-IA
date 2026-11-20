"""Construcción e invocación del Agente de Triaje con LangChain/LangGraph (Unidad 5)."""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from src.tools import HERRAMIENTAS_AGENTE

load_dotenv()
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM_PROMPT = (
    "Eres el Agente de Triaje Inteligente de Banco Digital Boyacá. Responde siempre en español "
    "de forma clara, concisa y estructurada.\n"
    "Dispones exactamente de dos herramientas:\n"
    "1. `clasificar_ticket_neuronal`: úsala cuando el usuario entregue un ticket/queja de un cliente "
    "para clasificarlo con la red neuronal de la Unidad 4.\n"
    "2. `consultar_politica_y_reglas_triaje`: úsala para verificar SLA, reglas de escalamiento y "
    "consultar el manual interno de políticas (RAG).\n\n"
    "Reglas estrictas de decisión:\n"
    "- Si el usuario solo saluda o hace una pregunta general que no involucra tickets ni políticas "
    "del banco, responde cordialmente SIN invocar ninguna herramienta.\n"
    "- Si el usuario solo pregunta por una política, SLA, requisito o reembolso del banco (sin "
    "entregar un ticket para clasificar), invoca ÚNICAMENTE `consultar_politica_y_reglas_triaje`.\n"
    "- Si el usuario entrega un ticket o queja de un cliente para triaje, invoca PRIMERO "
    "`clasificar_ticket_neuronal` y LUEGO `consultar_politica_y_reglas_triaje` pasando la categoría "
    "y la confianza devueltas por la red neuronal.\n"
    "- Si la confianza de la red neuronal es menor a 0.65 (65%), advierte explícitamente que el "
    "ticket debe escalarse a un Analista Humano Nivel 2 según la política de gobernanza."
)


def crear_agente_triaje(
    model_name: str | None = None,
    checkpointer: InMemorySaver | None = None,
):
    """Construye el agente ReAct con create_agent y memoria persistente (InMemorySaver)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró GOOGLE_API_KEY en el entorno. "
            "Crea el archivo .env a partir de .env.example."
        )

    llm = ChatGoogleGenerativeAI(
        model=model_name or DEFAULT_GEMINI_MODEL,
        api_key=api_key,
    )

    memoria = checkpointer if checkpointer is not None else InMemorySaver()
    agente = create_agent(
        model=llm,
        tools=HERRAMIENTAS_AGENTE,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memoria,
    )
    return agente


def extraer_texto_mensaje(mensaje: Any) -> str:
    """Extrae texto limpio de un mensaje de LangChain v1+ (compatibilidad .text / .content)."""
    texto = getattr(mensaje, "text", None)
    if isinstance(texto, str) and texto.strip():
        return texto.strip()
    contenido = getattr(mensaje, "content", "")
    if isinstance(contenido, str):
        return contenido.strip()
    if isinstance(contenido, list):
        partes = [
            bloque.get("text", "") if isinstance(bloque, dict) else str(bloque)
            for bloque in contenido
        ]
        return "\n".join(p for p in partes if p).strip()
    return str(contenido).strip()


def invocar_agente(agente, mensaje_usuario: str, thread_id: str) -> dict[str, Any]:
    """Invoca el agente en un thread_id y extrae trayectoria (tool_calls) del turno actual."""
    config = {"configurable": {"thread_id": thread_id}}

    # Determinar cuántos mensajes existían antes de este turno en el checkpointer
    estado_previo = agente.get_state(config)
    n_previos = len(estado_previo.values.get("messages", [])) if estado_previo and estado_previo.values else 0

    resultado = agente.invoke(
        {"messages": [{"role": "user", "content": mensaje_usuario}]},
        config=config,
    )

    mensajes_turno = resultado["messages"][n_previos:]
    herramientas_invocadas: list[dict[str, Any]] = []
    salidas_herramientas: dict[str, str] = {}

    for msg in mensajes_turno:
        if getattr(msg, "tool_calls", None):
            for llamada in msg.tool_calls:
                herramientas_invocadas.append(
                    {
                        "name": llamada["name"],
                        "args": llamada.get("args", {}),
                    }
                )
        if msg.__class__.__name__ == "ToolMessage":
            nombre_tool = getattr(msg, "name", "tool")
            salidas_herramientas[nombre_tool] = extraer_texto_mensaje(msg)

    respuesta_final = extraer_texto_mensaje(resultado["messages"][-1])

    return {
        "respuesta": respuesta_final,
        "herramientas_invocadas": herramientas_invocadas,
        "nombres_herramientas": [h["name"] for h in herramientas_invocadas],
        "salidas_herramientas": salidas_herramientas,
        "thread_id": thread_id,
    }
