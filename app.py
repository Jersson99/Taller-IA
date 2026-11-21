"""Aplicación Interactiva en Streamlit — Sistema Neuro-Agéntico de Triaje Bancario (U4 + U5)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agent_u5 import crear_agente_triaje, invocar_agente
from src.model_u4 import DEFAULT_METADATA_PATH, cargar_modelo_y_metadatos, predecir_ticket

load_dotenv()

st.set_page_config(
    page_title="Triaje Bancario Neuro-Agéntico · U4 + U5",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Recursos en caché (@st.cache_resource) — evita recargar pesos en cada turno
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando red neuronal entrenada (Unidad 4)...")
def obtener_modelo_u4_cacheado():
    return cargar_modelo_y_metadatos()


@st.cache_resource(show_spinner="Inicializando agente LangGraph + memoria (Unidad 5)...")
def obtener_agente_cacheado():
    return crear_agente_triaje()


def cargar_caso_fallo_referencia() -> dict | None:
    """Lee el caso de fallo real guardado en metadata_u4.json para probarlo con un clic."""
    meta_path = Path(DEFAULT_METADATA_PATH)
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("error_analysis_case")


# ---------------------------------------------------------------------------
# Estado de sesión (thread_id + historial de chat + última inferencia U4)
# ---------------------------------------------------------------------------

def reiniciar_sesion():
    st.session_state.thread_id = f"sesion-{uuid.uuid4().hex[:8]}"
    st.session_state.chat_history = []
    st.session_state.ultima_inferencia_u4 = None


if "thread_id" not in st.session_state:
    reiniciar_sesion()


# ---------------------------------------------------------------------------
# Carga inicial y verificación
# ---------------------------------------------------------------------------

try:
    _, metadatos_u4 = obtener_modelo_u4_cacheado()
    agente = obtener_agente_cacheado()
except Exception as exc:
    st.error(f"Error al inicializar el sistema: {exc}")
    st.stop()

caso_fallo = cargar_caso_fallo_referencia()


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("Sistema Neuro-Agéntico de Triaje PQRS Bancarias")
st.caption(
    "Integración Unidad 4 (Red Neuronal con Embeddings en Keras) + "
    "Unidad 5 (Agente LangGraph con 2 @tools, RAG + Reglas y Memoria por thread_id) · "
    "Andres Plazas & Jersson Montenegro"
)


# ---------------------------------------------------------------------------
# Barra lateral: Sesión, Casos de Evaluación e Inferencia Directa U4
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Control de Sesión (Memoria)")
    st.code(f"thread_id: {st.session_state.thread_id}", language="text")
    if st.button("Reiniciar conversación (nuevo thread_id)", use_container_width=True):
        reiniciar_sesion()
        st.rerun()

    st.divider()
    st.header("Casos de Prueba Rápidos")
    st.caption("Haz clic en un caso para enviarlo directamente al agente:")

    ejemplos = {
        "Caso 1 · Saludo (0 tools)": "Hola, ¿qué puedes hacer por mí hoy?",
        "Caso 2 · Solo normativa (1 tool)": (
            "¿Cuál es la política y el SLA para comisiones y reembolsos, y cuándo aplica "
            "la exención de cuota de manejo?"
        ),
        "Caso 3 · Ticket Fraude (2 tools)": (
            "Por favor clasifica y realiza el triaje de este ticket de cliente: "
            "'I saw a card payment on my statement that I do not recognise at all, "
            "I think my card details were stolen.'"
        ),
    }

    if caso_fallo:
        ejemplos["Caso 4 · Fallo Real U4 (Error Analysis)"] = (
            f"Clasifica y aplica protocolo de triaje a este ticket (caso de análisis de fallo "
            f"ID {caso_fallo['test_index']}): \"{caso_fallo['text']}\""
        )

    prompt_rapido = None
    for etiqueta, texto_prompt in ejemplos.items():
        if st.button(etiqueta, use_container_width=True):
            prompt_rapido = texto_prompt

    st.divider()
    st.header("Inferencia Directa (Red U4)")
    st.caption(
        f"Modelo activo: **{metadatos_u4.get('best_model_name', 'Embedding + LSTM')}** "
        f"(Accuracy test: **{metadatos_u4.get('best_test_accuracy', 0) * 100:.1f}%**)"
    )
    ticket_directo = st.text_area(
        "Ingresa un ticket crudo (inglés, estilo Banking77):",
        placeholder="Ej: My card arrived yesterday but I cannot activate it in the app.",
        height=90,
        key="ticket_directo_input",
    )
    c_btn1, c_btn2 = st.columns(2)
    if c_btn1.button("Predecir U4", use_container_width=True) and ticket_directo.strip():
        st.session_state.ultima_inferencia_u4 = predecir_ticket(ticket_directo.strip())
    if c_btn2.button("Enviar al Agente", type="primary", use_container_width=True) and ticket_directo.strip():
        st.session_state.ultima_inferencia_u4 = predecir_ticket(ticket_directo.strip())
        prompt_rapido = (
            f"Realiza el triaje completo (clasificación + política/SLA) para este ticket: "
            f"\"{ticket_directo.strip()}\""
        )


# ---------------------------------------------------------------------------
# Cuerpo principal: Chat (izq) + Panel de Transparencia Técnica (der)
# ---------------------------------------------------------------------------

col_chat, col_panel = st.columns([1.45, 1.0], gap="large")

with col_chat:
    st.subheader("Conversación con el Agente de Triaje")

    for turno in st.session_state.chat_history:
        with st.chat_message(turno["role"]):
            st.markdown(turno["content"])
            if turno["role"] == "assistant":
                tools_usadas = turno.get("nombres_herramientas", [])
                if tools_usadas:
                    badges = "  ".join(f"`{t}`" for t in tools_usadas)
                    st.caption(f"**Herramientas (@tool) invocadas:** {badges}")
                else:
                    st.caption("**Herramientas (@tool) invocadas:** `Ninguna (respuesta directa)`")

                if turno.get("herramientas_invocadas"):
                    with st.expander("Ver traza técnica (tool_calls y salidas)"):
                        for llamada in turno["herramientas_invocadas"]:
                            st.markdown(f"- **`{llamada['name']}`** — args: `{llamada['args']}`")
                        for nombre_t, salida_t in turno.get("salidas_herramientas", {}).items():
                            st.markdown(f"**Salida de `{nombre_t}`:**")
                            st.code(salida_t, language="text")

    entrada_usuario = st.chat_input(
        "Escribe una consulta, un ticket de cliente o pregunta sobre políticas bancarias..."
    )
    mensaje_a_enviar = prompt_rapido or entrada_usuario

    if mensaje_a_enviar:
        st.session_state.chat_history.append({"role": "user", "content": mensaje_a_enviar})
        with st.chat_message("user"):
            st.markdown(mensaje_a_enviar)

        with st.chat_message("assistant"):
            with st.spinner("Razonando y ejecutando herramientas..."):
                res = invocar_agente(agente, mensaje_a_enviar, st.session_state.thread_id)

            # Si el agente invocó clasificar_ticket_neuronal, sincronizar el panel U4
            for llamada in res["herramientas_invocadas"]:
                if llamada["name"] == "clasificar_ticket_neuronal":
                    texto_eval = llamada.get("args", {}).get("texto_ticket", "")
                    if texto_eval:
                        st.session_state.ultima_inferencia_u4 = predecir_ticket(texto_eval)

            st.markdown(res["respuesta"])
            if res["nombres_herramientas"]:
                badges = "  ".join(f"`{t}`" for t in res["nombres_herramientas"])
                st.caption(f"**Herramientas (@tool) invocadas:** {badges}")
            else:
                st.caption("**Herramientas (@tool) invocadas:** `Ninguna (respuesta directa)`")

            if res["herramientas_invocadas"]:
                with st.expander("Ver traza técnica (tool_calls y salidas)"):
                    for llamada in res["herramientas_invocadas"]:
                        st.markdown(f"- **`{llamada['name']}`** — args: `{llamada['args']}`")
                    for nombre_t, salida_t in res["salidas_herramientas"].items():
                        st.markdown(f"**Salida de `{nombre_t}`:**")
                        st.code(salida_t, language="text")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": res["respuesta"],
                "herramientas_invocadas": res["herramientas_invocadas"],
                "nombres_herramientas": res["nombres_herramientas"],
                "salidas_herramientas": res["salidas_herramientas"],
            }
        )
        st.rerun()


with col_panel:
    st.subheader("Transparencia Técnica (Hemisferio U4)")

    inf = st.session_state.ultima_inferencia_u4
    if inf is None:
        st.info(
            "Envía un ticket al agente o usa **Inferencia Directa (Red U4)** en la barra lateral "
            "para visualizar aquí la predicción y probabilidades reales de la red neuronal."
        )
    else:
        st.markdown(f"**Ticket analizado:** _{inf['texto_entrada']}_")
        m1, m2 = st.columns(2)
        m1.metric("Categoría U4", inf["categoria"])
        m2.metric("Confianza", inf["confianza_pct"], delta=f"Certeza: {inf['nivel_certeza']}")

        if inf["requiere_escalamiento_baja_confianza"]:
            st.warning(
                "**Alerta de Gobernanza IA:** Confianza inferior al 65%. "
                "Las reglas de negocio ordenan escalamiento a **Analista Humano Nivel 2**."
            )
        else:
            st.success("Confianza suficiente para enrutamiento automático según SLA.")

        st.markdown("**Distribución Softmax por categoría:**")
        df_probs = pd.DataFrame(
            {
                "Categoría": list(inf["distribucion_probabilidades"].keys()),
                "Probabilidad": list(inf["distribucion_probabilidades"].values()),
            }
        ).set_index("Categoría")
        st.bar_chart(df_probs, height=230)

    if caso_fallo:
        st.divider()
        with st.expander("Referencia: Caso de Fallo Real (Sección 3 del README)"):
            st.write(f"**ID en conjunto de prueba:** `{caso_fallo['test_index']}`")
            st.write(f"**Texto:** _{caso_fallo['text']}_")
            st.write(f"**Categoría Real:** `{caso_fallo['true_label']}`")
            st.write(
                f"**Predicción Modelo:** `{caso_fallo['predicted_label']}` "
                f"(`{caso_fallo['confidence'] * 100:.1f}%`)"
            )
