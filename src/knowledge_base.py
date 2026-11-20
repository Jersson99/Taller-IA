"""Base de conocimiento (RAG ligero) y motor de reglas de escalamiento/SLA bancario."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MANUAL_POLITICAS_PQRS = """
MANUAL INTERNO DE POLÍTICAS DE ATENCIÓN Y TRIAJE PQRS — BANCO DIGITAL BOYACÁ (EDICIÓN 2026)

1. POLÍTICA DE FRAUDE Y SEGURIDAD
- Alcance: Tarjetas comprometidas, pagos con tarjeta no reconocidos, retiros en cajero no autorizados, débitos directos desconocidos y tarjetas perdidas o robadas.
- Acción inmediata obligatoria: Bloqueo preventivo instantáneo del plástico o canal digital comprometido y emisión de número de radicado de contracargo.
- SLA máximo de atención: 2 horas calendario (Prioridad Crítica).
- Compensación / Reembolso provisional: Se otorga abono temporal hasta por $2.000.000 COP en 24 horas hábiles tras validar identidad biométrica del titular.
- Área responsable: Mesa Antifraude y Seguridad 24/7.

2. POLÍTICA DE TRANSFERENCIAS Y PAGOS
- Alcance: Transferencias fallidas, transferencias pendientes por más de 2 horas, transferencias no recibidas por el destinatario, pagos declinados o pagos revertidos.
- Acción obligatoria: Consultar código de rastreo en cámara de compensación ACH/PSE; si la falla es atribuible al banco, activar reversión automática en máximo 24 horas.
- SLA máximo de atención: 12 horas hábiles (Prioridad Alta).
- Compensación: Condonación de comisiones bancarias de la operación y reconocimiento de intereses moratorios si el retraso supera 48 horas hábiles.
- Área responsable: Operaciones de Pagos y Tesorería.

3. POLÍTICA DE TARJETAS Y ACTIVACIÓN
- Alcance: Activación de nueva tarjeta, estado de envío y entrega física, tarjeta que no funciona en datáfono, PIN bloqueado o solicitud de tarjeta física.
- Acción obligatoria: Guiar activación desde la app móvil con CVV dinámico; ante falla física de chip o banda, reexpedición gratuita con entrega en 3 días hábiles.
- SLA máximo de atención: 24 horas hábiles (Prioridad Media).
- Compensación: Envío exprés sin costo ($0 COP) cuando el retraso en entrega supera 5 días hábiles.
- Área responsable: Gestión de Medios de Pago.

4. POLÍTICA DE COMISIONES Y REEMBOLSOS
- Alcance: Cobros de comisiones por pago con tarjeta, cargos por retiro en efectivo, cobros extra en extracto, cargos por tasa de cambio o reembolsos que no aparecen reflejados.
- Acción obligatoria: Auditar el movimiento contra el tarifario vigente; si existe cobro duplicado o indebido, programar devolución directa a la cuenta en máximo 5 días hábiles.
- Exención especial: Clientes con saldo promedio mensual superior a $1.500.000 COP tienen exención del 100% en cuota de manejo.
- SLA máximo de atención: 48 horas hábiles (Prioridad Media).
- Área responsable: Facturación, Tarifas y Reembolsos.

5. POLÍTICA DE CUENTA Y VERIFICACIÓN
- Alcance: Verificación de identidad (KYC), problemas al subir documentos, olvido de clave/passcode, actualización de datos personales o cancelación de cuenta.
- Acción obligatoria: Enviar enlace seguro de verificación biométrica (cédula + selfie) o desbloqueo de clave en 15 minutos tras validación de doble factor (OTP).
- SLA máximo de atención: 24 horas hábiles (Prioridad Baja-Media).
- Área responsable: Onboarding y Soporte de Cuentas.

6. REGLA TRANSVERSAL DE GOBERNANZA DE IA (ESCALAMIENTO POR BAJA CONFIANZA)
- Cuando el modelo de red neuronal arroje una confianza inferior al 65% (0.65), el ticket se considera ambiguo o de riesgo de clasificación errónea.
- En estos casos el agente DEBE indicar explícitamente que el caso requiere revisión obligatoria por un Analista Humano Nivel 2 antes de ejecutar cierres automáticos o devoluciones.
"""

REGLAS_NEGOCIO_POR_CATEGORIA: dict[str, dict[str, Any]] = {
    "Fraude y Seguridad": {
        "prioridad": "CRÍTICA",
        "sla_horas": 2,
        "area_responsable": "Mesa Antifraude 24/7",
        "accion_automatica": "Bloqueo preventivo inmediato y radicado de contracargo",
        "tope_reembolso_provisional_cop": 2_000_000,
    },
    "Transferencias y Pagos": {
        "prioridad": "ALTA",
        "sla_horas": 12,
        "area_responsable": "Operaciones de Pagos (ACH/PSE)",
        "accion_automatica": "Rastreo interbancario y reversión automática si aplica",
        "tope_reembolso_provisional_cop": 500_000,
    },
    "Tarjetas y Activación": {
        "prioridad": "MEDIA",
        "sla_horas": 24,
        "area_responsable": "Gestión de Medios de Pago",
        "accion_automatica": "Guía de activación en app o reexpedición gratuita",
        "tope_reembolso_provisional_cop": 0,
    },
    "Comisiones y Reembolsos": {
        "prioridad": "MEDIA",
        "sla_horas": 48,
        "area_responsable": "Facturación, Tarifas y Reembolsos",
        "accion_automatica": "Auditoría de extracto y abono en máx. 5 días hábiles",
        "tope_reembolso_provisional_cop": 300_000,
    },
    "Cuenta y Verificación": {
        "prioridad": "BAJA-MEDIA",
        "sla_horas": 24,
        "area_responsable": "Onboarding y Soporte de Cuentas",
        "accion_automatica": "Envío de enlace seguro KYC / restablecimiento OTP",
        "tope_reembolso_provisional_cop": 0,
    },
}


def _normalizar_categoria(categoria: str) -> str:
    """Empareja el texto de categoría con una de las 5 llaves oficiales."""
    cat_low = categoria.strip().lower()
    for oficial in REGLAS_NEGOCIO_POR_CATEGORIA:
        if oficial.lower() in cat_low or cat_low in oficial.lower():
            return oficial
    if any(k in cat_low for k in ("fraud", "segur", "robo", "comprom", "no reconoc")):
        return "Fraude y Seguridad"
    if any(k in cat_low for k in ("transfer", "pago", "declin", "pend")):
        return "Transferencias y Pagos"
    if any(k in cat_low for k in ("tarjet", "card", "pin", "activ")):
        return "Tarjetas y Activación"
    if any(k in cat_low for k in ("comisi", "cargo", "fee", "reembols", "refund")):
        return "Comisiones y Reembolsos"
    return "Cuenta y Verificación"


def evaluar_reglas_escalamiento(categoria: str, confianza: float) -> dict[str, Any]:
    """Aplica reglas deterministas de SLA, prioridad y escalamiento por incertidumbre."""
    cat_oficial = _normalizar_categoria(categoria)
    regla = dict(REGLAS_NEGOCIO_POR_CATEGORIA[cat_oficial])
    requiere_humano = confianza < 0.65

    if requiere_humano:
        estado_escalamiento = (
            "ESCALAR A ANALISTA HUMANO NIVEL 2 "
            f"(confianza del modelo {confianza * 100:.1f}% < 65.0%)"
        )
    elif cat_oficial == "Fraude y Seguridad":
        estado_escalamiento = "ESCALAMIENTO INMEDIATO A MESA ANTIFRAUDE (Prioridad Crítica)"
    else:
        estado_escalamiento = f"Enrutamiento estándar a {regla['area_responsable']}"

    return {
        "categoria_normalizada": cat_oficial,
        "prioridad": regla["prioridad"],
        "sla_horas": regla["sla_horas"],
        "area_responsable": regla["area_responsable"],
        "accion_automatica": regla["accion_automatica"],
        "tope_reembolso_provisional_cop": regla["tope_reembolso_provisional_cop"],
        "requiere_revision_humana_n2": requiere_humano,
        "estado_escalamiento": estado_escalamiento,
    }


@lru_cache(maxsize=1)
def _obtener_fragmentos() -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=380, chunk_overlap=45)
    return splitter.split_text(MANUAL_POLITICAS_PQRS)


@lru_cache(maxsize=1)
def construir_vectorstore() -> InMemoryVectorStore:
    """Construye la base vectorial en memoria con Gemini Embeddings (patrón Unidad 5)."""
    fragmentos = _obtener_fragmentos()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return InMemoryVectorStore.from_texts(fragmentos, embedding=embeddings)


def buscar_en_manual_rag(consulta: str, k: int = 2) -> tuple[list[str], str]:
    """Recupera los k fragmentos más relevantes del manual usando RAG vectorial."""
    try:
        vs = construir_vectorstore()
        docs = vs.similarity_search(consulta, k=k)
        return [d.page_content.strip() for d in docs], "RAG vectorial (InMemoryVectorStore + gemini-embedding-001)"
    except Exception:
        # Respaldo determinista por coincidencia léxica si el endpoint de embeddings falla
        fragmentos = _obtener_fragmentos()
        palabras = {p.lower().strip(".,:;¿?¡!") for p in consulta.split() if len(p) > 3}
        puntuados = sorted(
            fragmentos,
            key=lambda frag: sum(1 for p in palabras if p in frag.lower()),
            reverse=True,
        )
        return [f.strip() for f in puntuados[:k]], "RAG léxico (respaldo local)"
