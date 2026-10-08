"""Respuestas con recuperación (RAG): busca fragmentos en la copia local y se los da al
modelo de chat local para que redacte la respuesta."""
import logging

from . import backends, vectors
from .retrieval import search

log = logging.getLogger("wikichat.llm")

SYSTEM_PROMPT = """Eres un asistente que responde en español usando ÚNICAMENTE la información de
los fragmentos de la wiki que se te proporcionan. Si la respuesta no está en los fragmentos,
dilo claramente en lugar de inventar. Cita los artículos usados entre corchetes, por ejemplo
[Ciudad de Guatemala]. Sé claro y conciso."""


def build_context(sources):
    return "\n\n".join(
        f"[{s['title']}] ({s['section']})\n{s['text']}" for s in sources
    ) or "(no se encontraron fragmentos relevantes)"


def retrieve(conn, cfg, query, index=None):
    """Búsqueda híbrida si hay índice semántico; si el modelo de embeddings no responde,
    se usa solo la búsqueda por palabras."""
    semantic = None
    if index is not None and len(index):
        try:
            semantic = index.search(conn, vectors.embed_query(cfg, query), cfg["top_k"] * 2)
        except backends.BackendUnavailable as e:
            log.warning("Búsqueda semántica no disponible (%s); se usa solo BM25", e)
    return search(conn, query, cfg["top_k"], semantic)


def retrieve_for(conn, cfg, messages, index=None):
    """Usa la última pregunta; en preguntas de seguimiento cortas añade la anterior."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    query = users[-1]
    if len(users) > 1 and len(query.split()) < 6:
        query = users[-2] + " " + query
    return retrieve(conn, cfg, query, index)


def stream_answer(cfg, messages, sources):
    """Genera la respuesta token a token. Si no hay modelo de chat, devuelve los fragmentos."""
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages[:-1][-6:],
        {"role": "user", "content": f"Fragmentos de la wiki:\n\n{build_context(sources)}\n\n"
                                    f"Pregunta: {messages[-1]['content']}"},
    ]
    try:
        yield from backends.chat_stream(cfg, prompt_msgs)
    except backends.BackendUnavailable as e:
        yield (f"⚠️ No hay modelo de chat disponible ({e}). Mostrando los fragmentos más "
               "relevantes de la copia local:\n\n")
        for s in sources:
            yield f"**{s['title']}** — {s['section']}\n{s['text'][:600]}…\n\n"
        if not sources:
            yield "No se encontró nada relacionado en la copia local."
