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


REWRITE_PROMPT = """Reescribe la última pregunta del usuario para que se entienda sola, sin
la conversación: reemplaza pronombres y referencias ("él", "eso", "y cuándo…") por lo que
nombran. Responde SOLO con la pregunta reescrita, en español, en una línea."""

SUMMARY_PROMPT = """Resume en español, en un párrafo breve, la conversación siguiente: los temas
tratados, los datos importantes que se dieron y lo que quería saber el usuario. Solo el resumen."""


def _complete(cfg, messages, max_chars=2000):
    """Respuesta completa (no en streaming) del modelo de chat."""
    out = "".join(backends.chat_stream(cfg, messages)).strip()
    return out[:max_chars]


def search_query(cfg, messages, summary=None):
    """La consulta de búsqueda para la última pregunta.

    Con `rewrite_followups`, el modelo reescribe las preguntas de seguimiento para que se
    entiendan solas (mejor búsqueda, pero una llamada más al modelo). Si no, o si falla,
    las preguntas cortas se combinan con la anterior.
    """
    users = [m["content"] for m in messages if m["role"] == "user"]
    question = users[-1]
    if len(users) == 1:
        return question
    if cfg.get("rewrite_followups"):
        convo = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:500]}"
                          for m in messages[-5:-1])
        if summary:
            convo = f"Resumen previo: {summary}\n{convo}"
        try:
            rewritten = _complete(cfg, [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"{convo}\n\nÚltima pregunta: {question}"},
            ], max_chars=300).splitlines()
            if rewritten and rewritten[0].strip():
                return rewritten[0].strip().strip('"«»')
        except backends.BackendUnavailable as e:
            log.warning("No se pudo reescribir la pregunta: %s", e)
    if len(question.split()) < 6:
        return users[-2] + " " + question
    return question


def summarize(cfg, old_summary, messages):
    """Resumen actualizado de la conversación (lo antiguo + los mensajes dados)."""
    convo = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:800]}"
                      for m in messages)
    if old_summary:
        convo = f"Resumen de lo anterior: {old_summary}\n\n{convo}"
    return _complete(cfg, [{"role": "system", "content": SUMMARY_PROMPT},
                           {"role": "user", "content": convo}])


def retrieve_for(conn, cfg, messages, index=None, summary=None):
    """(fragmentos, consulta usada) para la última pregunta de la conversación."""
    query = search_query(cfg, messages, summary)
    return retrieve(conn, cfg, query, index), query


def stream_answer(cfg, messages, sources, summary=None):
    """Genera la respuesta token a token. `messages` es el historial reciente más la pregunta
    actual; `summary`, el resumen de lo anterior. Sin modelo de chat, devuelve los fragmentos."""
    system = SYSTEM_PROMPT
    if summary:
        system += f"\n\nResumen de la conversación anterior: {summary}"
    prompt_msgs = [
        {"role": "system", "content": system},
        *messages[:-1],
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
