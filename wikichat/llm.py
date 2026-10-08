"""Respuestas con recuperación (RAG): busca fragmentos en la copia local y se los da al
modelo de chat local para que redacte la respuesta."""
import logging

from . import backends, vectors
from .retrieval import search

log = logging.getLogger("wikichat.llm")

# Medido con `wikichat eval` (evals/quimica.json, qwen2.5:3b): frente a las instrucciones
# anteriores deja de inventar, de hablar de "los fragmentos" y de copiar las cabeceras del
# contexto, y cita el artículo en el 73 % de las respuestas en vez del 7 % (con temperatura 0,2).
SYSTEM_PROMPT = """Eres un asistente que responde preguntas usando solo los textos de Wikipedia que se te dan.

Cómo responder:
1. Empieza con la respuesta directa a la pregunta en la primera oración.
2. Usa únicamente datos que aparezcan en los textos. No agregues nada de tu propio conocimiento, aunque lo sepas.
3. Ignora los textos que no tengan que ver con la pregunta.
4. Si los textos no contienen la respuesta, responde exactamente: "No encontré esa información en la wiki." y nada más.
5. Después de cada dato, escribe entre corchetes el artículo de donde sale, por ejemplo: El oro tiene número atómico 79 [Oro].
6. No menciones los textos ni la wiki en la respuesta; responde como si lo supieras.
7. Responde en español, en 2 a 4 oraciones, salvo que pidan más detalle."""


def build_context(sources):
    """Los textos recuperados, cada uno con su artículo de origen. Antes iban como
    "[Título] (Sección)" y los modelos pequeños copiaban esa cabecera tal cual en la respuesta."""
    return "\n\n".join(
        f"Artículo: {s['title']}" + (f" (sección «{s['section']}»)" if s["section"] != "Introducción" else "")
        + f"\n{s['text']}" for s in sources
    ) or "(no se encontró ningún texto relacionado)"


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
    system = cfg.get("system_prompt") or SYSTEM_PROMPT
    if summary:
        system += f"\n\nResumen de la conversación anterior: {summary}"
    prompt_msgs = [
        {"role": "system", "content": system},
        *messages[:-1],
        # La pregunta va antes y después de los textos: a los modelos pequeños les cuesta
        # recordar qué se preguntó después de leer varios párrafos.
        {"role": "user", "content": f"Pregunta: {messages[-1]['content']}\n\n"
                                    f"Textos de la wiki:\n\n{build_context(sources)}\n\n"
                                    f"Responde a la pregunta: {messages[-1]['content']}"},
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
