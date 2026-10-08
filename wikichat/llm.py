"""Generación de respuestas con un LLM local vía Ollama (RAG)."""
import json
import urllib.error
import urllib.request

from .retrieval import search

SYSTEM_PROMPT = """Eres un asistente que responde en español usando ÚNICAMENTE la información de
los fragmentos de la wiki que se te proporcionan. Si la respuesta no está en los fragmentos,
dilo claramente en lugar de inventar. Cita los artículos usados entre corchetes, por ejemplo
[Ciudad de Guatemala]. Sé claro y conciso."""


def build_context(sources):
    return "\n\n".join(
        f"[{s['title']}] ({s['section']})\n{s['text']}" for s in sources
    ) or "(no se encontraron fragmentos relevantes)"


def retrieve_for(conn, messages, k):
    """Usa la última pregunta (y la anterior, para preguntas de seguimiento)."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    sources = search(conn, users[-1], k)
    if len(sources) < k and len(users) > 1:
        seen = {(s["title"], s["section"]) for s in sources}
        for s in search(conn, users[-2] + " " + users[-1], k):
            if (s["title"], s["section"]) not in seen and len(sources) < k:
                sources.append(s)
    return sources


def stream_answer(cfg, messages, sources):
    """Genera la respuesta token a token. Si Ollama no está, devuelve los fragmentos."""
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages[:-1][-6:],
        {"role": "user", "content": f"Fragmentos de la wiki:\n\n{build_context(sources)}\n\n"
                                    f"Pregunta: {messages[-1]['content']}"},
    ]
    body = json.dumps({"model": cfg["chat_model"], "messages": prompt_msgs, "stream": True})
    req = urllib.request.Request(
        f"{cfg['ollama_url']}/api/chat", data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text
                if chunk.get("done"):
                    return
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        yield (f"⚠️ No hay LLM disponible ({e}). Mostrando los fragmentos más relevantes "
               "de la copia local:\n\n")
        for s in sources:
            yield f"**{s['title']}** — {s['section']}\n{s['text'][:600]}…\n\n"
        if not sources:
            yield "No se encontró nada relacionado en la copia local."
