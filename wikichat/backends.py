"""Acceso a los modelos locales: Ollama o cualquier servidor compatible con la API de
OpenAI (llama.cpp `llama-server`, LM Studio, vLLM, Jan…)."""
import json
import urllib.error
import urllib.request


class BackendUnavailable(Exception):
    pass


def _post(cfg, path, payload, timeout):
    headers = {"Content-Type": "application/json"}
    if cfg.get("llm_api_key"):
        headers["Authorization"] = f"Bearer {cfg['llm_api_key']}"
    req = urllib.request.Request(
        cfg["llm_url"].rstrip("/") + path, data=json.dumps(payload).encode(), headers=headers,
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise BackendUnavailable(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, OSError) as e:
        raise BackendUnavailable(str(e)) from e


def chat_stream(cfg, messages):
    """Genera la respuesta del modelo de chat, fragmento a fragmento."""
    if cfg["llm_backend"] == "openai":
        resp = _post(cfg, "/chat/completions",
                     {"model": cfg["chat_model"], "messages": messages, "stream": True}, 600)
        with resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                choices = json.loads(data).get("choices") or [{}]
                text = (choices[0].get("delta") or {}).get("content")
                if text:
                    yield text
    else:
        # think=False evita que modelos con razonamiento (qwen3, deepseek-r1…) lo mezclen
        # en la respuesta; los modelos sin razonamiento lo ignoran.
        resp = _post(cfg, "/api/chat", {"model": cfg["chat_model"], "messages": messages,
                                        "stream": True, "think": False}, 600)
        with resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise BackendUnavailable(chunk["error"])
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text
                if chunk.get("done"):
                    return


def embed(cfg, texts):
    """Lista de vectores (listas de float) para los textos dados."""
    if cfg["llm_backend"] == "openai":
        with _post(cfg, "/embeddings", {"model": cfg["embed_model"], "input": texts}, 300) as r:
            data = json.load(r)["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
    with _post(cfg, "/api/embed",
               {"model": cfg["embed_model"], "input": texts, "truncate": True}, 300) as r:
        return json.load(r)["embeddings"]
