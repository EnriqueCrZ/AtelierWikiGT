import json
import os

DEFAULTS = {
    "api_url": "https://es.wikipedia.org/w/api.php",
    "user_agent": "AtelierWikiGT/0.1 (https://github.com/EnriqueCrZ/AtelierWikiGT)",
    "db_path": "data/wiki.db",
    "seed_categories": [],
    "category_depth": 1,
    "seed_titles": [],
    "track_all_changes": False,
    "skip_bot_edits": True,
    "update_interval_hours": 12,
    "request_delay_seconds": 0.5,
    # Modelos locales: "ollama" o "openai" (cualquier servidor compatible: llama.cpp,
    # LM Studio, vLLM…; en ese caso llm_url termina en /v1).
    "llm_backend": "ollama",
    "llm_url": "http://localhost:11434",
    "llm_api_key": "",
    # Ollama descargado y administrado por `setup` (se inicia y detiene con `serve`).
    "manage_ollama": False,
    "ollama_dir": "data/ollama",
    "chat_model": "qwen2.5:7b",
    # Cuánto tiempo mantiene Ollama los modelos en memoria sin uso (por defecto los descarga a
    # los 5 minutos y la siguiente pregunta espera a que se vuelvan a cargar). "-1" = siempre.
    "keep_alive": "2h",
    "top_k": 4,
    # Búsqueda semántica (vacío desactiva). Los prefijos dependen del modelo.
    "embed_model": "embeddinggemma",
    "embed_dims": 384,
    "embed_chars": 400,
    "embed_doc_template": "title: {title} | text: {text}",
    "embed_query_prefix": "task: search result | query: ",
    "zim_path": "",
    # Conversaciones guardadas.
    "chats_db_path": "data/chats.db",
    "history_messages": 6,        # mensajes recientes que recibe el modelo tal cual
    "summarize_history": True,    # resume lo más antiguo para no perder el contexto
    "rewrite_followups": False,   # el modelo reescribe preguntas de seguimiento (más lento)
    "host": "127.0.0.1",
    "port": 8800,
}


def load_config(path="config.json"):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg
