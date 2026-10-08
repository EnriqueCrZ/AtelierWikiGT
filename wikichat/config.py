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
    "update_interval_hours": 12,
    "request_delay_seconds": 0.5,
    "ollama_url": "http://localhost:11434",
    "chat_model": "qwen2.5:7b",
    "top_k": 6,
    "host": "127.0.0.1",
    "port": 8080,
}


def load_config(path="config.json"):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg
