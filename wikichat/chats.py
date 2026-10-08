"""Conversaciones guardadas. Viven en su propio archivo (data/chats.db), separado de la
wiki, para que actualizarla o reimportarla nunca toque el historial."""
import json
import os
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    summary TEXT,                    -- resumen de los mensajes antiguos
    summary_upto INTEGER DEFAULT 0   -- id del último mensaje incluido en el resumen
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,              -- user | assistant
    content TEXT NOT NULL,
    sources TEXT,                    -- JSON: [{title, section}]
    images TEXT,                     -- JSON: [{id, title, caption}]
    query TEXT,                      -- búsqueda usada (si se reescribió la pregunta)
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
"""

TITLE_CHARS = 60


class ChatStore:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def create(self, title="Nuevo chat"):
        chat_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self.lock:
            self.conn.execute("INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                              (chat_id, title, now, now))
            self.conn.commit()
        return chat_id

    def list(self):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC").fetchall()
        return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

    def exists(self, chat_id):
        with self.lock:
            return self.conn.execute("SELECT 1 FROM chats WHERE id=?", (chat_id,)).fetchone() is not None

    def get(self, chat_id):
        with self.lock:
            chat = self.conn.execute(
                "SELECT id, title, summary, summary_upto FROM chats WHERE id=?", (chat_id,)).fetchone()
            if not chat:
                return None
            rows = self.conn.execute(
                "SELECT id, role, content, sources, images FROM messages WHERE chat_id=? ORDER BY id",
                (chat_id,)).fetchall()
        return {
            "id": chat[0], "title": chat[1], "summary": chat[2], "summary_upto": chat[3] or 0,
            "messages": [{"id": r[0], "role": r[1], "content": r[2],
                          "sources": json.loads(r[3] or "[]"), "images": json.loads(r[4] or "[]")}
                         for r in rows],
        }

    def add_message(self, chat_id, role, content, sources=None, images=None, query=None):
        now = time.time()
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO messages (chat_id, role, content, sources, images, query, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chat_id, role, content, json.dumps(sources or [], ensure_ascii=False),
                 json.dumps(images or [], ensure_ascii=False), query, now))
            # El primer mensaje del usuario da nombre al chat (si no se renombró a mano).
            if role == "user":
                self.conn.execute(
                    "UPDATE chats SET title=? WHERE id=? AND title='Nuevo chat'",
                    (_title_from(content), chat_id))
            self.conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
            self.conn.commit()
        return cur.lastrowid

    def rename(self, chat_id, title):
        title = " ".join(title.split())[:120] or "Sin título"
        with self.lock:
            cur = self.conn.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
            self.conn.commit()
        return cur.rowcount > 0

    def delete(self, chat_id):
        with self.lock:
            cur = self.conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
            self.conn.commit()
        return cur.rowcount > 0

    def set_summary(self, chat_id, summary, upto):
        with self.lock:
            self.conn.execute("UPDATE chats SET summary=?, summary_upto=? WHERE id=?",
                              (summary, upto, chat_id))
            self.conn.commit()


def _title_from(text):
    text = " ".join(text.split())
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1].rstrip() + "…"


def context_for(chat, recent):
    """Lo que el modelo recibe de la conversación: el resumen de lo antiguo (si hay) y los
    últimos `recent` mensajes, sin contar la pregunta actual."""
    history = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
    previous = history[:-1]
    kept = previous[-recent:] if recent else []
    summary = chat.get("summary") if len(previous) > len(kept) else None
    return summary, kept


def needs_summary(chat, recent, slack=4):
    """True si hay suficientes mensajes fuera de la ventana reciente sin resumir."""
    msgs = chat["messages"]
    outside = msgs[:-recent] if recent else msgs
    pending = [m for m in outside if m["id"] > chat["summary_upto"]]
    return len(pending) >= slack
