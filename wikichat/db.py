"""Almacenamiento local: SQLite con índice de texto completo FTS5.

Pensado para millones de artículos: los fragmentos viven en una tabla normal
indexada por página y FTS5 los indexa como "external content", así borrar o
reemplazar un artículo no requiere recorrer todo el índice.
"""
import os
import re
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    revid INTEGER,          -- NULL si viene de un .zim (no trae número de revisión)
    snapshot_ts REAL        -- fecha del contenido guardado (fecha del .zim o de descarga)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    page INTEGER NOT NULL,
    title TEXT, section TEXT, text TEXT
);
CREATE INDEX IF NOT EXISTS chunks_page ON chunks(page);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, section, text,
    content = 'chunks', content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title, section, text)
    VALUES (new.id, new.title, new.section, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, section, text)
    VALUES ('delete', old.id, old.title, old.section, old.text);
END;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

CHUNK_CHARS = 1500
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
# Secciones que solo tienen listas de enlaces o citas: no aportan al chat y ocupan espacio.
SKIP_SECTIONS = {
    "referencias", "notas", "enlaces externos", "bibliografía", "véase también",
    "fuentes", "notas y referencias", "referencias y notas", "bibliografía adicional",
    "lecturas adicionales", "enlaces", "obras citadas",
}


def connect(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn, key, value, commit=True):
    if value is None:
        conn.execute("DELETE FROM meta WHERE key=?", (key,))
    else:
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))
    if commit:
        conn.commit()


def chunk_text(text):
    """Divide el texto plano por secciones (== Título ==) y luego por tamaño."""
    parts = []
    section = "Introducción"
    last = 0
    for m in _HEADING.finditer(text):
        parts.append((section, text[last:m.start()]))
        section, last = m.group(2), m.end()
    parts.append((section, text[last:]))

    chunks = []
    for section, body in parts:
        if section.strip().lower() in SKIP_SECTIONS:
            continue
        buf = ""
        for para in (p.strip() for p in body.split("\n")):
            if not para:
                continue
            if buf and len(buf) + len(para) > CHUNK_CHARS:
                chunks.append((section, buf))
                buf = ""
            buf = f"{buf}\n{para}" if buf else para
        if buf:
            chunks.append((section, buf))
    return chunks


def delete_page(conn, title, commit=True):
    row = conn.execute("SELECT id FROM pages WHERE title=?", (title,)).fetchone()
    if row:
        conn.execute("DELETE FROM chunks WHERE page=?", (row[0],))
        conn.execute("DELETE FROM pages WHERE id=?", (row[0],))
    if commit:
        conn.commit()


def upsert_page(conn, title, revid, text, snapshot_ts=None, commit=True):
    delete_page(conn, title, commit=False)
    cur = conn.execute(
        "INSERT INTO pages (title, revid, snapshot_ts) VALUES (?, ?, ?)",
        (title, revid, snapshot_ts or time.time()),
    )
    conn.executemany(
        "INSERT INTO chunks (page, title, section, text) VALUES (?, ?, ?, ?)",
        [(cur.lastrowid, title, s, t) for s, t in chunk_text(text)],
    )
    if commit:
        conn.commit()


def has_page(conn, title):
    return conn.execute("SELECT 1 FROM pages WHERE title=?", (title,)).fetchone() is not None


def page_titles(conn):
    return {r[0] for r in conn.execute("SELECT title FROM pages")}


def stats(conn):
    pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    chunks = conn.execute("SELECT MAX(id) FROM chunks").fetchone()[0] or 0
    return {
        "pages": pages,
        "chunks_approx": chunks,
        "last_sync": get_meta(conn, "last_sync"),
        "zim_source": get_meta(conn, "zim_name"),
        "catchup_pending": get_meta(conn, "check_after") is not None,
    }
