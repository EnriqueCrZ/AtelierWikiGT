"""Almacenamiento local: SQLite con índice de texto completo FTS5."""
import os
import re
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    pageid INTEGER PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    revid INTEGER,
    updated_at REAL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    title, section, text,
    pageid UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

CHUNK_CHARS = 1500
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)


def connect(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))
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


def upsert_page(conn, pageid, title, revid, text):
    conn.execute("DELETE FROM chunks WHERE pageid=?", (pageid,))
    conn.execute("DELETE FROM pages WHERE pageid=? OR title=?", (pageid, title))
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)", (pageid, title, revid, time.time())
    )
    conn.executemany(
        "INSERT INTO chunks (title, section, text, pageid) VALUES (?, ?, ?, ?)",
        [(title, s, t, pageid) for s, t in chunk_text(text)],
    )
    conn.commit()


def delete_page(conn, pageid):
    conn.execute("DELETE FROM chunks WHERE pageid=?", (pageid,))
    conn.execute("DELETE FROM pages WHERE pageid=?", (pageid,))
    conn.commit()


def tracked_pages(conn):
    return conn.execute("SELECT pageid, title, revid FROM pages").fetchall()


def stats(conn):
    pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"pages": pages, "chunks": chunks, "last_sync": get_meta(conn, "last_sync")}
