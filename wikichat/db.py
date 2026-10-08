"""Almacenamiento local: SQLite con índice de texto completo FTS5.

Pensado para millones de artículos: los fragmentos viven en una tabla normal
indexada por página y FTS5 los indexa como "external content", así borrar o
reemplazar un artículo no requiere recorrer todo el índice.
"""
import hashlib
import os
import re
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT UNIQUE NOT NULL,
    revid INTEGER,          -- NULL si viene de un .zim (no trae número de revisión)
    snapshot_ts REAL,       -- fecha del contenido guardado (fecha del .zim o de descarga)
    size INTEGER,           -- largo del texto; los artículos largos se vectorizan primero
    hash TEXT,              -- huella del texto: al importar otro .zim, si no cambió se conserva
    gen INTEGER             -- en qué importación de .zim se vio por última vez
);
CREATE INDEX IF NOT EXISTS pages_size ON pages(size DESC);
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
-- Un vector por artículo (título + introducción), cuantizado a int8.
CREATE TABLE IF NOT EXISTS page_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page INTEGER UNIQUE NOT NULL,
    vec BLOB NOT NULL
);
-- src: "zim:<ruta dentro del .zim>" o URL; data: copia local (caché) si se descargó.
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page INTEGER NOT NULL,
    src TEXT NOT NULL,
    caption TEXT,
    mime TEXT,
    data BLOB,
    vec BLOB                -- vector del pie de foto (se calcula la primera vez que se necesita)
);
CREATE INDEX IF NOT EXISTS images_page ON images(page);
"""

CHUNK_CHARS = 1500
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
# Secciones que solo tienen listas de enlaces o citas: no aportan al chat y ocupan espacio.
SKIP_SECTIONS = {
    "referencias", "notas", "enlaces externos", "bibliografía", "véase también",
    "fuentes", "notas y referencias", "referencias y notas", "bibliografía adicional",
    "lecturas adicionales", "enlaces", "obras citadas",
}


# Columnas añadidas después de la primera versión: se agregan a bases ya existentes.
MIGRATIONS = {"pages": {"size": "INTEGER", "hash": "TEXT", "gen": "INTEGER"},
              "images": {"vec": "BLOB"}}


def connect(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    for table, columns in MIGRATIONS.items():
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone()
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, kind in columns.items():
            if exists and col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {kind}")
    conn.executescript(SCHEMA)
    return conn


def text_hash(text):
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


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


def _remove(conn, page_id, keep_images=False):
    conn.execute("DELETE FROM chunks WHERE page=?", (page_id,))
    conn.execute("DELETE FROM page_vectors WHERE page=?", (page_id,))
    if not keep_images:
        conn.execute("DELETE FROM images WHERE page=?", (page_id,))
    conn.execute("DELETE FROM pages WHERE id=?", (page_id,))


def page_id(conn, title):
    row = conn.execute("SELECT id FROM pages WHERE title=?", (title,)).fetchone()
    return row[0] if row else None


def delete_page(conn, title, commit=True):
    pid = page_id(conn, title)
    if pid is not None:
        _remove(conn, pid)
    if commit:
        conn.commit()


def upsert_page(conn, title, revid, text, snapshot_ts=None, images=None, commit=True,
                gen=None, hash=None):
    """Guarda o reemplaza un artículo.

    images: lista de (src, pie de foto). None conserva las imágenes que ya tenía
    (las actualizaciones por API no traen las del .zim y no queremos perderlas).
    """
    old = page_id(conn, title)
    if old is not None:
        if gen is None:
            row = conn.execute("SELECT gen FROM pages WHERE id=?", (old,)).fetchone()
            gen = row[0] if row else None
        _remove(conn, old, keep_images=images is None)
    cur = conn.execute(
        "INSERT INTO pages (title, revid, snapshot_ts, size, hash, gen) VALUES (?, ?, ?, ?, ?, ?)",
        (title, revid, snapshot_ts or time.time(), len(text), hash or text_hash(text), gen),
    )
    new = cur.lastrowid
    conn.executemany(
        "INSERT INTO chunks (page, title, section, text) VALUES (?, ?, ?, ?)",
        [(new, title, s, t) for s, t in chunk_text(text)],
    )
    if images is None:
        if old is not None:
            conn.execute("UPDATE images SET page=? WHERE page=?", (new, old))
    else:
        conn.executemany(
            "INSERT INTO images (page, src, caption) VALUES (?, ?, ?)",
            [(new, src, caption) for src, caption in images],
        )
    if commit:
        conn.commit()
    return new


def set_images(conn, page_id, images):
    """Reemplaza las imágenes del .zim de un artículo sin tocar su texto ni su vector.

    Con imágenes nuevas, sustituye todas. Sin ellas (p. ej. un .zim sin fotos), quita solo
    las que apuntaban al .zim anterior y conserva las descargadas de internet.
    """
    if images:
        conn.execute("DELETE FROM images WHERE page=?", (page_id,))
        conn.executemany("INSERT INTO images (page, src, caption) VALUES (?, ?, ?)",
                         [(page_id, src, caption) for src, caption in images])
    else:
        conn.execute("DELETE FROM images WHERE page=? AND src LIKE 'zim:%'", (page_id,))


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
        "zim_date": get_meta(conn, "zim_date"),
        "catchup_pending": get_meta(conn, "check_after") is not None,
        "embedded_pages": conn.execute("SELECT COUNT(*) FROM page_vectors").fetchone()[0],
        "images": conn.execute("SELECT MAX(id) FROM images").fetchone()[0] or 0,
    }
