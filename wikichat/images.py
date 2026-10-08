"""Imágenes de los artículos: se leen del .zim (versión "maxi") o, para artículos
actualizados por la API, de Wikimedia la primera vez que se muestran (y quedan en caché)."""
import logging
import threading
import urllib.request

from . import db

log = logging.getLogger("wikichat.images")

MAX_CACHE_BYTES = 2 * 1024 * 1024
_archives = {}
_lock = threading.Lock()


def _archive(path):
    with _lock:
        if path not in _archives:
            from libzim.reader import Archive
            _archives[path] = Archive(path)
        return _archives[path]


def for_pages(conn, page_ids, per_page=2, total=6):
    """Imágenes para mostrar junto a una respuesta: [{id, title, caption}]."""
    out = []
    for pid in page_ids:
        rows = conn.execute(
            """SELECT i.id, p.title, i.caption FROM images i JOIN pages p ON p.id = i.page
               WHERE i.page = ? ORDER BY i.id LIMIT ?""", (pid, per_page)).fetchall()
        out += [{"id": r[0], "title": r[1], "caption": r[2] or r[1]} for r in rows]
        if len(out) >= total:
            break
    return out[:total]


def load(conn, cfg, image_id):
    """(bytes, tipo MIME) de una imagen, o None si no está disponible sin conexión."""
    row = conn.execute("SELECT src, mime, data FROM images WHERE id=?", (image_id,)).fetchone()
    if not row:
        return None
    src, mime, data = row
    if data:
        return data, mime or "image/jpeg"
    try:
        if src.startswith("zim:"):
            path = cfg.get("zim_path") or db.get_meta(conn, "zim_path")
            if not path:
                return None
            item = _archive(path).get_entry_by_path(src[4:]).get_item()
            return bytes(item.content), item.mimetype
        if src.startswith(("http://", "https://")):
            req = urllib.request.Request(src, headers={"User-Agent": cfg["user_agent"]})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read(MAX_CACHE_BYTES + 1)
                mime = resp.headers.get_content_type()
            if len(data) <= MAX_CACHE_BYTES:
                conn.execute("UPDATE images SET data=?, mime=? WHERE id=?", (data, mime, image_id))
                conn.commit()
            return data, mime
    except Exception as e:  # sin red, .zim movido, etc.: simplemente no se muestra
        log.debug("imagen %s no disponible: %s", image_id, e)
    return None
