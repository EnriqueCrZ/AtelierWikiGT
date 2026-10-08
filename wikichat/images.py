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


# Pesos de la selección. La similitud entre pregunta y pie de foto va de ~0 a ~0,7.
FIRST_IMAGE_BONUS = 0.08   # la primera imagen suele ser la de la ficha: la más representativa
SOURCE_RANK_STEP = 0.03    # cuanto más arriba está el artículo entre las fuentes, más pesa
KEYWORD_BONUS = 0.05       # por cada palabra de la pregunta que aparece en el pie de foto
KEEP_WITHIN = 0.12         # se descartan las que quedan más de esto por debajo de la mejor


def _caption_text(cfg, title, caption):
    return cfg["embed_doc_template"].format(title=title, text=caption or title)


def _caption_vectors(conn, cfg, rows):
    """{id: vector} de los pies de foto; calcula y guarda los que falten."""
    from . import backends, vectors
    np = vectors.np
    dims = cfg["embed_dims"]
    out, missing = {}, []
    for image_id, title, caption, page_rank, position, blob in rows:
        if blob and len(blob) == dims:
            out[image_id] = np.frombuffer(blob, dtype=np.int8).astype(np.float32) / 127
        else:
            missing.append((image_id, title, caption))
    if missing:
        vecs = backends.embed(cfg, [_caption_text(cfg, t, c) for _, t, c in missing])
        for (image_id, _, _), v in zip(missing, vecs):
            q = vectors.quantize(v, dims)
            conn.execute("UPDATE images SET vec=? WHERE id=?", (q.tobytes(), image_id))
            out[image_id] = q.astype(np.float32) / 127
        conn.commit()
    return out


def rank(conn, cfg, page_ids, query, qvec=None, per_page=2, total=6, exclude=()):
    """Imágenes de los artículos fuente que mejor ilustran la pregunta: [{id, title, caption}].

    Compara el pie de foto de cada imagen con la pregunta (con el modelo de embeddings, si
    está; si no, por palabras en común) y da un poco de ventaja a la primera imagen de cada
    artículo y a los artículos mejor ubicados entre las fuentes.
    """
    from . import backends, vectors
    from .retrieval import keywords

    rows = []
    for page_rank, pid in enumerate(page_ids):
        for position, (image_id, title, caption, blob) in enumerate(conn.execute(
                """SELECT i.id, p.title, i.caption, i.vec FROM images i JOIN pages p ON p.id = i.page
                   WHERE i.page = ? ORDER BY i.id""", (pid,))):
            if image_id not in exclude:
                rows.append((image_id, title, caption, page_rank, position, blob))
    if not rows:
        return []

    words = set(keywords(query))
    sims = {}
    if qvec is not None and vectors.enabled(cfg):
        try:
            cap_vecs = _caption_vectors(conn, cfg, rows)
            q = vectors.np.asarray(qvec, dtype=vectors.np.float32)[: cfg["embed_dims"]]
            sims = {i: float(v @ q) for i, v in cap_vecs.items()}
        except backends.BackendUnavailable:
            sims = {}

    scored = []
    for image_id, title, caption, page_rank, position, _ in rows:
        cap_words = set(keywords(caption or ""))
        s = sims.get(image_id, 0.0)
        s += KEYWORD_BONUS * len(words & cap_words)
        s += FIRST_IMAGE_BONUS if position == 0 else 0
        s -= SOURCE_RANK_STEP * page_rank
        scored.append((s, image_id, title, caption))
    scored.sort(key=lambda x: -x[0])

    best = scored[0][0]
    out, per = [], {}
    for s, image_id, title, caption in scored:
        if s < best - KEEP_WITHIN or len(out) == total:
            break
        if per.get(title, 0) >= per_page:
            continue
        per[title] = per.get(title, 0) + 1
        out.append({"id": image_id, "title": title, "caption": caption or title})
    return out


def for_pages(conn, page_ids, per_page=2, total=6, exclude=()):
    """Imágenes para mostrar junto a una respuesta: [{id, title, caption}]."""
    out = []
    for pid in page_ids:
        rows = conn.execute(
            """SELECT i.id, p.title, i.caption FROM images i JOIN pages p ON p.id = i.page
               WHERE i.page = ? ORDER BY i.id""", (pid,)).fetchall()
        rows = [r for r in rows if r[0] not in exclude][:per_page]
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
