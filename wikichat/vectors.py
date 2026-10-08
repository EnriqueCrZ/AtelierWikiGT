"""Búsqueda semántica: un vector por artículo (título + introducción).

Los vectores se recortan a `embed_dims` dimensiones (modelos Matryoshka como
embeddinggemma lo permiten sin perder mucha calidad) y se guardan cuantizados a int8:
con 384 dimensiones, 2 millones de artículos ocupan ~0,7 GB en disco y en RAM, y una
búsqueda recorre la matriz completa en ~1 s en una CPU de 4 núcleos.
"""
import logging
import threading
import time

from . import backends

try:
    import numpy as np
except ImportError:  # la búsqueda semántica es opcional
    np = None

log = logging.getLogger("wikichat.vectors")

BLOCK = 65536


def enabled(cfg):
    return bool(cfg.get("embed_model")) and np is not None


def quantize(vec, dims):
    v = np.asarray(vec, dtype=np.float32)[:dims]
    v /= np.linalg.norm(v) or 1.0
    return np.clip(np.round(v * 127), -127, 127).astype(np.int8)


def embed_query(cfg, text):
    v = backends.embed(cfg, [cfg["embed_query_prefix"] + text])[0]
    v = np.asarray(v, dtype=np.float32)[: cfg["embed_dims"]]
    return v / (np.linalg.norm(v) or 1.0)


def embed_pending(conn, cfg, batch=32, max_pages=None, stop=None, pause=None):
    """Calcula los vectores de los artículos que no tienen, de los más extensos (suelen
    ser los más importantes) a los más cortos. Reanudable: se puede cortar en cualquier
    momento. pause(): si devuelve True se espera (p. ej. mientras alguien usa el chat, para
    no competir por la CPU/GPU). Devuelve cuántos artículos procesó."""
    if not enabled(cfg):
        return 0
    done, t0 = 0, time.time()
    chars = cfg["embed_chars"]
    max_size = 1 << 62  # cursor: evita revisar una y otra vez los ya vectorizados
    while max_pages is None or done < max_pages:
        while pause is not None and pause() and not (stop is not None and stop.is_set()):
            time.sleep(0.5)
        if stop is not None and stop.is_set():
            break
        rows = conn.execute(
            """SELECT p.id, p.title,
                      (SELECT text FROM chunks WHERE page = p.id ORDER BY id LIMIT 1), p.size
               FROM pages p LEFT JOIN page_vectors v ON v.page = p.id
               WHERE v.page IS NULL AND p.size <= ? ORDER BY p.size DESC LIMIT ?""",
            (max_size, batch),
        ).fetchall()
        if not rows:
            break
        texts = [
            cfg["embed_doc_template"].format(title=title, text=(text or title)[:chars])
            for _, title, text, _ in rows
        ]
        vecs = backends.embed(cfg, texts)
        conn.executemany(
            "INSERT OR REPLACE INTO page_vectors (page, vec) VALUES (?, ?)",
            [(pid, quantize(v, cfg["embed_dims"]).tobytes()) for (pid, _, _, _), v in zip(rows, vecs)],
        )
        conn.commit()
        max_size = rows[-1][3]
        done += len(rows)
        if done % (batch * 30) < batch:
            log.info("embeddings: %d artículos (%.1f/s)", done, done / max(time.time() - t0, 1e-9))
    return done


class VectorIndex:
    """Matriz en memoria con todos los vectores; se refresca con los nuevos."""

    def __init__(self, dims):
        self.dims = dims
        self.lock = threading.Lock()
        self.vec_ids = np.empty(0, dtype=np.int64)
        self.pages = np.empty(0, dtype=np.int64)
        self.matrix = np.empty((0, dims), dtype=np.int8)
        self.last_id = 0
        self.refreshed_at = 0.0

    def __len__(self):
        return len(self.pages)

    def refresh(self, conn):
        with self.lock:
            ids, pages, blobs = [], [], []
            for vid, page, blob in conn.execute(
                "SELECT id, page, vec FROM page_vectors WHERE id > ? ORDER BY id", (self.last_id,)
            ):
                if len(blob) != self.dims:
                    continue  # vector de otra configuración de dimensiones
                ids.append(vid)
                pages.append(page)
                blobs.append(blob)
            self.refreshed_at = time.time()
            if not ids:
                return 0
            new = np.frombuffer(b"".join(blobs), dtype=np.int8).reshape(-1, self.dims)
            self.vec_ids = np.concatenate([self.vec_ids, np.array(ids, dtype=np.int64)])
            self.pages = np.concatenate([self.pages, np.array(pages, dtype=np.int64)])
            self.matrix = np.concatenate([self.matrix, new])
            self.last_id = ids[-1]
            return len(ids)

    def search(self, conn, qvec, n=10):
        """[(page_id, similitud)] de los artículos más parecidos a la consulta."""
        vec_ids, pages, matrix = self.vec_ids, self.pages, self.matrix
        if not len(pages):
            return []
        q = np.asarray(qvec, dtype=np.float32)
        scores = np.empty(len(pages), dtype=np.float32)
        for s in range(0, len(pages), BLOCK):
            scores[s:s + BLOCK] = matrix[s:s + BLOCK].astype(np.float32) @ q
        want = min(len(pages), n * 3)
        top = np.argpartition(-scores, want - 1)[:want]
        top = top[np.argsort(-scores[top])]
        # Descarta vectores de artículos que se borraron o reemplazaron desde que se cargaron.
        cand = [int(vec_ids[i]) for i in top]
        alive = {r[0] for r in conn.execute(
            f"SELECT id FROM page_vectors WHERE id IN ({','.join('?' * len(cand))})", cand)}
        out = [(int(pages[i]), float(scores[i]) / 127) for i in top if int(vec_ids[i]) in alive]
        return out[:n]
