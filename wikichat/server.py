"""Servidor web local con interfaz de chat y actualización en segundo plano."""
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import backends, db, images, sync, vectors
from .llm import retrieve_for, stream_answer

log = logging.getLogger("wikichat.server")
STATIC = os.path.join(os.path.dirname(__file__), "static")
INDEX_REFRESH_SECONDS = 120


class Activity:
    """Cuenta las preguntas en curso; el trabajo de fondo espera mientras haya alguna."""

    def __init__(self):
        self._n = 0
        self._lock = threading.Lock()

    def __enter__(self):
        with self._lock:
            self._n += 1

    def __exit__(self, *exc):
        with self._lock:
            self._n -= 1

    def busy(self):
        return self._n > 0


def start_background(cfg, index, auto_update=True, activity=None):
    """Hilo que carga el índice semántico y, si se pide, actualiza la wiki cada N horas y
    calcula los embeddings que falten. Cualquier fallo se registra y se ignora."""
    stop = threading.Event()

    def loop():
        conn = db.connect(cfg["db_path"])
        if index is not None:
            n = index.refresh(conn)
            log.info("Índice semántico: %d artículos cargados", n)
        while not stop.is_set():
            if auto_update:
                ok = sync.try_update(conn, cfg)
                log.info("Actualización %s", "completada" if ok else "omitida (sin conexión)")
            if index is not None:
                try:
                    # Por tandas, refrescando el índice para que la búsqueda mejore mientras avanza.
                    while not stop.is_set() and vectors.embed_pending(
                            conn, cfg, max_pages=2000, stop=stop,
                            pause=activity.busy if activity else None):
                        index.refresh(conn)
                except backends.BackendUnavailable as e:
                    log.warning("Embeddings pendientes, el modelo no responde: %s", e)
                index.refresh(conn)
            if not auto_update:
                break
            stop.wait(cfg["update_interval_hours"] * 3600)

    threading.Thread(target=loop, daemon=True, name="background").start()
    return stop


def make_handler(cfg, db_lock, index=None, activity=None):
    activity = activity or Activity()
    conn = db.connect(cfg["db_path"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _send(self, body, ctype, status=200, cache=False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=200):
            self._send(json.dumps(obj, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8", status)

        def do_GET(self):
            if self.path == "/":
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    self._send(f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/stats":
                with db_lock:
                    stats = db.stats(conn)
                stats["semantic_index"] = len(index) if index is not None else None
                self._json(stats)
            elif m := re.fullmatch(r"/api/image/(\d+)", self.path):
                with db_lock:
                    img = images.load(conn, cfg, int(m.group(1)))
                if img:
                    self._send(img[0], img[1], cache=True)
                else:
                    self._json({"error": "imagen no disponible"}, 404)
            else:
                self._json({"error": "no encontrado"}, 404)

        def do_POST(self):
            if self.path != "/api/chat":
                return self._json({"error": "no encontrado"}, 404)
            with activity:
                self._chat()

        def _chat(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                messages = json.loads(self.rfile.read(length))["messages"]
                assert messages and messages[-1]["role"] == "user"
            except (ValueError, KeyError, AssertionError, TypeError):
                return self._json({"error": "petición inválida"}, 400)

            with db_lock:
                if index is not None and time.time() - index.refreshed_at > INDEX_REFRESH_SECONDS:
                    index.refresh(conn)
                sources = retrieve_for(conn, cfg, messages, index)
                pages = list(dict.fromkeys(s["page"] for s in sources))[:3]
                pics = images.for_pages(conn, pages)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.end_headers()

            def send(obj):
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

            try:
                send({"type": "sources",
                      "sources": [{"title": s["title"], "section": s["section"]} for s in sources],
                      "images": pics})
                for text in stream_answer(cfg, messages, sources):
                    send({"type": "token", "text": text})
                send({"type": "done"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(cfg, auto_update=True):
    db_lock = threading.Lock()
    index = vectors.VectorIndex(cfg["embed_dims"]) if vectors.enabled(cfg) else None
    if cfg.get("embed_model") and index is None:
        log.warning("Búsqueda semántica desactivada: instala numpy (pip install numpy)")
    activity = Activity()
    start_background(cfg, index, auto_update, activity)
    httpd = ThreadingHTTPServer((cfg["host"], cfg["port"]),
                                make_handler(cfg, db_lock, index, activity))
    print(f"Chat disponible en http://{cfg['host']}:{cfg['port']}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
