"""Servidor web local con interfaz de chat y actualización en segundo plano."""
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import db, sync
from .llm import retrieve_for, stream_answer

log = logging.getLogger("wikichat.server")
STATIC = os.path.join(os.path.dirname(__file__), "static")


def start_updater(cfg):
    """Hilo que actualiza la copia local cada N horas. Los fallos se ignoran."""
    interval = cfg["update_interval_hours"] * 3600
    stop = threading.Event()

    def loop():
        conn = db.connect(cfg["db_path"])
        while not stop.is_set():
            # Conexión propia sin bloqueo: con WAL el chat puede leer mientras se actualiza.
            ok = sync.try_update(conn, cfg)
            log.info("Actualización %s", "completada" if ok else "omitida (sin conexión)")
            stop.wait(interval)

    threading.Thread(target=loop, daemon=True, name="updater").start()
    return stop


def make_handler(cfg, db_lock):
    conn = db.connect(cfg["db_path"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/stats":
                with db_lock:
                    self._json(db.stats(conn))
            else:
                self._json({"error": "no encontrado"}, 404)

        def do_POST(self):
            if self.path != "/api/chat":
                return self._json({"error": "no encontrado"}, 404)
            length = int(self.headers.get("Content-Length", 0))
            try:
                messages = json.loads(self.rfile.read(length))["messages"]
                assert messages and messages[-1]["role"] == "user"
            except (ValueError, KeyError, AssertionError, TypeError):
                return self._json({"error": "petición inválida"}, 400)

            with db_lock:
                sources = retrieve_for(conn, messages, cfg["top_k"])
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.end_headers()

            def send(obj):
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

            try:
                send({"type": "sources", "sources": [
                    {"title": s["title"], "section": s["section"]} for s in sources]})
                for text in stream_answer(cfg, messages, sources):
                    send({"type": "token", "text": text})
                send({"type": "done"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(cfg, auto_update=True):
    db_lock = threading.Lock()
    if auto_update:
        start_updater(cfg)
    httpd = ThreadingHTTPServer((cfg["host"], cfg["port"]), make_handler(cfg, db_lock))
    print(f"Chat disponible en http://{cfg['host']}:{cfg['port']}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
