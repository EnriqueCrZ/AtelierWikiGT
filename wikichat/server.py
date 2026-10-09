"""Servidor web local con interfaz de chat y actualización en segundo plano."""
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import backends, chats, db, images, sync, vectors
from .llm import cited_titles, citation_warning, refused, retrieve_for, stream_answer, summarize

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
    """Hilos de fondo: precarga los modelos, actualiza la wiki cada N horas (si se pide) y
    calcula los embeddings que falten. Cualquier fallo se registra y se ignora."""
    stop = threading.Event()

    def warm_up():
        try:
            t = time.time()
            backends.preload(cfg)
            log.info("Modelos cargados en memoria (%.0f s)", time.time() - t)
        except backends.BackendUnavailable as e:
            log.warning("No se pudieron precargar los modelos: %s", e)

    changed = threading.Event()  # la actualización trajo artículos nuevos o cambiados

    def update_loop():
        """Actualiza la wiki cada N horas. Va en su propio hilo: una puesta al día puede durar
        horas y no debe frenar la vectorización (antes la vectorización esperaba a que terminara)."""
        conn = db.connect(cfg["db_path"])
        while not stop.is_set():
            ok = sync.try_update(conn, cfg)
            log.info("Actualización %s", "completada" if ok else "omitida (sin conexión)")
            changed.set()
            stop.wait(cfg["update_interval_hours"] * 3600)

    def embed_loop():
        """Vectoriza lo que falte, por tandas, refrescando el índice para que la búsqueda
        semántica mejore mientras avanza. Se pausa mientras alguien usa el chat."""
        conn = db.connect(cfg["db_path"])
        n = index.refresh(conn)
        total = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        log.info("Índice semántico: %d de %d artículos vectorizados", n, total)
        t0, done = time.time(), 0
        while not stop.is_set():
            try:
                batch = vectors.embed_pending(conn, cfg, max_pages=2000, stop=stop,
                                              pause=activity.busy if activity else None)
            except backends.BackendUnavailable as e:
                log.warning("Embeddings en pausa, el modelo no responde (se reintenta en 1 min): %s", e)
                stop.wait(60)
                continue
            index.refresh(conn)
            if batch:
                done += batch
                rate = done / max(time.time() - t0, 1e-9)
                left = total - len(index)
                log.info("Búsqueda semántica: %d de %d artículos (%.0f/s, faltan ~%.1f h)",
                         len(index), total, rate, left / max(rate, 1e-9) / 3600)
            else:
                # Al día: espera a que la actualización traiga cambios (o revisa cada 10 min).
                changed.wait(600)
                changed.clear()
                total = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

    threading.Thread(target=warm_up, daemon=True, name="precarga").start()
    if auto_update:
        threading.Thread(target=update_loop, daemon=True, name="actualizacion").start()
    if index is not None:
        threading.Thread(target=embed_loop, daemon=True, name="embeddings").start()
    return stop


def make_handler(cfg, db_lock, index=None, activity=None, store=None):
    activity = activity or Activity()
    conn = db.connect(cfg["db_path"])
    store = store or chats.ChatStore(cfg["chats_db_path"])
    summarizing = set()
    summarizing_lock = threading.Lock()

    def maybe_summarize(chat_id):
        """Resume en segundo plano los mensajes que quedaron fuera de la ventana reciente."""
        recent = cfg["history_messages"]
        chat = store.get(chat_id)
        if not cfg["summarize_history"] or not chat or not chats.needs_summary(chat, recent):
            return
        with summarizing_lock:
            if chat_id in summarizing:
                return
            summarizing.add(chat_id)

        def work():
            try:
                with activity:  # que la vectorización de fondo no compita con el resumen
                    _summarize()
            except backends.BackendUnavailable as e:
                log.warning("No se pudo resumir el chat %s: %s", chat_id, e)
            finally:
                with summarizing_lock:
                    summarizing.discard(chat_id)

        def _summarize():
            outside = chat["messages"][:-recent] if recent else chat["messages"]
            pending = [m for m in outside if m["id"] > chat["summary_upto"]]
            text = summarize(cfg, chat["summary"], pending)
            if text:
                store.set_summary(chat_id, text, pending[-1]["id"])

        threading.Thread(target=work, daemon=True, name=f"resumen-{chat_id}").start()

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
            elif self.path == "/api/chats":
                self._json(store.list())
            elif m := re.fullmatch(r"/api/chats/(\w+)", self.path):
                chat = store.get(m.group(1))
                if chat:
                    chat.pop("summary_upto")
                    self._json(chat)
                else:
                    self._json({"error": "chat no encontrado"}, 404)
            elif m := re.fullmatch(r"/api/image/(\d+)", self.path):
                with db_lock:
                    img = images.load(conn, cfg, int(m.group(1)))
                if img:
                    self._send(img[0], img[1], cache=True)
                else:
                    self._json({"error": "imagen no disponible"}, 404)
            else:
                self._json({"error": "no encontrado"}, 404)

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                return body if isinstance(body, dict) else None
            except ValueError:
                return None

        def do_POST(self):
            if self.path == "/api/chats":
                return self._json({"id": store.create()}, 201)
            if self.path != "/api/chat":
                return self._json({"error": "no encontrado"}, 404)
            body = self._body()
            with activity:
                self._chat(body)

        def do_PATCH(self):
            m = re.fullmatch(r"/api/chats/(\w+)", self.path)
            body = self._body()
            if not m or not body or not isinstance(body.get("title"), str):
                return self._json({"error": "petición inválida"}, 400)
            ok = store.rename(m.group(1), body["title"])
            self._json({"ok": ok}, 200 if ok else 404)

        def do_DELETE(self):
            m = re.fullmatch(r"/api/chats/(\w+)", self.path)
            ok = bool(m) and store.delete(m.group(1))
            self._json({"ok": ok}, 200 if ok else 404)

        def _chat(self, body):
            """Dos formas: {"chat_id"?, "message"} guarda la conversación en el servidor;
            {"messages": [...]} responde sin guardar nada (el cliente lleva el historial)."""
            chat_id, summary, chat = None, None, None
            try:
                if "message" in body:
                    question = body["message"].strip()
                    assert question
                    chat_id = body.get("chat_id")
                    if not chat_id or not store.exists(chat_id):
                        chat_id = store.create()
                    store.add_message(chat_id, "user", question)
                    chat = store.get(chat_id)
                    summary, recent = chats.context_for(chat, cfg["history_messages"])
                    everything = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
                    messages = recent + [everything[-1]]
                else:
                    everything = messages = body["messages"]
                    assert messages and messages[-1]["role"] == "user"
                    messages = messages[-(cfg["history_messages"] + 1):]
            except (KeyError, AssertionError, TypeError, AttributeError):
                return self._json({"error": "petición inválida"}, 400)

            with db_lock:
                if index is not None and time.time() - index.refreshed_at > INDEX_REFRESH_SECONDS:
                    index.refresh(conn)
                sources, query, qvec = retrieve_for(conn, cfg, everything, index, summary)
                pages = list(dict.fromkeys(s["page"] for s in sources))
                # Las que mejor ilustran la pregunta, sin repetir las que ya salieron en el chat.
                shown = {im["id"] for m in (chat["messages"] if chat_id else []) for im in m["images"]}
                pics = images.rank(conn, cfg, pages, query, qvec, exclude=shown)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.end_headers()

            def send(obj):
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

            brief = [{"title": s["title"], "section": s["section"]} for s in sources]
            answer = []
            try:
                if chat_id:
                    send({"type": "chat", "id": chat_id, "title": store.get(chat_id)["title"]})
                send({"type": "sources", "sources": brief, "images": pics, "query": query,
                      "model_loaded": backends.chat_model_loaded(cfg)})
                for text in stream_answer(cfg, messages, sources, summary):
                    answer.append(text)
                    send({"type": "token", "text": text})
                text = "".join(answer)
                warning = citation_warning(text, sources)
                if warning:
                    answer.append("\n\n" + warning)
                    send({"type": "warning", "text": warning})
                # Si no respondió con la wiki, las imágenes de las "fuentes" no vienen al caso.
                if warning or refused(text):
                    pics = []
                    send({"type": "hide_images"})
                else:
                    # Solo las fuentes e imágenes de los artículos que la respuesta cita: los
                    # demás fueron candidatos de la búsqueda, no necesariamente relevantes.
                    cited = cited_titles(text, sources)
                    if cited and any(b["title"] not in cited for b in brief):
                        brief = [b for b in brief if b["title"] in cited]
                        pics = [p for p in pics if p["title"] in cited]
                        send({"type": "cited", "titles": sorted(cited)})
                send({"type": "done"})
            except (BrokenPipeError, ConnectionResetError):
                answer.append(" [respuesta interrumpida]")
            if chat_id:
                store.add_message(chat_id, "assistant", "".join(answer), brief, pics, query)
                maybe_summarize(chat_id)

    return Handler


def _exit_on_signals():
    """Que `kill` (SIGTERM) o cerrar la terminal (SIGHUP) cierren igual que Ctrl+C, para
    detener ordenadamente el Ollama administrado en vez de dejarlo huérfano."""
    import signal

    def stop(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    # Lanzado en segundo plano desde un script, Ctrl+C llega ignorado: se restablece.
    signal.signal(signal.SIGINT, signal.default_int_handler)


def serve(cfg, auto_update=True):
    _exit_on_signals()
    from . import ollama_manager
    manager = ollama_manager.ensure_running(cfg)  # solo si setup dejó Ollama administrado
    db_lock = threading.Lock()
    index = vectors.VectorIndex(cfg["embed_dims"]) if vectors.enabled(cfg) else None
    if cfg.get("embed_model") and index is None:
        log.warning("Búsqueda semántica desactivada: instala numpy (pip install numpy)")
    activity = Activity()
    start_background(cfg, index, auto_update, activity)
    store = chats.ChatStore(cfg["chats_db_path"])
    httpd = ThreadingHTTPServer((cfg["host"], cfg["port"]),
                                make_handler(cfg, db_lock, index, activity, store))
    print(f"Chat disponible en http://{cfg['host']}:{cfg['port']}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if manager:
            manager.stop()
