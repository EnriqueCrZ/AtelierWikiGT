import json
import os
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from wikichat import backends, chats, db, images, sync, vectors
from wikichat.llm import retrieve
from wikichat.config import DEFAULTS
from wikichat.retrieval import search
from wikichat.server import make_handler
from wikichat.wiki_api import WikiUnavailable
from wikichat.zim_import import html_to_text


class FakeWiki:
    """Simula la API de MediaWiki en memoria."""

    def __init__(self):
        self.pages = {
            "Antigua Guatemala": (10, "La ciudad colonial.\n== Historia ==\nFundada en 1543 tras el traslado."),
            "Lago de Atitlán": (20, "Lago volcánico rodeado de pueblos mayas."),
        }
        self.redirects = {}
        self.thumbs = {}
        self.rev_ts = {}
        self.recent = []
        self.offline = False
        self.fetched = []

    def _check(self):
        if self.offline:
            raise WikiUnavailable("sin red")

    def category_titles(self, category, depth):
        self._check()
        return set(self.pages)

    def page_text(self, title):
        self._check()
        self.fetched.append(title)
        title = self.redirects.get(title, title)
        if title not in self.pages:
            return None
        revid, text = self.pages[title]
        return title, revid, text, self.thumbs.get(title)

    def latest_revisions(self, titles):
        self._check()
        out = {}
        for t in titles:
            if t in self.pages and t not in self.redirects:
                out[t] = (self.pages[t][0], self.rev_ts.get(t, "2020-01-01T00:00:00Z"))
            else:
                out[t] = None
        return out

    def recent_changes(self, since_iso, skip_bots=True):
        self._check()
        return [e for e in self.recent if e[0] >= since_iso]


class WikiChatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = dict(DEFAULTS, db_path=os.path.join(self.tmp, "wiki.db"),
                        chats_db_path=os.path.join(self.tmp, "chats.db"),
                        seed_categories=["Guatemala"], llm_url="http://127.0.0.1:9")
        self.conn = db.connect(self.cfg["db_path"])
        self.wiki = FakeWiki()

    def test_chunk_by_sections_skips_references(self):
        chunks = db.chunk_text("Intro.\n== Historia ==\nTexto histórico.\n== Referencias ==\nCita 1.")
        self.assertEqual(chunks, [("Introducción", "Intro."), ("Historia", "Texto histórico.")])

    def test_initial_sync_and_search_ignores_accents(self):
        self.assertTrue(sync.try_update(self.conn, self.cfg, self.wiki))
        self.assertEqual(db.stats(self.conn)["pages"], 2)
        results = search(self.conn, "¿Cuándo se fundó la antigua guatemala?")
        self.assertEqual(results[0]["title"], "Antigua Guatemala")
        self.assertEqual(search(self.conn, "atitlan")[0]["title"], "Lago de Atitlán")

    def test_incremental_update_via_recent_changes(self):
        sync.update(self.conn, self.cfg, self.wiki)
        self.wiki.pages["Lago de Atitlán"] = (21, "Lago endorreico en Sololá.")
        now = sync.iso(time.time())
        self.wiki.recent = [(now, "edit", "Lago de Atitlán", None),
                            (now, "edit", "Artículo que no seguimos", None)]
        self.wiki.fetched.clear()
        sync.update(self.conn, self.cfg, self.wiki)
        self.assertEqual(search(self.conn, "Solola")[0]["title"], "Lago de Atitlán")
        self.assertEqual(search(self.conn, "mayas"), [])
        self.assertNotIn("Artículo que no seguimos", self.wiki.fetched)

    def test_full_wiki_mode_follows_all_changes_moves_and_deletes(self):
        cfg = dict(self.cfg, track_all_changes=True, seed_categories=[])
        db.upsert_page(self.conn, "Antigua Guatemala", None, "Texto viejo del zim", snapshot_ts=0)
        db.upsert_page(self.conn, "Lago de Atitlán", None, "Lago volcánico", snapshot_ts=0)
        db.set_meta(self.conn, "rc_cursor", sync.iso(time.time() - 86400))
        self.wiki.pages["Volcán de Fuego"] = (30, "Volcán activo cerca de Antigua.")
        self.wiki.pages["Lago Atitlán"] = self.wiki.pages.pop("Lago de Atitlán")
        now = sync.iso(time.time())
        self.wiki.recent = [
            (now, "edit", "Volcán de Fuego", None),
            (now, "move", "Lago de Atitlán", "Lago Atitlán"),
            (now, "delete", "Antigua Guatemala", None),
        ]
        sync.update(self.conn, cfg, self.wiki)
        self.assertEqual(db.page_titles(self.conn), {"Volcán de Fuego", "Lago Atitlán"})

    def test_catch_up_after_long_offline_detects_edits_deletions_redirects(self):
        cfg = dict(self.cfg, track_all_changes=True, seed_categories=[])
        snapshot = time.time() - 60 * 86400  # .zim de hace 60 días
        for t in ("Antigua Guatemala", "Lago de Atitlán", "Borrado", "Ahora redirección"):
            db.upsert_page(self.conn, t, None, f"Texto de {t} en el zim", snapshot_ts=snapshot)
        db.set_meta(self.conn, "rc_cursor", sync.iso(snapshot))
        self.wiki.rev_ts["Antigua Guatemala"] = sync.iso(time.time() - 86400)  # editado después
        self.wiki.pages["Ahora redirección"] = (5, "x")
        self.wiki.redirects["Ahora redirección"] = "Lago de Atitlán"
        sync.update(self.conn, cfg, self.wiki)
        self.assertEqual(db.page_titles(self.conn), {"Antigua Guatemala", "Lago de Atitlán"})
        self.assertEqual(self.wiki.fetched, ["Antigua Guatemala"])  # el otro no cambió
        self.assertIsNone(db.get_meta(self.conn, "check_after"))
        self.assertGreater(sync.parse_iso(db.get_meta(self.conn, "rc_cursor")), time.time() - 7200)

    def test_catch_up_resumes_after_interruption(self):
        cfg = dict(self.cfg, track_all_changes=True, seed_categories=[])
        for t in ("A", "B", "C"):
            db.upsert_page(self.conn, t, None, f"Texto {t}", snapshot_ts=time.time())
            self.wiki.pages[t] = (1, f"Texto {t}")
        old = sync.CHECK_BATCH
        sync.CHECK_BATCH = 1
        calls = []
        real = self.wiki.latest_revisions

        def flaky(titles):
            titles = list(titles)
            calls.append(titles)
            if len(calls) == 2:
                raise WikiUnavailable("se cayó la red")
            return real(titles)

        self.wiki.latest_revisions = flaky
        try:
            self.assertFalse(sync.try_update(self.conn, cfg, self.wiki))
            self.assertEqual(db.get_meta(self.conn, "check_after"), "A")
            self.assertTrue(sync.try_update(self.conn, cfg, self.wiki))
        finally:
            sync.CHECK_BATCH = old
        self.assertEqual(calls, [["A"], ["B"], ["B"], ["C"]])

    def test_offline_update_keeps_local_copy(self):
        sync.update(self.conn, self.cfg, self.wiki)
        self.wiki.offline = True
        self.assertFalse(sync.try_update(self.conn, self.cfg, self.wiki))
        self.assertEqual(db.stats(self.conn)["pages"], 2)

    def test_html_to_text_like_kiwix_wikipedia(self):
        html = """<html><body><header><h1>Título</h1></header>
        <div class="mw-content-ltr mw-parser-output">
          <p>El <b>lago</b> mide 18 km<sup>2</sup><sup class="mw-ref reference"><a>[1]</a></sup>.</p>
          <h2 id="Historia">Historia</h2>
          <p>Fórmula <math alttext="{\\displaystyle x^{2}}"><mi>x</mi></math> aquí.</p>
          <table class="wikitable"><tr><th>Año</th><td>1543</td></tr></table>
          <div class="navbox">Navegación</div>
          <ol class="mw-references references"><li>Cita</li></ol>
        </div><div class="zim-footer">Pie</div></body></html>"""
        text = html_to_text(html)
        self.assertIn("El lago mide 18 km2.", text)
        self.assertIn("== Historia ==", text)
        self.assertIn("{\\displaystyle x^{2}}", text)
        self.assertIn("Año | 1543", text)
        for unwanted in ("[1]", "Navegación", "Cita", "Pie", "Título"):
            self.assertNotIn(unwanted, text)

    def test_chat_endpoint_falls_back_without_llm(self):
        sync.update(self.conn, self.cfg, self.wiki)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.cfg, threading.Lock()))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_port}/api/chat",
                data=json.dumps({"messages": [{"role": "user", "content": "lago volcánico"}]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                events = [json.loads(line) for line in resp]
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(events[0]["sources"][0]["title"], "Lago de Atitlán")
        text = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertIn("No hay modelo de chat disponible", text)
        self.assertEqual(events[-1]["type"], "done")


# --- Búsqueda semántica, imágenes y modelos -------------------------------------------

TOPICS = ["volcan", "lago", "ciudad", "ave", "maiz", "cafe", "mar", "selva"]
SYNONYMS = {"montaña de fuego": "volcan", "pájaro": "ave", "grano amarillo": "maiz"}


def fake_embed(cfg, texts):
    """Embeddings de juguete: un eje por tema, con sinónimos que BM25 no conoce."""
    out = []
    for t in texts:
        t = t.lower()
        for syn, topic in SYNONYMS.items():
            t = t.replace(syn, topic)
        out.append([1.0 + t.count(topic) * 5 for topic in TOPICS] + [0.0] * 8)
    return out


class SemanticAndImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = dict(DEFAULTS, db_path=os.path.join(self.tmp, "wiki.db"),
                        llm_url="http://127.0.0.1:9", embed_dims=8, top_k=2)
        self.conn = db.connect(self.cfg["db_path"])
        db.upsert_page(self.conn, "Volcán de Fuego", None, "El volcan de Fuego es un volcan activo. " * 3)
        db.upsert_page(self.conn, "Quetzal", None, "El quetzal es un ave de la selva.")
        db.upsert_page(self.conn, "Maíz", None, "El maiz es la base de la comida y un cultivo antiguo de Mesoamérica. " * 5)

    def test_embed_pending_longest_first_and_resumable(self):
        with mock.patch.object(backends, "embed", side_effect=fake_embed):
            self.assertEqual(vectors.embed_pending(self.conn, self.cfg, batch=1, max_pages=1), 1)
            first = self.conn.execute("SELECT p.title FROM page_vectors v JOIN pages p ON p.id=v.page").fetchall()
            self.assertEqual(first, [("Maíz",)])
            self.assertEqual(vectors.embed_pending(self.conn, self.cfg), 2)
            self.assertEqual(vectors.embed_pending(self.conn, self.cfg), 0)

    def test_hybrid_finds_synonyms_keyword_search_misses(self):
        with mock.patch.object(backends, "embed", side_effect=fake_embed):
            vectors.embed_pending(self.conn, self.cfg)
            index = vectors.VectorIndex(8)
            index.refresh(self.conn)
            query = "¿qué pájaro es símbolo nacional?"
            self.assertNotIn("Quetzal", [s["title"] for s in retrieve(self.conn, self.cfg, query)])
            self.assertEqual(retrieve(self.conn, self.cfg, query, index)[0]["title"], "Quetzal")

    def test_replaced_article_vector_is_ignored_until_reembedded(self):
        with mock.patch.object(backends, "embed", side_effect=fake_embed):
            vectors.embed_pending(self.conn, self.cfg)
            index = vectors.VectorIndex(8)
            index.refresh(self.conn)
            db.upsert_page(self.conn, "Quetzal", 2, "Ahora habla de otra cosa.")
            pages = {p for p, _ in index.search(self.conn, vectors.embed_query(self.cfg, "ave"), 5)}
            self.assertNotIn(db.page_id(self.conn, "Quetzal"), pages)

    def test_exact_title_intro_comes_first(self):
        db.upsert_page(self.conn, "Maíz transgénico", None, "El maíz maíz maíz maíz modificado. " * 9)
        self.assertEqual(retrieve(self.conn, self.cfg, "¿Qué es el maíz?")[0]["title"], "Maíz")

    def test_without_embedding_model_falls_back_to_keywords(self):
        index = vectors.VectorIndex(8)
        index.matrix = vectors.np.ones((1, 8), dtype=vectors.np.int8)
        index.pages = index.vec_ids = vectors.np.array([1])
        self.assertEqual(retrieve(self.conn, self.cfg, "volcan activo", index)[0]["title"], "Volcán de Fuego")

    def test_api_update_keeps_zim_images_and_adds_lead_image_when_none(self):
        db.upsert_page(self.conn, "Tikal", None, "Templos mayas.", images=[("zim:_assets_/tikal.webp", "Templo I")])
        wiki = FakeWiki()
        wiki.pages = {"Tikal": (5, "Templos mayas en Petén."), "Quetzal": (6, "Ave.")}
        wiki.thumbs = {"Tikal": "https://upload.example/t.jpg", "Quetzal": "https://upload.example/q.jpg"}
        sync.fetch(self.conn, wiki, "Tikal")
        sync.fetch(self.conn, wiki, "Quetzal")
        pics = images.for_pages(self.conn, [db.page_id(self.conn, "Tikal"), db.page_id(self.conn, "Quetzal")])
        self.assertEqual([p["caption"] for p in pics], ["Templo I", "Quetzal"])

    def test_remote_image_is_cached_and_offline_returns_none(self):
        pid = db.upsert_page(self.conn, "Lago", None, "Lago.", images=[("http://127.0.0.1:9/x.jpg", "x")])
        image_id = images.for_pages(self.conn, [pid])[0]["id"]
        self.assertIsNone(images.load(self.conn, self.cfg, image_id))  # sin red: no rompe
        self.conn.execute("UPDATE images SET data=?, mime='image/png' WHERE id=?", (b"PNG", image_id))
        self.assertEqual(images.load(self.conn, self.cfg, image_id), (b"PNG", "image/png"))


class FakeModelServer(BaseHTTPRequestHandler):
    """Responde como Ollama (/api/...) o como un servidor compatible con OpenAI (/v1/...)."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.end_headers()
        if self.path == "/v1/chat/completions":
            for word in ("Hola", " mundo"):
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": word}}]}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        elif self.path == "/v1/embeddings":
            data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(body["input"]))]
            self.wfile.write(json.dumps({"data": data[::-1]}).encode())
        elif self.path == "/api/chat":
            assert body["think"] is False
            for word in ("Hola", " mundo"):
                self.wfile.write(json.dumps({"message": {"content": word}, "done": False}).encode() + b"\n")
            self.wfile.write(json.dumps({"done": True}).encode() + b"\n")
        elif self.path == "/api/embed":
            self.wfile.write(json.dumps({"embeddings": [[float(i), 1.0] for i in range(len(body["input"]))]}).encode())


class BackendsTest(unittest.TestCase):
    def test_ollama_and_openai_protocols(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelServer)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_port}"
        try:
            for backend, url in (("ollama", base), ("openai", base + "/v1")):
                cfg = dict(DEFAULTS, llm_backend=backend, llm_url=url)
                self.assertEqual("".join(backends.chat_stream(cfg, [])), "Hola mundo")
                self.assertEqual(backends.embed(cfg, ["a", "b"]), [[0.0, 1.0], [1.0, 1.0]])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_unreachable_backend_raises_clear_error(self):
        cfg = dict(DEFAULTS, llm_url="http://127.0.0.1:9")
        with self.assertRaises(backends.BackendUnavailable):
            backends.embed(cfg, ["x"])


class ChatStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = chats.ChatStore(os.path.join(tempfile.mkdtemp(), "chats.db"))

    def test_create_title_rename_delete(self):
        cid = self.store.create()
        self.store.add_message(cid, "user", "¿Cuándo se fundó Antigua Guatemala y por qué la trasladaron a otro valle?")
        self.assertTrue(self.store.get(cid)["title"].startswith("¿Cuándo se fundó Antigua"))
        self.assertTrue(self.store.get(cid)["title"].endswith("…"))
        self.store.add_message(cid, "user", "otra pregunta")  # no cambia el título
        self.assertTrue(self.store.rename(cid, "  Antigua  "))
        self.store.add_message(cid, "assistant", "x", [{"title": "Antigua", "section": "Historia"}], [{"id": 1}])
        chat = self.store.get(cid)
        self.assertEqual(chat["title"], "Antigua")
        self.assertEqual(chat["messages"][-1]["sources"][0]["title"], "Antigua")
        self.assertEqual([c["id"] for c in self.store.list()], [cid])
        self.assertTrue(self.store.delete(cid))
        self.assertIsNone(self.store.get(cid))
        self.assertFalse(self.store.delete(cid))

    def test_context_window_and_summary_trigger(self):
        cid = self.store.create()
        for i in range(5):
            self.store.add_message(cid, "user", f"pregunta {i}")
            self.store.add_message(cid, "assistant", f"respuesta {i}")
        self.store.add_message(cid, "user", "pregunta actual")
        chat = self.store.get(cid)
        summary, recent = chats.context_for(chat, 4)
        self.assertEqual([m["content"] for m in recent], ["pregunta 3", "respuesta 3", "pregunta 4", "respuesta 4"])
        self.assertIsNone(summary)
        self.assertTrue(chats.needs_summary(chat, 4))
        upto = chat["messages"][-5]["id"]  # todo lo que quedó fuera de la ventana
        self.store.set_summary(cid, "Hablaron de preguntas 0 a 2.", upto)
        chat = self.store.get(cid)
        self.assertEqual(chats.context_for(chat, 4)[0], "Hablaron de preguntas 0 a 2.")
        self.assertFalse(chats.needs_summary(chat, 4))


class ChatServerTest(unittest.TestCase):
    """Flujo completo del chat guardado contra un servidor de modelos simulado."""

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.models = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelServer)
        threading.Thread(target=self.models.serve_forever, daemon=True).start()
        self.cfg = dict(DEFAULTS, db_path=os.path.join(tmp, "wiki.db"),
                        chats_db_path=os.path.join(tmp, "chats.db"), embed_model="",
                        llm_url=f"http://127.0.0.1:{self.models.server_port}",
                        history_messages=2)
        conn = db.connect(self.cfg["db_path"])
        db.upsert_page(conn, "Tikal", None, "Tikal es una ciudad maya en Petén.")
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.cfg, threading.Lock()))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self):
        for s in (self.httpd, self.models):
            s.shutdown()
            s.server_close()

    def call(self, method, path, body=None):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, None
        if path == "/api/chat":
            return 200, [json.loads(line) for line in raw.splitlines()]
        return 200, json.loads(raw)

    def ask(self, message, chat_id=None):
        return self.call("POST", "/api/chat", {"chat_id": chat_id, "message": message})[1]

    def test_conversation_is_saved_and_reopened(self):
        events = self.ask("¿Qué es Tikal?")
        chat_id = events[0]["id"]
        self.assertEqual(events[0]["title"], "¿Qué es Tikal?")
        self.assertEqual(events[1]["sources"][0]["title"], "Tikal")
        self.ask("¿Dónde está?", chat_id)
        _, chat = self.call("GET", f"/api/chats/{chat_id}")
        self.assertEqual([m["role"] for m in chat["messages"]], ["user", "assistant"] * 2)
        self.assertEqual(chat["messages"][1]["content"], "Hola mundo")
        self.assertEqual(chat["messages"][1]["sources"][0]["title"], "Tikal")
        _, listing = self.call("GET", "/api/chats")
        self.assertEqual(listing[0]["id"], chat_id)
        self.assertEqual(self.call("PATCH", f"/api/chats/{chat_id}", {"title": "Mayas"})[1], {"ok": True})
        self.assertEqual(self.call("GET", f"/api/chats/{chat_id}")[1]["title"], "Mayas")
        self.assertEqual(self.call("DELETE", f"/api/chats/{chat_id}")[1], {"ok": True})
        self.assertEqual(self.call("GET", f"/api/chats/{chat_id}")[0], 404)

    def test_images_are_not_repeated_within_a_chat(self):
        conn = db.connect(self.cfg["db_path"])
        db.upsert_page(conn, "Tikal", None, "Tikal es una ciudad maya en Petén.",
                       images=[(f"zim:{n}.webp", f"foto {n}") for n in range(3)])
        first = self.ask("¿Qué es Tikal?")
        chat_id = first[0]["id"]
        self.assertEqual([i["caption"] for i in first[1]["images"]], ["foto 0", "foto 1"])
        self.assertEqual([i["caption"] for i in self.ask("Tikal maya", chat_id)[1]["images"]], ["foto 2"])

    def test_old_messages_get_summarized_in_background(self):
        chat_id = self.ask("¿Qué es Tikal?")[0]["id"]
        for q in ("¿Dónde está?", "¿Quién la construyó?", "¿Cuándo?"):
            self.ask(q, chat_id)
        for _ in range(50):
            if self.call("GET", f"/api/chats/{chat_id}")[1]["summary"]:
                break
            time.sleep(0.1)
        self.assertEqual(self.call("GET", f"/api/chats/{chat_id}")[1]["summary"], "Hola mundo")

    def test_followup_rewrite_uses_model_when_enabled(self):
        chat_id = self.ask("¿Qué es Tikal?")[0]["id"]
        self.assertEqual(self.ask("¿y dónde?", chat_id)[1]["query"], "¿Qué es Tikal? ¿y dónde?")
        self.cfg["rewrite_followups"] = True
        self.assertEqual(self.ask("¿y dónde?", chat_id)[1]["query"], "Hola mundo")

    def test_invalid_requests(self):
        self.assertEqual(self.call("POST", "/api/chat", {"message": "  "})[0], 400)
        self.assertEqual(self.call("PATCH", "/api/chats/nope", {"title": "x"})[0], 404)
        self.assertEqual(self.call("DELETE", "/api/chats/nope")[0], 404)


def make_zim(path, date, pages, assets=()):
    """Crea un .zim de prueba. pages: {título: (texto, [ruta de imagen])}."""
    from libzim.writer import Creator, Hint, Item, StringProvider

    class Page(Item):
        def __init__(self, path, title, content, mime="text/html", front=True):
            super().__init__()
            self.p, self.t, self.c, self.m, self.f = path, title, content, mime, front
        def get_path(self): return self.p
        def get_title(self): return self.t
        def get_mimetype(self): return self.m
        def get_contentprovider(self): return StringProvider(self.c)
        def get_hints(self): return {Hint.FRONT_ARTICLE: self.f}

    with Creator(path).config_indexing(False, "spa") as c:
        c.set_mainpath(next(iter(pages)))
        c.add_metadata("Date", date)
        for title, (text, imgs) in pages.items():
            figs = "".join(f"<figure><img class='mw-file-element' width='200' height='150' src='./{src}'>"
                           f"<figcaption>{title}</figcaption></figure>" for src in imgs)
            c.add_item(Page(title, title, f"<div class='mw-parser-output'><p>{text}</p>{figs}</div>"))
        for src in assets:
            c.add_item(Page(src, "", "IMG", "image/webp", False))


class ZimUpdateTest(unittest.TestCase):
    """Importar un .zim nuevo sobre una copia existente solo reprocesa lo que cambió."""

    def setUp(self):
        try:
            import libzim  # noqa: F401
        except ImportError:
            self.skipTest("libzim no instalado")
        self.tmp = tempfile.mkdtemp()
        self.conn = db.connect(os.path.join(self.tmp, "wiki.db"))
        self.cfg = dict(DEFAULTS, db_path=os.path.join(self.tmp, "wiki.db"), embed_dims=8)

    def test_second_zim_only_reprocesses_changes(self):
        from wikichat.zim_import import import_zim
        long = lambda s: (s + " ") * 20
        v1 = os.path.join(self.tmp, "wiki_2026-01.zim")
        make_zim(v1, "2026-01-01", {
            "Igual": (long("Texto que no cambia"), ["_assets_/v1/igual.webp"]),
            "Cambia": (long("Versión vieja"), []),
            "Desaparece": (long("Contenido eliminado posteriormente"), []),
            "Editado por internet": (long("Texto del zim viejo"), []),
        }, assets=["_assets_/v1/igual.webp"])
        import_zim(self.conn, v1, workers=1)
        with mock.patch.object(backends, "embed", side_effect=fake_embed):
            self.assertEqual(vectors.embed_pending(self.conn, self.cfg), 4)
        ids = {t: db.page_id(self.conn, t) for t in ("Igual", "Cambia", "Editado por internet")}

        # Un artículo se actualiza por internet después de la fecha del .zim nuevo.
        db.upsert_page(self.conn, "Editado por internet", 99, long("Texto más nuevo de la API"),
                       snapshot_ts=sync.parse_iso("2026-03-01T00:00:00Z"))
        # Y otro existe solo porque se descargó por internet (no está en ningún .zim).
        db.upsert_page(self.conn, "Solo internet", 7, long("Nuevo en la wiki"))

        v2 = os.path.join(self.tmp, "wiki_2026-02.zim")
        make_zim(v2, "2026-02-01", {
            "Igual": (long("Texto que no cambia"), ["_assets_/v2/igual.webp"]),
            "Cambia": (long("Versión nueva"), []),
            "Nuevo": (long("Artículo creado"), []),
            "Editado por internet": (long("Texto del zim nuevo"), []),
        }, assets=["_assets_/v2/igual.webp"])
        counts = import_zim(self.conn, v2, workers=1)

        self.assertEqual(counts, {"igual": 1, "cambiado": 1, "nuevo": 1, "local más reciente": 1, "borrado": 1})
        self.assertEqual(db.page_titles(self.conn),
                         {"Igual", "Cambia", "Nuevo", "Editado por internet", "Solo internet"})
        # Sin cambios: conserva su id (y por lo tanto su vector); las imágenes apuntan al .zim nuevo.
        self.assertEqual(db.page_id(self.conn, "Igual"), ids["Igual"])
        pics = images.for_pages(self.conn, [ids["Igual"]])
        self.assertEqual(images.load(self.conn, self.cfg, pics[0]["id"]), (b"IMG", "image/webp"))
        self.assertTrue(search(self.conn, "versión nueva"))
        self.assertFalse(search(self.conn, "posteriormente"))
        self.assertTrue(search(self.conn, "más nuevo de la API"))  # no se pisó con el .zim
        # Solo se vectoriza lo que cambió de texto: "Cambia" y "Nuevo" por el .zim, y los dos
        # actualizados por internet. "Igual" conserva su vector.
        with mock.patch.object(backends, "embed", side_effect=fake_embed):
            self.assertEqual(vectors.embed_pending(self.conn, self.cfg), 4)
        self.assertEqual(db.get_meta(self.conn, "rc_cursor"), "2026-02-01T00:00:00Z")
        self.assertEqual(import_zim(self.conn, v2, workers=1)["igual"], 1)  # repetir: no hace nada


class ZimImportTest(unittest.TestCase):
    """Crea un .zim pequeño con libzim y lo importa (se omite si libzim no está)."""

    def test_import_zim_then_resume_is_noop(self):
        try:
            from libzim.writer import Creator, Hint, Item, StringProvider
        except ImportError:
            self.skipTest("libzim no instalado")
        from wikichat.zim_import import import_zim

        class Page(Item):
            def __init__(self, path, title, html):
                super().__init__()
                self.p, self.t, self.h = path, title, html
            def get_path(self): return self.p
            def get_title(self): return self.t
            def get_mimetype(self): return "text/html"
            def get_contentprovider(self): return StringProvider(self.h)
            def get_hints(self): return {Hint.FRONT_ARTICLE: True}

        class Asset(Page):
            def get_mimetype(self): return "image/webp"
            def get_hints(self): return {Hint.FRONT_ARTICLE: False}

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "prueba.zim")
        body = "<div class='mw-parser-output'><p>%s</p></div>"
        figure = ("<figure><img class='mw-file-element' width='250' height='160' "
                  "src='./_assets_/a/Templo_I.webp'><figcaption>Gran Jaguar</figcaption></figure>"
                  "<img class='mw-file-element' width='20' height='20' src='./_assets_/a/icono.png'>")
        with Creator(path).config_indexing(False, "spa") as c:
            c.set_mainpath("Tikal")
            c.add_metadata("Date", "2026-08-26")
            c.add_item(Page("Tikal", "Tikal", body % ("Tikal es un sitio arqueológico maya. " * 10) + figure))
            c.add_item(Asset("_assets_/a/Templo_I.webp", "", "IMAGEN"))
            c.add_item(Page("Quetzal", "Quetzal", body % ("El quetzal es el ave nacional. " * 10)))
            c.add_item(Page("Corta", "Corta", body % "muy corta"))
            c.add_redirection("Mundo_Perdido", "Mundo Perdido", "Tikal", {Hint.FRONT_ARTICLE: True})

        conn = db.connect(os.path.join(tmp, "wiki.db"))
        import_zim(conn, path, workers=1)
        self.assertEqual(db.page_titles(conn), {"Tikal", "Quetzal"})
        self.assertEqual(search(conn, "ave nacional")[0]["title"], "Quetzal")
        self.assertEqual(db.get_meta(conn, "rc_cursor"), "2026-08-26T00:00:00Z")
        pics = images.for_pages(conn, [db.page_id(conn, "Tikal")])
        self.assertEqual([p["caption"] for p in pics], ["Gran Jaguar"])  # sin el ícono
        cfg = dict(DEFAULTS, db_path=os.path.join(tmp, "wiki.db"))
        self.assertEqual(images.load(conn, cfg, pics[0]["id"]), (b"IMAGEN", "image/webp"))
        import_zim(conn, path, workers=1)  # ya importado: no hace nada
        self.assertEqual(db.stats(conn)["pages"], 2)


if __name__ == "__main__":
    unittest.main()
