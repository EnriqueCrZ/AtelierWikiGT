import json
import os
import re
import sqlite3
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

    def test_migration_is_safe_with_concurrent_connections(self):
        # Base "vieja" sin las columnas nuevas, abierta a la vez por varios hilos (como serve).
        path = os.path.join(self.tmp, "vieja.db")
        old = sqlite3.connect(path)
        old.executescript("CREATE TABLE pages (id INTEGER PRIMARY KEY, title TEXT UNIQUE, revid INTEGER, snapshot_ts REAL);"
                          "CREATE TABLE images (id INTEGER PRIMARY KEY, page INTEGER, src TEXT, caption TEXT, mime TEXT, data BLOB);")
        old.close()
        errors = []

        def open_db():
            try:
                db.connect(path)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=open_db) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])
        cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(images)")}
        self.assertIn("vec", cols)

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

    def test_question_about_named_article_gets_its_introduction(self):
        from wikichat.retrieval import entity_intros
        db.upsert_page(self.conn, "Tikal", None,
                       "Tikal es una antigua ciudad de la civilización maya.\n== Templos ==\n" + "Templo IV. " * 200)
        db.upsert_page(self.conn, "La Gioconda", None, "La Gioconda es un óleo de Leonardo da Vinci.")
        db.upsert_page(self.conn, "Civilización", None, "Una civilización es una sociedad compleja.")
        self.conn.execute("INSERT INTO redirects (title_key, target) VALUES ('mona lisa', 'La Gioconda')")
        intros = lambda q: [c["title"] for c in entity_intros(self.conn, q)]
        # Nombres propios (con mayúscula), no palabras comunes como "civilización".
        self.assertEqual(intros("¿Qué civilización construyó Tikal?"), ["Tikal"])
        self.assertEqual(entity_intros(self.conn, "¿Qué civilización construyó Tikal?")[0]["section"], "Introducción")
        self.assertEqual(intros("¿Quién pintó la Mona Lisa?"), ["La Gioconda"])  # por redirección
        self.assertEqual(intros("¿Qué es la civilización?"), ["Civilización"])  # la pregunta entera
        self.assertEqual(retrieve(self.conn, self.cfg, "¿Quién pintó la Mona Lisa?")[0]["title"], "La Gioconda")

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

    def test_images_ranked_by_caption_relevance(self):
        pid = db.upsert_page(self.conn, "Volcanes", None, "Texto sobre volcanes.", images=[
            ("zim:a.webp", "Mapa político de la región"),        # primera, pero no viene al caso
            ("zim:b.webp", "Erupción del volcán con lava"),
            ("zim:c.webp", "Ave sobre un árbol de la selva"),
        ])
        cfg = dict(self.cfg, embed_model="fake")
        with mock.patch.object(backends, "embed", side_effect=fake_embed):
            qvec = vectors.embed_query(cfg, "¿Cómo es una erupción de un volcan?")
            pics = images.rank(self.conn, cfg, [pid], "¿Cómo es una erupción de un volcan?", qvec)
            self.assertEqual(pics[0]["caption"], "Erupción del volcán con lava")
            self.assertNotIn("Ave sobre un árbol de la selva", [p["caption"] for p in pics])
            cached = self.conn.execute("SELECT COUNT(*) FROM images WHERE vec IS NOT NULL").fetchone()[0]
            self.assertEqual(cached, 3)  # la próxima vez no se vuelven a calcular

    def test_images_ranked_by_words_without_embeddings(self):
        pid = db.upsert_page(self.conn, "Lago", None, "Texto.", images=[
            ("zim:a.webp", "Mapa"), ("zim:b.webp", "Atardecer en el lago Atitlán")])
        pics = images.rank(self.conn, dict(self.cfg, embed_model=""), [pid], "atardecer en Atitlán")
        self.assertEqual(pics[0]["caption"], "Atardecer en el lago Atitlán")

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

    def test_preload_and_loaded_check(self):
        loaded = {"models": []}

        class Ps(FakeModelServer):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(loaded).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert body["keep_alive"] == "2h"
                if self.path == "/api/chat" and body["messages"] == []:
                    loaded["models"].append({"name": body["model"]})
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"embeddings": [[1.0] for _ in body["input"]]}).encode())

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Ps)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            cfg = dict(DEFAULTS, llm_url=f"http://127.0.0.1:{httpd.server_port}", chat_model="qwen2.5:3b")
            self.assertFalse(backends.chat_model_loaded(cfg))
            backends.preload(cfg)
            self.assertTrue(backends.chat_model_loaded(cfg))
            self.assertIsNone(backends.chat_model_loaded(dict(cfg, llm_backend="openai")))
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
        counts.pop("redirecciones", None)  # el .zim de prueba trae la de su página principal

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


class ZimRedirectsTest(unittest.TestCase):
    def test_redirects_imported_and_redirects_only_mode(self):
        try:
            from libzim.writer import Creator, Hint
        except ImportError:
            self.skipTest("libzim no instalado")
        from wikichat.retrieval import entity_intros
        from wikichat.zim_import import import_redirects, import_zim
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "r.zim")
        make_zim(path, "2026-01-01", {"La Gioconda": ("La Gioconda es un óleo de Leonardo da Vinci. " * 10, [])})
        # make_zim no crea redirecciones: se agregan con un segundo .zim que las incluye
        path2 = os.path.join(tmp, "r2.zim")
        from libzim.writer import StringProvider, Item

        class Page(Item):
            def get_path(self): return "La_Gioconda"
            def get_title(self): return "La Gioconda"
            def get_mimetype(self): return "text/html"
            def get_contentprovider(self):
                return StringProvider("<div class='mw-parser-output'><p>" + "La Gioconda es un óleo de Leonardo. " * 10 + "</p></div>")
            def get_hints(self): return {Hint.FRONT_ARTICLE: True}

        with Creator(path2).config_indexing(False, "spa") as c:
            c.set_mainpath("La_Gioconda")
            c.add_metadata("Date", "2026-02-01")
            c.add_item(Page())
            c.add_redirection("Mona_Lisa", "Mona Lisa", "La_Gioconda", {Hint.FRONT_ARTICLE: True})
        conn = db.connect(os.path.join(tmp, "wiki.db"))
        import_zim(conn, path, workers=1)
        self.assertEqual(entity_intros(conn, "¿Quién pintó la Mona Lisa?"), [])
        self.assertEqual(import_redirects(conn, path2, workers=1), 1)  # copia ya importada
        self.assertEqual([c["title"] for c in entity_intros(conn, "¿Quién pintó la Mona Lisa?")], ["La Gioconda"])
        conn.execute("DELETE FROM redirects")
        counts = import_zim(conn, path2, workers=1)                  # importación normal
        self.assertEqual(counts["redirecciones"], 1)
        self.assertEqual(db.stats(conn)["redirects"], 1)


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


# --- Instalación: hardware, perfil y Ollama ---------------------------------------------

from wikichat import hardware, ollama_manager  # noqa: E402

GiB = 1024 ** 3


def hw_with(ram_gb, cores=4, gpus=()):
    return {"arch": "amd64", "cpu": "x", "cores": cores, "avx2": True, "avx512": False,
            "ram_total": ram_gb * GiB, "ram_free": ram_gb * GiB, "disk_free": 100 * GiB,
            "gpus": list(gpus)}


class HardwareTest(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(hardware.parse_meminfo("MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\n"),
                         (16384000 * 1024, 8192000 * 1024))
        self.assertEqual(hardware.parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288\n"),
                         [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 3060", "vram": 12 * GiB}])
        self.assertEqual(hardware.parse_rocm_smi("device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
                                                 "card0,17163091968,123\n")[0]["vram"], 17163091968)
        gpus = hardware.parse_lspci(
            "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620\n"
            "01:00.0 3D controller: NVIDIA Corporation GP108M [GeForce MX150]\n"
            "00:1f.3 Audio device: Intel Corporation Sunrise Point\n")
        self.assertEqual([g["vendor"] for g in gpus], ["intel", "nvidia"])

    def test_profiles(self):
        self.assertEqual(hardware.choose_profile(hw_with(6))["chat_model"], "qwen2.5:1.5b")
        self.assertEqual(hardware.choose_profile(hw_with(8))["chat_model"], "qwen2.5:3b")
        self.assertEqual(hardware.choose_profile(hw_with(16, cores=4))["chat_model"], "qwen2.5:3b")
        self.assertEqual(hardware.choose_profile(hw_with(16, cores=8))["chat_model"], "qwen2.5:7b")
        self.assertEqual(hardware.choose_profile(hw_with(4))["top_k"], 3)
        gpu = hardware.choose_profile(hw_with(32, gpus=[{"vendor": "nvidia", "name": "x", "vram": 8 * GiB}]))
        self.assertEqual((gpu["name"], gpu["chat_model"], gpu["rewrite_followups"]), ("gpu", "qwen2.5:7b", True))
        six = hardware.choose_profile(hw_with(32, gpus=[{"vendor": "nvidia", "name": "x", "vram": 6 * GiB}]))
        self.assertEqual(six["chat_model"], "qwen2.5:3b")
        self.assertEqual(hardware.choose_profile(hw_with(10, cores=8))["chat_model"], "qwen2.5:7b")
        big = hardware.choose_profile(hw_with(32, gpus=[{"vendor": "nvidia", "name": "x", "vram": 24 * GiB}]))
        self.assertEqual(big["chat_model"], "qwen2.5:14b")
        # Una GPU de 4 GB no alcanza para el perfil GPU, pero ayuda: con 32 GB se prueba el 7B
        # aunque haya pocos núcleos (la medición de setup decide si se queda).
        small = hardware.choose_profile(hw_with(32, cores=6, gpus=[{"vendor": "nvidia", "name": "x", "vram": 4 * GiB}]))
        self.assertEqual((small["name"], small["chat_model"]), ("cpu+gpu", "qwen2.5:7b"))
        # Una GPU Intel integrada (sin VRAM medida) no cuenta como GPU para los modelos.
        igpu = hardware.choose_profile(hw_with(16, gpus=[{"vendor": "intel", "name": "x", "vram": 0}]))
        self.assertEqual(igpu["name"], "cpu")
        self.assertEqual(hardware.smaller_chat_model("qwen2.5:7b"), "qwen2.5:3b")
        self.assertEqual(hardware.smaller_chat_model("qwen2.5:1.5b"), "qwen2.5:1.5b")

    def test_package_filter_keeps_only_needed_gpu_libraries(self):
        member = lambda n: type("M", (), {"name": n})()
        names = ["bin/ollama", "lib/ollama/libggml-cpu-haswell.so", "lib/ollama/cuda_v12/libggml-cuda.so",
                 "lib/ollama/vulkan/libggml-vulkan.so", "lib/ollama/mlx_cuda_v13/x.so"]
        keep = lambda gpus: [n for n in names if ollama_manager._keep(member(n), gpus)]
        self.assertEqual(keep([]), names[:2])
        self.assertEqual(keep([{"vendor": "nvidia"}]), names[:3] + [names[4]])
        self.assertEqual(keep([{"vendor": "intel"}]), names[:2] + [names[3]])

    def test_serve_without_managed_ollama_does_nothing(self):
        self.assertIsNone(ollama_manager.ensure_running(dict(DEFAULTS)))


class FakeOllama(BaseHTTPRequestHandler):
    """Lo mínimo de la API de Ollama que usa `setup`."""
    pulled = set()

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._json({"version": "0.0-prueba"})
        elif self.path == "/api/tags":
            self._json({"models": [{"name": n} for n in self.pulled]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/pull":
            self.pulled.add(body["model"])
            self.send_response(200)
            self.end_headers()
            for ev in ({"status": "pulling", "total": 100, "completed": 50},
                       {"status": "pulling", "total": 100, "completed": 100}, {"status": "success"}):
                self.wfile.write(json.dumps(ev).encode() + b"\n")
        elif self.path == "/api/embed":
            self._json({"embeddings": [[1.0] * 8 for _ in body["input"]]})
        elif self.path == "/api/chat":
            # El 3b lee 40 tokens/s y escribe 10: 1500/40 + 200/10 ≈ 58 s por respuesta (aceptable).
            # El 7b lee 5 tokens/s: más de 5 minutos, demasiado lento.
            slow = "7b" in body["model"]
            self._json({"message": {"content": "ok"}, "prompt_eval_count": 1000,
                        "prompt_eval_duration": (200 if slow else 25) * 1e9,
                        "eval_count": 80, "eval_duration": 8e9})


class SetupTest(unittest.TestCase):
    def test_setup_uses_running_ollama_measures_and_writes_config(self):
        from wikichat import setup as setup_mod
        FakeOllama.pulled = set()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_port}"
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump({"db_path": os.path.join(tmp, "data", "wiki.db"), "port": 9999}, f)
        many_cores = hw_with(32, cores=16)
        try:
            with mock.patch.object(ollama_manager, "SYSTEM_URL", url), \
                 mock.patch.object(hardware, "detect", return_value=many_cores), \
                 mock.patch("builtins.print"):
                setup_mod.run(path, yes=True)
        finally:
            httpd.shutdown()
            httpd.server_close()
        with open(path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["port"], 9999)  # lo que el usuario ya tenía se respeta
        self.assertEqual(cfg["llm_url"], url)
        self.assertFalse(cfg["manage_ollama"])
        # Empezó con el 7b (16 núcleos, 32 GB), lo midió lento y bajó al 3b.
        self.assertEqual(cfg["chat_model"], "qwen2.5:3b")
        self.assertEqual(FakeOllama.pulled, {"embeddinggemma", "qwen2.5:7b", "qwen2.5:3b"})
        self.assertFalse(cfg["rewrite_followups"])  # 400/40 + 30/10 = 13 s > 6 s
        self.assertTrue(os.path.exists(os.path.join(tmp, "data", "hardware.json")))


class CitationCheckTest(unittest.TestCase):
    def test_warns_only_when_every_citation_is_unsupported(self):
        from wikichat.llm import citation_warning
        src = [{"title": "Ciclo del mercurio"}, {"title": "Tornasol"}, {"title": "Manganato"}]
        ok = ["El mercurio es líquido [Mercurio].", "Se pone rojo [Tornasol (sección «Aplicaciones»)].",
              "Sin citas, no hay nada que comprobar.", "Rojo [Tornasol] y algo más [Oro]."]
        for answer in ok:
            self.assertIsNone(citation_warning(answer, src), answer)
        warn = citation_warning("La capital es Ulaanbaatar [Ulaanbaatar (Mongolía)].", src)
        self.assertIn("Ulaanbaatar (Mongolía)", warn)


class EvaluateTest(unittest.TestCase):
    def test_refusal_detection(self):
        from wikichat.evaluate import refused
        for text in ("No encontré esa información en la wiki.", "Los fragmentos no mencionan quién ganó.",
                     "Lo siento, no tengo información sobre eso.", "La información proporcionada no contiene ese dato."):
            self.assertTrue(refused(text), text)
        for text in ("El bronce es una aleación de cobre y estaño [Bronce].", "No es un metal noble, sino un gas."):
            self.assertFalse(refused(text), text)

    def test_scoring(self):
        from wikichat.evaluate import score, summarize
        item = {"pregunta": "¿De qué es el bronce?", "articulos": ["Bronce"], "datos": [["cobre"], ["estano"]]}
        ok = score(item, "Es una aleación de cobre y estaño [Bronce].", ["Bronce"])
        self.assertTrue(ok["acierto"] and ok["busqueda"] and ok["cita"])
        half = score(item, "Es una aleación de cobre.", ["Latón"])
        self.assertEqual((half["acierto"], half["datos"], half["busqueda"]), (False, "1/2", False))
        lazy = score(item, "No encontré esa información en la wiki.", ["Bronce"])
        self.assertTrue(lazy["se_niega_de_mas"])
        trap = {"pregunta": "¿Quién ganó el Mundial?", "sin_respuesta": True}
        invent = score(trap, "Lo ganó España.", [])
        honest = score(trap, "No encontré esa información en la wiki.", [])
        self.assertEqual((invent["acierto"], invent["inventa"], honest["acierto"]), (False, True, True))
        for r in (ok, half, lazy, invent, honest):
            r["total_s"] = 1.0
        s = summarize([ok, half, lazy, invent, honest])
        self.assertEqual((s["aciertos_pct"], s["con_respuesta_pct"], s["sin_respuesta_pct"], s["inventa"]),
                         (40, 33, 50, 1))

    def test_question_sets_are_valid(self):
        import re as _re
        for name in ("quimica.json", "general.json"):
            with open(os.path.join(os.path.dirname(__file__), "..", "evals", name), encoding="utf-8") as f:
                items = json.load(f)["preguntas"]
            self.assertGreaterEqual(len(items), 15)
            for item in items:
                self.assertTrue(item.get("sin_respuesta") or item.get("datos"), item)
                for group in item.get("datos", []):
                    for pattern in group:
                        _re.compile(pattern)
                        self.assertEqual(pattern, pattern.lower(), "los patrones van sin mayúsculas ni acentos")
