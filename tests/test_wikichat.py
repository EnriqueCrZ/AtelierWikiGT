import json
import os
import re
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from wikichat import backends, db, images, sync, vectors
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
