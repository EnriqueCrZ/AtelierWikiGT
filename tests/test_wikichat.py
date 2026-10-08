import json
import os
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from wikichat import db, sync
from wikichat.config import DEFAULTS
from wikichat.retrieval import search
from wikichat.server import make_handler
from wikichat.wiki_api import WikiUnavailable


class FakeWiki:
    """Simula la API de MediaWiki en memoria."""

    def __init__(self):
        self.pages = {
            1: ("Antigua Guatemala", 10, "La ciudad colonial.\n== Historia ==\nFundada en 1543 tras el traslado."),
            2: ("Lago de Atitlán", 20, "Lago volcánico rodeado de pueblos mayas."),
        }
        self.recent = set()
        self.offline = False

    def _check(self):
        if self.offline:
            raise WikiUnavailable("sin red")

    def category_titles(self, category, depth):
        self._check()
        return {t for t, _, _ in self.pages.values()}

    def page_text(self, title=None, pageid=None):
        self._check()
        for pid, (t, rev, text) in self.pages.items():
            if pid == pageid or t == title:
                return pid, t, rev, text
        return None

    def latest_revids(self, pageids):
        self._check()
        return {pid: self.pages[pid][1] if pid in self.pages else None for pid in pageids}

    def recent_changes(self, since_iso):
        self._check()
        return set(self.recent)


class WikiChatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = dict(DEFAULTS, db_path=os.path.join(self.tmp, "wiki.db"),
                        seed_categories=["Guatemala"], ollama_url="http://127.0.0.1:9")
        self.conn = db.connect(self.cfg["db_path"])
        self.wiki = FakeWiki()

    def test_chunk_by_sections(self):
        chunks = db.chunk_text("Intro.\n== Historia ==\nTexto histórico.")
        self.assertEqual(chunks, [("Introducción", "Intro."), ("Historia", "Texto histórico.")])

    def test_initial_sync_and_search_ignores_accents(self):
        self.assertTrue(sync.try_update(self.conn, self.cfg, self.wiki))
        self.assertEqual(db.stats(self.conn)["pages"], 2)
        results = search(self.conn, "¿Cuándo se fundó la antigua guatemala?")
        self.assertEqual(results[0]["title"], "Antigua Guatemala")
        self.assertEqual(search(self.conn, "atitlan")[0]["title"], "Lago de Atitlán")

    def test_incremental_update_via_recent_changes(self):
        sync.update(self.conn, self.cfg, self.wiki)
        self.wiki.pages[2] = ("Lago de Atitlán", 21, "Lago endorreico en Sololá.")
        self.wiki.recent = {"Lago de Atitlán"}
        sync.update(self.conn, self.cfg, self.wiki)
        self.assertEqual(search(self.conn, "Solola")[0]["title"], "Lago de Atitlán")
        self.assertEqual(search(self.conn, "mayas"), [])

    def test_full_revision_check_detects_edits_and_deletions(self):
        sync.update(self.conn, self.cfg, self.wiki)
        db.set_meta(self.conn, "last_full_check_ts", 0)  # fuerza revisión completa
        self.wiki.pages[1] = ("Antigua Guatemala", 11, "Patrimonio de la Humanidad.")
        del self.wiki.pages[2]
        sync.update(self.conn, self.cfg, self.wiki)
        self.assertEqual(db.stats(self.conn)["pages"], 1)
        self.assertTrue(search(self.conn, "patrimonio"))

    def test_offline_update_keeps_local_copy(self):
        sync.update(self.conn, self.cfg, self.wiki)
        self.wiki.offline = True
        self.assertFalse(sync.try_update(self.conn, self.cfg, self.wiki))
        self.assertEqual(db.stats(self.conn)["pages"], 2)

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
        self.assertEqual(events[0]["sources"][0]["title"], "Lago de Atitlán")
        text = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertIn("No hay LLM disponible", text)
        self.assertEqual(events[-1]["type"], "done")


if __name__ == "__main__":
    unittest.main()
