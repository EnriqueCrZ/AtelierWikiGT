"""Cliente mínimo de la API de MediaWiki (solo librería estándar)."""
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request


log = logging.getLogger("wikichat.api")


class WikiUnavailable(Exception):
    """La wiki no responde (sin red, rate-limit persistente, etc.)."""


class WikiClient:
    def __init__(self, api_url, user_agent, delay=0.5, timeout=30):
        self.api_url = api_url
        self.user_agent = user_agent
        self.delay = delay
        self.timeout = timeout

    def get(self, **params):
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(4):
            time.sleep(self.delay)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 3:
                    wait = int(e.headers.get("Retry-After") or 5 * 2**attempt)
                    log.warning("La wiki pide esperar (HTTP %d); reintento en %d s", e.code, wait)
                    time.sleep(wait)
                    continue
                raise WikiUnavailable(f"HTTP {e.code}") from e
            except (urllib.error.URLError, OSError, ValueError) as e:
                raise WikiUnavailable(str(e)) from e
            if data.get("error", {}).get("code") == "maxlag" and attempt < 3:
                time.sleep(5)
                continue
            if "error" in data:
                raise WikiUnavailable(data["error"].get("info", "error de API"))
            return data
        raise WikiUnavailable("demasiados reintentos")

    def query_all(self, **params):
        """Itera sobre todas las páginas de resultados siguiendo 'continue'."""
        cont = {}
        while True:
            data = self.get(action="query", **params, **cont)
            yield data.get("query", {})
            if "continue" not in data:
                return
            cont = data["continue"]

    def category_titles(self, category, depth):
        """Títulos de artículos (ns 0) en una categoría y sus subcategorías."""
        seen_cats, titles = set(), set()
        pending = [(f"Categoría:{category}" if ":" not in category else category, 0)]
        while pending:
            cat, level = pending.pop()
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            for q in self.query_all(
                list="categorymembers", cmtitle=cat, cmtype="page|subcat",
                cmnamespace="0|14", cmlimit="500",
            ):
                for m in q.get("categorymembers", []):
                    if m["ns"] == 0:
                        titles.add(m["title"])
                    elif m["ns"] == 14 and level < depth:
                        pending.append((m["title"], level + 1))
        return titles

    def page_text(self, title):
        """Devuelve (título, revid, texto plano, URL de la imagen principal o None), o None
        si no existe. Si el título es una redirección, devuelve el artículo de destino."""
        data = self.get(
            action="query", prop="extracts|info|pageimages", explaintext="1",
            exsectionformat="wiki", redirects="1", titles=title,
            piprop="thumbnail", pithumbsize="480",
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or "extract" not in pages[0]:
            return None
        p = pages[0]
        thumb = (p.get("thumbnail") or {}).get("source")
        return p["title"], p.get("lastrevid"), p["extract"], thumb

    def latest_revisions(self, titles):
        """{título: (revid, timestamp ISO)} en lotes de 50; None si ya no existe o es redirección."""
        result = {}
        titles = list(titles)
        for i in range(0, len(titles), 50):
            batch = titles[i:i + 50]
            data = self.get(
                action="query", prop="revisions|info", rvprop="ids|timestamp",
                titles="|".join(batch),
            ).get("query", {})
            original = {n["to"]: n["from"] for n in data.get("normalized", [])}
            for p in data.get("pages", []):
                title = original.get(p["title"], p["title"])
                if p.get("missing") or p.get("invalid") or p.get("redirect"):
                    result[title] = None
                else:
                    rev = p["revisions"][0]
                    result[title] = (rev["revid"], rev["timestamp"])
        return result

    def recent_changes(self, since_iso, skip_bots=True):
        """Cambios en artículos desde 'since_iso' (la wiki guarda ~30 días), del más viejo al
        más nuevo. Produce (timestamp, tipo, título, destino) con tipo edit, delete o move."""
        params = dict(
            list="recentchanges", rcstart=since_iso, rcdir="newer", rcnamespace="0",
            rctype="edit|new|log", rcprop="title|timestamp|loginfo", rclimit="500",
        )
        if skip_bots:
            params["rcshow"] = "!bot"
        for q in self.query_all(**params):
            for c in q.get("recentchanges", []):
                kind, target = "edit", None
                if c.get("type") == "log":
                    if c.get("logtype") == "delete" and c.get("logaction") == "delete":
                        kind = "delete"
                    elif c.get("logtype") == "move":
                        kind = "move"
                        params = c.get("logparams", {})
                        if params.get("target_ns") == 0:
                            target = params.get("target_title")
                    elif not (c.get("logtype") == "delete" and c.get("logaction") == "restore"):
                        continue
                yield c["timestamp"], kind, c["title"], target
