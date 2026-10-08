"""Cliente mínimo de la API de MediaWiki (solo librería estándar)."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request


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
                    time.sleep(int(e.headers.get("Retry-After") or 5 * 2**attempt))
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

    def page_text(self, title=None, pageid=None):
        """Devuelve (pageid, title, revid, texto plano) o None si no existe."""
        key = {"pageids": str(pageid)} if pageid else {"titles": title}
        data = self.get(
            action="query", prop="extracts|info", explaintext="1",
            exsectionformat="wiki", redirects="1", **key,
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or "extract" not in pages[0]:
            return None
        p = pages[0]
        return p["pageid"], p["title"], p.get("lastrevid"), p["extract"]

    def latest_revids(self, pageids):
        """{pageid: lastrevid} para lotes de 50; None si la página ya no existe."""
        result = {}
        ids = list(pageids)
        for i in range(0, len(ids), 50):
            batch = ids[i:i + 50]
            data = self.get(action="query", prop="info", pageids="|".join(map(str, batch)))
            for p in data.get("query", {}).get("pages", []):
                result[p["pageid"]] = None if p.get("missing") else p.get("lastrevid")
        return result

    def recent_changes(self, since_iso):
        """Títulos de artículos editados/creados desde 'since_iso' (máx. ~30 días)."""
        titles = set()
        for q in self.query_all(
            list="recentchanges", rcstart=since_iso, rcdir="newer",
            rcnamespace="0", rctype="edit|new", rcprop="title", rclimit="500",
        ):
            titles.update(c["title"] for c in q.get("recentchanges", []))
        return titles
