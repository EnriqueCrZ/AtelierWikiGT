"""Importa un .zim de Kiwix (p. ej. wikipedia_es_all_nopic) a la base local.

Requiere `pip install libzim`. La conversión de HTML a texto se reparte entre varios
procesos y la importación es reanudable: si se interrumpe, al repetir el comando
continúa donde quedó.
"""
import calendar
import logging
import multiprocessing
import os
import time
from html.parser import HTMLParser

from . import db

log = logging.getLogger("wikichat.zim")

BATCH = 2000

# Elementos cuyo contenido se descarta por completo.
SKIP_TAGS = {"script", "style", "noscript", "button", "svg", "img", "audio", "video", "h1"}
SKIP_CLASSES = {
    "reference", "references", "mw-references", "navbox", "navbox-inner", "mw-cite-backlink",
    "mw-editsection", "noprint", "metadata", "mw-authority-control", "zim-footer",
    "mw-empty-elt", "mwe-math-fallback-image-inline", "mwe-math-fallback-image-display",
    "hatnote", "dablink", "ambox", "listaref", "reflist", "toc", "gallery",
}
BLOCK_TAGS = {
    "p", "div", "section", "li", "ul", "ol", "dl", "dd", "dt", "blockquote", "pre",
    "table", "caption", "tr", "br", "figcaption", "details", "summary",
}
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "wbr", "source", "area", "col", "embed"}


class TextExtractor(HTMLParser):
    """Convierte el HTML de un artículo en texto plano con títulos «== Sección ==»."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.stack = []        # (tag, descarta)
        self.skip_depth = 0
        self.in_body = False
        self.heading = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if not self.in_body:
            if "mw-parser-output" in classes or tag == "body" and not classes:
                self.in_body = True
            if tag not in VOID_TAGS:
                self.stack.append((tag, False))
            return
        if self.skip_depth:
            if tag not in VOID_TAGS:
                self.stack.append((tag, False))
            return
        if tag == "math":
            # Las fórmulas se guardan como su TeX (atributo alttext).
            alt = attrs.get("alttext")
            if alt:
                self.out.append(f" {alt} ")
            self._push_skip(tag)
            return
        if tag in SKIP_TAGS or classes & SKIP_CLASSES:
            if tag not in VOID_TAGS:
                self._push_skip(tag)
            return
        if tag in ("h2", "h3", "h4", "h5", "h6"):
            self.heading = (int(tag[1]), [])
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in BLOCK_TAGS:
            self.out.append("\n")
        if tag not in VOID_TAGS:
            self.stack.append((tag, False))

    def _push_skip(self, tag):
        self.stack.append((tag, True))
        self.skip_depth += 1

    def handle_endtag(self, tag):
        # Cierra hasta la etiqueta correspondiente (tolera HTML mal anidado).
        if not any(t == tag for t, _ in self.stack):
            return
        while self.stack:
            t, skipping = self.stack.pop()
            if skipping:
                self.skip_depth -= 1
            if t == tag:
                break
        if self.skip_depth:
            return
        if self.heading and tag == f"h{self.heading[0]}":
            level, parts = self.heading
            title = " ".join("".join(parts).split())
            marks = "=" * level
            self.out.append(f"\n{marks} {title} {marks}\n")
            self.heading = None
        elif tag in BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.in_body or self.skip_depth:
            return
        if self.heading:
            self.heading[1].append(data)
        else:
            self.out.append(data)

    def text(self):
        lines = (" ".join(line.split()) for line in "".join(self.out).split("\n"))
        return "\n".join(line for line in lines if line.strip(" |"))


def html_to_text(html):
    parser = TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


# --- Trabajo en paralelo -------------------------------------------------------------

_archive = None


def _init_worker(path):
    global _archive
    from libzim.reader import Archive
    _archive = Archive(path)


def _process_range(bounds):
    """Convierte las entradas [start, end) del .zim. Devuelve (end, [(título, texto)])."""
    start, end = bounds
    out = []
    for i in range(start, end):
        try:
            entry = _archive._get_entry_by_id(i)
            if entry.is_redirect:
                continue
            item = entry.get_item()
            if not item.mimetype.startswith("text/html"):
                continue
            text = html_to_text(bytes(item.content).decode("utf-8", "replace"))
        except Exception as e:  # una entrada dañada no debe detener la importación
            log.debug("entrada %d omitida: %s", i, e)
            continue
        if len(text) >= 200 and "#" not in entry.title:  # descarta vacías y anclas
            out.append((entry.title, text))
    return end, out


def zim_date(archive):
    try:
        date = bytes(archive.get_metadata("Date")).decode()
        return calendar.timegm(time.strptime(date, "%Y-%m-%d"))
    except Exception:
        return os.path.getmtime(archive.filename)


def import_zim(conn, path, workers=None):
    try:
        from libzim.reader import Archive
    except ImportError:
        raise SystemExit("Falta libzim: instala con  pip install libzim")

    archive = Archive(path)
    name = os.path.basename(path)
    snapshot = zim_date(archive)
    total = archive.all_entry_count

    if db.get_meta(conn, "zim_name") != name:
        db.set_meta(conn, "zim_name", name)
        db.set_meta(conn, "zim_next_id", 0)
    start = int(db.get_meta(conn, "zim_next_id") or 0)
    if start >= total:
        log.info("%s ya estaba importado por completo", name)
        return

    log.info("Importando %s (%d entradas, fecha %s) desde la entrada %d",
             name, total, time.strftime("%Y-%m-%d", time.gmtime(snapshot)), start)
    ranges = [(i, min(i + BATCH, total)) for i in range(start, total, BATCH)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0, imported = time.time(), 0
    with multiprocessing.Pool(workers, _init_worker, (path,)) as pool:
        for end, pages in pool.imap(_process_range, ranges):
            for title, text in pages:
                db.upsert_page(conn, title, None, text, snapshot_ts=snapshot, commit=False)
            imported += len(pages)
            db.set_meta(conn, "zim_next_id", end, commit=False)
            conn.commit()
            rate = (end - start) / max(time.time() - t0, 1e-9)
            log.info("%.1f%% · %d artículos importados · %.0f entradas/s",
                     100 * end / total, imported, rate)

    # Cambios posteriores al .zim: se piden desde su fecha (o se hace la puesta al día).
    cursor = db.get_meta(conn, "rc_cursor")
    if not cursor or db.get_meta(conn, "last_sync") is None:
        db.set_meta(conn, "rc_cursor", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(snapshot)))
    log.info("Importación terminada: %d artículos. Optimizando índice…", imported)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    conn.commit()
