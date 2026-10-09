"""Importa un .zim de Kiwix (p. ej. wikipedia_es_all_nopic) a la base local.

Requiere `pip install libzim`. La conversión de HTML a texto se reparte entre varios
procesos y la importación es reanudable: si se interrumpe, al repetir el comando
continúa donde quedó.
"""
import calendar
import json
import logging
import multiprocessing
import os
import posixpath
import time
import urllib.parse
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
MAX_IMAGES = 6
MIN_IMAGE_SIDE = 100  # px mostrados; descarta íconos, banderitas y diagramas diminutos
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "wbr", "source", "area", "col", "embed"}


class TextExtractor(HTMLParser):
    """Convierte el HTML de un artículo en texto plano con títulos «== Sección ==»."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.images = []       # [src, pie de foto]
        self.figure_img = None
        self.caption = None
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
        if tag == "img":
            self._image(attrs, classes)
            return
        if tag == "figure":
            self.figure_img = None
        elif tag == "figcaption":
            self.caption = []
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

    def _image(self, attrs, classes):
        if "mw-file-element" not in classes or len(self.images) >= MAX_IMAGES:
            return
        try:
            if min(int(attrs.get("width", 0)), int(attrs.get("height", 0))) < MIN_IMAGE_SIDE:
                return
        except ValueError:
            return
        src = attrs.get("src")
        if not src:
            return
        for i, img in enumerate(self.images):
            if img[0] == src:  # repetida (p. ej. ficha y figura): conserva el mejor pie
                self.figure_img = i
                return
        self.images.append([src, attrs.get("alt") or ""])
        self.figure_img = len(self.images) - 1

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
        if tag == "figcaption" and self.caption is not None:
            if self.figure_img is not None:
                text = " ".join("".join(self.caption).split())
                if text:
                    self.images[self.figure_img][1] = text
            self.caption = None
        elif tag == "figure":
            self.figure_img = None
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
        if self.caption is not None:
            self.caption.append(data)
        if self.heading:
            self.heading[1].append(data)
        else:
            self.out.append(data)

    def text(self):
        lines = (" ".join(line.split()) for line in "".join(self.out).split("\n"))
        return "\n".join(line for line in lines if line.strip(" |"))


def html_to_text(html):
    return parse_article(html)[0]


def parse_article(html, path=""):
    """Devuelve (texto, [(ruta de la imagen dentro del .zim, pie de foto)])."""
    parser = TextExtractor()
    parser.feed(html)
    parser.close()
    base = posixpath.dirname(path)
    images = []
    for src, caption in parser.images:
        if "://" in src:
            continue
        src = urllib.parse.unquote(src.split("?")[0])
        images.append((posixpath.normpath(posixpath.join(base, src)).lstrip("/"), caption))
    return parser.text(), images


# --- Trabajo en paralelo -------------------------------------------------------------

_archive = None


def _init_worker(path):
    global _archive
    from libzim.reader import Archive
    _archive = Archive(path)


def _redirect(entry):
    """(clave del nombre alternativo, título del artículo destino), o None."""
    if entry.path == "mainPage":  # entrada interna del .zim que apunta a la portada
        return None
    target = entry.get_redirect_entry()
    if "#" in entry.title or "#" in target.title or entry.title == target.title:
        return None
    key = db.title_key(entry.title)
    return (key, target.title) if key else None


def _redirects_range(bounds):
    """Solo las redirecciones de las entradas [start, end) (modo --solo-redirecciones)."""
    start, end = bounds
    out = []
    for i in range(start, end):
        try:
            entry = _archive._get_entry_by_id(i)
            if entry.is_redirect and (r := _redirect(entry)):
                out.append(r)
        except Exception as e:
            log.debug("entrada %d omitida: %s", i, e)
    return end, out


def _process_range(bounds):
    """Convierte las entradas [start, end) del .zim.

    Devuelve (end, [(título, texto, huella del texto, imágenes)], [redirecciones])."""
    start, end = bounds
    out, redirects = [], []
    for i in range(start, end):
        try:
            entry = _archive._get_entry_by_id(i)
            if entry.is_redirect:
                if r := _redirect(entry):
                    redirects.append(r)
                continue
            item = entry.get_item()
            if not item.mimetype.startswith("text/html"):
                continue
            text, images = parse_article(bytes(item.content).decode("utf-8", "replace"), entry.path)
        except Exception as e:  # una entrada dañada no debe detener la importación
            log.debug("entrada %d omitida: %s", i, e)
            continue
        if len(text) >= 200 and "#" not in entry.title:  # descarta vacías y anclas
            out.append((entry.title, text, db.text_hash(text), [(f"zim:{p}", c) for p, c in images]))
    return end, out, redirects


def zim_date(archive):
    try:
        date = bytes(archive.get_metadata("Date")).decode()
        return calendar.timegm(time.strptime(date, "%Y-%m-%d"))
    except Exception:
        return os.path.getmtime(archive.filename)


def apply_article(conn, title, text, digest, images, snapshot, gen):
    """Aplica un artículo del .zim sobre la copia local. Devuelve qué pasó:
    "nuevo", "cambiado", "igual" o "local más reciente".

    Solo "nuevo" y "cambiado" reemplazan el texto (y obligan a recalcular su vector); en los
    demás casos se conservan texto y vector y solo se actualizan las rutas de las imágenes,
    que cambian entre versiones del .zim.
    """
    row = conn.execute("SELECT id, hash, snapshot_ts FROM pages WHERE title=?", (title,)).fetchone()
    if row is None:
        db.upsert_page(conn, title, None, text, snapshot, images, commit=False, gen=gen, hash=digest)
        return "nuevo"
    pid, old_hash, old_ts = row
    if (old_ts or 0) > snapshot:
        # Se actualizó por internet después de la fecha de este .zim: el texto local es más nuevo.
        db.set_images(conn, pid, images)
        conn.execute("UPDATE pages SET gen=? WHERE id=?", (gen, pid))
        return "local más reciente"
    if old_hash == digest:
        db.set_images(conn, pid, images)
        conn.execute("UPDATE pages SET gen=?, snapshot_ts=? WHERE id=?", (gen, snapshot, pid))
        return "igual"
    db.upsert_page(conn, title, None, text, snapshot, images, commit=False, gen=gen, hash=digest)
    return "cambiado"


def remove_missing(conn, gen, snapshot):
    """Borra los artículos que no venían en el .zim recién importado, salvo los que se
    actualizaron por internet después de su fecha (son más nuevos que el .zim)."""
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM pages WHERE (gen IS NULL OR gen < ?) AND COALESCE(snapshot_ts, 0) <= ?",
        (gen, snapshot))]
    for pid in ids:
        db._remove(conn, pid)
    conn.commit()
    return len(ids)


def import_zim(conn, path, workers=None):
    """Importa un .zim. Si ya había una copia (de otro .zim o actualizada por internet),
    solo reprocesa lo que cambió: el resto conserva su texto y su vector."""
    try:
        from libzim.reader import Archive
    except ImportError:
        raise SystemExit("Falta libzim: instala con  pip install libzim")

    archive = Archive(path)
    name = os.path.basename(path)
    snapshot = zim_date(archive)
    day = time.strftime("%Y-%m-%d", time.gmtime(snapshot))
    total = archive.all_entry_count

    db.set_meta(conn, "zim_path", os.path.abspath(path))  # de aquí se leen las imágenes
    if db.get_meta(conn, "zim_name") != name:
        # Importación nueva: nueva "generación" para saber qué artículos dejaron de venir.
        db.set_meta(conn, "zim_gen", int(db.get_meta(conn, "zim_gen") or 0) + 1, commit=False)
        db.set_meta(conn, "zim_name", name, commit=False)
        db.set_meta(conn, "zim_next_id", 0, commit=False)
        db.set_meta(conn, "zim_counts", "{}")
    gen = int(db.get_meta(conn, "zim_gen") or 1)
    start = int(db.get_meta(conn, "zim_next_id") or 0)
    counts = json.loads(db.get_meta(conn, "zim_counts") or "{}")
    if start >= total:
        log.info("%s ya estaba importado por completo", name)
        return counts

    had_pages = conn.execute("SELECT 1 FROM pages LIMIT 1").fetchone() is not None
    log.info("%s %s (%d entradas, fecha %s) desde la entrada %d",
             "Actualizando con" if had_pages else "Importando", name, total, day, start)
    ranges = [(i, min(i + BATCH, total)) for i in range(start, total, BATCH)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    with multiprocessing.Pool(workers, _init_worker, (path,)) as pool:
        for end, pages, redirects in pool.imap(_process_range, ranges):
            conn.executemany("INSERT INTO redirects (title_key, target, gen) VALUES (?, ?, ?)",
                             [(key, target, gen) for key, target in redirects])
            counts["redirecciones"] = counts.get("redirecciones", 0) + len(redirects)
            for title, text, digest, images in pages:
                what = apply_article(conn, title, text, digest, images, snapshot, gen)
                counts[what] = counts.get(what, 0) + 1
            db.set_meta(conn, "zim_next_id", end, commit=False)
            db.set_meta(conn, "zim_counts", json.dumps(counts), commit=False)
            conn.commit()
            rate = (end - start) / max(time.time() - t0, 1e-9)
            log.info("%.1f%% · %s · %.0f entradas/s", 100 * end / total, _fmt(counts), rate)

    counts["borrado"] = remove_missing(conn, gen, snapshot)
    conn.execute("DELETE FROM redirects WHERE gen IS NULL OR gen < ?", (gen,))  # las del .zim anterior
    db.set_meta(conn, "zim_counts", json.dumps(counts), commit=False)
    db.set_meta(conn, "zim_date", day, commit=False)

    # Cambios posteriores al .zim: se piden por internet desde su fecha. Si la copia ya
    # estaba más al día que el .zim, se sigue desde donde iba.
    cursor = db.get_meta(conn, "rc_cursor")
    zim_cursor = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(snapshot))
    if not cursor or cursor < zim_cursor:
        db.set_meta(conn, "rc_cursor", zim_cursor, commit=False)
        # Una puesta al día pendiente quedó cubierta por el .zim nuevo.
        db.set_meta(conn, "check_after", None, commit=False)
        db.set_meta(conn, "check_started", None, commit=False)
    conn.commit()
    log.info("Importación terminada: %s. Optimizando índice…", _fmt(counts))
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    conn.commit()
    pending = conn.execute(
        "SELECT COUNT(*) FROM pages p LEFT JOIN page_vectors v ON v.page = p.id WHERE v.page IS NULL"
    ).fetchone()[0]
    log.info("Artículos por vectorizar (en segundo plano con `serve`, o con `embed`): %d", pending)
    return counts


def import_redirects(conn, path, workers=None):
    """Solo las redirecciones ("Mona Lisa" → "La Gioconda") de un .zim, sin tocar los
    artículos: para copias importadas con versiones anteriores, que no las guardaban."""
    try:
        from libzim.reader import Archive
    except ImportError:
        raise SystemExit("Falta libzim: instala con  pip install libzim")
    total = Archive(path).all_entry_count
    gen = int(db.get_meta(conn, "zim_gen") or 1)
    log.info("Leyendo las redirecciones de %s (%d entradas)…", os.path.basename(path), total)
    conn.execute("DELETE FROM redirects")
    ranges = [(i, min(i + BATCH * 5, total)) for i in range(0, total, BATCH * 5)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    found = 0
    with multiprocessing.Pool(workers, _init_worker, (path,)) as pool:
        for end, redirects in pool.imap(_redirects_range, ranges):
            conn.executemany("INSERT INTO redirects (title_key, target, gen) VALUES (?, ?, ?)",
                             [(key, target, gen) for key, target in redirects])
            found += len(redirects)
            if end == total or end % (BATCH * 250) == 0:
                log.info("%.0f%% · %d redirecciones", 100 * end / total, found)
    conn.commit()
    log.info("Listo: %d redirecciones guardadas", found)
    return found


def _fmt(counts):
    order = ("nuevo", "cambiado", "igual", "local más reciente", "borrado", "redirecciones")
    return ", ".join(f"{counts[k]} {k}" for k in order if counts.get(k)) or "sin artículos"
