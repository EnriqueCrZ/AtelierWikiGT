"""Actualización incremental de la copia local. Si falla, la copia local sigue sirviendo.

Dos estrategias, elegidas automáticamente:

* Cambios recientes: si la última sincronización (o la fecha del .zim) tiene menos de
  ~29 días, se pide a la wiki la lista de artículos editados, borrados o movidos desde
  entonces y solo se descargan esos. Es lo normal en el día a día.
* Puesta al día: si pasó más tiempo, se comparan las revisiones de todas las páginas
  locales en lotes de 50 y se descargan las que cambiaron. Es reanudable: si se corta,
  continúa donde quedó en la siguiente ejecución.
"""
import calendar
import logging
import time

from . import db
from .wiki_api import WikiClient, WikiUnavailable

log = logging.getLogger("wikichat.sync")

# recentchanges guarda ~30 días; dejamos margen.
RC_WINDOW_SECONDS = 29 * 86400
# En modo subconjunto se hace una comparación completa semanal (es barata) para detectar
# cambios que recentchanges no reporta (p. ej. ediciones de bots si se omiten).
FULL_CHECK_SECONDS = 7 * 86400
CHECK_BATCH = 500


def client_from(cfg):
    return WikiClient(cfg["api_url"], cfg["user_agent"], cfg["request_delay_seconds"])


def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def parse_iso(s):
    return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


def fetch(conn, client, title):
    """Descarga un artículo y lo guarda; si ya no existe (o ahora es redirección) lo borra."""
    page = client.page_text(title)
    if page is None:
        db.delete_page(conn, title)
        return False
    new_title, revid, text = page
    if new_title != title:
        db.delete_page(conn, title, commit=False)
    db.upsert_page(conn, new_title, revid, text)
    log.debug("actualizado: %s", new_title)
    return True


def discover_titles(client, cfg):
    titles = set(cfg["seed_titles"])
    for cat in cfg["seed_categories"]:
        titles |= client.category_titles(cat, cfg["category_depth"])
    return titles


def _changed(stored_revid, snapshot_ts, latest):
    revid, ts = latest
    if stored_revid is not None:
        return revid != stored_revid
    return parse_iso(ts) > (snapshot_ts or 0)


def catch_up(conn, client, started):
    """Compara revisiones de todas las páginas locales. Reanudable."""
    check_started = float(db.get_meta(conn, "check_started") or started)
    db.set_meta(conn, "check_started", check_started)
    after = db.get_meta(conn, "check_after") or ""
    db.set_meta(conn, "check_after", after)
    checked = refreshed = 0
    while True:
        rows = conn.execute(
            "SELECT title, revid, snapshot_ts FROM pages WHERE title > ? ORDER BY title LIMIT ?",
            (after, CHECK_BATCH),
        ).fetchall()
        if not rows:
            break
        latest = client.latest_revisions(r[0] for r in rows)
        for title, revid, snapshot_ts in rows:
            if title not in latest:
                continue
            if latest[title] is None:
                db.delete_page(conn, title)
            elif _changed(revid, snapshot_ts, latest[title]):
                fetch(conn, client, title)
                refreshed += 1
        checked += len(rows)
        after = rows[-1][0]
        db.set_meta(conn, "check_after", after)
        log.info("puesta al día: %d revisadas, %d actualizadas (vamos en «%s»)",
                 checked, refreshed, after)
    db.set_meta(conn, "check_after", None)
    db.set_meta(conn, "check_started", None)
    db.set_meta(conn, "last_full_check_ts", started)
    db.set_meta(conn, "rc_cursor", iso(check_started - 3600))


def apply_recent_changes(conn, client, cfg, cursor, started):
    """Aplica los cambios desde 'cursor'. Avanza el cursor a medida que procesa."""
    full = cfg["track_all_changes"]
    known = None if full else db.page_titles(conn)

    # Nos quedamos con el último evento de cada título, en orden cronológico.
    last = {}
    for ts, kind, title, target in client.recent_changes(cursor, cfg["skip_bot_edits"]):
        last.pop(title, None)
        last[title] = (ts, kind, target)
    events = sorted(last.items(), key=lambda kv: kv[1][0])
    log.info("cambios recientes desde %s: %d artículos", cursor, len(events))

    for n, (title, (ts, kind, target)) in enumerate(events, 1):
        relevant = full or title in known
        if kind == "delete" or kind == "move":
            db.delete_page(conn, title)
            if kind == "move" and target and relevant:
                fetch(conn, client, target)
        elif relevant:
            fetch(conn, client, title)
        if n % 50 == 0:
            db.set_meta(conn, "rc_cursor", ts)
            log.info("cambios recientes: %d/%d", n, len(events))
    db.set_meta(conn, "rc_cursor", iso(started - 120))


def update(conn, cfg, client=None):
    """Sincroniza la copia local. Lanza WikiUnavailable si no hay acceso a la wiki."""
    client = client or client_from(cfg)
    started = time.time()

    if not cfg["track_all_changes"] and (cfg["seed_categories"] or cfg["seed_titles"]):
        new_titles = discover_titles(client, cfg) - db.page_titles(conn)
        for title in sorted(new_titles):
            fetch(conn, client, title)

    cursor = db.get_meta(conn, "rc_cursor")
    last_full = float(db.get_meta(conn, "last_full_check_ts") or 0)
    if (
        db.get_meta(conn, "check_after") is not None          # puesta al día a medias
        or not cursor
        or started - parse_iso(cursor) > RC_WINDOW_SECONDS
        or (not cfg["track_all_changes"] and started - last_full > FULL_CHECK_SECONDS)
    ):
        catch_up(conn, client, started)
    else:
        apply_recent_changes(conn, client, cfg, cursor, started)

    db.set_meta(conn, "last_sync", iso(started))


def try_update(conn, cfg, client=None):
    """Como update(), pero nunca falla: si no hay red se sigue con lo que hay."""
    try:
        update(conn, cfg, client)
        return True
    except WikiUnavailable as e:
        log.warning("No se pudo actualizar (se usa la copia local): %s", e)
    except Exception:
        log.exception("Error inesperado al actualizar; se usa la copia local")
    return False
