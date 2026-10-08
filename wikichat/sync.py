"""Descarga inicial y actualización incremental. Si falla, la copia local sigue sirviendo."""
import logging
import time
from datetime import datetime, timezone

from . import db
from .wiki_api import WikiClient, WikiUnavailable

log = logging.getLogger("wikichat.sync")

# recentchanges solo guarda ~30 días; por encima de esto se comparan revisiones.
RC_WINDOW_SECONDS = 25 * 86400
FULL_CHECK_SECONDS = 7 * 86400


def client_from(cfg):
    return WikiClient(cfg["api_url"], cfg["user_agent"], cfg["request_delay_seconds"])


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch(conn, client, title=None, pageid=None):
    page = client.page_text(title=title, pageid=pageid)
    if page is None:
        if pageid:
            db.delete_page(conn, pageid)
        return False
    db.upsert_page(conn, *page)
    log.info("actualizado: %s", page[1])
    return True


def discover_titles(client, cfg):
    titles = set(cfg["seed_titles"])
    for cat in cfg["seed_categories"]:
        titles |= client.category_titles(cat, cfg["category_depth"])
    return titles


def update(conn, cfg, client=None):
    """Sincroniza la copia local. Lanza WikiUnavailable si no hay acceso a la wiki."""
    client = client or client_from(cfg)
    started = time.time()
    known = {title: (pageid, revid) for pageid, title, revid in db.tracked_pages(conn)}

    # 1. Páginas nuevas (de las categorías semilla) que aún no tenemos.
    new_titles = discover_titles(client, cfg) - known.keys()
    for title in sorted(new_titles):
        _fetch(conn, client, title=title)

    # 2. Cambios en páginas existentes.
    last_sync = float(db.get_meta(conn, "last_sync_ts", 0))
    last_full = float(db.get_meta(conn, "last_full_check_ts", 0))
    if last_sync and started - last_sync < RC_WINDOW_SECONDS and started - last_full < FULL_CHECK_SECONDS:
        changed = client.recent_changes(_iso(last_sync - 3600))
        if not cfg["track_all_changes"]:
            changed &= known.keys()
        for title in sorted(changed - new_titles):
            _fetch(conn, client, title=title)
    else:
        # Comparación completa de revisiones: detecta ediciones y borrados.
        by_id = {pid: rev for pid, rev in known.values()}
        latest = client.latest_revids(by_id)
        for pid, rev in latest.items():
            if rev is None:
                db.delete_page(conn, pid)
            elif rev != by_id.get(pid):
                _fetch(conn, client, pageid=pid)
        db.set_meta(conn, "last_full_check_ts", started)

    db.set_meta(conn, "last_sync_ts", started)
    db.set_meta(conn, "last_sync", _iso(started))


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
