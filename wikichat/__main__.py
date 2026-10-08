"""Uso: python -m wikichat {sync,serve,ask,search,stats} ..."""
import argparse
import json
import logging
import sys

from . import db, sync
from .config import load_config
from .llm import retrieve_for, stream_answer
from .retrieval import search


def main(argv=None):
    p = argparse.ArgumentParser(prog="wikichat", description="Chat offline sobre la wiki")
    p.add_argument("-c", "--config", default="config.json")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync", help="descarga/actualiza la copia local")
    s = sub.add_parser("serve", help="inicia la interfaz web de chat")
    s.add_argument("--no-update", action="store_true", help="no actualizar en segundo plano")
    a = sub.add_parser("ask", help="pregunta desde la terminal")
    a.add_argument("question", nargs="+")
    q = sub.add_parser("search", help="búsqueda sin LLM (como Kiwix)")
    q.add_argument("query", nargs="+")
    sub.add_parser("stats", help="estado de la copia local")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)

    if args.cmd == "serve":
        from .server import serve
        return serve(cfg, auto_update=not args.no_update)

    conn = db.connect(cfg["db_path"])
    if args.cmd == "sync":
        return 0 if sync.try_update(conn, cfg) else 1
    if args.cmd == "stats":
        print(json.dumps(db.stats(conn), ensure_ascii=False, indent=2))
    elif args.cmd == "search":
        for r in search(conn, " ".join(args.query), cfg["top_k"]):
            print(f"\n## {r['title']} — {r['section']}\n{r['text'][:400]}")
    elif args.cmd == "ask":
        messages = [{"role": "user", "content": " ".join(args.question)}]
        sources = retrieve_for(conn, messages, cfg["top_k"])
        for text in stream_answer(cfg, messages, sources):
            sys.stdout.write(text)
            sys.stdout.flush()
        print("\n\nFuentes:", ", ".join(sorted({s["title"] for s in sources})) or "ninguna")
    return 0


if __name__ == "__main__":
    sys.exit(main())
