"""Uso: python -m wikichat {sync,serve,ask,search,stats} ..."""
import argparse
import json
import logging
import os
import sys
import time

from . import db, sync, vectors
from .config import load_config
from .llm import retrieve, stream_answer


def main(argv=None):
    p = argparse.ArgumentParser(prog="wikichat", description="Chat offline sobre la wiki")
    p.add_argument("-c", "--config", default="config.json")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("setup", help="detecta el hardware, prepara Ollama y los modelos, y configura")
    st.add_argument("-y", "--yes", action="store_true", help="no preguntar antes de descargar")
    st.add_argument("--no-benchmark", action="store_true", help="no medir la velocidad")
    sub.add_parser("sync", help="descarga/actualiza la copia local")
    z = sub.add_parser("import-zim", help="importa un .zim de Kiwix como base inicial")
    z.add_argument("path")
    z.add_argument("--workers", type=int, help="procesos para convertir HTML (por defecto: núcleos - 1)")
    s = sub.add_parser("serve", help="inicia la interfaz web de chat")
    s.add_argument("--no-update", action="store_true", help="no actualizar en segundo plano")
    a = sub.add_parser("ask", help="pregunta desde la terminal")
    a.add_argument("question", nargs="+")
    q = sub.add_parser("search", help="búsqueda sin LLM (como Kiwix)")
    q.add_argument("query", nargs="+")
    e = sub.add_parser("embed", help="calcula los embeddings pendientes (búsqueda semántica)")
    e.add_argument("--max", type=int, help="máximo de artículos en esta ejecución")
    ev = sub.add_parser("eval", help="mide la calidad de las respuestas con un conjunto de preguntas")
    ev.add_argument("archivo", nargs="?", default="evals/general.json",
                    help="conjunto de preguntas (por defecto evals/general.json)")
    ev.add_argument("--limite", type=int, help="evaluar solo las primeras N preguntas")
    ev.add_argument("--instrucciones", help="archivo de texto con otras instrucciones para el modelo")
    ev.add_argument("--temperatura", type=float, help="temperatura del modelo de chat")
    ev.add_argument("--salida", help="dónde guardar el detalle (por defecto data/evals/<fecha>.json)")
    sub.add_parser("stats", help="estado de la copia local")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)

    if args.cmd == "setup":
        from .setup import run
        run(args.config, yes=args.yes, benchmark=not args.no_benchmark)
        return 0
    if args.cmd == "serve":
        from .server import serve
        return serve(cfg, auto_update=not args.no_update)

    conn = db.connect(cfg["db_path"])
    if args.cmd in ("ask", "search", "embed", "eval"):
        # Si setup dejó Ollama administrado, se inicia para este comando y se detiene al salir.
        from .ollama_manager import ensure_running
        ensure_running(cfg)
    if args.cmd == "sync":
        return 0 if sync.try_update(conn, cfg) else 1
    if args.cmd == "import-zim":
        from .zim_import import import_zim
        import_zim(conn, args.path, args.workers)
    elif args.cmd == "stats":
        print(json.dumps(db.stats(conn), ensure_ascii=False, indent=2))
    elif args.cmd == "eval":
        from . import evaluate
        if args.instrucciones:
            with open(args.instrucciones, encoding="utf-8") as f:
                cfg["system_prompt"] = f.read().strip()
        if args.temperatura is not None:
            cfg["temperature"] = args.temperatura
        out = args.salida or os.path.join(os.path.dirname(cfg["db_path"]) or ".", "evals",
                                          time.strftime("%Y%m%d-%H%M%S") + ".json")
        evaluate.run(conn, cfg, args.archivo, args.limite, out)
    elif args.cmd == "embed":
        if not vectors.enabled(cfg):
            raise SystemExit("Configura embed_model en config.json e instala numpy")
        n = vectors.embed_pending(conn, cfg, max_pages=args.max)
        print(f"{n} artículos vectorizados")
    elif args.cmd == "search":
        for r in retrieve(conn, cfg, " ".join(args.query), _index(conn, cfg)):
            print(f"\n## {r['title']} — {r['section']}\n{r['text'][:400]}")
    elif args.cmd == "ask":
        question = " ".join(args.question)
        messages = [{"role": "user", "content": question}]
        sources = retrieve(conn, cfg, question, _index(conn, cfg))
        for text in stream_answer(cfg, messages, sources):
            sys.stdout.write(text)
            sys.stdout.flush()
        print("\n\nFuentes:", ", ".join(sorted({s["title"] for s in sources})) or "ninguna")
    return 0


def _index(conn, cfg):
    if not vectors.enabled(cfg):
        return None
    index = vectors.VectorIndex(cfg["embed_dims"])
    index.refresh(conn)
    return index


if __name__ == "__main__":
    sys.exit(main())
