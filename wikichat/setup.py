"""`python -m wikichat setup`: detecta el hardware, prepara Ollama y los modelos, mide la
velocidad real y escribe config.json con los ajustes adecuados para este equipo."""
import json
import os
import shutil
import sqlite3
import time
import urllib.request

from . import hardware, ollama_manager as om
from .config import DEFAULTS

GB = hardware.GB
WIKI_ARTICLES = 2_000_000          # Wikipedia en español, para estimar tiempos
CHARS_PER_TOKEN = 4
ANSWER_TOKENS = 200                # largo típico de una respuesta
MAX_ANSWER_SECONDS = 90            # por encima, se prueba un modelo más chico
MAX_REWRITE_SECONDS = 6            # reescribir una pregunta de seguimiento
MIN_FREE_DISK = 40 * GB            # .zim nopic (11) + base (~15) + modelos (~6) + margen

SAMPLE = [
    "Tikal es uno de los mayores yacimientos arqueológicos y centros urbanos de la civilización "
    "maya precolombina. Está situado en el municipio de Flores, en el departamento de Petén.",
    "El quetzal es un ave de la familia de los trogónidos que habita en los bosques nubosos de "
    "Mesoamérica; es el ave nacional de Guatemala y da nombre a su moneda.",
    "El lago de Atitlán es un lago endorreico de origen volcánico, rodeado por los volcanes "
    "Atitlán, Tolimán y San Pedro, y por pueblos de origen maya.",
    "La fotosíntesis es el proceso por el cual las plantas convierten la luz en energía química, "
    "liberando oxígeno a partir del agua y fijando dióxido de carbono.",
]


def _post(url, path, payload, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def bench_embed(url, model, n=32, rounds=2):
    """Artículos por segundo. Una tanda completa de calentamiento (justo después de descargar
    los modelos la primera medición sale hasta 2 veces más lenta) y la mejor de `rounds`."""
    texts = [f"title: Prueba {i} | text: {(SAMPLE[i % len(SAMPLE)] + ' ') * 3}"[:420] for i in range(n)]
    _post(url, "/api/embed", {"model": model, "input": texts})
    best = 0.0
    for _ in range(rounds):
        t = time.time()
        _post(url, "/api/embed", {"model": model, "input": texts})
        best = max(best, n / (time.time() - t))
    return best


def bench_chat(url, model, top_k):
    """(tokens/s leyendo el contexto, tokens/s escribiendo) con un contexto como el real."""
    context = " ".join(SAMPLE * 40)[: top_k * 1500]
    _post(url, "/api/chat", {"model": model, "stream": False, "think": False,
                             "messages": [{"role": "user", "content": "Hola"}],
                             "options": {"num_predict": 1}})  # carga el modelo
    r = _post(url, "/api/chat", {
        "model": model, "stream": False, "think": False, "options": {"num_predict": 80},
        "messages": [{"role": "system", "content": "Responde en español usando solo los fragmentos."},
                     {"role": "user", "content": f"Fragmentos:\n{context}\n\nPregunta: ¿qué es Tikal?"}]})
    prefill = r["prompt_eval_count"] / max(r["prompt_eval_duration"] / 1e9, 1e-6)
    gen = r["eval_count"] / max(r["eval_duration"] / 1e9, 1e-6)
    return prefill, gen


def answer_seconds(prefill, gen, top_k):
    return top_k * 1500 / CHARS_PER_TOKEN / prefill + ANSWER_TOKENS / gen


def _confirm(question, yes):
    if yes:
        print(f"{question} sí")
        return True
    return input(f"{question} [S/n] ").strip().lower() in ("", "s", "si", "sí", "y", "yes")


def _existing_dims(db_path):
    """Dimensiones de los vectores ya calculados (no se cambian: habría que rehacerlos)."""
    if not os.path.exists(db_path):
        return None
    try:
        row = sqlite3.connect(db_path).execute("SELECT length(vec) FROM page_vectors LIMIT 1").fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def run(config_path="config.json", yes=False, benchmark=True):
    raw = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
    cfg = {**DEFAULTS, **raw}
    data_dir = os.path.dirname(cfg["db_path"]) or "."
    os.makedirs(data_dir, exist_ok=True)
    # Ollama administrado junto a los datos (salvo que se haya indicado otra carpeta).
    ollama_dir = raw.get("ollama_dir") or os.path.join(data_dir, "ollama")

    print("== Hardware ==")
    hw = hardware.detect(data_dir)
    print(hardware.describe(hw))
    if hw["disk_free"] < MIN_FREE_DISK:
        print(f"⚠️  Hay {hw['disk_free'] / GB:.0f} GB libres; para toda la Wikipedia conviene tener "
              f"al menos {MIN_FREE_DISK / GB:.0f} GB (más 38 GB si usas el .zim con imágenes).")
    profile = hardware.choose_profile(hw)
    print(f"\nPerfil inicial: {'con GPU' if profile['name'] == 'gpu' else 'solo CPU'} · "
          f"chat {profile['chat_model']} · embeddings {hardware.EMBED_MODEL[0]}")

    print("\n== Ollama ==")
    manager, managed = None, False
    if om.version(om.SYSTEM_URL):
        url = om.SYSTEM_URL
        print(f"Se usará el Ollama que ya está corriendo en {url} (versión {om.version(url)}).")
    else:
        binary = om.managed_binary(ollama_dir) or shutil.which("ollama")
        if not binary:
            if not _confirm("Ollama no está instalado. ¿Descargarlo en "
                            f"{ollama_dir}? (~1,4 GB de descarga; sin GPU ocupa ~60 MB)", yes):
                raise SystemExit("Sin Ollama no se pueden preparar los modelos. Instálalo y repite.")
            binary = om.install(ollama_dir, hw["arch"], hw["gpus"])
        manager = om.ManagedOllama(binary, ollama_dir)
        url = manager.start()
        managed = True
        print(f"Ollama administrado en {url} (se inicia y se detiene junto con el chat).")

    try:
        print("\n== Modelos ==")
        embed_model = hardware.EMBED_MODEL[0]
        chat_model = profile["chat_model"]
        gb = lambda m: f"{hardware.DOWNLOAD_GB.get(m, 2):.1f}".replace(".", ",")
        if not _confirm(f"¿Descargar {embed_model} (~{gb(embed_model)} GB) y {chat_model} "
                        f"(~{gb(chat_model)} GB) si faltan?", yes):
            raise SystemExit("Cancelado.")
        om.pull(url, embed_model)
        om.pull(url, chat_model)

        bench = {}
        if benchmark:
            print("\n== Midiendo la velocidad real ==")
            bench["embed_per_s"] = bench_embed(url, embed_model)
            while True:
                prefill, gen = bench_chat(url, chat_model, profile["top_k"])
                secs = answer_seconds(prefill, gen, profile["top_k"])
                print(f"{chat_model}: lee {prefill:.0f} tokens/s, escribe {gen:.1f} tokens/s "
                      f"→ ~{secs:.0f} s por respuesta")
                smaller = hardware.smaller_chat_model(chat_model)
                if secs <= MAX_ANSWER_SECONDS or smaller == chat_model:
                    break
                print(f"Demasiado lento; se prueba {smaller}.")
                chat_model = smaller
                om.pull(url, chat_model)
            bench.update(chat_model=chat_model, prefill_tps=prefill, gen_tps=gen, answer_s=secs)
            rewrite_s = 400 / prefill + 30 / gen
            profile["rewrite_followups"] = rewrite_s <= MAX_REWRITE_SECONDS
            print(f"Embeddings: {bench['embed_per_s']:.1f} artículos/s → toda la Wikipedia en español "
                  f"en ~{WIKI_ARTICLES / bench['embed_per_s'] / 3600:.0f} h (en segundo plano).")
            print(f"Reescribir preguntas de seguimiento tardaría ~{rewrite_s:.0f} s: "
                  f"{'activado' if profile['rewrite_followups'] else 'desactivado'}.")
    finally:
        if manager:
            manager.stop()

    dims = profile["embed_dims"]
    existing = _existing_dims(cfg["db_path"])
    if existing and existing != dims:
        print(f"Se mantienen {existing} dimensiones porque ya hay vectores calculados con ellas.")
        dims = existing

    raw.update({
        "llm_backend": "ollama", "llm_url": url, "manage_ollama": managed, "ollama_dir": ollama_dir,
        "chat_model": chat_model, "embed_model": embed_model, "embed_dims": dims,
        "top_k": profile["top_k"], "rewrite_followups": profile["rewrite_followups"],
    })
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
        f.write("\n")
    report = {"hardware": hw, "profile": profile, "benchmark": bench, "date": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(data_dir, "hardware.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Configuración guardada en {config_path}.")
    print("Siguientes pasos:")
    if not os.path.exists(cfg["db_path"]):
        print("  python -m wikichat import-zim wikipedia_es_all_nopic_AAAA-MM.zim")
    print("  python -m wikichat serve")
    return raw
