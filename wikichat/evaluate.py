"""`python -m wikichat eval`: mide la calidad de las respuestas con un conjunto de preguntas.

Califica con reglas fijas (sin otro modelo como juez), así los resultados son reproducibles
y se pueden comparar entre versiones de las instrucciones, modelos o ajustes:

* acierto: la respuesta contiene todos los datos esperados; en las preguntas que la wiki no
  puede responder, acierta si dice que no encontró la información.
* búsqueda: alguno de los artículos esperados está entre las fuentes recuperadas.
* inventa: responde algo a una pregunta sin respuesta en la wiki.
* se niega de más: dice que no lo sabe cuando el dato sí estaba en las fuentes.
* cita: menciona al menos un artículo entre corchetes.
* habla de fragmentos: menciona la maquinaria ("según los fragmentos…") en vez de responder.

Cada pregunta puede traer "no_debe_decir": patrones que delatan que el modelo usó su propio
conocimiento (p. ej. la capital de Mongolia cuando la wiki local no la tiene). Si aparecen,
la respuesta cuenta como inventada aunque luego diga que no encontró la información.
"""
import json
import os
import re
import time
import unicodedata

from . import vectors
from .llm import retrieve, stream_answer

REFUSAL = re.compile(
    r"no (lo )?(encontre|encuentro|tengo|hay|dispongo|puedo (responder|dar|proporcionar)|se (menciona|encuentra|especifica|indica|proporciona)"
    r"|aparece|contiene|incluye|esta (disponible|en))"
    r"|(fragmentos|wiki|informacion) (proporcionad[oa]s? )?no (contiene|incluye|menciona|tiene|dice|habla)"
    r"|no cuento con|sin informacion|fuera del alcance|desconozco"
)


def norm(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def refused(answer):
    return bool(REFUSAL.search(norm(answer)))


def score(item, answer, source_titles):
    a = norm(answer)
    result = {"cita": bool(re.search(r"\[[^\]]+\]", answer)), "se_niega": refused(answer),
              "habla_de_fragmentos": bool(re.search(
                  r"fragmento|\btextos? (de la wiki|proporcionad|del articulo)|segun (la wiki|los textos)"
                  r"|se deriva de los textos|que se me proporciona", a)),
              "palabras": len(answer.split())}
    leaked = any(re.search(p, a) for p in item.get("no_debe_decir", []))
    if item.get("sin_respuesta"):
        result["inventa"] = leaked or not result["se_niega"]
        result["acierto"] = not result["inventa"]
        return result
    groups = item.get("datos", [])
    found = [any(re.search(p, a) for p in group) for group in groups]
    result["datos"] = f"{sum(found)}/{len(groups)}"
    result["acierto"] = all(found) and not leaked
    expected = item.get("articulos")
    if expected:
        result["busqueda"] = any(t in source_titles for t in expected)
    result["se_niega_de_mas"] = result["se_niega"] and not result["acierto"]
    return result


def run(conn, cfg, path, limit=None, out=None, progress=print):
    with open(path, encoding="utf-8") as f:
        items = json.load(f)["preguntas"]
    if limit:
        items = items[:limit]
    index = None
    if vectors.enabled(cfg):
        index = vectors.VectorIndex(cfg["embed_dims"])
        index.refresh(conn)
    present = lambda t: conn.execute("SELECT 1 FROM pages WHERE title=?", (t,)).fetchone()

    results = []
    for n, item in enumerate(items, 1):
        if item.get("articulos") and not any(present(t) for t in item["articulos"]):
            progress(f"[{n:2}/{len(items)}] omitida (el artículo no está en esta copia): {item['pregunta']}")
            continue
        t0 = time.time()
        sources = retrieve(conn, cfg, item["pregunta"], index)
        first, parts = None, []
        for text in stream_answer(cfg, [{"role": "user", "content": item["pregunta"]}], sources):
            first = first or time.time() - t0
            parts.append(text)
        answer = "".join(parts).strip()
        titles = list(dict.fromkeys(s["title"] for s in sources))
        r = score(item, answer, titles)
        r.update(pregunta=item["pregunta"], respuesta=answer, fuentes=titles,
                 primera_palabra_s=round(first or 0, 1), total_s=round(time.time() - t0, 1))
        results.append(r)
        mark = "✓" if r["acierto"] else "✗"
        extra = " (inventa)" if r.get("inventa") else " (se niega de más)" if r.get("se_niega_de_mas") else ""
        progress(f"[{n:2}/{len(items)}] {mark}{extra} {item['pregunta']}  ({r['total_s']:.0f} s)")

    summary = summarize(results)
    progress("\n" + format_summary(summary))
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"fecha": time.strftime("%Y-%m-%d %H:%M"), "conjunto": os.path.basename(path),
                       "modelo": cfg["chat_model"], "temperatura": cfg.get("temperature"),
                       "instrucciones": cfg.get("system_prompt") or "(por defecto)",
                       "resumen": summary, "resultados": results}, f, ensure_ascii=False, indent=2)
        progress(f"Detalle guardado en {out}")
    return summary


def summarize(results):
    pct = lambda xs: round(100 * sum(xs) / len(xs)) if xs else None
    answerable = [r for r in results if "datos" in r]
    unanswerable = [r for r in results if "inventa" in r]
    searched = [r["busqueda"] for r in answerable if "busqueda" in r]
    return {
        "preguntas": len(results),
        "aciertos_pct": pct([r["acierto"] for r in results]),
        "con_respuesta_pct": pct([r["acierto"] for r in answerable]),
        "sin_respuesta_pct": pct([r["acierto"] for r in unanswerable]),
        "busqueda_pct": pct(searched),
        "inventa": sum(r["inventa"] for r in unanswerable),
        "se_niega_de_mas": sum(r["se_niega_de_mas"] for r in answerable),
        "cita_pct": pct([r["cita"] for r in answerable]),
        "habla_de_fragmentos_pct": pct([r["habla_de_fragmentos"] for r in results]),
        "palabras_promedio": round(sum(r["palabras"] for r in answerable) / len(answerable)) if answerable else None,
        "segundos_promedio": round(sum(r["total_s"] for r in results) / len(results), 1) if results else None,
    }


def format_summary(s):
    if not s["preguntas"]:
        return "No se evaluó ninguna pregunta."
    s = {k: ("—" if v is None else v) for k, v in s.items()}
    return (f"Aciertos: {s['aciertos_pct']}% de {s['preguntas']} preguntas\n"
            f"  con respuesta en la wiki: {s['con_respuesta_pct']}%  ·  sin respuesta (debe decir que no sabe): "
            f"{s['sin_respuesta_pct']}%\n"
            f"  la búsqueda encontró el artículo: {s['busqueda_pct']}%  ·  cita fuentes: {s['cita_pct']}%\n"
            f"  habla de «fragmentos»: {s['habla_de_fragmentos_pct']}%  ·  "
            f"{s['palabras_promedio']} palabras por respuesta\n"
            f"  inventó en {s['inventa']}  ·  se negó de más en {s['se_niega_de_mas']}  ·  "
            f"{s['segundos_promedio']} s por pregunta")
