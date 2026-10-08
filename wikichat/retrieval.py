"""Búsqueda de fragmentos relevantes (BM25 sobre FTS5)."""
import re
import unicodedata

STOPWORDS = set("""
a al algo algun alguna algunas alguno algunos ante antes aqui asi cada como con contra cual cuales
cuando cuanto de del desde donde dos el ella ellas ellos en entre era eran es esa esas ese eso esos
esta estaba estan estas este esto estos fue fueron ha habia han hay hasta la las le les lo los mas me
mi muy nada ni no nos o otra otro para pero poco por porque que quien quienes se sea segun ser si sin
sobre son su sus tambien tan te tiene tienen todo todos tu un una uno unos y ya yo dime cuentame
explica explicame sabes saber quiero puedes hablame acerca informacion the of and is what who
""".split())


def _normalize(word):
    word = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in word if not unicodedata.combining(c))


def keywords(text):
    words = re.findall(r"\w+", text, re.UNICODE)
    seen, out = set(), []
    for w in words:
        n = _normalize(w)
        if len(n) > 1 and n not in STOPWORDS and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def search(conn, query, k=6):
    """Devuelve [{title, section, text, score}] ordenados por relevancia."""
    terms = keywords(query)
    if not terms:
        return []
    # Prefijos (term*) para tolerar plurales y conjugaciones simples.
    fts = " OR ".join(f'"{t}"*' if len(t) > 3 else f'"{t}"' for t in terms)
    rows = conn.execute(
        """SELECT title, section, text, bm25(chunks, 8.0, 3.0, 1.0) AS score
           FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?""",
        (fts, k * 3),
    ).fetchall()
    # Evita que un solo artículo acapare todos los resultados.
    per_title, results = {}, []
    for title, section, text, score in rows:
        if per_title.get(title, 0) >= 2:
            continue
        per_title[title] = per_title.get(title, 0) + 1
        results.append({"title": title, "section": section, "text": text, "score": score})
        if len(results) == k:
            break
    return results
