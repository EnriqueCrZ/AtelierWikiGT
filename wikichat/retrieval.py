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


# Constante de Reciprocal Rank Fusion. Más baja que el 60 habitual para que los primeros
# puestos de cada lista dominen: queremos lo mejor de ambas búsquedas en solo top_k fragmentos.
RRF_K = 10


def _fts_query(terms, op):
    # Prefijos (term*) para tolerar plurales y conjugaciones simples.
    return f" {op} ".join(f'"{t}"*' if len(t) > 4 else f'"{t}"' for t in terms)


_SELECT = """SELECT c.id, c.page, c.title, c.section, c.text, bm25(chunks_fts, 8.0, 3.0, 1.0) AS score
             FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
             WHERE chunks_fts MATCH ?"""


def _row(r):
    return {"id": r[0], "page": r[1], "title": r[2], "section": r[3], "text": r[4]}


def keyword_search(conn, terms, limit):
    """Fragmentos por BM25. Primero exige todas las palabras (rápido y preciso incluso con
    millones de fragmentos); si no alcanza, acepta cualquiera de ellas.

    Devuelve (fragmentos, estricta): estricta indica que todos coinciden con todas las palabras.
    """
    rows, strict = [], True
    for op in ("AND", "OR") if len(terms) > 1 else ("AND",):
        found = conn.execute(_SELECT + " ORDER BY score LIMIT ?",
                             (_fts_query(terms, op), limit)).fetchall()
        if op == "AND":
            rows = found
        else:
            # Los que tienen todas las palabras van primero; el resto completa.
            seen = {r[0] for r in rows}
            rows += [r for r in found if r[0] not in seen][: limit - len(rows)]
            strict = False
        if len(rows) >= limit:
            break
    return [_row(r) for r in rows], strict


def best_chunk(conn, page, terms):
    """El fragmento de un artículo que mejor coincide con la consulta (o su introducción)."""
    if terms:
        row = conn.execute(_SELECT + " AND c.page = ? ORDER BY score LIMIT 1",
                           (_fts_query(terms, "OR"), page)).fetchone()
        if row:
            return _row(row)
    row = conn.execute(
        "SELECT id, page, title, section, text FROM chunks WHERE page = ? ORDER BY id LIMIT 1",
        (page,)).fetchone()
    return _row(row) if row else None


def exact_title_intro(conn, terms):
    """Introducción del artículo cuyo título son exactamente las palabras clave
    ("¿Qué es el oro?" → Oro, "tabla periódica de los elementos" → ese artículo)."""
    if not terms or len(terms) > 6:
        return None
    wanted = set(terms)
    rows = conn.execute(
        """SELECT DISTINCT c.page, c.title FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
           WHERE chunks_fts MATCH ? LIMIT 200""",
        ("title : (" + " AND ".join(f'"{t}"' for t in terms) + ")",),
    ).fetchall()
    for page, title in rows:
        if set(keywords(title)) == wanted:
            return best_chunk(conn, page, [])
    return None


def _diverse(chunks, k):
    """Evita que un solo artículo acapare todos los resultados."""
    per_title, out = {}, []
    for c in chunks:
        if per_title.get(c["title"], 0) >= 2:
            continue
        per_title[c["title"]] = per_title.get(c["title"], 0) + 1
        out.append(c)
        if len(out) == k:
            break
    return out


def search(conn, query, k=6, semantic_pages=None):
    """Fragmentos más relevantes para la consulta.

    semantic_pages: [(page_id, similitud)] de la búsqueda semántica, si está disponible.
    Las dos listas se combinan con Reciprocal Rank Fusion: un fragmento que aparece arriba
    en cualquiera de ellas (o en ambas) sube en el resultado final.
    """
    terms = keywords(query)
    keyword, strict = keyword_search(conn, terms, k * 3) if terms else ([], True)
    exact = exact_title_intro(conn, terms)
    first = [exact] if exact else []
    if not semantic_pages:
        return _diverse(first + [c for c in keyword if not exact or c["id"] != exact["id"]], k)

    # Si no hubo fragmentos con todas las palabras, la evidencia por palabras es débil
    # (preguntas en lenguaje natural): la búsqueda semántica pesa más.
    weight = 1.0 if strict else 0.8
    rrf, chunks = {}, {}
    for rank, c in enumerate(keyword):
        rrf[c["id"]] = rrf.get(c["id"], 0) + weight / (RRF_K + rank)
        chunks[c["id"]] = c
    for rank, (page, _sim) in enumerate(semantic_pages):
        c = best_chunk(conn, page, terms)
        if c:
            rrf[c["id"]] = rrf.get(c["id"], 0) + 1 / (RRF_K + rank)
            chunks[c["id"]] = c
    ranked = sorted(chunks.values(), key=lambda c: -rrf[c["id"]])
    return _diverse(first + [c for c in ranked if not exact or c["id"] != exact["id"]], k)
