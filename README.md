# AtelierWikiGT — Chat offline sobre Wikipedia en español

Un "Kiwix con chat": guarda una copia local de **toda la Wikipedia en español** y responde
preguntas con un LLM que corre en tu máquina. Las respuestas salen **solo de la copia
local**; Internet se usa únicamente para mantenerla al día, y si no hay conexión no pasa
nada.

```
 .zim de Kiwix ──(importación inicial, una vez)──┐
                                                 ▼
 API de Wikipedia ──(solo lo que cambió)──▶ SQLite + FTS5 (data/wiki.db)
                                                 │ búsqueda BM25
 Navegador / terminal ──pregunta──▶ servidor ────┤
                                        │        ▼
                                        └──▶ Ollama (LLM local) ──▶ respuesta con fuentes
```

## Cómo se mantiene actualizada sin volver a bajar el .zim

El `.zim` se usa **una sola vez** como punto de partida. Después, el sistema elige solo:

- **Cambios recientes** (lo normal): si la copia tiene menos de ~29 días, pide a Wikipedia
  la lista de artículos editados, creados, borrados o movidos desde la última vez y baja
  solo esos (en es.wikipedia son unos miles al día sin contar bots).
- **Puesta al día**: si la copia es más vieja (p. ej. un `.zim` de hace dos meses, o estuviste
  mucho tiempo sin Internet), compara las revisiones de todos los artículos en lotes de 50
  y descarga los que cambiaron. Es reanudable: si se corta, sigue donde quedó.

**Si la actualización falla** (sin Internet, Wikipedia limita las peticiones, etc.) se
registra un aviso y todo sigue funcionando con la copia local.

## Requisitos

| | |
|---|---|
| Python | 3.10+ |
| `libzim` | solo para importar: `pip install libzim` |
| Disco | ~11 GB del `.zim` (se puede borrar después) + base de tamaño similar (estimado: 12–18 GB) |
| [Ollama](https://ollama.com) | para el chat: `ollama pull qwen2.5:7b` (con poca RAM: `qwen2.5:3b`) |

Sin Ollama todo funciona igual, pero en vez de una respuesta redactada verás los fragmentos
encontrados (como una búsqueda en Kiwix).

## Puesta en marcha

```bash
pip install libzim
cp config.example.json config.json

# 1. Descarga el .zim más reciente "all_nopic" (texto completo, sin imágenes):
#    https://download.kiwix.org/zim/wikipedia/  →  wikipedia_es_all_nopic_AAAA-MM.zim
# 2. Impórtalo (estimado 1–2 h con 4 núcleos; si se interrumpe, repite el comando y continúa):
python -m wikichat import-zim wikipedia_es_all_nopic_2026-08.zim

# 3. Inicia el chat; se actualiza solo en segundo plano cada 6 h:
python -m wikichat serve        # http://127.0.0.1:8080
```

La primera actualización tras importar un `.zim` de más de un mes hace la **puesta al
día**: unas 40 000 consultas para revisar ~2 M artículos más la descarga de los que
cambiaron. Puede tardar horas, pero corre en segundo plano, se reanuda sola y el chat
funciona mientras tanto. Cuanto más reciente sea el `.zim`, menos trabajo.

Otros comandos:

```bash
python -m wikichat sync                    # actualizar ahora (útil con cron / Programador de tareas)
python -m wikichat ask "¿Cuándo se fundó Antigua Guatemala?"
python -m wikichat search lago atitlan     # búsqueda tipo Kiwix, sin LLM
python -m wikichat stats
python -m wikichat serve --no-update       # modo 100 % offline
```

## Configuración (`config.json`)

| Clave | Para qué sirve |
|---|---|
| `api_url` | API de la wiki (por defecto Wikipedia en español). |
| `user_agent` | Wikimedia exige un User-Agent identificable; sin él responde 429. |
| `track_all_changes` | `true` (Wikipedia completa): sigue cualquier artículo que cambie, incluidos los nuevos. `false`: solo los que ya tienes y los de `seed_categories` / `seed_titles`. |
| `skip_bot_edits` | Ignora ediciones de bots (casi siempre mantenimiento), reduce mucho las descargas. |
| `seed_categories`, `category_depth`, `seed_titles` | Para copiar solo un tema en vez de toda la wiki (con `track_all_changes: false`, sin `.zim`). |
| `update_interval_hours` | Cada cuánto actualiza el servidor en segundo plano. |
| `request_delay_seconds` | Pausa entre peticiones a Wikipedia (sé amable con sus servidores). |
| `chat_model`, `ollama_url`, `top_k` | Modelo local y cuántos fragmentos se le pasan como contexto. |

## Cómo se arma una respuesta

1. Los artículos se guardan como texto plano dividido por secciones (~1 500 caracteres);
   se omiten referencias, enlaces externos, cajas de navegación, etc. Las fórmulas se
   guardan como TeX.
2. La pregunta se reduce a palabras clave (sin acentos ni palabras vacías) y SQLite FTS5
   busca con BM25 los fragmentos más relevantes, con más peso en título y sección. Primero
   exige todas las palabras y, si no alcanza, acepta cualquiera.
3. Esos fragmentos van al LLM con la instrucción de responder solo con ellos y citar los
   artículos; la interfaz muestra las fuentes usadas.

## Pruebas

```bash
python -m unittest -v      # la prueba del importador se omite si no está libzim
```

## Límites conocidos

- Los artículos creados entre la fecha del `.zim` y hace 29 días no se detectan en la puesta
  al día (Wikipedia solo guarda 29–30 días de cambios recientes). Usa el `.zim` más
  reciente o reimporta uno nuevo de vez en cuando para recogerlos.
- Con `skip_bot_edits` las correcciones hechas por bots no se descargan hasta la siguiente
  edición humana del artículo.
- La búsqueda es por palabras, no entiende sinónimos; el siguiente paso natural es añadir
  embeddings locales para búsqueda híbrida.
- El contenido de Wikipedia es CC BY-SA: si publicas respuestas, cita los artículos.
