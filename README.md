# AtelierWikiGT — Chat offline sobre la wiki

Un "Kiwix con chat": guarda una copia local de artículos de Wikipedia (o de cualquier
wiki MediaWiki) y responde preguntas con un LLM que corre en tu máquina. Las respuestas
salen **solo de la copia local**, nunca de Internet en el momento de preguntar.

```
 Wikipedia API ──(sync incremental, si hay red)──▶ SQLite + FTS5 (data/wiki.db)
                                                         │ búsqueda BM25
 Navegador / terminal ──pregunta──▶ servidor local ──────┤
                                        │                ▼
                                        └──▶ Ollama (LLM local) ──▶ respuesta con fuentes
```

## Por qué no un .zim

Un `.zim` es una foto fija: para actualizar hay que bajar el archivo completo otra vez.
Aquí la primera descarga trae los artículos que te interesan y luego **solo se bajan los
que cambiaron**:

- Si la última sincronización fue hace menos de ~25 días, se consulta `recentchanges`
  (una sola lista de títulos editados) y se descargan únicamente esos.
- Una vez por semana, o si pasó mucho tiempo sin conexión, se comparan los números de
  revisión de todas las páginas en lotes de 50, lo que detecta ediciones y borrados.
- Las categorías semilla se vuelven a recorrer para detectar artículos nuevos.

**Si la actualización falla** (sin Internet, Wikipedia limita las peticiones, etc.) se
registra un aviso y todo sigue funcionando con la copia local.

## Requisitos

- Python 3.10+ (solo librería estándar, sin `pip install`).
- [Ollama](https://ollama.com) para el chat, con un modelo descargado, por ejemplo:
  `ollama pull qwen2.5:7b` (buen español; con poca RAM usa `qwen2.5:3b` o `llama3.2:3b`).
  Sin Ollama el sistema funciona igual, pero muestra los fragmentos encontrados en vez de
  una respuesta redactada.

## Uso

```bash
cp config.example.json config.json   # define qué parte de la wiki copiar
python -m wikichat sync              # primera descarga (se puede interrumpir y repetir)
python -m wikichat serve             # chat en http://127.0.0.1:8080, se actualiza solo
```

Otros comandos:

```bash
python -m wikichat ask "¿Cuándo se fundó Antigua Guatemala?"
python -m wikichat search lago atitlan    # búsqueda tipo Kiwix, sin LLM
python -m wikichat stats
python -m wikichat serve --no-update      # modo 100 % offline
```

Para actualizar sin tener el servidor abierto, programa `python -m wikichat sync`
con cron o el Programador de tareas de Windows.

## Configuración (`config.json`)

| Clave | Para qué sirve |
|---|---|
| `api_url` | API de la wiki (por defecto Wikipedia en español). |
| `user_agent` | Wikimedia exige un User-Agent identificable; sin él responde 429. |
| `seed_categories`, `category_depth` | Categorías a copiar y cuántos niveles de subcategorías recorrer. |
| `seed_titles` | Artículos sueltos a incluir. |
| `track_all_changes` | `true` descarga cualquier artículo editado en la wiki, no solo los que ya tienes (crece sin límite). |
| `update_interval_hours` | Cada cuánto actualiza el servidor en segundo plano. |
| `chat_model`, `ollama_url`, `top_k` | Modelo local y cuántos fragmentos se le pasan como contexto. |

## Cómo se arma una respuesta

1. La pregunta se reduce a palabras clave (sin acentos ni palabras vacías).
2. SQLite FTS5 busca con BM25 los fragmentos más relevantes, dando más peso al título y
   a la sección. Los artículos se dividen por secciones de ~1500 caracteres.
3. Esos fragmentos van al LLM con la instrucción de responder solo con ellos y citar los
   artículos; la interfaz muestra las fuentes usadas.

## Pruebas

```bash
python -m unittest -v
```

## Límites y siguientes pasos

- Pensado para un subconjunto (miles o decenas de miles de artículos). Para la Wikipedia
  completa en español (~2 M artículos) la descarga inicial por API tardaría días: conviene
  importar primero un dump o `.zim` y usar este sistema solo para las actualizaciones.
- La búsqueda es por palabras; si hace falta entender sinónimos se pueden añadir
  embeddings locales (p. ej. `nomic-embed-text` en Ollama) para búsqueda híbrida.
- El contenido de Wikipedia es CC BY-SA: si publicas respuestas, cita los artículos.
