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

## ¿Qué necesito instalar?

| Pieza | Para qué | ¿Obligatoria? |
|---|---|---|
| Python 3.10+ | todo el sistema | sí |
| `pip install -r requirements.txt` | `libzim` (importar el `.zim` y mostrar imágenes) y `numpy` (búsqueda semántica) | sí para importar; numpy opcional |
| Un servidor de modelos local | redactar respuestas y búsqueda semántica | no: sin él funciona como un buscador tipo Kiwix |

**Servidor de modelos:** el más sencillo es [Ollama](https://ollama.com), pero sirve cualquiera compatible con la
API de OpenAI (llama.cpp `llama-server`, LM Studio, vLLM, Jan). Se necesitan dos modelos:

```bash
ollama pull embeddinggemma   # embeddings, 621 MB: búsqueda semántica
ollama pull qwen2.5:7b       # chat, 4,7 GB (con poca RAM: qwen2.5:3b, 1,9 GB)
```

Con llama.cpp, LM Studio u otro servidor compatible con OpenAI:

```json
"llm_backend": "openai", "llm_url": "http://localhost:8080/v1"
```

Si un modelo no responde, el sistema no se cae: sin modelo de embeddings usa solo la búsqueda
por palabras, y sin modelo de chat muestra los fragmentos encontrados.

### Cuánto pide cada parte (medido)

Medido en un servidor de 4 núcleos (Xeon 2,8 GHz, AVX-512, **sin GPU**), con artículos reales de
Wikipedia en español. Un PC con GPU será mucho más rápido; aquí no se pudo medir.

| | Modelo | Velocidad en esa CPU | Memoria |
|---|---|---|---|
| Embeddings | `embeddinggemma` (recomendado) | 6,5 artículos/s → **~86 h para 2 M artículos** | 0,65 GB |
| Embeddings | `qwen3-embedding:0.6b` | 1,4 artículos/s (~16 días) y no acertó más en la prueba | 0,64 GB |
| Chat | `qwen2.5:3b` | lee 66 tokens/s, escribe ~10 tokens/s → **~27 s por respuesta** | 2 GB |
| Chat | `qwen3:4b` | lee 49 tokens/s, escribe 8 tokens/s (más lento) | 2,5 GB |
| Índice vectorial | 2 M artículos × 384 dimensiones int8 | ~1,2 s por búsqueda | ~0,7 GB |

En la prueba de calidad (10 preguntas parafraseadas sobre 410 artículos de matemáticas),
embeddinggemma encontró el artículo correcto entre los 5 primeros en 9/10 casos, con 256 a 768
dimensiones; 384 es un buen equilibrio entre acierto y RAM.

**Qwen:** el modelo de chat por defecto ya es Qwen (`qwen2.5:7b`). Los Qwen3 "piensan" antes de
responder; el sistema les pide no hacerlo (`think: false`), pero con `qwen3:4b` el razonamiento se
coló en la respuesta durante las pruebas, así que se recomienda qwen2.5 o una variante "instruct" de Qwen3.

**Vectorizar toda la Wikipedia toma tiempo** (días en CPU, horas con GPU), pero no hay que
esperar: se hace en segundo plano, de los artículos más extensos a los más cortos, la búsqueda
semántica mejora a medida que avanza y se pausa sola mientras alguien usa el chat para no
quitarle velocidad. También se puede adelantar con `python -m wikichat embed`.

## Imágenes

Las imágenes vienen del `.zim`. Hay tres versiones de la Wikipedia en español:

| Versión | Tamaño | Imágenes |
|---|---|---|
| `wikipedia_es_all_nopic` | 11 GB | ninguna |
| `wikipedia_es_all_mini` | 3,5 GB | solo la introducción de cada artículo |
| `wikipedia_es_all_maxi` | 38 GB | todas (en miniatura, webp) |

Con la versión **maxi**, el importador guarda hasta 6 imágenes por artículo con su pie de foto,
descartando íconos y fórmulas. Las imágenes no se copian: se leen del `.zim` al mostrarlas, así
que hay que conservarlo (si lo mueves, indica la nueva ruta en `zim_path`). Debajo de cada
respuesta, el chat muestra las imágenes de los artículos usados como fuente.

Para artículos que se actualizan después por la API se conservan las imágenes del `.zim`, y si
no tenía ninguna se guarda la imagen principal del artículo: se descarga la primera vez que se
muestra y queda guardada para usarla sin conexión.

## Conversaciones guardadas

Cada conversación se guarda en `data/chats.db`, un archivo aparte de la wiki: actualizar o
reimportar la wiki nunca la toca. En la barra lateral puedes abrir, renombrar y borrar chats,
y cada chat tiene su propia dirección (`http://127.0.0.1:8800/#<id>`), así que sobrevive a
recargar la página o reiniciar el servidor. Con cada respuesta se guardan también sus fuentes
e imágenes, y las imágenes no se repiten dentro de una misma conversación.

Cómo se mantiene el contexto sin que el modelo reciba la conversación entera:

- **Mensajes recientes:** el modelo recibe tal cual los últimos `history_messages` (6).
- **Resumen de lo antiguo:** cuando quedan suficientes mensajes fuera de esa ventana, el propio
  modelo los resume en segundo plano y el resumen acompaña a las preguntas siguientes
  (`summarize_history`).
- **Preguntas de seguimiento:** "¿y en qué se usa?" no sirve para buscar sola. Por defecto, si la
  pregunta es corta se busca junto con la anterior ("¿Qué es el oro? ¿Y en qué se usa?"). Con
  `rewrite_followups` el modelo la reescribe para que se entienda sola; busca mejor, pero
  cuesta una llamada más al modelo (unos segundos en CPU).

## Puesta en marcha

```bash
pip install -r requirements.txt
cp config.example.json config.json

# 1. Descarga el .zim más reciente desde https://download.kiwix.org/zim/wikipedia/
#    (maxi si quieres imágenes, nopic si no)
# 2. Impórtalo (estimado 1–2 h con 4 núcleos para nopic; si se interrumpe, repite el comando):
python -m wikichat import-zim wikipedia_es_all_maxi_2026-05.zim

# 3. Inicia el chat: se actualiza solo y vectoriza en segundo plano.
python -m wikichat serve        # http://127.0.0.1:8800
```

La primera actualización tras importar un `.zim` de más de un mes hace la **puesta al día**:
unas 40 000 consultas para revisar ~2 M artículos, más la descarga de los que cambiaron. Puede
tardar horas, pero corre en segundo plano, se reanuda sola y el chat funciona mientras tanto.
Cuanto más reciente sea el `.zim`, menos trabajo.

Otros comandos:

```bash
python -m wikichat sync                    # actualizar ahora (útil con cron / Programador de tareas)
python -m wikichat embed [--max N]         # adelantar la vectorización
python -m wikichat ask "¿Cuándo se fundó Antigua Guatemala?"
python -m wikichat search lago atitlan     # búsqueda sin LLM
python -m wikichat stats
python -m wikichat serve --no-update       # sin conexión a Wikipedia
```

## Configuración (`config.json`)

| Clave | Para qué sirve |
|---|---|
| `api_url`, `user_agent` | API de la wiki y User-Agent identificable (Wikimedia lo exige; sin él responde 429). |
| `track_all_changes` | `true` (Wikipedia completa): sigue cualquier artículo que cambie, incluidos los nuevos. `false`: solo los que ya tienes y los de `seed_categories` / `seed_titles`. |
| `skip_bot_edits` | Ignora ediciones de bots (casi siempre mantenimiento), reduce mucho las descargas. |
| `seed_categories`, `category_depth`, `seed_titles` | Para copiar solo un tema en vez de toda la wiki (con `track_all_changes: false`, sin `.zim`). |
| `update_interval_hours`, `request_delay_seconds` | Frecuencia de actualización y pausa entre peticiones a Wikipedia. |
| `llm_backend`, `llm_url`, `llm_api_key` | `ollama` o `openai` (servidor compatible) y su dirección. |
| `chat_model`, `top_k` | Modelo de chat y cuántos fragmentos recibe como contexto (más = respuestas más completas pero más lentas en CPU). |
| `embed_model` | Modelo de embeddings; vacío desactiva la búsqueda semántica. |
| `embed_dims`, `embed_chars` | Dimensiones guardadas (RAM del índice) y caracteres de cada artículo que se vectorizan. |
| `embed_doc_template`, `embed_query_prefix` | Formato que espera el modelo de embeddings (los valores por defecto son los de embeddinggemma; para otros modelos consulta su documentación). Si cambias de modelo o de dimensiones, borra la tabla `page_vectors` para volver a vectorizar. |
| `zim_path` | Ruta del `.zim` si lo moviste después de importarlo (para las imágenes). |
| `chats_db_path` | Dónde se guardan las conversaciones. |
| `history_messages` | Cuántos mensajes recientes recibe el modelo tal cual. |
| `summarize_history` | Resumir los mensajes antiguos para conservar el contexto en chats largos. |
| `rewrite_followups` | Que el modelo reescriba las preguntas de seguimiento antes de buscar (mejor búsqueda, más lento en CPU). |

## Cómo se arma una respuesta

1. Los artículos se guardan como texto plano dividido por secciones (~1 500 caracteres); se
   omiten referencias, enlaces externos y cajas de navegación. Las fórmulas se guardan como TeX.
2. **Búsqueda por palabras:** la pregunta se reduce a palabras clave (sin acentos ni palabras
   vacías) y SQLite FTS5 busca con BM25, primero exigiendo todas las palabras.
3. **Búsqueda semántica:** la pregunta se convierte en un vector y se compara con el vector de
   cada artículo (título + introducción), así encuentra artículos aunque no compartan palabras
   ("pájaro símbolo nacional" → Quetzal). De cada artículo se toma el fragmento que mejor
   coincide.
4. Ambas listas se combinan con *Reciprocal Rank Fusion*; si la búsqueda por palabras no encontró
   fragmentos con todas las palabras, la semántica pesa más. Si las palabras clave son exactamente
   el título de un artículo ("¿Qué es el oro?" → Oro), su introducción va primero.
5. Los mejores fragmentos van al modelo de chat con la instrucción de responder solo con ellos y
   citar los artículos; la interfaz muestra las fuentes y sus imágenes.

## Pruebas

```bash
python -m unittest -v      # no necesita modelos; la prueba del importador se omite sin libzim
```

## Límites conocidos

- Los artículos creados entre la fecha del `.zim` y hace 29 días no se detectan en la puesta
  al día (Wikipedia solo guarda 29–30 días de cambios recientes). Usa el `.zim` más
  reciente o reimporta uno nuevo de vez en cuando para recogerlos.
- Con `skip_bot_edits` las correcciones hechas por bots no se descargan hasta la siguiente
  edición humana del artículo.
- La búsqueda semántica usa un vector por artículo (título + introducción): encuentra bien de
  qué artículo se trata, pero dentro del artículo el fragmento se elige por palabras.
- Las imágenes se muestran junto a la respuesta, pero el modelo no las "ve"; para eso haría falta
  un modelo con visión.
- El contenido de Wikipedia es CC BY-SA: si publicas respuestas, cita los artículos.
