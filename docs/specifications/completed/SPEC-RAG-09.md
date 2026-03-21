# SPEC-RAG-09: Qwen3-Embedding + Qwen3-Reranker + Chunking

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Draft
> **Цель:** Обновить embedding и reranker на Qwen3 family, внедрить two-tier chunking при ingest,
> подключить reranker в agent pipeline между search и compose_context.
> **Источники:** R-07 (Blocks 1-3), R-08, DEC-0026
> **Зависит от:** SPEC-RAG-08 (function calling agent)

---

## 0. Мотивация

### Embedding: e5-large → Qwen3-Embedding-0.6B

Per R-07 Block 3:
- `multilingual-e5-large` (2023): MMTEB retrieval 57.12, max context **512 tokens** — длинные посты обрезаются
- `Qwen3-Embedding-0.6B` (2025): MMTEB retrieval **64.64** (+7.5), max context **32K tokens**, MRL support
- Тот же размер (~1.5-1.8GB VRAM), TEI совместим с v1.8.0+
- Instruction format: `"Instruct: <task>\nQuery: <text>"` для запросов, без prefix для документов

### Reranker: bge-v2-m3 → Qwen3-Reranker-0.6B

Per R-07 Block 2:
- `bge-reranker-v2-m3` (2023): MMTEB-R 58.36
- `Qwen3-Reranker-0.6B` (2025): MMTEB-R **66.36** (+8.0), Russian support
- TEI совместим через seq-cls конверсию: `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
- Тот же VRAM footprint (~1.5GB)

### Reranker в agent pipeline

Сейчас: `search → compose_context` (reranker пропускается).
Нужно: `search → rerank → compose_context` (+15-20pp precision per R-07).

### Chunking

Per R-07 Block 1:
- Посты <1500 chars → один вектор (single topic, fits in any context window)
- Посты >1500 chars → recursive split по `\n\n`, target ~400 tokens per chunk
- Без overlap (hybrid search компенсирует)
- Point ID с chunking: `"{channel}:{msg_id}:{chunk_idx}"`

---

## 1. Изменяемые файлы

| Файл | Характер |
|------|----------|
| `src/adapters/tei/embedding_client.py` | Обновить prefix format для Qwen3-Embedding |
| `src/adapters/tei/reranker_client.py` | Без изменений (HTTP API тот же) |
| `src/services/agent_service.py` | Reranker в _normalize_tool_params для rerank: fix minor P3 |
| `scripts/ingest_telegram.py` | Two-tier chunking logic |
| `src/core/settings.py` | Новые settings: chunk_threshold, embedding instruction |
| `deploy/compose/compose.dev.yml` | TEI image/model env vars |
| `agent_context/core/always_on.md` | Обновить модели TEI |
| `AGENTS.md` | Обновить модели |
| `docs/ai/project_brief.md` | Обновить embedding/reranker |
| `docs/ai/agent_technical_spec.md` | Обновить embedding/reranker, pipeline |
| `docs/ai/modules/src/adapters/tei/embedding_client.py.md` | Обновить (если существует) |
| `docs/ai/modules/scripts/ingest_telegram.py.md` | Обновить chunking |
| `agent_context/modules/retrieval.md` | Обновить models |

### Что НЕ менять

- `src/adapters/qdrant/store.py` — schema не меняется (1024-dim dense + sparse). Qwen3-Embedding тоже 1024-dim.
- `src/services/tools/search.py` — HybridRetriever API не меняется
- `src/services/tools/compose_context.py` — coverage metric не меняется
- `src/services/tools/tool_runner.py` — не меняется

---

## 2. Embedding Client — Qwen3-Embedding prefix format

### 2.1 Текущий формат (e5-large)

```python
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "
```

### 2.2 Новый формат (Qwen3-Embedding-0.6B)

Per [Qwen3-Embedding docs](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B):
- **Документы**: БЕЗ prefix (plain text)
- **Запросы**: `"Instruct: <task_description>\nQuery: <query_text>"`
- Instruction пишется на **английском** даже для русского контента

```python
# Instruction для нашего домена (ML/AI Telegram news retrieval)
_QUERY_INSTRUCTION = (
    "Instruct: Given a user question about ML, AI, LLM or tech news, "
    "retrieve relevant Telegram channel posts\n"
    "Query: "
)
_PASSAGE_PREFIX = ""  # Qwen3-Embedding не требует prefix для документов
```

### 2.3 Изменения в `TEIEmbeddingClient`

```python
async def embed_query(self, text: str) -> List[float]:
    """Встраивает поисковый запрос с instruction prefix."""
    prefixed = _QUERY_INSTRUCTION + text
    vectors = await self._embed_batch([prefixed], normalize=True)
    return vectors[0]

async def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """Батчевое встраивание документов — БЕЗ prefix (Qwen3-Embedding)."""
    # Qwen3-Embedding не использует prefix для documents (в отличие от e5-large)
    return await self._embed_batch(texts, normalize=True)
```

### 2.4 Обновить docstring модуля

Заменить все упоминания `multilingual-e5-large` на `Qwen3-Embedding-0.6B`.
Обновить описание prefix format.

---

## 3. Chunking в ingest

### 3.1 Новая функция `_smart_chunk()`

Добавить в `scripts/ingest_telegram.py`:

```python
# Порог для чанкования: посты длиннее этого значения разбиваются
CHUNK_CHAR_THRESHOLD = 1500
# Целевой размер чанка в символах (~400 tokens для Qwen3-Embedding)
CHUNK_TARGET_SIZE = 1200

def _smart_chunk(text: str, threshold: int = CHUNK_CHAR_THRESHOLD,
                 target: int = CHUNK_TARGET_SIZE) -> List[str]:
    """Two-tier chunking: короткие посты целиком, длинные — recursive split.

    Стратегия (per R-07):
    - text <= threshold: возвращаем как есть (одна тема, один вектор)
    - text > threshold: recursive split по иерархии сепараторов ["\n\n", "\n", ". ", " "]

    Без overlap — hybrid search (BM25 sparse) компенсирует стыки.
    """
    if len(text) <= threshold:
        return [text]

    # Recursive split: сначала по двойным переносам (абзацы)
    chunks = _recursive_split(text, target, ["\n\n", "\n", ". ", " "])
    return [c.strip() for c in chunks if c.strip()]


def _recursive_split(text: str, target: int, separators: List[str]) -> List[str]:
    """Рекурсивно делит текст по иерархии сепараторов."""
    if len(text) <= target:
        return [text]

    if not separators:
        # Fallback: hard split по target size
        return [text[i:i+target] for i in range(0, len(text), target)]

    sep = separators[0]
    rest_seps = separators[1:]
    parts = text.split(sep)

    chunks = []
    current = ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= target:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Если отдельная часть > target, рекурсивно делим мельче
            if len(part) > target:
                chunks.extend(_recursive_split(part, target, rest_seps))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)

    return chunks
```

### 3.2 Заменить `_split_text()` в `ingest_batches()`

Текущий `_split_text()` делает fixed-size split по `--chunk-size`. Заменить на `_smart_chunk()`:

```python
# Было:
for part in _split_text(text_full, chunk_size):
    texts.append(part)
    source_messages.append(message)

# Стало:
for part in _smart_chunk(text_full):
    texts.append(part)
    source_messages.append(message)
```

`--chunk-size` CLI аргумент **оставить** как override: если задан > 0, использовать старый `_split_text()`.
Если не задан (default 0) — использовать `_smart_chunk()`.

### 3.3 Settings

```python
# Chunking
self.chunk_char_threshold: int = int(os.getenv("CHUNK_CHAR_THRESHOLD", "1500"))
self.chunk_target_size: int = int(os.getenv("CHUNK_TARGET_SIZE", "1200"))
```

---

## 4. Reranker в agent pipeline

### 4.1 Текущий flow (SPEC-RAG-08)

Модель видит tool `rerank` в schema, но может его не вызывать. Pipeline зависит от решения LLM.

### 4.2 Что нужно сделать

Reranker уже в AGENT_TOOLS schema (SPEC-RAG-08). LLM должен вызывать его по system prompt instruction:
"2. search → 3. rerank → 4. compose_context".

Никаких программных изменений для этого не нужно — LLM уже видит tool и описание.

### 4.3 Fix minor P3 из SPEC-RAG-08 ревью

В `_normalize_tool_params` для `rerank`: если `docs` передан как пустой список `[]`,
fallback на `_last_search_hits` не срабатывает.

**Было:**
```python
if not normalized.get("docs"):
    normalized["docs"] = [...]
```

**Стало:**
```python
if not normalized.get("docs") or not any(normalized["docs"]):
    normalized["docs"] = [...]
```

---

## 5. Deploy — TEI модели

### 5.1 WSL2 команды запуска

**Embedding (Qwen3-Embedding-0.6B):**
```bash
docker run -d --gpus all --name tei-embedding -p 8082:80 \
  ghcr.io/huggingface/text-embeddings-inference:1.9 \
  --model-id Qwen/Qwen3-Embedding-0.6B
```

**Reranker (Qwen3-Reranker-0.6B-seq-cls):**
```bash
docker run -d --gpus all --name tei-reranker -p 8083:80 \
  ghcr.io/huggingface/text-embeddings-inference:1.9 \
  --model-id tomaarsen/Qwen3-Reranker-0.6B-seq-cls
```

**Примечание:** `Qwen3-Reranker-0.6B` оригинальная модель использует causal LM scoring,
не совместимый с TEI напрямую. Конверсия `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` оборачивает
её в SequenceClassification format для TEI (per R-07 Block 2).

### 5.2 compose.dev.yml

Env vars не меняются (TEI URL те же). Сами модели меняются в WSL2 контейнерах.

### 5.3 Re-ingest

После смены embedding модели коллекция **должна быть пересоздана**:
```bash
# Удалить старую коллекцию
curl -X DELETE http://localhost:16333/collections/news

# API пересоздаст при старте (ensure_collection)
docker restart rag-dev-api-1

# Re-ingest
docker compose -f deploy/compose/compose.dev.yml run -it --rm ingest \
  --channels "@protechietich,@data_secrets,@ai_machinelearning_big_data,@data_easy,@xor_journal" \
  --since 2026-01-01 --until 2026-03-17
```

---

## 6. Документация

### 6.1 Обязательно обновить

- `docs/ai/project_brief.md` — Embedding: Qwen3-Embedding-0.6B, Reranker: Qwen3-Reranker-0.6B
- `docs/ai/agent_technical_spec.md` — embedding model, reranker model, pipeline с rerank
- `agent_context/core/always_on.md` — модели TEI, команды запуска WSL2
- `AGENTS.md` — модели
- `agent_context/modules/retrieval.md` — embedding, reranker, chunking
- `agent_context/modules/ingest_eval.md` — chunking в ingest
- `docs/ai/modules/scripts/ingest_telegram.py.md` — _smart_chunk()
- `docs/ai/modules/src/services/reranker_service.py.md` — модель reranker

### 6.2 Проверить/обновить при необходимости

- `docs/architecture/04-system/overview.md` — если упоминает конкретные модели
- `docs/architecture/05-flows/FLOW-01-ingest.md` — chunking, embedding model

---

## 7. Тесты

### 7.1 `src/tests/test_embedding_client.py` (обновить существующий или новый)

- `test_embed_query_uses_instruction_prefix` — проверить что запрос получает "Instruct: ...\nQuery: "
- `test_embed_documents_no_prefix` — проверить что документы идут без prefix

### 7.2 `src/tests/test_chunking.py` (новый)

- `test_short_post_no_chunking` — пост <1500 chars → [один элемент]
- `test_long_post_splits_by_paragraphs` — пост 3000 chars с \n\n → несколько чанков
- `test_chunk_target_size` — каждый чанк <= target size
- `test_no_empty_chunks` — пустые чанки не возвращаются
- `test_single_long_paragraph_hard_split` — нет \n\n, есть ". " → split по предложениям

Не запускать pytest.

---

## 8. Чеклист реализации

### Embedding
- [ ] `embedding_client.py` — prefix format обновлён для Qwen3-Embedding
- [ ] `_QUERY_INSTRUCTION` — instruction на английском для нашего домена
- [ ] `_PASSAGE_PREFIX = ""` — документы без prefix
- [ ] Docstring обновлён

### Reranker
- [ ] Модель в WSL2: `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
- [ ] Код `reranker_client.py` — без изменений (HTTP API тот же)
- [ ] `reranker_service.py` — без изменений

### Chunking
- [ ] `_smart_chunk()` добавлена в `ingest_telegram.py`
- [ ] `_recursive_split()` добавлена
- [ ] `ingest_batches()` использует `_smart_chunk()` по умолчанию
- [ ] `--chunk-size` CLI аргумент сохранён как override
- [ ] `settings.py` — `chunk_char_threshold`, `chunk_target_size`

### Agent pipeline
- [ ] `_normalize_tool_params` rerank — fix empty docs fallback

### Documentation
- [ ] `project_brief.md` — embedding/reranker models
- [ ] `agent_technical_spec.md` — models, pipeline
- [ ] `always_on.md` — TEI models, WSL2 commands
- [ ] `AGENTS.md` — models
- [ ] `retrieval.md` — models, chunking
- [ ] `ingest_eval.md` — chunking
- [ ] `ingest_telegram.py.md` — _smart_chunk
- [ ] `reranker_service.py.md` — model

### Tests
- [ ] Embedding prefix tests
- [ ] Chunking tests (5 cases)
- [ ] НЕ запускать pytest
