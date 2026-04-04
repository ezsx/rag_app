# rag_app — Architecture Brief (Phase 1 Migration)

> **Дата:** 2026-03-17
> **Статус:** Актуально
> **Вход:** `docs/architecture/11-decisions/decision-log.md` (DEC-0015 – DEC-0026),
>            `docs/research/reports/R00-synthesis.md`, R01–R06
> **Назначение:** reference-документ для написания SPEC-RAG-01..07.
>             Каждая спецификация начинает чтение с этого файла.

---

## 1. Что такое rag_app

`rag_app` — FastAPI-платформа Retrieval-Augmented Generation для поиска и агрегации новостей из Telegram-каналов. Основной flow: пользователь задаёт вопрос → ReAct-агент выполняет цикл (router → plan → search → rerank → compose → verify → answer) → ответ стримится через SSE.

**Публичный контракт:**
- `POST /v1/agent/stream` — SSE стрим: `thought / tool_invoked / observation / citations / final`
- `POST /v1/qa` — простой QA без агента
- Аутентификация: JWT (ADMIN_KEY)

**Инварианты (нельзя нарушать):**
- SSE event типы не меняются
- `lru_cache` на всех фабриках в `deps.py` — смена настроек через `settings.update_*()`
- `coverage_threshold=0.65`, `max_refinements=2` (DEC-0019)
- Point ID в Qdrant: `"{channel_name}:{message_id}"` (уникальность)

---

## 2. Состояние кода: Phase 0 (сейчас) vs Phase 1 (цель)

| Слой | Phase 0 (текущий код) | Phase 1 (целевой) | Spec |
|------|-----------------------|-------------------|------|
| Settings | `coverage_threshold=0.8`, `max_refinements=1`, ChromaDB поля, нет TEI URL | `coverage_threshold=0.65`, `max_refinements=2`, Qdrant поля, `EMBEDDING_TEI_URL`, `RERANKER_TEI_URL` | SPEC-RAG-01 |
| Docker | chroma service, GPU reservation, bm25-index volume | qdrant service (named volume), без GPU, TEI через env | SPEC-RAG-01 |
| Embedding | `SentenceTransformer` в-процессе, `BAAI/bge-base-en-v1.5` | `TEIEmbeddingClient` HTTP → `multilingual-e5-large` @ `:8082` | SPEC-RAG-02 |
| Reranker | `CrossEncoder` в-процессе, `BAAI/bge-reranker-v2-m3` | `TEIRerankerClient` HTTP → `bge-reranker-v2-m3` @ `:8083` | SPEC-RAG-02 + 05 |
| Vector Store | ChromaDB HTTP + BM25IndexManager disk | Qdrant (dense 1024-dim + sparse BM25 russian, named vectors) | SPEC-RAG-03 |
| HybridRetriever | ChromaDB dense + BM25 + manual RRF | Qdrant prefetch + FusionQuery(RRF) + MmrQuery native | SPEC-RAG-04 |
| Ingest | ChromaDB + local SentenceTransformer | Qdrant upsert (dense + sparse) + TEI HTTP | SPEC-RAG-06 |
| Coverage | `citation_coverage` (doc count ratio) | 5-signal composite metric | SPEC-RAG-07 |

---

## 3. ADR-карта: решения → файлы → спецификации

### DEC-0015 — Qdrant вместо ChromaDB + BM25
**Суть:** ChromaDB + BM25IndexManager = ~400 строк кода ради того, что Qdrant делает нативно.
`Qdrant/bm25` с `language="russian"` — корректный sparse для русского (BM42 English-only).
Нативный RRF через `prefetch + FusionQuery` в одном HTTP-запросе.

**Файлы к удалению:**
- `src/adapters/chroma/` — весь каталог
- `src/adapters/search/bm25_index.py`
- `src/adapters/search/bm25_retriever.py`
- `docker-compose.yml` → service `chroma`, volume `./chroma-data`, `./bm25-index`

**Файлы к созданию:**
- `src/adapters/qdrant/__init__.py`
- `src/adapters/qdrant/store.py` — `QdrantStore`

**Источник:** `docs/research/reports/R01-qdrant-hybrid-rag.md`

---

### DEC-0016 — TEI для embedding (multilingual-e5-large)
**Суть:** TEI (text-embeddings-inference) от HuggingFace — production HTTP-сервер для embedding,
значительно быстрее `sentence-transformers` в-процессе. Запускается в WSL2 native на RTX 5060 Ti.

**Файлы к удалению:** `SentenceTransformerEmbeddingFunction` из `src/adapters/chroma/retriever.py`

**Файлы к созданию:**
- `src/adapters/tei/embedding_client.py` — `TEIEmbeddingClient`

**Конфиг:** `EMBEDDING_TEI_URL=http://host.docker.internal:8082`
**Модель:** `intfloat/multilingual-e5-large` (1024-dim, cosine)

**Источник:** `docs/architecture/04-system/overview.md`

---

### DEC-0017 — TEI для reranker (bge-reranker-v2-m3)
**Суть:** Аналогично DEC-0016 — CrossEncoder локально слишком тяжёлый для CPU Docker.
TEI reranker на том же RTX 5060 Ti, отдельный порт.

**Файлы к изменению:**
- `src/services/reranker_service.py` — убрать CrossEncoder, использовать `TEIRerankerClient`
- `src/core/deps.py` — `get_reranker()` → `RerankerService(TEIRerankerClient)`

**Файлы к удалению (или зачистке):**
- `src/utils/model_downloader.py` — функция `download_reranker_model()` становится ненужной

**Конфиг:** `RERANKER_TEI_URL=http://host.docker.internal:8083`

**Источник:** `docs/architecture/04-system/overview.md`

---

### DEC-0018 — Composite coverage metric (5 сигналов)
**Суть:** `citation_coverage` (количество документов) не отражает достаточность контекста.
Нужен weighted composite из 5 сигналов.

**5 сигналов и веса:**
| Сигнал | Вес | Описание |
|--------|-----|----------|
| `cosine_sim` | 0.35 | mean cosine sim query↔doc (Qdrant `with_vectors=True`, L2-normalized) |
| `answer_coverage` | 0.25 | доля предложений ответа, покрытых docs |
| `term_coverage` | 0.20 | доля query-термов (без stopwords) в retrieved text |
| `source_diversity` | 0.10 | нормализованное число уникальных источников |
| `passage_relevance` | 0.10 | топ reranker-скор (0–1 нормализованный) |

**Формула:** `coverage = sum(weight_i * signal_i)`

**Файлы к изменению:** `src/services/agent_service.py` (или новый `src/services/coverage.py`)

**Источник:** `docs/research/reports/R04-coverage-metrics.md`

---

### DEC-0019 — coverage_threshold=0.65, max_refinements=2
**Суть:** Старый threshold 0.80 слишком агрессивен с composite metric (score compression).
Asymmetric error: false-negative (пропущенный поиск) → 66.1% галлюцинаций.

**Файлы к изменению:**
- `src/core/settings.py`: `coverage_threshold=0.65`, `max_refinements=2`
- `docker-compose.yml`: `COVERAGE_THRESHOLD=0.65`, `MAX_REFINEMENTS=2`

---

### DEC-0024 — Docker GPU blocker
**Суть:** V100 в TCC-режиме блокирует NVML-enumeration для **всех** GPU при старте Docker Desktop,
включая RTX 5060 Ti. GPU в Docker-контейнерах недоступны пока V100 подключена.
**Решение**: embedding + reranker = WSL2-native процессы, Docker-контейнеры = CPU only.

**Файлы к изменению:**
- `docker-compose.yml`: убрать `deploy.resources.reservations.devices: nvidia` из `api` и `ingest`
- Добавить в README/комментарии: TEI запускается отдельно в WSL2

---

### DEC-0026 — Qwen3-Embedding (future, вне scope Phase 1)
Целевая embedding-модель — `Qwen3/Qwen3-Embedding-0.6B` (tops MTEB Multilingual 2025).
**Отложено** до R-embed research. Коллекция Qdrant после смены модели пересоздаётся полностью.
Требует instruction prefix: `query: <текст>` / `passage: <текст>`.

---

## 4. Целевая схема данных (Phase 1)

### Qdrant Collection: `news`

```
VectorParams (named):
  "dense_vector":  size=1024, distance=Cosine
  "sparse_vector": SparseVectorParams(modifier=Modifier.IDF, index=SparseIndexParams(on_disk=False))
                   model: Qdrant/bm25, language="russian"

Point:
  id:      str  — "{channel_name}:{message_id}"
  vector:  {"dense_vector": list[float], "sparse_vector": SparseVector}
  payload: {"text": str, "channel": str, "date": str, "author": str | None, ...}
```

### Hybrid Search запрос

```python
client.query_points(
    collection_name="news",
    prefetch=[
        Prefetch(query=dense_vector, using="dense_vector", limit=20),
        Prefetch(query=sparse_vector, using="sparse_vector", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    with_vectors=True,  # нужно для cosine_sim в coverage
    limit=10,
)
```

---

## 5. Hardware & Deployment контекст

```
[Windows Host]
  └── llama-server.exe → V100 SXM2 32GB (TCC, device 1)  :8080

[Ubuntu WSL2 — нативно]
  ├── TEI embedding  → RTX 5060 Ti (device 0)  :8082  (intfloat/multilingual-e5-large)
  └── TEI reranker   → RTX 5060 Ti (device 0)  :8083  (BAAI/bge-reranker-v2-m3)

[Docker Desktop / WSL2 — CPU only]
  ├── api    :8000
  └── qdrant :6333
```

**Критично:** Docker-контейнеры обращаются к WSL2-native и Windows-хосту через `host.docker.internal`.
GPU в Docker недоступны (DEC-0024) — все модели выведены за пределы Docker.

---

## 6. Карта источников по доменам

| Домен | Первичный источник | Вторичный |
|-------|--------------------|-----------|
| Qdrant schema, hybrid search, RRF | `docs/research/reports/R01-qdrant-hybrid-rag.md` | `docs/architecture/07-data-model/data-model.md` |
| TEI embedding/reranker | `docs/architecture/04-system/overview.md` | `docs/architecture/05-flows/FLOW-01-ingest.md` |
| Coverage metric (5 сигналов) | `docs/research/reports/R04-coverage-metrics.md` | `docs/architecture/11-decisions/decision-log.md` (DEC-0018/19) |
| ReAct agent цикл, tools | `docs/ai/agent_technical_spec.md` | `docs/architecture/05-flows/FLOW-02-agent.md` |
| Ingest pipeline | `docs/architecture/05-flows/FLOW-01-ingest.md` | `docs/architecture/07-data-model/data-model.md` |
| Settings Phase 1 | `docs/architecture/07-data-model/data-model.md` | `docs/architecture/11-decisions/decision-log.md` |
| Все решения сводно | `docs/research/reports/R00-synthesis.md` | `docs/architecture/11-decisions/decision-log.md` |

---

## 7. Что НЕ меняется в Phase 1

- `src/adapters/llm/llama_server_client.py` — уже Phase 1, не трогать
- `src/services/agent_service.py` — только coverage logic (SPEC-RAG-07), остальное без изменений
- `src/services/query_planner_service.py` — без изменений
- `src/api/` — без изменений
- `src/schemas/` — без изменений
- `src/core/auth.py`, `rate_limit.py`, `security.py` — без изменений
- SSE event contract: `thought / tool_invoked / observation / citations / final` — не ломать
