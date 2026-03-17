# Spec Plan — rag_app Phase 0 → Phase 1 Migration

> **Дата:** 2026-03-17
> **Статус:** Active
> **Scope:** Полная миграция кодовой базы Phase 0 (ChromaDB + BM25 + local models) → Phase 1 (Qdrant + TEI HTTP)
> **Arch Brief:** [arch-brief.md](arch-brief.md) — читать перед написанием любой спецификации

---

## Контекст

Текущий код (`src/`) находится в состоянии **Phase 0**:
- Vector store: ChromaDB HTTP
- Lexical index: BM25IndexManager (disk-based)
- Embedding: local SentenceTransformer (`BAAI/bge-base-en-v1.5`)
- Reranker: local CrossEncoder (`BAAI/bge-reranker-v2-m3`)
- Settings: `coverage_threshold=0.8`, `max_refinements=1`

Целевое состояние **Phase 1** (задокументировано в ADR DEC-0015 – DEC-0026):
- Vector store: **Qdrant** (dense 1024-dim + sparse BM25, named vectors, нативный RRF+MMR)
- Embedding: **TEI HTTP** → multilingual-e5-large (WSL2, :8082)
- Reranker: **TEI HTTP** → bge-reranker-v2-m3 (WSL2, :8083)
- Settings: `coverage_threshold=0.65`, `max_refinements=2`
- Docker: без GPU reservation (DEC-0024)

---

## Карта зависимостей

```
SPEC-RAG-01 (Settings + Compose)
    ↓
SPEC-RAG-02 (TEI Adapter) ──────────┐
    ↓                                │
SPEC-RAG-03 (Qdrant Store) ─────────┤
    ↓                                │
SPEC-RAG-04 (HybridRetriever + DI) ←┘
    ↓
SPEC-RAG-05 (RerankerService) ←── SPEC-RAG-02
    ↓
SPEC-RAG-06 (Ingest Pipeline) ←── SPEC-RAG-02 + SPEC-RAG-03
    ↓
SPEC-RAG-07 (Coverage Metric) ←── SPEC-RAG-04 (Qdrant with_vectors)
```

---

## Таблица спецификаций

| # | Файл | Название | Ключевые изменения | ADR |
|---|------|----------|--------------------|-----|
| 01 | [SPEC-RAG-01.md](SPEC-RAG-01.md) | Settings & Docker Compose | `settings.py` + `docker-compose.yml` | DEC-0019, DEC-0024 |
| 02 | [SPEC-RAG-02.md](SPEC-RAG-02.md) | TEI Adapter Layer | Новый `src/adapters/tei/` | DEC-0016, DEC-0017 |
| 03 | [SPEC-RAG-03.md](SPEC-RAG-03.md) | Qdrant Store Adapter | Новый `src/adapters/qdrant/` | DEC-0015 |
| 04 | [SPEC-RAG-04.md](SPEC-RAG-04.md) | HybridRetriever & DI | `hybrid_retriever.py` + `deps.py` | DEC-0015, DEC-0020 |
| 05 | [SPEC-RAG-05.md](SPEC-RAG-05.md) | RerankerService Migration | `reranker_service.py` + `deps.py` | DEC-0017 |
| 06 | [SPEC-RAG-06.md](SPEC-RAG-06.md) | Ingest Pipeline | `scripts/ingest_telegram.py` | DEC-0015, DEC-0016 |
| 07 | [SPEC-RAG-07.md](SPEC-RAG-07.md) | Composite Coverage Metric | `agent_service.py` coverage logic | DEC-0018, DEC-0019 |

---

## Порядок реализации

Спецификации реализуются строго по номеру: каждая следующая зависит от предыдущей.

### Критический путь

```
01 → 02 → 03 → 04 → 06   (retrieval pipeline)
             ↘ 05         (reranker)
                ↘ 07     (coverage metric — требует with_vectors=True из Qdrant)
```

### Phase 1 baseline (минимальный запуск)

Для первого запуска достаточно **SPEC-RAG-01 + 02 + 03 + 04 + 06** — система будет отвечать на запросы.
SPEC-RAG-05 + 07 улучшают качество (reranking + composite coverage).

---

## Что удаляется после миграции

| Компонент | Файл/Сервис | Удаляется в |
|-----------|-------------|-------------|
| ChromaDB adapter | `src/adapters/chroma/` | SPEC-RAG-04 |
| BM25IndexManager | `src/adapters/search/bm25_index.py` | SPEC-RAG-04 |
| BM25Retriever | `src/adapters/search/bm25_retriever.py` | SPEC-RAG-04 |
| local SentenceTransformer | `src/adapters/chroma/retriever.py` | SPEC-RAG-04 |
| local CrossEncoder | `src/services/reranker_service.py` | SPEC-RAG-05 |
| ChromaDB docker service | `docker-compose.yml` | SPEC-RAG-01 |
| BM25 index volume | `docker-compose.yml` | SPEC-RAG-01 |
| GPU reservation | `docker-compose.yml` | SPEC-RAG-01 |
| `src/utils/model_downloader.py` | (util для local HF download) | SPEC-RAG-05 |

---

## Что НЕ меняется

- `src/adapters/llm/llama_server_client.py` — уже Phase 1
- `src/services/agent_service.py` — кроме coverage (SPEC-RAG-07)
- `src/services/query_planner_service.py` — без изменений
- `src/api/` — без изменений
- `src/schemas/` — без изменений
- JWT / Auth — без изменений

---

## Будущее (вне scope Phase 1)

- **SPEC-RAG-08** (отложено): Qwen3-Embedding-0.6B миграция — после R-embed research (DEC-0026)
- **SPEC-RAG-09** (отложено): Evaluation Framework — generate_eval_dataset + LLM-judge (FLOW-03)
- Phase 2: vLLM + Hermes tool calling (после Proxmox VFIO)

---

---

# Как писать спецификации — инструкция для агента

> Этот раздел обязателен к прочтению **перед** написанием любой SPEC-RAG-XX.
> Цель: другой агент читает спецификацию и реализует её без уточняющих вопросов.

---

## Обязательный контекст перед написанием

Перед написанием спецификации агент **обязан** прочитать:

1. **`docs/specifications/arch-brief.md`** — карта ADR → файлы, схемы данных, hardware контекст
2. **Первичные источники**, указанные в заголовке спецификации (R01–R06, FLOW-XX, data-model.md)
3. **Текущий код** затрагиваемых файлов (через `hybrid_search_code` + `find_symbol`)

Использовать `repo-semantic-search` для поиска контекста. Не писать по памяти.

---

## Структура спецификации (обязательные секции)

```
# SPEC-RAG-XX: Название

> Версия, Дата, Статус, Цель (1 предложение), Источники

---

## 0. Implementation Pointers — что уже существует

### 0.1 Текущие файлы (что есть сейчас)
Таблица: файл → текущее поведение → что будет после

### 0.2 Новые файлы (создать)
Дерево новых файлов/модулей

### 0.3 Что удалить
Список файлов/блоков к удалению

---

## 1. Обзор

### 1.1 Задача
Numbered list: конкретные шаги реализации (1, 2, 3...)

### 1.2 Контекст — почему именно так
Объяснение решения. Ссылки на ADR.

### 1.3 Ключевые решения
Таблица: Решение | Выбор | Обоснование

### 1.4 Что НЕ делать
Explicit список анти-паттернов для этой задачи

---

## 2..N. Детальная реализация (по файлу/компоненту)

Для каждого файла/класса/функции:
- Полный Python-код реализации (не псевдокод, а готовый код)
- Сигнатуры методов с типами
- Docstring на русском (если тело > 5 строк)
- Инлайн-комментарии для нетривиальной логики

---

## N+1. Интеграция (deps.py и точки подключения)

Конкретный код изменений в deps.py (factory functions).

---

## N+2. Тесты

Минимальный набор тестов (unit/integration):
- Что проверяет каждый тест
- Полный код теста или точная заготовка

---

## N+3. Чеклист реализации

- [ ] Файл A создан
- [ ] Функция B реализована
- [ ] deps.py обновлён
- [ ] Старый код удалён
- [ ] Тест X проходит
```

---

## Требования к качеству кода в спецификации

- **Не псевдокод.** Каждый метод — рабочий Python с typing, обработкой ошибок, логированием.
- **Обработка ошибок.** `httpx.ConnectError`, `TimeoutException`, Qdrant exceptions — явно описать fallback.
- **Typing.** Все публичные методы аннотированы. `list[float]`, `dict[str, Any]`, `SparseVector` — конкретные типы.
- **Логирование.** `logger.info` на инициализацию, `logger.error` на исключения.
- **Async.** Все HTTP-клиенты — `async def` с `httpx.AsyncClient`.
- **Константы.** URL, таймауты, имена векторов — через `Settings`, не хардкодить.

---

## Требования к объёму

- **Целевой объём:** 400–900 строк на спецификацию
- **Меньше 400** — значит код неполный, есть pseudocode или пропущены секции
- **Больше 900** — значит спецификация слишком широкая, нужно разбить

---

## Чего НЕ должно быть в спецификации

- Дублирования arch-brief.md (не пересказывать ADR, только ссылаться)
- Предположений о будущих требованиях (только Phase 1 scope)
- Обратной совместимости с Phase 0 (ChromaDB/BM25 уходят полностью)
- Абстракций ради абстракций (не создавать ABC если один конкретный класс)
- Изменений в файлах вне scope спецификации
