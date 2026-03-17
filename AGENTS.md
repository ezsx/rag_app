# rag_app Agent Bootstrap

Этот файл должен оставаться коротким. Always-on guardrails и карта контекста.

## Проект

`rag_app` — FastAPI-платформа RAG с агентским ReAct-пайплайном.
**Суть**: поисковик/агрегатор новостей из Telegram-каналов с применением RAG + ReAct.
- Telegram-каналы → ingest → **Qdrant** (Phase 1) → Hybrid Retrieval → ReAct Agent → SSE ответ

## Always-On

### Архитектура
- `docs/architecture/04-system/overview.md` — эталонная системная схема. Читать при неясности.
- `docs/ai/agent_technical_spec.md` — детальная спецификация агента.
- LLM, ретриверы создаются через `lru_cache` в `src/core/deps.py`. Смена — через `settings.update_*()`.

### Код и модели
- LLM: **Qwen3-8B GGUF** через llama-server.exe (V100, Windows Host, порт 8080).
- Embedding: **multilingual-e5-large** через TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8082).
- Reranker: **bge-reranker-v2-m3** через TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8083).
- Vector store: **Qdrant** (Docker, CPU), dense + sparse named vectors, native RRF+MMR.
- **GPU blocker**: RTX 5060 Ti недоступна в Docker Desktop (V100 TCC блокирует NVML для всех GPU).
  Embedding/Reranker = WSL2-native. Docker-контейнеры = CPU only. (DEC-0024)

### ReAct агент
- Цикл: router_select → query_plan → search → rerank → compose_context → verify → final_answer.
- Coverage threshold **0.65**, max **2** refinements (DEC-0019). Не менять без ресерча.
- Не ломать SSE контракт событий: `thought/tool_invoked/observation/citations/final`.

### Deploy и запуск
- **Порядок запуска** (важно):
  1. Windows Host: `llama-server.exe` (V100, порт 8080)
  2. Ubuntu WSL2: TEI embedding (порт 8082) + TEI reranker (порт 8083) через `docker run --gpus all`
  3. Docker Desktop: `docker compose -f deploy/compose/compose.dev.yml up` (порт 8000, CPU)
- Ingest: `docker compose -f deploy/compose/compose.dev.yml run --rm ingest --channel @name --since YYYY-MM-DD --until YYYY-MM-DD`
- `.env` в корне — не коммитить секреты.

### Тесты
- `pytest` в контейнере: `docker compose -f deploy/compose/compose.test.yml run --rm test`
- Evaluation: `python scripts/evaluate_agent.py` (требует запущенного API).

### Безопасность
- Не логировать JWT, API-ключи, PII.
- `SecurityManager` / `sanitize_for_logging` — для внешнего ввода.
- Не использовать destructive git-команды без явного запроса.

### Стиль
- Python docstring на русском, если тело > ~5 строк.
- Комментарии на русском для ReAct цикла, RRF fusion, нетривиальной логики.

## Tool Policy

- MCP-first: сначала `repo-semantic-search`, затем `serena`, `ast-grep`, `ripgrep`.
- Читать файлы кусками и только после нахождения релевантного scope.

### repo-semantic-search — инструменты

| Инструмент | Назначение |
|-----------|-----------|
| `hybrid_search_code(query, path_prefix?, domain_tags?)` | Поиск по коду |
| `hybrid_search_docs(query, path_prefix?, domain_tags?)` | Поиск по документации |
| `hybrid_search(query, scope="all")` | Cross-domain поиск |
| `read_chunk(scope, chunk_id)` | Полный текст чанка |
| `index_status()` | Статус индекса |
| `rebuild_index()` | Полный пересброс индекса |
| `reindex_paths(paths)` | Точечная переиндексация |
| `update_include_globs(globs)` | Изменить что индексируется + rebuild |

**domain_tags** строятся из пути: `src/services/*` → `["src","services"]`, `src/api/*` → `["src","api"]`, `docs/*` → `["docs"]`, `scripts/*` → `["scripts"]`.

**Include globs** по умолчанию — авто-детект из структуры репо. При смене проекта — `update_include_globs(["auto"])` или `update_include_globs(["src/**","docs/**"])`.

**При смене репозитория** сервис автоматически удаляет коллекции старого проекта из Qdrant при рестарте.

## Task-Specific Context

Читай только нужный модуль:

- `agent_context/modules/agent.md` — ReAct агент, ToolRunner, инструменты, цикл
- `agent_context/modules/retrieval.md` — Qdrant, HybridRetriever, TEI embedding/reranker, QueryPlanner
- `agent_context/modules/ingest_eval.md` — Telegram ingest, evaluation скрипт, датасет
- `agent_context/modules/api.md` — FastAPI endpoints, SSE, JWT, DI, schemas

## Repository Layout

```
apps/
  api/                — Dockerfile + requirements для API
  ingest/             — Dockerfile + requirements для ingest
deploy/
  compose/            — services.yml, compose.dev.yml, compose.test.yml
src/
  api/v1/endpoints/   — FastAPI эндпоинты
  core/               — settings, deps, auth, security
  services/           — agent_service, qa_service, query_planner_service, reranker_service
  services/tools/     — 7+1 инструментов агента
  adapters/qdrant/    — QdrantStore
  adapters/search/    — HybridRetriever (Qdrant RRF)
  adapters/tei/       — TEIEmbeddingClient, TEIRerankerClient
  adapters/llm/       — LlamaServerClient
  schemas/            — Pydantic схемы
scripts/
  evaluate_agent.py   — CLI evaluation
  ingest_telegram.py  — Telegram → Qdrant ingestion
datasets/
  eval_dataset.json   — датасет для оценки
docs/                 — architecture, specifications, research, ai docs
agent_context/        — контекст для агентов (Claude/Codex)
```
