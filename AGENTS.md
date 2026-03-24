# rag_app Agent Bootstrap

Этот файл должен оставаться коротким. Always-on guardrails и карта контекста.

## Проект

`rag_app` — FastAPI-платформа RAG с агентским ReAct-пайплайном.
**Суть**: поисковик/агрегатор новостей из Telegram-каналов с применением RAG + ReAct.
- Telegram-каналы → ingest → **Qdrant** → Hybrid Retrieval (BM25+Dense+ColBERT) → ReAct Agent → SSE ответ

## Always-On

### Архитектура
- `docs/architecture/04-system/overview.md` — эталонная системная схема. Читать при неясности.
- LLM, ретриверы создаются через `lru_cache` в `src/core/deps.py`. Смена — через `settings.update_*()`.

### Код и модели
- LLM: **Qwen3-30B-A3B GGUF** через llama-server.exe (V100, Windows Host, порт 8080).
- Embedding: **Qwen3-Embedding-0.6B** через gpu_server.py (WSL2 native, RTX 5060 Ti, порт 8082).
- Reranker: **bge-reranker-v2-m3** (dedicated cross-encoder) через gpu_server.py (порт 8082).
- ColBERT: **jina-colbert-v2** (560M, 128-dim per-token MaxSim) через gpu_server.py (порт 8082).
- Vector store: **Qdrant** (Docker, CPU), dense + sparse + ColBERT named vectors, **weighted RRF** (BM25 3:1).
- **GPU blocker**: RTX 5060 Ti недоступна в Docker Desktop (V100 TCC блокирует NVML).
  GPU-сервисы = WSL2-native через gpu_server.py. Docker = CPU only. (DEC-0024)

### ReAct агент
- Оркестрация: native function calling через `/v1/chat/completions`.
- 11 LLM tools: `query_plan`, `search`, `temporal_search`, `channel_search`, `cross_channel_compare`, `summarize_channel`, `list_channels`, `rerank`, `related_posts`, `compose_context`, `final_answer`.
- **Dynamic visibility**: phase-based (pre-search / post-search), max 5 видимых, signal + keyword routing.
- **Forced search**: если LLM не вызывает tools, принудительный search с оригинальным запросом.
- **Original query injection**: оригинальный запрос пользователя всегда в subqueries (BM25 match).
- **Multi-query search**: все LLM subqueries через round-robin merge.
- Retrieval: `query_plan → search (BM25 top-100 + dense top-20 → weighted RRF 3:1 → ColBERT rerank) → cross-encoder rerank → channel dedup`.
- Coverage threshold **0.65**, max **2** refinements (DEC-0019).
- **Recall@5**: v1=0.76, v2=0.61, retrieval=0.73. Target 0.80+ с adaptive retrieval.
- Не ломать SSE контракт: `thought/tool_invoked/observation/citations/final`.

### Deploy и запуск
- **Docker GPU НЕ ИСПОЛЬЗУЕТСЯ.** V100 TCC отравляет NVML.
- **Порядок запуска**:
  1. Windows Host: `llama-server.exe` (V100, порт 8080, `--jinja --reasoning-budget 0`)
  2. WSL2 native: `gpu_server.py` — embedding + reranker + ColBERT (RTX 5060 Ti, порт 8082)
  3. Docker Desktop (CPU only): `docker compose -f deploy/compose/compose.dev.yml up`
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

## Documentation Governance

**ОБЯЗАТЕЛЬНО**: `docs/architecture/00-meta/02-documentation-governance.md`

При создании/изменении файлов — следовать правилам размещения:
- Research → `docs/research/` (prompts/ и reports/)
- Specification → `docs/specifications/` (active/ и completed/)
- Текущее состояние системы → `docs/architecture/`
- Операционные планы → `docs/planning/`
- **Не создавать файлы в других местах без согласования.**

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

## Task-Specific Context

Читай только нужный модуль:

- `agent_context/modules/agent.md` — ReAct агент, ToolRunner, инструменты, цикл
- `agent_context/modules/retrieval.md` — Qdrant, HybridRetriever, TEI embedding/reranker, QueryPlanner
- `agent_context/modules/ingest_eval.md` — Telegram ingest, evaluation скрипт, датасет
- `agent_context/modules/api.md` — FastAPI endpoints, SSE, JWT, DI, schemas

## Repository Layout

```
src/
  api/v1/endpoints/   — FastAPI эндпоинты
  core/               — settings, deps, auth, security
  services/           — agent_service, qa_service, query_planner, reranker
  services/tools/     — 11 LLM tools (search, cross_channel_compare, summarize_channel, list_channels, related_posts, ...)
  adapters/qdrant/    — QdrantStore
  adapters/search/    — HybridRetriever (Qdrant weighted RRF + ColBERT)
  adapters/tei/       — TEIEmbeddingClient, TEIRerankerClient
  adapters/llm/       — LlamaServerClient
  schemas/            — Pydantic схемы
  static/             — Web UI (index.html)
  tests/              — pytest тесты
apps/
  api/                — Dockerfile + requirements для API
  ingest/             — Dockerfile + requirements для ingest
deploy/
  compose/            — services.yml, compose.dev.yml, compose.test.yml
  mcp.env             — env для repo-semantic-search MCP
scripts/
  gpu_server.py       — Embedding + Reranker + ColBERT HTTP API (PyTorch cu128, RTX 5060 Ti)
  evaluate_agent.py   — Agent eval (full pipeline через LLM)
  evaluate_retrieval.py — Retrieval eval (прямые Qdrant queries)
  ingest_telegram.py  — Telegram → Qdrant ingestion
  validate_channels.py — Валидация доступности каналов
datasets/
  eval_dataset_quick.json    — Golden dataset v1 (10 Qs)
  eval_dataset_quick_v2.json — Golden dataset v2 (10 Qs, сложные)
  eval_retrieval_100.json    — Retrieval eval (100 auto-generated Qs)
sessions/             — Telethon session (для Telegram ingest)
docs/
  architecture/       — Источник правды: текущее состояние системы
  research/           — Промпты и отчёты исследований (R01-R14)
  specifications/     — Спецификации (active/ и completed/)
  planning/           — Операционные документы (scope, playbook, планы)
agent_context/        — Контекст для AI-агентов (Claude/Codex)
```
