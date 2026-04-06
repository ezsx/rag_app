# rag_app Agent Bootstrap

Этот файл должен оставаться коротким. Always-on guardrails и карта контекста.

## Preflight

**ОБЯЗАТЕЛЬНО перед существенной работой**: `agent_context/core/preflight.md`

- Определи `task type`: `debug / implementation / review / docs / research / eval`
- Подтяни нужные модули по триггерам из `preflight.md`
- Озвучь первый шаг перед exploration / edits

## Проект

`rag_app` — FastAPI-платформа RAG с агентским ReAct-пайплайном.
**Суть**: поисковик/агрегатор новостей из Telegram-каналов с применением RAG + ReAct.
- Telegram-каналы → ingest → **Qdrant** → Hybrid Retrieval (BM25+Dense+ColBERT) → ReAct Agent → SSE ответ

## Always-On

### Архитектура
- `docs/architecture/04-system/overview.md` — эталонная системная схема. Читать при неясности.
- LLM, ретриверы создаются через `lru_cache` в `src/core/deps.py`. Смена — через `settings.update_*()`.

### Код и модели
- LLM: **Qwen3.5-35B-A3B GGUF** Q4_K_M через llama-server.exe (V100, Windows Host, порт 8080). DEC-0039.
- Embedding: **pplx-embed-v1-0.6B** (bf16, mean pooling, без instruction prefix) через gpu_server.py (WSL2 native, RTX 5060 Ti, порт 8082). DEC-0042.
- Reranker: **Qwen3-Reranker-0.6B-seq-cls** (chat template, logit scoring) через gpu_server.py (порт 8082). DEC-0043.
- ColBERT: **jina-colbert-v2** (560M, 128-dim per-token MaxSim) через gpu_server.py (порт 8082).
- Vector store: **Qdrant** (Docker, CPU), dense + sparse + ColBERT named vectors, **weighted RRF** (BM25 3:1).
- **GPU blocker**: RTX 5060 Ti недоступна в Docker Desktop (V100 TCC блокирует NVML).
  GPU-сервисы = WSL2-native через gpu_server.py. Docker = CPU only. (DEC-0024)

### ReAct агент
- Оркестрация: native function calling через `/v1/chat/completions`.
- 15 LLM tools: `query_plan`, `search`, `temporal_search`, `channel_search`, `cross_channel_compare`, `summarize_channel`, `list_channels`, `rerank`, `related_posts`, `compose_context`, `final_answer`, `entity_tracker`, `arxiv_tracker`, `hot_topics`, `channel_expertise`.
- **Dynamic visibility**: phase-based (pre-search / post-search / analytics-complete / nav-complete), max 5 видимых, data-driven keyword routing из `datasets/tool_keywords.json`.
- **Forced search**: если LLM не вызывает tools, принудительный search. Bypass только для negative intent.
- **Original query injection**: оригинальный запрос пользователя всегда в subqueries (BM25 match).
- **Multi-query search**: все LLM subqueries через **MMR merge** (λ=0.7). Ablation phase 3.
- Retrieval: `query_plan → search (BM25 top-100 + dense top-**40** → weighted RRF 3:1 → ColBERT rerank) → **CE re-sort + adaptive filter** → channel dedup`.
- **Sparse normalization**: BM25 query нормализуется через lexicon (slang/aliases), dense — raw.
- **Ablation study**: 39+ экспериментов, R@5 0.833→0.900. Docs: `docs/progress/ablation_study.md`.
- Coverage: **LANCER nugget coverage** (query_plan subqueries = nuggets). Threshold **0.75**, max **1** targeted refinement. Module: `services/agent/coverage.py`.
- **Navigation short-circuit**: list_channels → navigation_answered → skip forced search, only final_answer visible.
- **Refusal policy**: explicit prompt rules + deterministic refusal trim + negative intent guard. Data-driven policies из `datasets/tool_keywords.json`.
- **Golden v2 baseline**: factual ~0.80, useful ~1.53/2, KTA 1.000 (36 Qs, consensus Claude+Codex). SPEC-RAG-18.
- **Eval pipeline v2** (SPEC-RAG-14): golden dataset v2 — 36 Qs (18 retrieval, 13 analytics, 2 navigation, 3 refusal), tool tracking, failure attribution.
- **Hot topics** (SPEC-RAG-16): BERTopic cron pipeline → Qdrant `weekly_digests` collection. `hot_topics` tool for trend queries.
- **Channel expertise** (SPEC-RAG-16): `channel_expertise` tool — per-channel profiles from `channel_profiles` collection.
- **Observability**: Langfuse v3 self-hosted (DEC-0040). UI `:3100`. 7 instrumentation points.
- Не ломать SSE контракт: `step_started/thought/tool_invoked/observation/citations/final`.

### Deploy и запуск
- **Docker GPU НЕ ИСПОЛЬЗУЕТСЯ.** V100 TCC отравляет NVML.
- **Порядок запуска**:
  1. Windows Host: `llama-server.exe` (V100, порт 8080, `--jinja --reasoning-budget 0 -c 32768`)
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
- Операционные планы → `docs/progress/`
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

- `agent_context/modules/debugging_protocol.md` — debugging/failure/unexpected behavior
- `agent_context/modules/parallel_agents.md` — review/handoff/multi-agent workflow
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
  services/tools/     — 15 LLM tools (search, entity_tracker, arxiv_tracker, hot_topics, channel_expertise, ...)
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
  research/           — Промпты и отчёты исследований (R00-R27, prompts 01-41)
  specifications/     — Спецификации (35 completed)
  progress/           — Project scope + experiment log (57 runs)
agent_context/        — Контекст для AI-агентов (Claude/Codex)
```
