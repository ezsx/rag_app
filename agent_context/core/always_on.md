# Always-On Guardrails

Этот файл загружается всегда. Короткий: только правила, которые нельзя нарушать.

## Архитектура
- `docs/architecture/04-system/overview.md` — эталонная системная схема. Читать при неясности.
- `docs/architecture/00-meta/02-documentation-governance.md` — правила ведения документации. Следовать при создании/изменении файлов в docs/.
- LLM, ретриверы и сервисы создаются лениво через `lru_cache` в `src/core/deps.py`.
  Смена настроек требует явного `cache_clear()` через `settings.update_*()`.

## Код и модели
- Основной LLM: **Qwen3-30B-A3B GGUF** (V100 SXM2 32GB, llama-server.exe на Windows хосте).
- Embedding: **Qwen3-Embedding-0.6B** через gpu_server.py → WSL2 native (RTX 5060 Ti, порт 8082).
- Reranker: **bge-reranker-v2-m3** (dedicated cross-encoder) через gpu_server.py (порт 8082).
- ColBERT: **jina-colbert-v2** (560M, 128-dim per-token MaxSim) через gpu_server.py (порт 8082).
- Хранилище: **Qdrant** (dense 1024 + sparse BM25 + ColBERT 128-dim, **weighted RRF** BM25 3:1).
- Коллекция: `news_colbert_v2` (enriched payload: entities, arxiv_ids, urls, year_week, lang + 16 payload indexes).
- **Docker GPU blocker**: RTX 5060 Ti недоступна в Docker Desktop (TCC V100 блокирует NVML).
  Все GPU-модели запускаются нативно через gpu_server.py в Ubuntu WSL2.
  Детали: DEC-0024 в `docs/architecture/11-decisions/decision-log.md`.

## ReAct агент
- Оркестрация: native function calling через `/v1/chat/completions`, без regex-парсинга.
- LLM tools schema: 11 tools — `query_plan`, `search`, `temporal_search`, `channel_search`, `cross_channel_compare`, `summarize_channel`, `list_channels`, `rerank`, `related_posts`, `compose_context`, `final_answer`.
- **Dynamic visibility**: phase-based (pre-search / post-search), max 5 видимых. Signal + keyword routing.
- **Forced search**: если LLM не вызывает tools, принудительный search с оригинальным запросом.
- **Original query injection**: оригинальный запрос всегда в subqueries для BM25 keyword match.
- **Multi-query search**: все LLM subqueries через round-robin merge (не только первый!).
- `verify` и `fetch_docs` вызываются системно внутри `AgentService`, не через schema для модели.
- Retrieval: `query_plan → search (BM25 top-100 + dense top-20 → RRF 3:1 → ColBERT rerank) → cross-encoder rerank → channel dedup (max 2/channel) → compose_context`.
- Coverage threshold: **0.65**; max refinements: **2** (DEC-0019; не менять без ресерча).
- `agent_service.py` — единственный владелец состояния шага; не дублировать логику снаружи.
- **Navigation short-circuit**: list_channels → `navigation_answered` → skip forced search, NAV-COMPLETE phase.
- **Refusal policy**: explicit prompt rules + temporal guard в `_execute_action` (dates outside corpus → empty hits).
- **Forced search bypass**: если LLM content содержит refusal markers → не форсить search.
- SSE стриминг через `/v1/agent/stream` — не ломать контракт событий (step_started/thought/tool_invoked/observation/citations/final).
- **Recall@5**: v1=0.76, v2=0.685, golden_v1=~0.43 (strict, занижен). Manual judge: factual=0.52, useful=1.14/2.
- **Eval pipeline v2** (SPEC-RAG-14): golden dataset 25 Qs, tool tracking, failure attribution, LLM judge. Подробности: `docs/specifications/active/SPEC-RAG-14-evaluation-pipeline.md`.

## Deploy и запуск
- **ВАЖНО: Docker GPU НЕ ИСПОЛЬЗУЕТСЯ.** V100 TCC отравляет NVML в WSL2.
  Подробности: `docs/research/reports/R10-gpu-docker-wsl2-troubleshooting.md`.
- **Порядок запуска:**
  1. llama-server.exe на Windows хосте (V100, порт 8080):
     `--jinja --reasoning-budget 0 --cache-type-k q8_0 --cache-type-v q8_0 -c 32768`
  2. gpu_server.py нативно в WSL2 (RTX 5060 Ti, порт 8082):
     `source /home/ezsx/infinity-env/bin/activate && CUDA_VISIBLE_DEVICES=0 python scripts/gpu_server.py`
     Embedding + Reranker + ColBERT в одном процессе. PyTorch cu128 + cuBLAS.
  3. Docker Desktop (CPU only): `docker compose -f deploy/compose/compose.dev.yml up`
- Ingest: `docker compose -f deploy/compose/compose.dev.yml run --rm ingest --channel @name --since YYYY-MM-DD --until YYYY-MM-DD`
- `.env` в корне репозитория — не коммитить, не логировать plaintext-секреты.

## Тесты
- Основные тесты в `src/tests/`.
- `pytest` в контейнере: `docker compose -f deploy/compose/compose.test.yml run --rm test`.
- Agent eval: `python scripts/evaluate_agent.py` (требует запущенного API).
- Retrieval eval: `python scripts/evaluate_retrieval.py` (прямые Qdrant queries).

## Безопасность
- Не логировать JWT-токены, API-ключи, промпты с PII.
- `SecurityManager` и `sanitize_for_logging` — использовать для всего внешнего ввода.
- Не использовать destructive git-команды без явного запроса.

## Стиль кода
- Python docstring на русском языке, если тело функции длиннее ~5 строк.
- Комментарии на русском для нетривиальной логики (особенно в ReAct цикле, RRF fusion).
