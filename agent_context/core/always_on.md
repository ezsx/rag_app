# Always-On Guardrails

Этот файл загружается всегда. Короткий: только правила, которые нельзя нарушать.

## Архитектура
- `docs/architecture/04-system/overview.md` — эталонная системная схема. Читать при неясности.
- `docs/architecture/00-meta/02-documentation-governance.md` — правила ведения документации. Следовать при создании/изменении файлов в docs/.
- LLM, ретриверы и сервисы создаются лениво через `lru_cache` в `src/core/deps.py`.
  Смена настроек требует явного `cache_clear()` через `settings.update_*()`.

## Код и модели
- Основной LLM: **Qwen3.5-35B-A3B GGUF** Q4_K_M (V100 SXM2 32GB, llama-server.exe на Windows хосте). DEC-0039.
- Embedding: **pplx-embed-v1-0.6B** (bf16, mean pooling, без instruction prefix) через gpu_server.py → WSL2 native (RTX 5060 Ti, порт 8082). DEC-0042.
- Reranker: **Qwen3-Reranker-0.6B-seq-cls** (chat template, padding_side=left, logit scoring) через gpu_server.py (порт 8082). DEC-0043.
- ColBERT: **jina-colbert-v2** (560M, 128-dim per-token MaxSim) через gpu_server.py (порт 8082).
- Хранилище: **Qdrant** (dense 1024 + sparse BM25 + ColBERT 128-dim, **weighted RRF** BM25 3:1).
- Коллекция: `news_colbert_v2` (enriched payload: entities, arxiv_ids, urls, year_week, lang + 16 payload indexes).
- Вспомогательные коллекции: `weekly_digests` (горячие темы по неделям), `channel_profiles` (экспертиза каналов).
- **Docker GPU blocker**: RTX 5060 Ti недоступна в Docker Desktop (TCC V100 блокирует NVML).
  Все GPU-модели запускаются нативно через gpu_server.py в Ubuntu WSL2.
  Детали: DEC-0024 в `docs/architecture/11-decisions/decision-log.md`.

## ReAct агент
- Оркестрация: native function calling через `/v1/chat/completions`, без regex-парсинга.
- LLM tools schema: 15 tools — `query_plan`, `search`, `temporal_search`, `channel_search`, `cross_channel_compare`, `summarize_channel`, `list_channels`, `rerank`, `related_posts`, `compose_context`, `final_answer`, `entity_tracker`, `arxiv_tracker`, `hot_topics`, `channel_expertise`.
- **Dynamic visibility**: phase-based (pre-search / post-search / analytics-complete / nav-complete), max 5 видимых. Signal + keyword routing из `datasets/tool_keywords.json`.
- **Analytics tools** (SPEC-RAG-15): `entity_tracker` (top/timeline/compare/co_occurrence) + `arxiv_tracker` (top/lookup). Facet API, point-level counts. Ответы без citations.
- **Digest tools** (SPEC-RAG-16): `hot_topics` (weekly digest из `weekly_digests` коллекции) + `channel_expertise` (профиль канала из `channel_profiles` коллекции). ANALYTICS-COMPLETE phase.
- **Forced search**: если LLM не вызывает tools, принудительный search. Bypass только для negative intent + refusal (не для обычных factual).
- **Original query injection**: оригинальный запрос всегда в subqueries для BM25 keyword match.
- **Multi-query search**: все LLM subqueries через **MMR merge** (λ=0.7, relevance + diversity). Заменил round-robin (ablation phase 3).
- **Planner language**: subqueries строго на языке запроса (fix: ранее генерировались на английском).
- `verify` и `fetch_docs` вызываются системно внутри `AgentService`, не через schema для модели.
- Retrieval: `query_plan → search (BM25 top-100 + dense top-**40** → RRF 3:1 → ColBERT rerank) → **CE re-sort + adaptive filter** → channel dedup (max 2/channel) → compose_context`.
- **Sparse normalization** (R2): BM25 query нормализуется через lexicon (`datasets/query_normalization_lexicon.json`), dense query остаётся raw. Ablation: R2 +0.009 R@1.
- Coverage: **LANCER-style nugget coverage** (query_plan subqueries как nuggets). Threshold **0.75** (3/4 nuggets); max refinements: **1** (targeted по uncovered nuggets). Модуль: `services/agent/coverage.py`.
- `agent_service.py` — единственный владелец состояния шага; не дублировать логику снаружи.
- **Request isolation** (SPEC-RAG-17 FIX-01): `RequestContext` на `ContextVar` — per-request state вместо instance-level. Каждый запрос изолирован.
- **Navigation short-circuit**: list_channels → `navigation_answered` → skip forced search, NAV-COMPLETE phase.
- **Analytics short-circuit**: entity_tracker/arxiv_tracker/hot_topics/channel_expertise → `analytics_done` → skip forced search, ANALYTICS-COMPLETE phase. Verify bypass для analytics-only ответов.
- **Refusal policy**: explicit prompt rules + temporal guard + deterministic refusal trim (обрезка альтернатив после отказа).
- **Forced search bypass**: только для negative intent queries + refusal markers. Data-driven из `datasets/tool_keywords.json` → `agent_policies`.
- SSE стриминг через `/v1/agent/stream` — не ломать контракт событий (step_started/thought/tool_invoked/observation/citations/final).
- **Recall@5**: v1=0.76, v2=0.685. **Golden v2 baseline**: factual ~0.80, useful ~1.53/2, KTA 1.000 (36 Qs, consensus Claude+Codex).
- **Eval pipeline v2** (SPEC-RAG-14): golden dataset v2 — 36 Qs (18 retrieval, 13 analytics, 2 navigation, 3 refusal), tool tracking, failure attribution. SPEC-RAG-18.
- **Observability**: Langfuse v3 self-hosted (DEC-0040). 7 instrumentation points. UI на `:3100`. Lazy imports + SafeSpan для graceful degradation.

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
