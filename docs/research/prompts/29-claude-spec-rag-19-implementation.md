# Prompt: SPEC-RAG-19 Langfuse Implementation

## Контекст сессии (2026-03-30)

Длинная сессия: R25 deep research, SPEC-RAG-18 (golden_v2 eval), Qwen3.5 model swap, judge consensus, routing fixes, auth removal, observability spec. Context заканчивается, нужен compact + fresh start на имплементацию.

## Что сделано сегодня

1. **R25 Deep Research** — production gap analysis vs Perplexity/Glean/Danswer. Retrieval на уровне, gaps в observability/eval/CRAG-lite
2. **SPEC-RAG-18** — golden_v2 dataset (36 Qs), offline judge workflow, реализован Codex'ом, ревью + debugging routing
3. **Qwen3.5-35B-A3B swap** — новая модель вместо Qwen3-30B-A3B. Починила q01 (false refusal), q21 (temporal refusal), q33 (tool selection). Команда запуска: `$env:CUDA_VISIBLE_DEVICES = "0"` + llama-server на V100
4. **Routing fixes** — q32 keywords, q33 tool description, q36 keywords (channel_expertise vs list_channels)
5. **Judge consensus** — Claude + Codex: factual ~0.80/1, useful ~1.53/2, KTA 1.000 (на старой Qwen3 модели)
6. **Auth removal** — убран весь JWT/token код из endpoints, router, UI, eval script
7. **Baseline skip** — evaluate_agent.py: `--run-baseline` opt-in (default off), убирает двойной инференс
8. **R28** — observability research (Langfuse vs Phoenix vs structlog)
9. **SPEC-RAG-19** — написан, прошёл 2 раунда ревью Codex, все findings исправлены. Ready for implementation

## Что нужно сделать СЕЙЧАС

**Реализовать SPEC-RAG-19: Langfuse observability integration.**

Spec: `docs/specifications/active/SPEC-RAG-19-observability-langfuse.md`

### Порядок реализации

1. **Docker Compose** — создать `deploy/compose/compose.langfuse.yml` (6 сервисов: web, worker, postgres, clickhouse, redis, minio). Шаблон в R28 §Phase 3
2. **`src/core/observability.py`** — lazy imports, graceful degradation, `observe_span()` и `observe_llm_call()` context managers
3. **`src/adapters/llm/llama_server_client.py`** — manual instrumentation через `observe_llm_call` в `__call__()` и `chat_completion()`
4. **`src/services/agent_service.py`** — root span в `stream_agent_response` (explicit enter/exit в finally, не with — SSE async safety)
5. **`src/services/tools/tool_runner.py`** — per-tool spans через `observe_span`
6. **`src/adapters/search/hybrid_retriever.py`** — retrieval span
7. **`src/adapters/tei/reranker_client.py`** — rerank spans
8. **`src/services/query_planner_service.py`** — planner span
9. **Verification** — поднять Langfuse, запустить 1 agent request, проверить trace в UI

### Ключевые design decisions из spec

- **Все langfuse imports ТОЛЬКО в `src/core/observability.py`** — lazy, catch ImportError
- **Runtime модули** импортируют только `from core.observability import observe_span, observe_llm_call`
- **LlamaServerClient** использует `requests.Session`, НЕ OpenAI SDK — drop-in wrapper не применим, manual instrumentation
- **SSE generator** — explicit span enter/exit в finally блоке, не with statement (ContextVar cleanup issue)
- **Networking** — API → Langfuse через `host.docker.internal:3000`
- **Graceful degradation** — если langfuse не установлен или сервер down → nullcontext, zero impact

## Текущее состояние инфры

- **LLM**: Qwen3.5-35B-A3B Q4_K_M на V100 (`$env:CUDA_VISIBLE_DEVICES = "0"`, порт 8080)
- **GPU server**: gpu_server.py на RTX 5060 Ti (embedding + reranker + ColBERT, порт 8082)
- **Docker**: compose.dev.yml (API порт 8001/8000, Qdrant порт 6333/16333)
- **Auth**: УДАЛЁН полностью — endpoints без JWT, eval без --api-key
- **Eval**: evaluate_agent.py с --run-baseline opt-in (default skip)

## Файлы для чтения перед началом

1. `docs/specifications/active/SPEC-RAG-19-observability-langfuse.md` — полная spec
2. `docs/research/reports/R28-deep-observability-langfuse-phoenix-structlog.md` — docker-compose template, SDK examples
3. `src/adapters/llm/llama_server_client.py` — текущий LLM client (requests-based)
4. `src/services/agent_service.py` — agent loop, stream_agent_response, ContextVar pattern
5. `src/services/tools/tool_runner.py` — tool execution
6. `src/adapters/search/hybrid_retriever.py` — retrieval

## Pending задачи (после observability)

1. **Полный eval прогон golden_v2 на Qwen3.5** — без baseline, с judge artifacts
2. **Judge consensus** на новом прогоне
3. **CRAG-lite spec** — на базе R13-deep, R14-deep, R25
4. **Eval expansion** 36→100 Qs
5. **NLI faithfulness** (R19)
6. **NDR/RSR/ROR** (R20)
