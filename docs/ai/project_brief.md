# Обзор проекта

`rag_app` — FastAPI-платформа Retrieval-Augmented Generation для поиска и агрегации новостей из Telegram-каналов.
Phase 1 стек: Telegram ingest → Qdrant (dense+sparse) → Hybrid Retrieval → AgentService → SSE-ответ.

## Архитектура
1. **Vector Store**: Qdrant с named vectors `dense_vector` и `sparse_vector`, native RRF+MMR.
2. **LLM**: Qwen3-30B-A3B GGUF через `llama-server.exe` на Windows Host (V100, OpenAI-compatible API).
3. **Embedding**: `Qwen/Qwen3-Embedding-0.6B` через TEI HTTP в WSL2.
4. **Reranker**: `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` через TEI HTTP в WSL2.
5. **Sparse**: `fastembed.SparseTextEmbedding("Qdrant/bm25", language="russian")`.
6. **Docker**: API и Qdrant работают на CPU; GPU в Docker Desktop недоступна по DEC-0024.

## Агент

`src/services/agent_service.py` использует native function calling через `/v1/chat/completions`.

- LLM-visible tools: `query_plan`, `search`, `rerank`, `compose_context`, `final_answer`.
- Системные инструменты: `verify`, `fetch_docs`.
- Retrieval path: `search → rerank → compose_context`.
- SSE контракт остаётся прежним: `thought`, `tool_invoked`, `observation`, `citations`, `final`.
- Coverage threshold: **0.65**; max refinements: **2**.
- После `compose_context` агент применяет deterministic guardrails:
  - composite coverage check
  - abort guard по `max_sim < 0.30`
  - refinement search при недостаточном coverage
  - hedged disclaimer при низком coverage после исчерпания refinements

## Ingest

- Короткие посты `<1500` символов индексируются целиком.
- Длинные посты режутся через `_smart_chunk()` с recursive split и target `1200` символов.
- При chunking point id имеет вид `{channel}:{message_id}:{chunk_idx}`.

## Сервисы

- `qa_service` — baseline QA поверх retrieval.
- `agent_service` — function-calling оркестрация и SSE-стриминг.
- `query_planner_service` — декомпозиция запроса.
- `reranker_service` — sync bridge над async `TEIRerankerClient`.

## Deploy

Порядок запуска:
1. Windows Host: `llama-server.exe` c `--jinja --reasoning-budget 0`
2. Ubuntu WSL2: TEI embedding `:8082` и reranker `:8083`
3. Docker Desktop: `docker compose -f deploy/compose/compose.dev.yml up`

Ingest запускается отдельно:
`docker compose -f deploy/compose/compose.dev.yml run --rm ingest --channel @name`
