## Glossary

| Термин | Определение |
|--------|-------------|
| **ReAct** | Reasoning + Acting: цикл LLM (Thought → Action → Observation) для агентского поиска |
| **native function calling** | LLM вызывает tools через `tools` parameter в `/v1/chat/completions` (OpenAI-compatible), без regex-парсинга текста |
| **SSE** | Server-Sent Events: однонаправленный push от сервера к клиенту (EventSource) |
| **coverage** | Float 0–1: composite из 6 cosine-сигналов о достаточности контекста. Если < `coverage_threshold` → refinement. Вычисляется в `compose_context`, требует `with_vectors=True` в Qdrant запросе. |
| **coverage_threshold** | `0.65` — порог для запуска refinement (DEC-0019). Bias toward retrieval: false-negative → 66% галлюцинаций. |
| **composite coverage** | Weighted sum: `max_sim×0.25 + mean_top_k×0.20 + term_coverage×0.20 + doc_count_adequacy×0.15 + score_gap×0.15 + above_threshold_ratio×0.05`. Использует cosine similarity, НЕ RRF-скоры. |
| **refinement** | Дополнительный поисковый раунд при `coverage < coverage_threshold` |
| **max_refinements** | `2` — максимальное число refinement-раундов (DEC-0019) |
| **RRF** | Reciprocal Rank Fusion: `score(d) = Σ 1/(k+rank_i)`, k=60. Weighted RRF: BM25 weight=3, dense weight=1. |
| **ColBERT** | Contextualized Late Interaction over BERT: per-token vectors + MaxSim scoring. Фундаментально решает attractor document problem. |
| **MaxSim** | ColBERT scoring: для каждого query token — максимальное dot product с document tokens, затем сумма. |
| **jina-colbert-v2** | ColBERT модель (560M, 89 языков, 128-dim per token). Загружается в gpu_server.py с manual linear projection 1024→128. |
| **channel dedup** | Post-retrieval: max 2 документа из одного канала. Улучшает diversity, не recall. |
| **round-robin merge** | Multi-query result interleaving: для rank 0..N, чередуем результаты из каждого subquery. Сохраняет per-query ColBERT ranking. |
| **attractor document** | Документ с высоким cosine similarity к большинству запросов (из-за embedding anisotropy). ColBERT + weighted RRF решают проблему. |
| **Qwen3-Reranker** | Qwen3-Reranker-0.6B-seq-cls: chat template reranker, logit scoring через seq-cls head. Запускается на GPU (RTX 5060 Ti). |
| **HybridRetriever** | `src/adapters/search/hybrid_retriever.py`: BM25 top-100 + dense top-20 → weighted RRF (3:1) → ColBERT MaxSim rerank → channel dedup. Fallback на RRF-only если ColBERT недоступен. |
| **gpu_server.py** | `scripts/gpu_server.py`: HTTP-сервер (stdlib http.server + PyTorch cu128) с 3 моделями: pplx-embed-v1-0.6B + Qwen3-Reranker-0.6B-seq-cls + jina-colbert-v2. Единый порт :8082. |
| **Qwen3.5-35B-A3B** | Основная LLM: MoE модель (35B total, 3B active params). GGUF Q4_K_M через llama-server на V100. Native function calling через `--jinja --reasoning-budget 0`. DEC-0039. |
| **pplx-embed-v1-0.6B** | Embedding модель (Perplexity, bf16, mean pooling, 1024-dim). Без instruction prefix. Через gpu_server.py на RTX 5060 Ti. DEC-0042. |
| **news_colbert_v2** | Qdrant коллекция: 3 named vectors (dense 1024 + sparse BM25 + colbert 128-dim multi-vector), 13777 точки из 36 каналов, 16 payload indexes. |
| **AgentService** | `src/services/agent_service.py`: единственный владелец ReAct цикла и SSE стриминга |
| **ToolRunner** | `src/services/tools/tool_runner.py`: реестр инструментов + единый запуск с timeout |
| **AgentState** | Внутреннее состояние одного запроса агента: coverage, refinement_count, search_count |
| **compose_context** | Инструмент: собирает контекст из hit_ids → prompt + citations + composite coverage |
| **verify** | Системный инструмент (не LLM tool): проверяет утверждения через допоиск в KB |
| **QueryPlannerService** | `src/services/query_planner_service.py`: LLM разбор запроса на sub-queries. Тот же V100 endpoint. |
| **SearchPlan** | Pydantic модель: subqueries из query planner |
| **lru_cache singleton** | Паттерн в `src/core/deps.py`: сервисы создаются один раз, кэшируются через `@lru_cache` |
| **GGUF** | Формат файлов для llama.cpp: квантованные LLM для CPU/GPU inference |
| **llama-server** | HTTP-сервер из llama.cpp (`llama-server.exe`), OpenAI-compatible API на V100. |
| **LlamaServerClient** | `src/adapters/llm/llama_server_client.py`: async HTTP-обёртка над llama-server (httpx.AsyncClient). |
| **Qdrant** | Vector database: named vectors (dense + sparse + ColBERT), нативный weighted RRF. |
| **evaluate_agent.py** | Agent eval скрипт: full pipeline через LLM, ~40с/запрос, recall@5 + coverage + latency. |
| **evaluate_retrieval.py** | Retrieval eval скрипт: прямые Qdrant queries без LLM, ~5с/запрос, recall@1/5/10/20. |
