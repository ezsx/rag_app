## Glossary

| Термин | Определение |
|--------|-------------|
| **ReAct** | Reasoning + Acting: цикл LLM (Thought → Action → Observation) для агентского поиска |
| **SSE** | Server-Sent Events: однонаправленный push от сервера к клиенту (EventSource) |
| **coverage** | Float 0–1: composite из 5 cosine-сигналов о достаточности контекста. Если < `coverage_threshold` → refinement. Вычисляется в `compose_context`, требует `with_vectors=True` в Qdrant запросе. |
| **coverage_threshold** | `0.65` — порог для запуска refinement. Был 0.80 (слишком агрессивный). Bias toward retrieval: false-negative (пропущенный поиск) → 66% галлюцинаций. |
| **composite coverage** | Weighted sum: `max_sim×0.25 + mean_top_k×0.20 + term_coverage×0.20 + doc_count_adequacy×0.15 + score_gap×0.15 + above_threshold_ratio×0.05`. Использует cosine similarity, НЕ RRF-скоры. |
| **refinement** | Дополнительный поисковый раунд при `coverage < coverage_threshold` |
| **max_refinements** | `2` — максимальное число refinement-раундов. Был 1; F1 растёт до 3 итераций. |
| **RRF** | Reciprocal Rank Fusion: `score(d) = Σ 1/(k+rank_i)`, k=60. Используется для ранжирования, **не** для coverage estimation (max RRF ≈ 0.0328, не cross-query сравним). |
| **MMR** | Maximum Marginal Relevance: баланс relevance и diversity. Нативен в Qdrant с v1.15.0 (`rescore: mmr`). |
| **BGE reranker** | BAAI/bge-reranker-v2-m3: CrossEncoder для финального ранжирования кандидатов |
| **HybridRetriever** | `src/adapters/search/hybrid_retriever.py`: единый вызов `qdrant_client.query_points()` с prefetch (dense + sparse) + FusionQuery(RRF). Заменил ChromaRetriever + BM25Retriever. |
| **QdrantRetriever** | Qdrant-based hybrid retriever: dense (`multilingual-e5-large`) + sparse (`Qdrant/bm25`, `language="russian"`) в одной named-vectors коллекции. |
| **AgentService** | `src/services/agent_service.py`: единственный владелец ReAct цикла и SSE стриминга |
| **ToolRunner** | `src/services/tools/tool_runner.py`: реестр инструментов + единый запуск с timeout и JSON-трейсом |
| **AgentState** | Внутреннее состояние одного запроса агента: coverage, refinement_count |
| **compose_context** | Инструмент: собирает контекст из hit_ids → prompt + citations + composite coverage. Принимает `query` для cosine computation. |
| **verify** | Инструмент: проверяет утверждения через дополнительный поиск в KB |
| **QueryPlannerService** | `src/services/query_planner_service.py`: LLM разбор запроса на sub-queries + filters. Использует тот же V100 endpoint (не отдельный CPU процесс). |
| **SearchPlan** | Pydantic модель: normalized_queries + filters из query planner |
| **lru_cache singleton** | Паттерн в `src/core/deps.py`: сервисы создаются один раз, кэшируются через `@lru_cache` |
| **GGUF** | Формат файлов для llama.cpp: квантованные LLM для CPU/GPU inference |
| **llama-server** | HTTP-сервер из llama.cpp (`llama-server.exe`), OpenAI-compatible API на хосте. Обращение через `LlamaServerClient` (httpx.AsyncClient). |
| **LlamaServerClient** | `src/adapters/llm/llama_server_client.py`: async HTTP-обёртка над llama-server. Использует `httpx.AsyncClient`. |
| **multilingual-e5-large** | HF embedding модель для dense retrieval, поддерживает RU+EN (1024 dims) |
| **Qdrant** | Vector database: named vectors (dense + sparse), нативный RRF (prefetch+FusionQuery), нативный MMR. Заменяет ChromaDB + BM25IndexManager. |
| **qdrant_collection** | Qdrant коллекция с новостями: named vectors (dense, sparse), payload {channel, date, author, message_id, text} |
| **eval_dataset.json** | `datasets/eval_dataset.json`: набор вопросов с expected_documents. Генерируется из Qdrant через `generate_eval_dataset.py`. Минимум 200 примеров, 5 типов с весами. |
| **AgentStepEvent** | Pydantic схема SSE события: type ∈ {thought, tool_invoked, observation, citations, final} |
| **Qwen3-8B** | Основная LLM: GGUF (Q8_0 или F16) через llama-server на V100. Заменяет Qwen2.5-7B (agent) и Qwen2.5-3B (planner). Thinking mode ОТКЛЮЧЁН (`/no_think` в system prompt). |
| **thinking mode** | Режим Qwen3, при котором модель эмитирует `<think>...</think>` блоки перед ответом. **Отключён** — ломает ReAct-парсер и тратит 250–1250 токенов. Управляется через `/no_think` в system prompt (llama-server) или `extra_body={"enable_thinking": False}` (vLLM). |
| **vLLM** | Целевой LLM-сервер после Proxmox (Phase 2). v0.15.1 pinned для V100 (SM7.0). Даёт xgrammar, prefix caching, нативный Hermes tool calling. Требует Linux. |
| **Hermes tool calling** | Нативный function calling Qwen3: `<tool_call>` XML теги, `--tool-call-parser hermes`. Требует AgentService rewrite. Целевое состояние после vLLM. |
| **cosine threshold** | Интерпретация cosine similarity: 0.85+ paraphrase, 0.75–0.85 strong, 0.60–0.75 moderate, 0.45–0.60 tangential, <0.45 irrelevant, <0.30 → abort (insufficient information). |
| **LLM-judge** | Qwen3-8B как судья для eval: faithfulness, relevance, completeness, citation accuracy. Промпты на русском. Обёртка в DeepEval BaseMetric для CI/CD. |
