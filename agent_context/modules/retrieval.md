# Retrieval Module — RAG Pipeline (Phase 1)

## Ключевые файлы

- `src/adapters/search/hybrid_retriever.py` — `HybridRetriever` (Qdrant weighted RRF: BM25 3:1)
- `src/adapters/qdrant/store.py` — `QdrantStore` (dense+sparse, named vectors)
- `src/adapters/tei/embedding_client.py` — `TEIEmbeddingClient` (async httpx → gpu_server.py :8082)
- `src/adapters/tei/reranker_client.py` — `TEIRerankerClient` (async httpx → gpu_server.py :8082)
- `src/services/query_planner_service.py` — GBNF-based планировщик подзапросов
- `src/services/reranker_service.py` — sync bridge над async TEIRerankerClient
- `scripts/gpu_server.py` — embedding + reranker HTTP API (PyTorch cu128, RTX 5060 Ti)
- `src/utils/ranking.py` — `rrf_merge`, `mmr_select` (mmr_select не используется — cosine MMR re-promotes attractors)
- `src/core/settings.py` — все гиперпараметры поиска
- `src/core/deps.py` — DI: `get_hybrid_retriever`, `get_reranker`, `get_query_planner`

## Pipeline (поиск в агенте)

```
query
  → QueryPlanner (Qwen3-30B-A3B, llama-server, GBNF grammar)
      → нормализованные подзапросы + MetadataFilters
  → search tool (original query всегда добавляется в subqueries)
      → HybridRetriever.search_with_plan()
          → TEI embed (Qwen3-Embedding-0.6B, instruction prefix для query)
          → fastembed sparse (Qdrant/bm25, language="russian")
          → Qdrant prefetch:
              ├─ dense_vector: top-20
              └─ sparse_vector: top-100
          → Weighted RRF (BM25 weight=3, dense weight=1)
          → top-N кандидатов (with_vectors=True для coverage cosine_sim)
  → Reranker (gpu_server.py → BAAI/bge-m3 AutoModelForSequenceClassification)
      Целевой: bge-reranker-v2-m3 (dedicated cross-encoder, +10 nDCG)
  → compose_context → coverage check (0.65, DEC-0019)
```

**ВАЖНО**: Dense re-score после RRF **убран** — стирал BM25 вклад (recall 0.33→0.15).
MMR post-processing **отключён** — cosine-based MMR re-promotes "attractor documents" (recall 0.33→0.11).

## Recall@5 = 0.70 (quick dataset, 10 вопросов)

Путь улучшений:
- 0.15 → 0.33: убрали dense re-score после RRF
- 0.33 → 0.59: оригинальный запрос в subqueries (BM25 keyword match)
- 0.59 → 0.70: weighted RRF (BM25 3:1) + forced search + dynamic tools

## Ключевые настройки

```python
# QueryPlanner
use_gbnf_planner      = True
planner_timeout       = 15.0
max_plan_subqueries   = 5

# Hybrid search (Qdrant)
hybrid_enabled        = True
hybrid_top_dense      = 20     # prefetch limit для dense
hybrid_top_sparse     = 100    # prefetch limit для sparse (BM25 — primary signal)
# Weighted RRF: BM25 weight=3, dense weight=1 (через RrfQuery)

# MMR — ОТКЛЮЧЁН (cosine-based MMR re-promotes attractor documents)
enable_mmr            = True   # в settings, но HybridRetriever не использует
mmr_lambda            = 0.7
# TODO: BM25-based diversity вместо cosine MMR

# Reranker (gpu_server.py HTTP)
enable_reranker       = True
reranker_tei_url      = "http://host.docker.internal:8082"  # ПОРТ 8082 (не 8083!)
reranker_top_n        = 80
```

## Embedding

- **Qwen3-Embedding-0.6B** — текущая модель (1024-dim, cosine)
- Через gpu_server.py HTTP (WSL2 native, RTX 5060 Ti, порт 8082)
- Query format: `Instruct: Given a user question about ML, AI, LLM or tech news, retrieve relevant Telegram channel posts\nQuery: ...`
- Documents идут без prefix
- **Проблема**: embedding anisotropy — все AI-тексты в cosine range [0.78-0.83]. "Attractor documents" попадают в top-10 любого запроса.
- **Планируемый fix**: Global PCA whitening (1024→512 dim), ожидание +5-15% recall

## Qdrant

- URL: `QDRANT_URL` (default: `http://qdrant:6333` в Docker)
- Named vectors: `dense_vector` (cosine, 1024-dim) + `sparse_vector` (Qdrant/bm25, russian)
- Коллекция: `news` (13124 точки, 36 каналов)
- Storage: named volume `qdrant_data`

## Sparse Encoding

- fastembed `SparseTextEmbedding` (модель: `Qdrant/bm25`, language="russian", CPU)
- Используется в HybridRetriever для sparse ветки prefetch
- **BM25 — наш самый надёжный retrieval signal** (лучше dense на entity queries)

## Reranker

- **Текущая модель: BAAI/bge-m3** (AutoModelForSequenceClassification) через gpu_server.py
- Загружается в gpu_server.py как `AutoModelForSequenceClassification`, logits для scoring
- **Целевой: bge-reranker-v2-m3** — dedicated cross-encoder, +10 nDCG на MIRACL
- Порт 8082 (тот же что embedding — gpu_server.py обслуживает оба)
- Используется после search и перед compose_context

## Chunking при ingest

- Посты `<1500` символов индексируются одним point
- Посты `>1500` символов режутся через `_smart_chunk()`
- Recursive split по `["\n\n", "\n", ". ", " "]`
- Target size чанка: `1200` символов
- Без overlap
- Point ID при chunking: `{channel}:{message_id}:{chunk_idx}`

## Hardware

- gpu_server.py: RTX 5060 Ti (WSL2 native, PyTorch cu128, порт 8082) — embedding + reranker
- Qdrant: Docker CPU (порт 6333 internal, 16333 external)
- GPU недоступна в Docker Desktop (DEC-0024: V100 TCC блокирует NVML)

## Типичные задачи

- Добавить фильтр по дате/каналу: `MetadataFilters` в `schemas/search.py`
- Сменить коллекцию: `settings.update_collection("new_name")`
- Проверить healthcheck: `GET http://localhost:8082/health`
- Прогнать eval: `python scripts/evaluate_agent.py --dataset datasets/eval_dataset_quick.json --agent-url http://localhost:8001/v1/agent/stream --api-key TOKEN`
- Roadmap улучшений: `docs/ai/planning/retrieval_improvement_playbook.md`
