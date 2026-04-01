## System Overview

### Stack (актуально 2026-03-28)

| Слой | Технология | Где работает |
|------|-----------|-------------|
| **API** | FastAPI + sse_starlette | Docker (CPU) |
| **LLM** | llama-server HTTP → Qwen3.5-35B-A3B GGUF (Q4_K_M, MoE 3B active) | **Windows Host** (V100 TCC) |
| **Embedding** | gpu_server.py → pplx-embed-v1-0.6B (bf16, mean pooling, 1024-dim) | **WSL2 native** (RTX 5060 Ti) → `:8082` |
| **Reranker** | gpu_server.py → Qwen3-Reranker-0.6B-seq-cls (chat template, logit scoring) | **WSL2 native** (RTX 5060 Ti) → `:8082` |
| **ColBERT** | gpu_server.py → jina-colbert-v2 (128-dim per-token MaxSim) | **WSL2 native** (RTX 5060 Ti) → `:8082` |
| **Vector Store** | Qdrant HTTP (dense + sparse + ColBERT named vectors) | Docker (CPU) |
| **Hybrid Retrieval** | Qdrant weighted RRF (BM25 3:1) → ColBERT MaxSim rerank | Docker (CPU) |
| **Agent** | ReAct loop, native function calling (15 LLM tools, phase-based visibility) | Docker (CPU) |
| **Query Planner** | JSON-guided LLM via HTTP (тот же endpoint) | Docker → Host |
| **Auth** | JWT (ADMIN_KEY) | Docker |
| **Config** | Settings singleton (os.getenv) | Docker |
| **DI** | lru_cache factories | Docker |

---

### Компонентная схема

```
[Windows Host]
  └── llama-server.exe → V100 SXM2 32GB (TCC, CUDA device 1)
       └── :8080/v1/chat/completions  (OpenAI-compatible)
            Model: Qwen3.5-35B-A3B GGUF Q4_K_M (~18 GB VRAM)
            Native function calling: --jinja --reasoning-budget 0

[Ubuntu WSL2 — нативно, не Docker]  ← RTX 5060 Ti 16GB (GPU-PV, CUDA device 0)
  └── gpu_server.py → :8082  (единый HTTP-сервер, stdlib http.server + PyTorch cu128)
       ├── POST /embed          → pplx-embed-v1-0.6B (bf16, mean pooling, 1024-dim)
       ├── POST /v1/embeddings  → OpenAI-compatible формат
       ├── POST /rerank         → Qwen3-Reranker-0.6B-seq-cls (chat template, logit scoring)
       └── POST /colbert-encode → jina-colbert-v2 (128-dim per-token vectors)

  3 модели в одном процессе, ~4-5 GB VRAM, ~11 GB свободно.
  PyTorch cu128 + cuBLAS. Не TEI, не Docker. Manual linear projection для ColBERT.

  Примечание: WSL2-native процессы видят 5060 Ti напрямую.
  Docker Desktop не видит 5060 Ti: V100 TCC-режим блокирует NVML-enumeration
  для всех GPU при старте Docker (DEC-0024).

[Docker Desktop / WSL2]
  │
  [Client] ──HTTP POST /v1/agent/stream + JWT──►
  │
  [FastAPI API] ──────────────────────────────── [JWT verify]
      │
      [AgentService]  (native function calling, 15 LLM tools, ContextVar isolation)
          │ httpx.AsyncClient /v1/chat/completions (tools parameter)
          ├─ LLM calls ─────────────────────────► [llama-server @ host.docker.internal:8080]
          │
          ├─ query_plan(query) ────────────────► [QueryPlannerService → host.docker.internal:8080]
          ├─ search(queries) ──────────────────► [HybridRetriever]
          │     for each subquery:                    ├── embed(query) ─► [gpu_server @ host.docker.internal:8082]
          │       round-robin merge results           └── [QdrantClient.query_points()]
          │                                                  prefetch: BM25 top-100 + dense top-20
          │                                                  RrfQuery(weights=[1.0, 3.0])
          │                                                  → ColBERT MaxSim rerank (if available)
          │                                                  → channel dedup (max 2/channel)
          ├─ rerank(query, docs) ──────────────► [gpu_server @ host.docker.internal:8082 /rerank]
          ├─ compose_context(hit_ids, query) ──► [Builds prompt + citations + composite coverage]
          │                                           with_vectors=True → cosine sim per doc
          ├─ verify(query, claim) ─────────────► [Qdrant search] (системный, не LLM tool)
          └─ final_answer(answer) ─────────────► [assembled SSE final event]

  [SSE Stream] ◄── thought / tool_invoked / observation / citations / final
```

---

### Hardware (актуально 2026-03-28)

| GPU | CUDA device | Режим | VRAM | Где доступен |
|-----|------------|-------|------|-------------|
| RTX 5060 Ti | 0 | WDDM | 16GB | Windows host + **WSL2 нативно** |
| V100 SXM2 | 1 | **TCC** | 32GB | **Только Windows host** (WSL2/Docker: ❌ TCC несовместим) |

> **⚠️ Docker GPU blocker**: V100 в TCC-режиме блокирует NVML-enumeration для **всех** GPU при старте
> Docker Desktop. Ни одна GPU не доступна в Docker-контейнерах. (DEC-0024)
> **Текущее решение**: gpu_server.py запускается как WSL2-native процесс.

- **LLM inference**: llama-server.exe на Windows хосте, V100
  - Qwen3.5-35B-A3B Q4_K_M: ~18 GB VRAM, `--jinja --reasoning-budget 0 -c 16384 --parallel 2`
- **GPU server**: gpu_server.py нативно в Ubuntu WSL2, RTX 5060 Ti, порт **:8082**
  - Embedding: pplx-embed-v1-0.6B (bf16, mean pooling, 1024-dim)
  - Reranker: Qwen3-Reranker-0.6B-seq-cls (chat template, logit scoring)
  - ColBERT: jina-colbert-v2 (560M, 128-dim per-token)
- **Qdrant, API, Ingest**: Docker (CPU), без GPU
- **Драйвер**: 581.80 — максимальный с поддержкой V100. **Не обновлять** (590+ дропнул V100)

Запуск llama-server:
```powershell
$env:GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F = "1"
llama-server.exe -m Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 --main-gpu 0 `
    --host 0.0.0.0 --port 8080 --jinja --reasoning-budget 0 `
    --cache-type-k q8_0 --cache-type-v q8_0 -c 16384 --parallel 2
```

Запуск gpu_server.py:
```bash
source /home/ezsx/infinity-env/bin/activate
CUDA_VISIBLE_DEVICES=0 python /mnt/c/llms/rag/rag_app/scripts/gpu_server.py
```

---

### Docker Compose Services

| Сервис | Профиль | Порт | GPU |
|--------|---------|------|-----|
| `api` | `api` | 8001 | — (GPU через WSL2-native gpu_server.py) |
| `qdrant` | `api`, `ingest` | 6333 | — |
| `ingest` | `ingest` | — | — (embedding через WSL2-native gpu_server.py) |

### WSL2-Native Services (запускаются вне Docker)

| Сервис | Технология | GPU | Порт |
|--------|-----------|-----|------|
| gpu_server.py | PyTorch cu128 + cuBLAS (stdlib http.server) | RTX 5060 Ti | 8082 |

**Один процесс, три модели:**
- pplx-embed-v1-0.6B (`/mnt/c/llms/models/pplx-embed-v1-0.6B`)
- Qwen3-Reranker-0.6B-seq-cls (`/mnt/c/llms/models/Qwen3-Reranker-0.6B-seq-cls`)
- jina-colbert-v2 (`/home/tei-models/jina-colbert-v2`) + manual linear projection 1024→128

Venv: `/home/ezsx/infinity-env/` (Python 3.11, torch 2.10.0+cu128, transformers 4.57.6)

---

### Файловые зависимости (volumes)

| Volume | Тип | Назначение |
|--------|-----|-----------|
| `qdrant_data` | **named volume** | Qdrant persistent storage (**обязательно** named — bind mounts → silent data corruption на Windows) |
| `./sessions` | bind mount | Telethon session files (для ingest) |

> **Удалено**: `./chroma-data` (ChromaDB), `./bm25-index` (BM25 disk index) — заменены Qdrant.
> **Удалено**: `./models` bind mount — модели на Linux FS (`/home/tei-models/`), не в docker-compose.

---

### Qdrant Collections

| Коллекция | Назначение | Размер |
|-----------|-----------|--------|
| `news_colbert_v2` | Основная: enriched Telegram posts (dense 1024 + sparse BM25 + ColBERT 128-dim) | ~тысячи points |
| `weekly_digests` | BERTopic topic clusters, агрегированные по неделям (`hot_topics` tool) | ~38 points |
| `channel_profiles` | Профили каналов: expertise areas, статистика (`channel_expertise` tool) | 36 points |

Auxiliary collections (`weekly_digests`, `channel_profiles`) генерируются **BERTopic cron pipeline** (`scripts/bertopic_pipeline.py`) и обновляются периодически.

---

### Request Isolation

`AgentService` использует `contextvars.ContextVar` для per-request изоляции состояния
(`_current_step`, `_current_request_id`). Реализовано в SPEC-RAG-17 — заменяет прежние shared class attributes.
