## System Overview

### Stack (целевой, Phase 1 — после Qdrant migration)

| Слой | Технология | Где работает |
|------|-----------|-------------|
| **API** | FastAPI + sse_starlette | Docker (WSL2) |
| **LLM** | llama-server HTTP → Qwen3-8B GGUF (Q8_0 или F16) | **Windows Host** (V100 TCC) |
| **LLM Planner** | тот же llama-server endpoint | **Windows Host** (V100 TCC) |
| **Embedding** | TEI HTTP → multilingual-e5-large | **WSL2 native** (RTX 5060 Ti) → `:8082` |
| **Vector Store** | Qdrant HTTP (dense + sparse named vectors) | Docker (CPU) |
| **Hybrid Retrieval** | Qdrant prefetch+FusionQuery(RRF) → MMR | Docker (CPU) |
| **Reranker** | TEI HTTP → BAAI/bge-reranker-v2-m3 | **WSL2 native** (RTX 5060 Ti) → `:8083` |
| **Agent** | ReAct loop (7 tools) | Docker (CPU) |
| **Query Planner** | JSON-guided LLM via HTTP | Docker → Host |
| **Auth** | JWT (ADMIN_KEY) | Docker |
| **Config** | Settings singleton (os.getenv) | Docker |
| **DI** | lru_cache factories | Docker |

### Stack (текущий, Phase 0 — до migration, для справки)

| Слой | Технология |
|------|-----------|
| **Vector Store** | ChromaDB HTTP |
| **Lexical Index** | BM25 disk-based (BM25IndexManager) |
| **LLM** | llama-server → Qwen2.5-7B-Instruct GGUF |

---

### Компонентная схема (Phase 1)

```
[Windows Host]
  └── llama-server.exe → V100 SXM2 32GB (TCC, CUDA device 1)
       └── :8080/v1/chat/completions  (OpenAI-compatible)
            Model: Qwen3-8B GGUF (Q8_0 ~9GB или F16 ~16.4GB)
            Thinking mode: ОТКЛЮЧЁН (/no_think)

[Ubuntu WSL2 — нативно, не Docker]  ← RTX 5060 Ti (GPU-PV, CUDA device 0)
  ├── TEI embedding service → :8082   (intfloat/multilingual-e5-large, 1024-dim)
  └── TEI reranker service  → :8083   (BAAI/bge-reranker-v2-m3)

  Примечание: WSL2-native процессы видят 5060 Ti напрямую.
  Docker Desktop не видит 5060 Ti: V100 TCC-режим блокирует NVML-enumeration
  для всех GPU при старте Docker (DEC-0024). Решение без Proxmox: нативный WSL2.

[Docker Desktop / WSL2]
  │
  [Client] ──HTTP POST /v1/agent/stream + JWT──►
  │
  [FastAPI API] ──────────────────────────────── [JWT verify]
      │
      [AgentService]
          │ httpx.AsyncClient /v1/chat/completions
          ├─ LLM calls ─────────────────────────► [llama-server @ host.docker.internal:8080]
          │
          ├─ router_select(query) ──────────────► [Heuristic]
          ├─ query_plan(query) ─────────────────► [QueryPlannerService → host.docker.internal:8080]
          ├─ search(queries, route) ────────────► [HybridRetriever]
          │                                           ├── embed(query) ─► [TEI @ host.docker.internal:8082]
          │                                           └── [QdrantClient.query_points()]
          │                                                  prefetch: dense + sparse
          │                                                  FusionQuery(RRF) → MMR
          ├─ rerank(query, docs) ───────────────► [TEI reranker @ host.docker.internal:8083]
          ├─ compose_context(hit_ids, query) ───► [Builds prompt + citations + composite coverage]
          │                                           with_vectors=True → cosine sim per doc
          ├─ verify(query, claim) ──────────────► [Qdrant search]
          └─ final_answer(answer) ──────────────► [assembled SSE final event]

  [SSE Stream] ◄── thought / tool_invoked / observation / citations / final
```

---

### Hardware (актуально 2026-03-16)

| GPU | CUDA device | Режим | VRAM | Где доступен |
|-----|------------|-------|------|-------------|
| RTX 5060 Ti | 0 | WDDM | 16GB | Windows host + **WSL2 нативно** |
| V100 SXM2 | 1 | **TCC** | 32GB | **Только Windows host** (WSL2/Docker: ❌ TCC несовместим) |

> **⚠️ Docker GPU blocker**: V100 в TCC-режиме блокирует NVML-enumeration для **всех** GPU при старте
> Docker Desktop — включая RTX 5060 Ti. Ни одна из GPU не доступна в Docker-контейнерах пока V100
> остаётся в TCC и подключена к той же системе. Фикс без физического отключения V100: Proxmox VFIO.
> **Текущее решение**: embedding + reranker запускаются как WSL2-native процессы (DEC-0024).

- **LLM inference**: llama-server.exe на Windows хосте, V100
  - Qwen3-8B Q8_0: ~9 GB VRAM, ~70–80 tok/s
  - Qwen3-8B F16: ~16.4 GB VRAM, ~60–70 tok/s
- **Embedding**: TEI нативно в Ubuntu WSL2, RTX 5060 Ti, порт **:8082**
- **Reranker**: TEI нативно в Ubuntu WSL2, RTX 5060 Ti, порт **:8083**
- **Qdrant, API, Ingest**: Docker (CPU), без GPU
- **Драйвер**: 581.80 — максимальный с поддержкой V100. **Не обновлять** (590+ дропнул V100)

Запуск llama-server (Qwen3-8B Q8_0):
```powershell
llama-server.exe -hf bartowski/Qwen3-8B-GGUF:Q8_0.gguf `
    -ngl 99 --main-gpu 1 --host 0.0.0.0 --port 8080
```

Запуск llama-server (Qwen3-8B F16):
```powershell
llama-server.exe -hf bartowski/Qwen3-8B-GGUF:F16.gguf `
    -ngl 99 --main-gpu 1 --host 0.0.0.0 --port 8080
```

---

### Docker Compose Services (Phase 1)

| Сервис | Профиль | Порт | GPU |
|--------|---------|------|-----|
| `api` | `api` | 8000 | — (GPU через WSL2-native TEI) |
| `qdrant` | `api`, `ingest` | 6333 | — |
| `ingest` | `ingest` | — | — (embedding через WSL2-native TEI) |

### WSL2-Native Services (запускаются вне Docker)

| Сервис | Технология | GPU | Порт |
|--------|-----------|-----|------|
| TEI embedding | text-embeddings-inference | RTX 5060 Ti | 8082 |
| TEI reranker | text-embeddings-inference | RTX 5060 Ti | 8083 |

**Prerequisites (однократно, Ubuntu WSL2):**
```bash
# 1. Установить Docker Engine (offline, из скачанных .deb на Windows):
#    docker-ce, docker-ce-cli, containerd.io, docker-compose-plugin
#    sudo dpkg -i /mnt/c/tmp/docker-pkgs/*.deb
#    sudo systemctl start docker

# 2. Сгенерировать CDI spec для GPU:
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 3. Установить CDI mode:
sudo sed -i 's/mode = "auto"/mode = "cdi"/' /etc/nvidia-container-runtime/config.toml
sudo systemctl restart docker

# 4. Скопировать модели в Linux FS (не /mnt/c/ — медленно):
sudo cp -rL /mnt/c/llms/rag/rag_app/models/hub/models--intfloat--multilingual-e5-large/snapshots/HASH/* /home/tei-models/e5-large/
sudo cp -rL /mnt/c/llms/rag/rag_app/models/hub/models--BAAI--bge-reranker-v2-m3/snapshots/HASH/* /home/tei-models/reranker/
```

**Запуск (Ubuntu WSL2 — каждый раз перед `docker compose up`):**
```bash
# Embedding — интфloat/multilingual-e5-large, 1024-dim
docker run -d --name tei-embedding \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8082:80 \
  -v /home/tei-models/e5-large:/model \
  ghcr.io/huggingface/text-embeddings-inference:120-1.9 \
  --model-id /model --port 80

# Reranker — BAAI/bge-reranker-v2-m3
docker run -d --name tei-reranker \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8083:80 \
  -v /home/tei-models/reranker:/model \
  ghcr.io/huggingface/text-embeddings-inference:120-1.9 \
  --model-id /model --port 80
```

> **Критично**: образ `120-1.9` — единственный рабочий для RTX 5060 Ti (SM 12.0, Blackwell).
> `cuda-1.9` deadlock: нет pre-compiled FlashAttention kernels для SM 12.0.
> Модели держать в `/home/tei-models/` (Linux FS) — `/mnt/c/` через 9P слишком медленный.
> `CUDA_VISIBLE_DEVICES=0` — изолирует от V100 (CUDA device 1, broken в WSL2).
> Docker Desktop не имеет доступа к GPU, Docker Engine в WSL2 — имеет (CDI mode + CDI spec).
> Интернет в WSL2 отсутствует (Tailscale routing). Модели скачивать через Docker Desktop контейнер.

---

### Файловые зависимости (volumes)

| Volume | Тип | Назначение |
|--------|-----|-----------|
| `./models` | bind mount | ~~HF snapshots~~ **Deprecated в Phase 1** — TEI управляет кэшем моделей самостоятельно (внутри WSL2-native контейнеров) |
| `qdrant_data` | **named volume** | Qdrant persistent storage (**обязательно** named — bind mounts → silent data corruption на Windows) |
| `./sessions` | bind mount | Telethon session files |

> **Удалено**: `./chroma-data` (ChromaDB), `./bm25-index` (BM25 disk index) — заменены Qdrant.
> **Deprecated**: `./models` bind mount — модели теперь кэшируются внутри TEI (WSL2-native), не нужны в docker-compose.

---

### Будущий стек (Phase 2 — после Proxmox + VFIO)

| Изменение | Детали |
|-----------|--------|
| V100 → Linux VM | Proxmox VFIO passthrough, изоляция от RTX 5060 Ti |
| llama-server → vLLM v0.15.1 | Pinned: v0.17.0+ сломал xformers (V100 SM7.0 требует xformers) |
| LlamaServerClient → AsyncOpenAI | `AsyncOpenAI(base_url=LLM_BASE_URL, api_key="EMPTY")` |
| text ReAct → Hermes tool calling | `--tool-call-parser hermes`, требует AgentService rewrite |
| httpx.AsyncClient → AsyncOpenAI | Финальный async фикс OPEN-02 |

vLLM флаги для V100:
```bash
vllm serve Qwen/Qwen3-8B \
  --dtype half \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```
