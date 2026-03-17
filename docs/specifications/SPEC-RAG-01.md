# SPEC-RAG-01: Settings & Docker Compose

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Ready for implementation
> **Цель:** Привести `src/core/settings.py` и `docker-compose.yml` в соответствие Phase 1 —
>           удалить ChromaDB/BM25/local-model конфигурацию, добавить Qdrant + TEI URLs,
>           исправить пороги coverage.
> **Источники:** `docs/specifications/arch-brief.md` (DEC-0019, DEC-0024),
>                `docs/architecture/07-data-model/data-model.md` §Settings Key Fields,
>                `docs/architecture/04-system/overview.md` §Docker Compose Services

---

## 0. Implementation Pointers — что уже есть

### 0.1 Затрагиваемые файлы

| Файл | Текущее состояние | После SPEC-RAG-01 |
|------|-------------------|-------------------|
| `src/core/settings.py` | Phase 0: ChromaDB поля, BM25 поля, `coverage_threshold=0.8`, `max_refinements=1`, `qwen2.5-7b-instruct` | Phase 1: Qdrant + TEI URL поля, `coverage_threshold=0.65`, `max_refinements=2`, `qwen3-8b` |
| `docker-compose.yml` | chroma service, bm25-index volume, GPU reservation на api + ingest | qdrant service (named volume), без GPU, TEI vars, без chroma |

### 0.2 Новые файлы

Нет. Только правки существующих файлов.

### 0.3 Что удалить из settings.py

| Поле | Причина |
|------|---------|
| `chroma_host`, `chroma_port`, `chroma_path` | ChromaDB уходит (DEC-0015) |
| `bm25_index_root`, `hybrid_top_bm25`, `bm25_default_top_k`, `bm25_reload_min_interval_sec` | BM25IndexManager уходит (DEC-0015); sparse теперь в Qdrant |
| `models_dir`, `cache_dir` | Локальные модели не нужны: TEI управляет кешем сам (DEC-0016, DEC-0017) |
| `planner_llm_device` | Planner больше не local-CPU, он идёт через тот же llama-server |

### 0.4 Что удалить из docker-compose.yml

| Элемент | Причина |
|---------|---------|
| Service `chroma` | ChromaDB заменяется Qdrant |
| `./chroma-data` volumes (api + ingest) | Данные переходят в Qdrant named volume |
| `./bm25-index` volumes (api + ingest) | BM25 disk index уходит |
| `./models` volumes (api + ingest) | Local-model кеш не нужен (DEC-0016, DEC-0017) |
| `deploy.resources.reservations.devices: nvidia` (api + ingest) | Docker GPU недоступен (DEC-0024) |
| `CHROMA_HOST`, `CHROMA_PORT`, `TRANSFORMERS_CACHE`, `HF_HOME`, `AUTO_DOWNLOAD_EMBEDDING` | Phase 0 переменные |
| `Dockerfile.chroma` (build context у chroma service) | Файл можно удалить после миграции |

---

## 1. Обзор

### 1.1 Задача

1. **settings.py**: удалить Phase 0 поля, добавить `qdrant_url`, `qdrant_collection`, `embedding_tei_url`, `reranker_tei_url`; исправить дефолты `coverage_threshold` → `0.65`, `max_refinements` → `2`, `current_llm_key` → `qwen3-8b`.
2. **docker-compose.yml**: убрать `chroma` service + все Phase 0 volumes + GPU reservation; добавить `qdrant` service с named volume; обновить env переменные в `api` и `ingest`.
3. **settings.py → `update_collection`**: убрать ссылку на ChromaDB в docstring (refs к `get_chroma_client` удаляются в SPEC-RAG-04).

### 1.2 Контекст

`settings.py` является единым источником конфигурации через `get_settings()` singleton (`lru_cache`). Все фабрики в `deps.py` читают настройки из него. Правки в этом файле немедленно влияют на то, что DI-фабрики будут создавать после `cache_clear()`.

`docker-compose.yml` определяет runtime среду контейнеров. GPU reservation (`deploy.resources.reservations.devices: nvidia`) ломает запуск на машинах с V100 TCC (DEC-0024): NVML enumeration падает для всех GPU, включая RTX 5060 Ti. Qdrant требует **named volume** (не bind mount) — на Windows bind mount вызывает silent data corruption при записи WAL.

### 1.3 Ключевые решения

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| `coverage_threshold` default | `0.65` | R04: composite metric сжимает скоры; 0.8 вызывает false-negative refinements (DEC-0019) |
| `max_refinements` default | `2` | R04: второй refinement даёт +12% recall при minimal overhead (DEC-0019) |
| Qdrant volume тип | named (`qdrant_data`) | Bind mount → silent WAL corruption на Windows (overview.md §volumes) |
| Qdrant image | `qdrant/qdrant:v1.15.3` | Pinned — проверена совместимость с Python client (R01) |
| GPU reservation | убрать полностью | V100 TCC блокирует NVML для всех GPU в Docker (DEC-0024) |
| `EMBEDDING_TEI_URL` default | `http://host.docker.internal:8082` | TEI embedding запущен в WSL2 native на этом порту |
| `RERANKER_TEI_URL` default | `http://host.docker.internal:8083` | TEI reranker запущен в WSL2 native на этом порту |
| `extra_hosts` в ingest | добавить `host.docker.internal:host-gateway` | Ingest тоже обращается к TEI на хосте |

### 1.4 Что НЕ делать

- Не трогать `src/core/deps.py` в этой спецификации — изменения DI-фабрик в SPEC-RAG-04 + SPEC-RAG-05
- Не удалять `reranker_model_key` из settings — он используется для логирования в RerankerService (удаляется в SPEC-RAG-05)
- Не менять схему Qdrant коллекции здесь — это SPEC-RAG-03
- Не удалять `redis_*` поля — Redis отключён по умолчанию, но может понадобиться позже
- Не пинить `qdrant/qdrant:latest` — использовать `v1.15.3`

---

## 2. src/core/settings.py — полная реализация

Полный файл. Заменяет текущий `src/core/settings.py` целиком.

```python
"""
Конфигурация приложения — Phase 1 (Qdrant + TEI HTTP).

Изменения по сравнению с Phase 0:
- Удалены: ChromaDB поля, BM25 поля, local-model пути
- Добавлены: qdrant_url, qdrant_collection, embedding_tei_url, reranker_tei_url
- Исправлены: coverage_threshold=0.65, max_refinements=2, LLM=qwen3-8b
"""

import os
from typing import List, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class Settings:
    """Настройки приложения Phase 1. Singleton через get_settings()."""

    def __init__(self):
        # === LLM — llama-server (V100 на Windows Host) ===
        # llama-server.exe запускается на хосте, Docker обращается через host.docker.internal.
        # V100 TCC недоступен в WSL2/Docker — только через HTTP на хосте.
        self.current_llm_key: str = os.getenv("LLM_MODEL_KEY", "qwen3-8b")
        self.llm_base_url: str = os.getenv(
            "LLM_BASE_URL", "http://host.docker.internal:8080"
        )
        self.llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen3-8b")
        self.llm_request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

        # Query Planner может использовать отдельный endpoint.
        # Если PLANNER_LLM_BASE_URL не задан — используется тот же llama-server.
        self.planner_llm_base_url: str = os.getenv("PLANNER_LLM_BASE_URL", "")
        self.planner_llm_key: str = os.getenv("PLANNER_LLM_MODEL_KEY", "qwen3-8b")

        # === Embedding — TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8082) ===
        # Модель: intfloat/multilingual-e5-large (1024-dim, cosine).
        # TEI запускается отдельно в WSL2, не в Docker (DEC-0024).
        self.current_embedding_key: str = os.getenv(
            "EMBEDDING_MODEL_KEY", "multilingual-e5-large"
        )
        self.embedding_tei_url: str = os.getenv(
            "EMBEDDING_TEI_URL", "http://host.docker.internal:8082"
        )

        # === Reranker — TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8083) ===
        # Модель: BAAI/bge-reranker-v2-m3.
        self.reranker_model_key: str = os.getenv(
            "RERANKER_MODEL_KEY", "BAAI/bge-reranker-v2-m3"
        )
        self.reranker_tei_url: str = os.getenv(
            "RERANKER_TEI_URL", "http://host.docker.internal:8083"
        )
        self.enable_reranker: bool = (
            os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        )
        self.reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "80"))
        self.reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))

        # === Qdrant (Docker CPU, порт 6333) ===
        self.qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "news")
        # Алиас для обратной совместимости с кодом, обращающимся к current_collection
        self.current_collection: str = self.qdrant_collection

        # === Redis кеширование (отключён по умолчанию) ===
        self.redis_enabled: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
        self.redis_host: str = os.getenv("REDIS_HOST", "localhost")
        self.redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
        self.cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))

        # === Query Planner / Fusion ===
        self.enable_query_planner: bool = (
            os.getenv("ENABLE_QUERY_PLANNER", "true").lower() == "true"
        )
        self.fusion_strategy: str = os.getenv("FUSION_STRATEGY", "rrf").lower()
        self.k_fusion: int = int(os.getenv("K_FUSION", "60"))

        # MMR — нативно через Qdrant MmrQuery
        self.enable_mmr: bool = os.getenv("ENABLE_MMR", "true").lower() == "true"
        try:
            self.mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))
        except Exception:
            self.mmr_lambda = 0.7
        self.mmr_top_n: int = int(os.getenv("MMR_TOP_N", "120"))
        self.mmr_output_k: int = int(os.getenv("MMR_OUTPUT_K", "60"))

        self.search_k_per_query_default: int = int(
            os.getenv("SEARCH_K_PER_QUERY_DEFAULT", "10")
        )
        self.max_plan_subqueries: int = int(os.getenv("MAX_PLAN_SUBQUERIES", "5"))

        # === Hybrid Retriever ===
        self.hybrid_enabled: bool = (
            os.getenv("HYBRID_ENABLED", "true").lower() == "true"
        )
        # Лимиты prefetch для dense и sparse в Qdrant prefetch запросе
        self.hybrid_top_dense: int = int(os.getenv("HYBRID_TOP_DENSE", "100"))
        self.hybrid_top_sparse: int = int(os.getenv("HYBRID_TOP_SPARSE", "100"))
        self.enforce_router_route: bool = (
            os.getenv("ENFORCE_ROUTER_ROUTE", "false").lower() == "true"
        )
        # Алиас для совместимости
        self.enable_hybrid_retriever: bool = self.hybrid_enabled

        # === Planner параметры декодинга ===
        self.use_gbnf_planner: bool = (
            os.getenv("USE_GBNF_PLANNER", "true").lower() == "true"
        )
        self.planner_timeout: float = float(os.getenv("PLANNER_TIMEOUT", "15.0"))
        self.planner_token_budget: int = int(os.getenv("PLANNER_TOKEN_BUDGET", "4096"))
        self.planner_temp: float = float(os.getenv("PLANNER_TEMP", "0.2"))
        self.planner_top_p: float = float(os.getenv("PLANNER_TOP_P", "0.9"))
        self.planner_top_k: int = int(os.getenv("PLANNER_TOP_K", "40"))
        self.planner_repeat_penalty: float = float(
            os.getenv("PLANNER_REPEAT_PENALTY", "1.1")
        )
        self.planner_stop: List[str] = os.getenv(
            "PLANNER_STOP", "Observation:"
        ).split("||")

        # === In-memory кеш (TTL) ===
        self.enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

        # === ReAct Agent ===
        self.enable_agent: bool = os.getenv("ENABLE_AGENT", "true").lower() == "true"
        self.agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "15"))
        self.agent_default_steps: int = int(os.getenv("AGENT_DEFAULT_STEPS", "8"))
        self.agent_tool_timeout: float = float(os.getenv("AGENT_TOOL_TIMEOUT", "15.0"))
        self.agent_token_budget: int = int(os.getenv("AGENT_TOKEN_BUDGET", "2000"))

        # Параметры декодинга для tool-шагов (короткие, детерминированные)
        self.agent_tool_temp: float = float(os.getenv("AGENT_TOOL_TEMP", "0.2"))
        self.agent_tool_top_p: float = float(os.getenv("AGENT_TOOL_TOP_P", "0.9"))
        self.agent_tool_top_k: int = int(os.getenv("AGENT_TOOL_TOP_K", "40"))
        self.agent_tool_repeat_penalty: float = float(
            os.getenv("AGENT_TOOL_REPEAT_PENALTY", "1.15")
        )
        self.agent_tool_max_tokens: int = int(os.getenv("AGENT_TOOL_MAX_TOKENS", "64"))

        # Параметры декодинга для финального ответа
        self.agent_final_temp: float = float(os.getenv("AGENT_FINAL_TEMP", "0.3"))
        self.agent_final_top_p: float = float(os.getenv("AGENT_FINAL_TOP_P", "0.9"))
        self.agent_final_max_tokens: int = int(
            os.getenv("AGENT_FINAL_MAX_TOKENS", "512")
        )

        # === Coverage / Refinement (DEC-0018, DEC-0019) ===
        # 0.65 — откалиброван под composite 5-signal metric (R04).
        # 0.8 был слишком агрессивен: вызывал false-negative refinements.
        self.coverage_threshold: float = float(
            os.getenv("COVERAGE_THRESHOLD", "0.65")
        )
        # 2 refinements дают +12% recall без существенного роста latency (R04).
        self.max_refinements: int = int(os.getenv("MAX_REFINEMENTS", "2"))
        self.enable_verify_step: bool = (
            os.getenv("ENABLE_VERIFY_STEP", "true").lower() == "true"
        )

        logger.info(
            "Настройки загружены: LLM=%s, Embedding=%s, Qdrant=%s/%s, "
            "EmbTEI=%s, RerankTEI=%s, Coverage=%.2f, MaxRefinements=%d",
            self.current_llm_key,
            self.current_embedding_key,
            self.qdrant_url,
            self.qdrant_collection,
            self.embedding_tei_url,
            self.reranker_tei_url,
            self.coverage_threshold,
            self.max_refinements,
        )

    def update_llm_model(self, model_key: str) -> None:
        """Горячая смена LLM модели. Сбрасывает lru_cache фабрик."""
        old_key = self.current_llm_key
        self.current_llm_key = model_key
        from core.deps import get_llm, get_qa_service
        try:
            from core.deps import get_agent_service
            get_agent_service.cache_clear()
        except ImportError:
            pass
        get_llm.cache_clear()
        get_qa_service.cache_clear()
        logger.info("LLM модель изменена: %s → %s", old_key, model_key)

    def update_embedding_model(self, model_key: str) -> None:
        """Горячая смена embedding модели. Сбрасывает lru_cache фабрик."""
        old_key = self.current_embedding_key
        self.current_embedding_key = model_key
        from core.deps import get_retriever, get_qa_service
        try:
            from core.deps import get_agent_service
            get_agent_service.cache_clear()
        except ImportError:
            pass
        get_retriever.cache_clear()
        get_qa_service.cache_clear()
        logger.info("Embedding модель изменена: %s → %s", old_key, model_key)

    def update_collection(self, collection_name: str) -> None:
        """Горячая смена Qdrant-коллекции. Сбрасывает lru_cache фабрик."""
        old = self.qdrant_collection
        self.qdrant_collection = collection_name
        self.current_collection = collection_name  # синхронизируем алиас
        from core.deps import get_retriever, get_qa_service
        try:
            from core.deps import get_agent_service
            get_agent_service.cache_clear()
        except ImportError:
            pass
        get_retriever.cache_clear()
        get_qa_service.cache_clear()
        logger.info("Qdrant коллекция изменена: %s → %s", old, collection_name)


@lru_cache()
def get_settings() -> Settings:
    """Singleton настроек приложения."""
    return Settings()
```

---

## 3. docker-compose.yml — полная реализация

Полный файл. Заменяет текущий `docker-compose.yml` целиком.

```yaml
# docker-compose.yml — Phase 1 (Qdrant + TEI HTTP)
#
# Порядок запуска (обязательно перед docker compose up):
#   1. Windows Host:  llama-server.exe -ngl 99 --main-gpu 1 --host 0.0.0.0 --port 8080
#   2. Ubuntu WSL2:   docker run -d --name tei-embedding ... -p 8082:80 (multilingual-e5-large)
#   3. Ubuntu WSL2:   docker run -d --name tei-reranker  ... -p 8083:80 (bge-reranker-v2-m3)
#   4. Docker Desktop: docker compose --profile api up
#
# GPU в Docker недоступен (V100 TCC блокирует NVML для всех GPU, DEC-0024).
# Embedding + Reranker работают вне Docker в WSL2 native (RTX 5060 Ti).

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    env_file:
      - .env
    profiles: ["api"]
    volumes:
      - ./scripts:/app/scripts
      - ./sessions:/app/sessions
      - ./src:/app
    ports:
      - "8000:8000"
    depends_on:
      qdrant:
        condition: service_healthy
    extra_hosts:
      - "host.docker.internal:host-gateway"  # llama-server + TEI на хосте/WSL2
    environment:
      - TG_SESSION=/app/sessions/telegram.session
      # LLM — llama-server на Windows Host (V100 TCC)
      - LLM_BASE_URL=http://host.docker.internal:8080
      - LLM_MODEL_NAME=qwen3-8b
      - LLM_REQUEST_TIMEOUT=120
      # Embedding TEI — WSL2 native, RTX 5060 Ti
      - EMBEDDING_TEI_URL=http://host.docker.internal:8082
      # Reranker TEI — WSL2 native, RTX 5060 Ti
      - RERANKER_TEI_URL=http://host.docker.internal:8083
      # Qdrant — Docker CPU
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_COLLECTION=news
      # Coverage (DEC-0019)
      - COVERAGE_THRESHOLD=0.65
      - MAX_REFINEMENTS=2
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --loop uvloop --reload

  qdrant:
    image: qdrant/qdrant:v1.15.3
    profiles: ["api", "ingest"]
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC (опционально)
    volumes:
      # Named volume обязателен — bind mount вызывает silent WAL corruption на Windows
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:6333/"]
      interval: 5s
      timeout: 3s
      retries: 12

  ingest:
    build:
      context: .
      dockerfile: Dockerfile.ingest
    env_file:
      - .env
    profiles: ["ingest"]
    volumes:
      - ./scripts:/app/scripts
      - ./src:/app/src
      - ./sessions:/app/sessions
    depends_on:
      qdrant:
        condition: service_healthy
    networks:
      - default
    extra_hosts:
      - "host.docker.internal:host-gateway"  # TEI embedding на хосте/WSL2
    environment:
      - TG_SESSION=/app/sessions/telegram.session
      # Qdrant
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_COLLECTION=news
      # Embedding TEI
      - EMBEDDING_TEI_URL=http://host.docker.internal:8082
    entrypoint: ["python3", "-m", "scripts.ingest_telegram"]

networks:
  default:
    name: rag-net

volumes:
  qdrant_data:
    # Named volume — Qdrant хранит WAL и snapshots здесь.
    # Не менять на bind mount (data corruption на Windows).
```

---

## 4. Порядок применения и совместимость

После применения SPEC-RAG-01 **код временно поломан**: `deps.py` ещё ссылается на
`settings.chroma_host`, `settings.chroma_port` — они удалены. Это ожидаемо.

**Порядок применения спецификаций для первого рабочего запуска:**

```
SPEC-RAG-01 → SPEC-RAG-02 → SPEC-RAG-03 → SPEC-RAG-04 → SPEC-RAG-06
```

После SPEC-RAG-04 (`deps.py` полностью переписан) система снова запускается.

**Не применять SPEC-RAG-01 изолированно** если планируется немедленный запуск.

---

## 5. Чеклист реализации

- [ ] `src/core/settings.py` заменён на версию из секции 2
- [ ] `coverage_threshold` default = `0.65` (проверить `os.getenv("COVERAGE_THRESHOLD", "0.65")`)
- [ ] `max_refinements` default = `2`
- [ ] `qdrant_url`, `qdrant_collection`, `embedding_tei_url`, `reranker_tei_url` присутствуют
- [ ] ChromaDB поля (`chroma_host`, `chroma_port`, `chroma_path`) отсутствуют
- [ ] BM25 поля (`bm25_index_root`, `hybrid_top_bm25`, `bm25_default_top_k`, `bm25_reload_min_interval_sec`) отсутствуют
- [ ] `models_dir`, `cache_dir` отсутствуют
- [ ] `update_collection` обновляет `qdrant_collection` и `current_collection` (алиас)
- [ ] `docker-compose.yml` заменён на версию из секции 3
- [ ] `chroma` service отсутствует в compose
- [ ] `qdrant` service присутствует с `qdrant/qdrant:v1.15.3` и named volume `qdrant_data`
- [ ] `deploy.resources.reservations.devices` отсутствует в `api` и `ingest`
- [ ] `./chroma-data`, `./bm25-index`, `./models` volumes отсутствуют в `api` и `ingest`
- [ ] `COVERAGE_THRESHOLD=0.65`, `MAX_REFINEMENTS=2` присутствуют в env `api`
- [ ] `EMBEDDING_TEI_URL`, `RERANKER_TEI_URL` присутствуют в env `api`
- [ ] `QDRANT_URL`, `QDRANT_COLLECTION` присутствуют в env `api` и `ingest`
- [ ] `extra_hosts: host.docker.internal:host-gateway` присутствует в `api` и `ingest`
- [ ] `volumes:` секция в конце compose содержит `qdrant_data:`
