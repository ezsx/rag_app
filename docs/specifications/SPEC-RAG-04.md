# SPEC-RAG-04: HybridRetriever & DI Migration

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Draft
> **Цель:** Переписать `hybrid_retriever.py` (Qdrant native RRF вместо ручного ChromaDB+BM25),
> заменить `deps.py` полностью (убрать chroma/bm25 фабрики, добавить qdrant/tei),
> обновить `verify.py` и `fetch_docs.py`, удалить `src/adapters/chroma/` и BM25-адаптеры.
>
> **Источники:**
> - `docs/specifications/arch-brief.md` (DEC-0015, схема hybrid search)
> - `docs/research/rag-stack/reports/R01-qdrant-hybrid-rag.md` (FilterQuery, prefetch, FusionQuery)
> - `docs/architecture/07-data-model/data-model.md` (Candidate, hybrid search query)
> - `docs/architecture/03-invariants/invariants.md` (INV-06)

---

## 0. Implementation Pointers

### 0.1 Текущие файлы (что есть сейчас)

| Файл | Текущее поведение | После SPEC-RAG-04 |
|------|------------------|-------------------|
| `src/adapters/search/hybrid_retriever.py` | BM25 + ChromaDB dense + ручной RRF в Python | **Перезаписать** — Qdrant prefetch + FusionQuery(RRF) |
| `src/core/deps.py` | chroma/bm25 фабрики | **Перезаписать** — qdrant/tei фабрики |
| `src/services/tools/verify.py` | `retriever: Retriever` (ChromaDB) | **Обновить** — `hybrid_retriever: HybridRetriever` |
| `src/services/tools/fetch_docs.py` | `retriever.get_by_ids()` (ChromaDB) | **Обновить** — `qdrant_store.get_by_ids()` |
| `src/adapters/chroma/` | ChromaDB HTTP клиент + SentenceTransformer | **Удалить** — весь каталог |
| `src/adapters/search/bm25_index.py` | BM25IndexManager disk-based | **Удалить** |
| `src/adapters/search/bm25_retriever.py` | BM25Retriever | **Удалить** |

### 0.2 Новые зависимости (pip)

В `requirements.txt` добавить:
```
fastembed>=0.3.0   # SparseTextEmbedding для Qdrant/bm25
```

`qdrant-client>=1.9.0` — уже добавлен в SPEC-RAG-03.

### 0.3 Что НЕ меняется

- `src/services/tools/search.py` — вызывает `hybrid_retriever.search_with_plan()` синхронно; интерфейс сохраняется.
- `src/schemas/search.py` — `SearchPlan`, `MetadataFilters`, `Candidate` — без изменений.
- `src/services/qa_service.py` — `без изменений`; deps.py передаёт `hybrid_retriever` как `retriever`.
- `src/utils/ranking.py` — более не используется, но НЕ удалять (может быть нужен в будущем).

---

## 1. Обзор

### 1.1 Задача

1. Перезаписать `src/adapters/search/hybrid_retriever.py`: `HybridRetriever` принимает `QdrantStore`,
   `TEIEmbeddingClient`, `SparseTextEmbedding` (fastembed), `Settings`.
2. `search_with_plan(query, plan)` — **sync** wrapper вокруг async `_async_search()`;
   работает в `ThreadPoolExecutor` (`ToolRunner`) через `asyncio.run()`.
3. `_async_search()` — embed query (TEI), encode sparse (fastembed BM25),
   `query_points(prefetch RRF, with_vectors=True)`, конвертация в `Candidate`.
4. `_build_filter()` — `MetadataFilters` → `models.Filter`
   (channel_usernames → MatchAny, date_from/date_to → DatetimeRange).
5. Перезаписать `src/core/deps.py` — убрать chroma/bm25 импорты и фабрики;
   добавить `get_tei_embedding_client()`, `get_tei_reranker_client()`,
   `get_hybrid_retriever()` (новый), обновить `get_agent_service()`.
6. Обновить `src/services/tools/verify.py` — заменить `Retriever` на `HybridRetriever`.
7. Обновить `src/services/tools/fetch_docs.py` — заменить `Retriever` на `QdrantStore`.
8. Удалить `src/adapters/chroma/`, `bm25_index.py`, `bm25_retriever.py`.

### 1.2 Контекст

Phase 0 `HybridRetriever` делал два HTTP-вызова (ChromaDB + BM25), затем сливал результаты через
`rrf_merge()` в Python (~130 строк). Qdrant выполняет то же самое в **одном** запросе на сервере:
`prefetch=[dense, sparse] + FusionQuery(RRF)` — нативный RRF в Rust.

`ToolRunner` запускает инструменты в `ThreadPoolExecutor` — функции должны быть **synchronous**.
`TEIEmbeddingClient` — async. Решение: `search_with_plan` остаётся синхронным, внутри вызывает
`asyncio.run(self._async_search(...))`. В worker-потоке ThreadPoolExecutor нет event loop —
`asyncio.run()` валиден.

`with_vectors=True` в запросе — обязательно: dense-вектор каждого результата нужен для
composite coverage metric в SPEC-RAG-07 (cosine_sim между query и doc vectors).

### 1.3 Ключевые решения

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| Sync/async bridge | `asyncio.run()` в `search_with_plan` | ToolRunner = ThreadPoolExecutor, потоки без event loop |
| Sparse encoding | fastembed `SparseTextEmbedding` | CPU-only, быстро, поддерживает `language="russian"` |
| MMR | Нет в Phase 1 | arch-brief не включает MMR в baseline; `plan.fusion="rrf"` → всегда RRF |
| `with_vectors=True` | Всегда | Нужен для cosine_sim в SPEC-RAG-07 (не только при coverage) |
| `_dense_vector` в metadata | Да | Proxy для coverage; HybridRetriever не знает про coverage — просто передаёт вектор |
| QAService compat | `search()` shim | QAService ожидает `.search()` — добавить тонкий wrapper без изменения qa_service.py |

### 1.4 Что НЕ делать

- **Не реализовывать** MMR в Phase 1 — только RRF.
- **Не оставлять** `utils/ranking.py` `rrf_merge()` вызов в новом коде — не нужен.
- **Не делать** `HybridRetriever` async на публичном уровне — ToolRunner sync.
- **Не трогать** `src/schemas/search.py` — `Candidate`, `SearchPlan` не меняются.
- **Не трогать** `src/services/tools/search.py` — интерфейс `search_with_plan` сохранён.
- **Не создавать** ABC или протокол для Retriever — один конкретный класс.
- **Не импортировать** `chromadb` нигде в новом коде.

---

## 2. HybridRetriever — полная реализация

Файл: `src/adapters/search/hybrid_retriever.py` (перезаписать полностью)

```python
"""
HybridRetriever Phase 1 — нативный RRF через Qdrant prefetch + FusionQuery.

Заменяет Phase 0 ChromaDB + BM25 + ручной rrf_merge (~130 строк Python).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastembed import SparseTextEmbedding
from qdrant_client import models

from adapters.qdrant.store import QdrantStore
from adapters.tei.embedding_client import TEIEmbeddingClient
from core.settings import Settings
from schemas.search import Candidate, MetadataFilters, SearchPlan

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Qdrant-based hybrid retriever: dense (TEI) + sparse (BM25 fastembed) → native RRF.

    Синхронный публичный интерфейс (совместим с ToolRunner / ThreadPoolExecutor).
    Async-логика инкапсулирована в _async_search() и вызывается через asyncio.run().
    """

    def __init__(
        self,
        store: QdrantStore,
        embedding_client: TEIEmbeddingClient,
        sparse_encoder: SparseTextEmbedding,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedding_client = embedding_client
        self._sparse_encoder = sparse_encoder
        self._settings = settings
        logger.info(
            "HybridRetriever инициализирован: collection=%s", store.collection
        )

    # ------------------------------------------------------------------
    # Публичный синхронный интерфейс (для ToolRunner + search.py)
    # ------------------------------------------------------------------

    def search_with_plan(self, query_text: str, plan: SearchPlan) -> list[Candidate]:
        """Выполняет hybrid search (dense + sparse RRF) и возвращает список Candidate.

        Синхронный wrapper — запускает asyncio.run() в worker-потоке ThreadPoolExecutor.
        Вызов из основного event loop-потока FastAPI недопустим — только из потоков.

        Args:
            query_text: текст поискового запроса.
            plan:        SearchPlan с normalized_queries, metadata_filters, k_per_query.

        Returns:
            Список Candidate, отсортированный по убыванию RRF score.
        """
        return asyncio.run(self._async_search(query_text, plan))

    def search(self, query: str, k: int = 10, **_kwargs) -> list[Candidate]:
        """Compatibility shim для QAService. Делегирует в search_with_plan.

        QAService ожидает .search(query, k) — этот метод обеспечивает совместимость
        без изменений qa_service.py.
        """
        plan = SearchPlan(
            normalized_queries=[query],
            k_per_query=k,
            fusion="rrf",
        )
        return self.search_with_plan(query, plan)

    # ------------------------------------------------------------------
    # Async ядро
    # ------------------------------------------------------------------

    async def _async_search(
        self, query_text: str, plan: SearchPlan
    ) -> list[Candidate]:
        """Async реализация hybrid search.

        Pipeline:
          1. embed_query(query_text) → dense_vector [1024-dim, L2-norm] (TEI HTTP)
          2. query_embed(query_text) → sparse SparseVector (fastembed BM25, CPU)
          3. query_points(prefetch=[dense, sparse], FusionQuery(RRF), with_vectors=True)
          4. ScoredPoint → Candidate (с _dense_vector в metadata для coverage)
        """
        # 1. Dense embedding (TEI добавляет "query: " prefix внутри)
        dense_vector: list[float] = await self._embedding_client.embed_query(query_text)

        # 2. Sparse encoding (fastembed, CPU, синхронный генератор)
        sparse_result = next(iter(self._sparse_encoder.query_embed(query_text)))
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        # 3. Фильтр из SearchPlan.metadata_filters
        query_filter = self._build_filter(plan.metadata_filters)

        # prefetch_limit: минимум 20, чтобы RRF имел достаточный пул
        prefetch_limit = max(plan.k_per_query * 2, 20)

        logger.debug(
            "HybridRetriever query: collection=%s prefetch_limit=%d k=%d filter=%s",
            self._store.collection,
            prefetch_limit,
            plan.k_per_query,
            bool(query_filter),
        )

        result = await self._store.client.query_points(
            collection_name=self._store.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=QdrantStore.DENSE_VECTOR,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=QdrantStore.SPARSE_VECTOR,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,   # обязательно для cosine_sim в SPEC-RAG-07
            limit=plan.k_per_query,
        )

        candidates = self._to_candidates(result.points)
        logger.debug(
            "HybridRetriever: %d результатов для '%s'", len(candidates), query_text[:60]
        )
        return candidates

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _build_filter(
        self, filters: Optional[MetadataFilters]
    ) -> Optional[models.Filter]:
        """Преобразует MetadataFilters в qdrant_client.models.Filter.

        Поддерживаемые условия:
          - channel_usernames → MatchAny (OR по каналам)
          - date_from / date_to → DatetimeRange
        Неподдерживаемые поля (channel_ids, min_views, reply_to) — игнорируются в Phase 1.
        """
        if not filters:
            return None

        conditions: list[models.FieldCondition] = []

        if filters.channel_usernames:
            # Нормализуем: payload хранит канал без '@' (ingest lstrip('@')),
            # но LLM / пользователь может передать "@news" — убираем @ для совпадения.
            clean_names = [u.lstrip("@") for u in filters.channel_usernames]
            conditions.append(
                models.FieldCondition(
                    key="channel",
                    match=models.MatchAny(any=clean_names),
                )
            )

        if filters.date_from or filters.date_to:
            conditions.append(
                models.FieldCondition(
                    key="date",
                    range=models.DatetimeRange(
                        gte=filters.date_from or None,
                        lte=filters.date_to or None,
                    ),
                )
            )

        return models.Filter(must=conditions) if conditions else None

    def _to_candidates(self, points: list[Any]) -> list[Candidate]:
        """Конвертирует ScoredPoint из Qdrant в Candidate.

        Поле `_dense_vector` в metadata — прокси для cosine_sim в SPEC-RAG-07.
        HybridRetriever не вычисляет coverage — только передаёт вектор.
        """
        candidates: list[Candidate] = []
        for point in points:
            payload: dict[str, Any] = point.payload or {}

            # Извлекаем dense vector (нужен для SPEC-RAG-07 coverage computation)
            dense_vec: list[float] | None = None
            if isinstance(point.vector, dict):
                dense_vec = point.vector.get(QdrantStore.DENSE_VECTOR)

            candidates.append(
                Candidate(
                    id=str(point.id),
                    text=payload.get("text", ""),
                    metadata={
                        "channel": payload.get("channel"),
                        "channel_id": payload.get("channel_id"),
                        "message_id": payload.get("message_id"),
                        "date": payload.get("date"),
                        "author": payload.get("author"),
                        "url": payload.get("url"),
                        # Передаётся в coverage metric без интерпретации здесь
                        "_dense_vector": dense_vec,
                    },
                    bm25_score=None,
                    dense_score=float(point.score),  # RRF score от Qdrant
                    source="hybrid",
                )
            )
        return candidates
```

---

## 3. `src/core/deps.py` — полная замена

```python
"""
Dependency Injection — Phase 1.

Фабрики всех сервисов и клиентов.
Все фабрики используют @lru_cache — синглтоны на процесс.
Смена настроек требует явного cache_clear() через settings.update_*().
"""

import logging
import os
from functools import lru_cache
from typing import Optional

from adapters.llm.llama_server_client import LlamaServerClient
from adapters.qdrant.store import QdrantStore
from adapters.search.hybrid_retriever import HybridRetriever
from adapters.tei.embedding_client import TEIEmbeddingClient
from adapters.tei.reranker_client import TEIRerankerClient
from core.settings import Settings, get_settings
from fastembed import SparseTextEmbedding
from services.agent_service import AgentService
from services.qa_service import QAService
from services.query_planner_service import QueryPlannerService
from services.reranker_service import RerankerService
from services.tools.tool_runner import ToolRunner

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core
# ------------------------------------------------------------------


@lru_cache
def get_llm() -> LlamaServerClient:
    """HTTP-клиент к llama-server (V100, Windows Host, :8080)."""
    settings = get_settings()
    logger.info(
        "LLM: llama-server → %s (model=%s)", settings.llm_base_url, settings.llm_model_name
    )
    return LlamaServerClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model_name,
        timeout=settings.llm_request_timeout,
    )


@lru_cache
def get_planner_llm() -> LlamaServerClient:
    """HTTP-клиент для QueryPlanner. Использует тот же llama-server если не задан PLANNER_LLM_BASE_URL."""
    settings = get_settings()
    base_url = settings.planner_llm_base_url or settings.llm_base_url
    logger.info("Planner LLM: llama-server → %s", base_url)
    return LlamaServerClient(
        base_url=base_url,
        model=settings.llm_model_name,
        timeout=settings.llm_request_timeout,
    )


@lru_cache
def get_query_planner() -> QueryPlannerService:
    settings = get_settings()
    try:
        planner_llm = get_planner_llm()
    except Exception:
        planner_llm = get_llm()
    return QueryPlannerService(planner_llm, settings)


# ------------------------------------------------------------------
# TEI clients (Phase 1: HTTP → WSL2-native RTX 5060 Ti)
# ------------------------------------------------------------------


@lru_cache
def get_tei_embedding_client() -> TEIEmbeddingClient:
    """Singleton TEIEmbeddingClient → embedding TEI service (:8082)."""
    settings = get_settings()
    logger.info("TEI embedding client: %s", settings.embedding_tei_url)
    return TEIEmbeddingClient(base_url=settings.embedding_tei_url)


@lru_cache
def get_tei_reranker_client() -> TEIRerankerClient:
    """Singleton TEIRerankerClient → reranker TEI service (:8083)."""
    settings = get_settings()
    logger.info("TEI reranker client: %s", settings.reranker_tei_url)
    return TEIRerankerClient(base_url=settings.reranker_tei_url)


# ------------------------------------------------------------------
# Qdrant
# ------------------------------------------------------------------


@lru_cache
def get_qdrant_store() -> QdrantStore:
    """Singleton QdrantStore. Смена URL требует cache_clear() через settings.update_*()."""
    settings = get_settings()
    return QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)


# ------------------------------------------------------------------
# SparseTextEmbedding (fastembed, CPU)
# ------------------------------------------------------------------


@lru_cache
def get_sparse_encoder() -> SparseTextEmbedding:
    """Singleton BM25 sparse encoder (fastembed, CPU).

    При первом вызове скачивает модель Qdrant/bm25 (~5 МБ).
    Последующие вызовы используют кэш fastembed.
    """
    logger.info("Инициализация sparse encoder: Qdrant/bm25 (language=russian)")
    return SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")


# ------------------------------------------------------------------
# HybridRetriever
# ------------------------------------------------------------------


@lru_cache
def get_hybrid_retriever() -> Optional[HybridRetriever]:
    """Singleton HybridRetriever (Qdrant + TEI + fastembed BM25).

    Возвращает None если hybrid_enabled=False в настройках.
    """
    settings = get_settings()
    if not settings.hybrid_enabled:
        logger.info("HybridRetriever отключён (hybrid_enabled=False)")
        return None
    try:
        return HybridRetriever(
            store=get_qdrant_store(),
            embedding_client=get_tei_embedding_client(),
            sparse_encoder=get_sparse_encoder(),
            settings=settings,
        )
    except Exception as exc:
        logger.error("HybridRetriever init failed: %s", exc)
        return None


# ------------------------------------------------------------------
# RerankerService (Phase 1: TEIRerankerClient)
# ------------------------------------------------------------------


@lru_cache
def get_reranker() -> Optional[RerankerService]:
    """Singleton RerankerService с TEIRerankerClient.

    Миграция TEIRerankerClient описана в SPEC-RAG-05.
    Возвращает None если enable_reranker=False.
    """
    settings = get_settings()
    if not settings.enable_reranker:
        return None
    reranker_client = get_tei_reranker_client()
    return RerankerService(reranker_client)


# ------------------------------------------------------------------
# QAService (для /v1/qa endpoint)
# ------------------------------------------------------------------


@lru_cache
def get_qa_service() -> QAService:
    settings = get_settings()
    hybrid_retriever = get_hybrid_retriever()

    def _llm_factory():
        return get_llm()

    top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    planner = get_query_planner() if settings.enable_query_planner else None
    reranker = get_reranker() if settings.enable_reranker else None

    # hybrid_retriever передаётся как retriever — QAService использует .search() shim
    return QAService(
        hybrid_retriever,
        _llm_factory,
        top_k,
        settings=settings,
        planner=planner,
        reranker=reranker,
        hybrid=hybrid_retriever,
    )


# ------------------------------------------------------------------
# Redis (без изменений — не в scope Phase 1 migration)
# ------------------------------------------------------------------


@lru_cache
def get_redis_client():
    """Redis-клиент если кеширование включено."""
    settings = get_settings()
    if not settings.redis_enabled:
        return None
    try:
        import redis

        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True,
        )
        client.ping()
        logger.info("Redis подключён: %s:%s", settings.redis_host, settings.redis_port)
        return client
    except Exception as exc:
        logger.warning("Redis недоступен: %s", exc)
        return None


# ------------------------------------------------------------------
# AgentService
# ------------------------------------------------------------------


@lru_cache
def get_agent_service() -> AgentService:
    """Singleton AgentService с полным набором инструментов Phase 1."""
    settings = get_settings()

    tool_runner = ToolRunner(default_timeout_sec=settings.agent_tool_timeout)

    from services.tools.compose_context import compose_context
    from services.tools.fetch_docs import fetch_docs
    from services.tools.final_answer import final_answer
    from services.tools.query_plan import query_plan
    from services.tools.rerank import rerank
    from services.tools.router_select import router_select
    from services.tools.search import search
    from services.tools.verify import verify

    qdrant_store = get_qdrant_store()
    hybrid_retriever = get_hybrid_retriever()
    reranker = get_reranker()
    query_planner = get_query_planner() if settings.enable_query_planner else None

    def search_wrapper(**kwargs):
        return search(hybrid_retriever=hybrid_retriever, **kwargs)

    def verify_wrapper(**kwargs):
        return verify(hybrid_retriever=hybrid_retriever, **kwargs)

    def fetch_docs_wrapper(**kwargs):
        return fetch_docs(qdrant_store=qdrant_store, **kwargs)

    def query_plan_wrapper(**kwargs):
        return query_plan(query_planner=query_planner, **kwargs)

    def rerank_wrapper(**kwargs):
        return rerank(reranker=reranker, **kwargs)

    tool_runner.register("router_select", router_select)
    tool_runner.register(
        "query_plan", query_plan_wrapper, timeout_sec=settings.planner_timeout
    )
    tool_runner.register(
        "search", search_wrapper, timeout_sec=settings.agent_tool_timeout
    )
    tool_runner.register(
        "rerank", rerank_wrapper, timeout_sec=settings.agent_tool_timeout
    )
    tool_runner.register("fetch_docs", fetch_docs_wrapper)
    tool_runner.register("compose_context", compose_context)
    tool_runner.register("verify", verify_wrapper)
    tool_runner.register("final_answer", final_answer)

    def _llm_factory():
        return get_llm()

    qa_service = get_qa_service()

    return AgentService(
        llm_factory=_llm_factory,
        tool_runner=tool_runner,
        settings=settings,
        qa_service=qa_service,
    )
```

---

## 4. `src/services/tools/verify.py` — обновление

Заменить полностью. Ключевые изменения: `Retriever` → `HybridRetriever`,
`retriever.search(claim, k)` → `hybrid_retriever.search_with_plan(claim, plan) -> list[Candidate]`.

```python
"""Инструмент verify — проверка утверждения через повторный поиск в базе знаний."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from adapters.search.hybrid_retriever import HybridRetriever
from schemas.search import SearchPlan

logger = logging.getLogger(__name__)


def verify(
    query: str,
    claim: str,
    hybrid_retriever: HybridRetriever,
    top_k: int = 3,
    docs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Проверяет утверждение через повторный поиск в базе знаний.

    Args:
        query:            исходный запрос пользователя.
        claim:            утверждение для проверки.
        hybrid_retriever: HybridRetriever для поиска.
        top_k:            количество документов для проверки.
        docs:             если переданы, поиск не выполняется (используются готовые docs).

    Returns:
        {verified: bool, confidence: float, evidence: List[str]}
    """
    if not claim.strip():
        return {
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "error": "Пустое утверждение",
        }

    try:
        retrieved_docs: List[Dict[str, Any]] = []

        # Используем переданные docs если доступны
        if docs:
            for item in docs:
                text = item.get("text") or item.get("snippet")
                if text:
                    retrieved_docs.append(
                        {
                            "text": str(text),
                            "distance": item.get("distance"),
                            "source_id": item.get("id"),
                        }
                    )

        # Если docs не переданы — выполняем поиск
        if not retrieved_docs:
            plan = SearchPlan(
                normalized_queries=[claim],
                k_per_query=top_k,
                fusion="rrf",
            )
            candidates = hybrid_retriever.search_with_plan(claim, plan)
            for idx, cand in enumerate(candidates[:top_k]):
                retrieved_docs.append(
                    {
                        "text": cand.text,
                        "distance": None,   # RRF score не является distance
                        "source_id": cand.id,
                        "rank": idx,
                    }
                )

        if not retrieved_docs:
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": [],
                "note": "Релевантные документы не найдены",
            }

        # Оцениваем confidence по рангу (RRF score не нормализован для расстояния)
        evidence: List[str] = []
        confidences: List[float] = []
        for idx, doc in enumerate(retrieved_docs[:top_k]):
            text = doc.get("text", "")
            # Используем ранговое убывание: rank 0 → 0.9, rank 1 → 0.8, ...
            confidence = max(0.3, 0.9 - idx * 0.1)
            confidences.append(confidence)
            evidence.append(text[:200] + "..." if len(text) > 200 else text)

        avg_confidence = sum(confidences) / len(confidences)
        threshold = 0.6
        verified = avg_confidence >= threshold

        return {
            "verified": verified,
            "confidence": round(avg_confidence, 3),
            "evidence": evidence,
            "threshold": threshold,
            "documents_found": len(retrieved_docs),
            "used_docs": min(len(retrieved_docs), top_k),
        }

    except Exception as exc:
        logger.error("Ошибка при проверке утверждения: %s", exc)
        return {
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "error": f"Ошибка поиска: {str(exc)}",
        }
```

---

## 5. `src/services/tools/fetch_docs.py` — обновление

Заменить полностью. `Retriever.get_by_ids()` → `asyncio.run(QdrantStore.get_by_ids())`.

```python
"""Инструмент fetch_docs — батч-выгрузка документов по ID из Qdrant."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from adapters.qdrant.store import QdrantStore

logger = logging.getLogger(__name__)


def fetch_docs(
    qdrant_store: QdrantStore,
    ids: Optional[List[str]] = None,
    window: Optional[List[int]] = None,
    doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Батч-выгрузка документов по IDs из Qdrant.

    Args:
        qdrant_store: QdrantStore для прямого доступа к точкам.
        ids:          список point ID вида "{channel}:{message_id}".
        window:       не используется в Phase 1 (оставлен для совместимости сигнатуры).
        doc_ids:      алиас для ids.

    Returns:
        {"docs": [{"id": ..., "text": ..., "metadata": {...}}]}
    """
    final_ids: List[str] = ids or doc_ids or []
    if not final_ids:
        logger.debug("fetch_docs вызван без ids")
        return {"docs": []}

    try:
        # asyncio.run() валиден в worker-потоке ThreadPoolExecutor (нет event loop)
        records = asyncio.run(qdrant_store.get_by_ids(final_ids))
        docs = []
        for record in records:
            payload = record.payload or {}
            docs.append(
                {
                    "id": str(record.id),
                    "text": payload.get("text", ""),
                    "metadata": {
                        "channel": payload.get("channel"),
                        "message_id": payload.get("message_id"),
                        "date": payload.get("date"),
                        "author": payload.get("author"),
                        "url": payload.get("url"),
                    },
                }
            )
        logger.debug(
            "fetch_docs: %d документов для %d ids", len(docs), len(final_ids)
        )
        return {"docs": docs}

    except Exception as exc:
        logger.error("fetch_docs ошибка для ids=%s: %s", final_ids, exc)
        return {
            "docs": [{"id": _id, "text": "", "metadata": {}} for _id in final_ids]
        }
```

---

## 6. Тесты

Файл: `src/tests/test_hybrid_retriever.py`

```python
# src/tests/test_hybrid_retriever.py

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models as qdrant_models

from adapters.search.hybrid_retriever import HybridRetriever
from schemas.search import MetadataFilters, SearchPlan


def make_retriever(
    collection: str = "news",
    k_per_query: int = 10,
) -> tuple[HybridRetriever, MagicMock, MagicMock, MagicMock]:
    """Возвращает (retriever, mock_store, mock_tei, mock_sparse)."""
    mock_store = MagicMock()
    mock_store.collection = collection
    mock_store.client = AsyncMock()

    # Имитируем query_points — возвращает объект с .points
    mock_result = MagicMock()
    mock_result.points = []
    mock_store.client.query_points = AsyncMock(return_value=mock_result)

    mock_tei = AsyncMock()
    mock_tei.embed_query = AsyncMock(return_value=[0.1] * 1024)

    mock_sparse_enc = MagicMock()
    mock_sparse_result = MagicMock()
    mock_sparse_result.indices = MagicMock(tolist=lambda: [1, 5, 10])
    mock_sparse_result.values = MagicMock(tolist=lambda: [0.5, 0.3, 0.2])
    mock_sparse_enc.query_embed = MagicMock(return_value=iter([mock_sparse_result]))

    settings = MagicMock()
    settings.hybrid_enabled = True

    retriever = HybridRetriever(
        store=mock_store,
        embedding_client=mock_tei,
        sparse_encoder=mock_sparse_enc,
        settings=settings,
    )
    return retriever, mock_store, mock_tei, mock_sparse_enc


def make_plan(k: int = 10, filters: MetadataFilters | None = None) -> SearchPlan:
    return SearchPlan(normalized_queries=["test query"], k_per_query=k, fusion="rrf")


# ------------------------------------------------------------------
# _async_search: основной путь
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_search_calls_embed_and_query_points() -> None:
    """_async_search вызывает embed_query и query_points с правильными параметрами."""
    retriever, mock_store, mock_tei, _ = make_retriever()

    await retriever._async_search("курс рубля", make_plan(k=5))

    mock_tei.embed_query.assert_awaited_once_with("курс рубля")
    mock_store.client.query_points.assert_awaited_once()
    call_kwargs = mock_store.client.query_points.call_args.kwargs
    assert call_kwargs["collection_name"] == "news"
    assert call_kwargs["with_vectors"] is True
    assert call_kwargs["limit"] == 5
    # Проверяем наличие prefetch с dense и sparse
    prefetch = call_kwargs["prefetch"]
    assert len(prefetch) == 2
    assert prefetch[0].using == "dense_vector"
    assert prefetch[1].using == "sparse_vector"


@pytest.mark.asyncio
async def test_async_search_uses_fusion_rrf() -> None:
    """query FusionQuery(RRF) используется всегда."""
    retriever, mock_store, _, _ = make_retriever()

    await retriever._async_search("test", make_plan())

    call_kwargs = mock_store.client.query_points.call_args.kwargs
    assert isinstance(call_kwargs["query"], qdrant_models.FusionQuery)
    assert call_kwargs["query"].fusion == qdrant_models.Fusion.RRF


# ------------------------------------------------------------------
# _build_filter
# ------------------------------------------------------------------

def test_build_filter_none_when_no_filters() -> None:
    """Без фильтров возвращает None."""
    retriever, *_ = make_retriever()
    result = retriever._build_filter(None)
    assert result is None


def test_build_filter_channel_usernames() -> None:
    """channel_usernames с @ → MatchAny без @ (payload хранит без @)."""
    retriever, *_ = make_retriever()
    filters = MetadataFilters(channel_usernames=["@news", "@finance"])
    result = retriever._build_filter(filters)
    assert result is not None
    cond = result.must[0]
    assert cond.key == "channel"
    # Фильтр нормализован: @ убран для совпадения с payload
    assert set(cond.match.any) == {"news", "finance"}


def test_build_filter_date_range() -> None:
    """date_from/date_to конвертируется в DatetimeRange."""
    retriever, *_ = make_retriever()
    filters = MetadataFilters(date_from="2026-01-01T00:00:00", date_to="2026-03-01T00:00:00")
    result = retriever._build_filter(filters)
    assert result is not None
    cond = result.must[0]
    assert cond.key == "date"
    assert cond.range.gte == "2026-01-01T00:00:00"


# ------------------------------------------------------------------
# _to_candidates
# ------------------------------------------------------------------

def test_to_candidates_extracts_fields() -> None:
    """ScoredPoint → Candidate с id, text, metadata, dense_score."""
    retriever, *_ = make_retriever()

    point = MagicMock()
    point.id = "channel:123"
    point.score = 0.42
    point.payload = {
        "text": "Новость", "channel": "@news",
        "message_id": 123, "date": "2026-01-01T00:00:00",
    }
    point.vector = {"dense_vector": [0.1] * 1024}

    candidates = retriever._to_candidates([point])

    assert len(candidates) == 1
    c = candidates[0]
    assert c.id == "channel:123"
    assert c.text == "Новость"
    assert c.dense_score == 0.42
    assert c.metadata["channel"] == "@news"
    assert "_dense_vector" in c.metadata
    assert len(c.metadata["_dense_vector"]) == 1024


# ------------------------------------------------------------------
# sync wrapper
# ------------------------------------------------------------------

def test_search_with_plan_is_sync() -> None:
    """search_with_plan возвращает список (sync), не корутину."""
    retriever, *_ = make_retriever()
    plan = make_plan()
    result = retriever.search_with_plan("test", plan)
    assert isinstance(result, list)
```

---

## 7. Чеклист реализации

- [ ] `src/adapters/search/hybrid_retriever.py` — перезаписан полностью
  - [ ] `HybridRetriever.__init__` принимает `store, embedding_client, sparse_encoder, settings`
  - [ ] `search_with_plan()` — sync, вызывает `asyncio.run(_async_search())`
  - [ ] `search()` — compatibility shim для QAService
  - [ ] `_async_search()` — embed → sparse → `query_points(prefetch RRF, with_vectors=True)`
  - [ ] `_build_filter()` — `channel_usernames → MatchAny`, `date → DatetimeRange`
  - [ ] `_to_candidates()` — включает `_dense_vector` в metadata
- [ ] `src/core/deps.py` — перезаписан полностью
  - [ ] `get_llm()`, `get_planner_llm()`, `get_query_planner()` — сохранены без изменений
  - [ ] `get_tei_embedding_client()` — `@lru_cache` → `TEIEmbeddingClient`
  - [ ] `get_tei_reranker_client()` — `@lru_cache` → `TEIRerankerClient`
  - [ ] `get_qdrant_store()` — `@lru_cache` → `QdrantStore`
  - [ ] `get_sparse_encoder()` — `@lru_cache` → `SparseTextEmbedding("Qdrant/bm25", "russian")`
  - [ ] `get_hybrid_retriever()` — новый, возвращает `Optional[HybridRetriever]`
  - [ ] `get_reranker()` — использует `TEIRerankerClient` (SPEC-RAG-05 детализирует)
  - [ ] `get_agent_service()` — использует `qdrant_store`, новый `hybrid_retriever`; нет bm25/chroma
  - [ ] Нет импортов `chromadb`, `BM25IndexManager`, `BM25Retriever`, `utils.model_downloader`
- [ ] `src/services/tools/verify.py` — перезаписан
  - [ ] Импортирует `HybridRetriever`, не `Retriever` (chroma)
  - [ ] Вызывает `hybrid_retriever.search_with_plan(claim, plan)`
- [ ] `src/services/tools/fetch_docs.py` — перезаписан
  - [ ] Импортирует `QdrantStore`, не `Retriever` (chroma)
  - [ ] Вызывает `asyncio.run(qdrant_store.get_by_ids(final_ids))`
- [ ] **Удалено:**
  - [ ] `src/adapters/chroma/` — весь каталог (retriever.py, __init__.py, ...)
  - [ ] `src/adapters/search/bm25_index.py`
  - [ ] `src/adapters/search/bm25_retriever.py`
- [ ] `fastembed>=0.3.0` добавлен в `requirements.txt`
- [ ] `src/tests/test_hybrid_retriever.py` — 8 тестов реализованы, `pytest` проходит
- [ ] `docker compose up` — приложение стартует, коллекция `news` создаётся в Qdrant
