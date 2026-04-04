# SPEC-RAG-02: TEI Adapter Layer

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Ready for implementation
> **Цель:** Создать `src/adapters/tei/` — HTTP-клиенты для TEI embedding и TEI reranker,
>           заменяющие локальные SentenceTransformer и CrossEncoder.
> **Зависит от:** SPEC-RAG-01 (settings: `embedding_tei_url`, `reranker_tei_url`)
> **Источники:** `docs/specifications/arch-brief.md` (DEC-0016, DEC-0017),
>                `docs/architecture/04-system/overview.md` §Stack,
>                `docs/research/reports/R06-async-architecture.md` §httpx patterns

---

## 0. Implementation Pointers

### 0.1 Текущее состояние

| Компонент | Где сейчас | Что делает |
|-----------|-----------|-----------|
| Embedding | `src/adapters/chroma/retriever.py` → `SentenceTransformerEmbeddingFunction` | Локальная загрузка модели в процессе |
| Reranker | `src/services/reranker_service.py` → `CrossEncoder` | Локальная загрузка модели в процессе |
| HTTP клиент | отсутствует для моделей | — |

### 0.2 Новые файлы (создать)

```
src/adapters/tei/
├── __init__.py          # экспорт: TEIEmbeddingClient, TEIRerankerClient
├── embedding_client.py  # TEIEmbeddingClient
└── reranker_client.py   # TEIRerankerClient
```

### 0.3 Что НЕ удаляется в этой спецификации

`SentenceTransformerEmbeddingFunction` и `CrossEncoder` **не удаляются здесь** — их удаление
происходит в SPEC-RAG-04 (`hybrid_retriever.py`) и SPEC-RAG-05 (`reranker_service.py`).
SPEC-RAG-02 только создаёт новый слой адаптеров.

---

## 1. Обзор

### 1.1 Задача

1. Создать `TEIEmbeddingClient` — async HTTP-клиент к TEI embedding service (`POST /embed`).
2. Создать `TEIRerankerClient` — async HTTP-клиент к TEI reranker service (`POST /rerank`).
3. Оба клиента используют `httpx.AsyncClient` с переиспользованием connection pool.
4. Добавить заготовки фабрик в `src/core/deps.py` (полная замена deps.py в SPEC-RAG-04).

### 1.2 Контекст

TEI (text-embeddings-inference) от HuggingFace — production HTTP-сервер для embedding и reranking.
Работает за пределами Docker в WSL2 native на RTX 5060 Ti. Docker-контейнер обращается через
`host.docker.internal`.

**Почему не локальные модели:**
- `SentenceTransformer` в-процессе: ~2s на инициализацию, занимает Docker CPU
- TEI: GPU inference, суб-миллисекундный warm latency для batches, не занимает CPU Docker

**Instruction prefix для multilingual-e5-large:**
`intfloat/multilingual-e5-large` — instruction-tuned модель, требует префикс для правильного
поиска. Без префикса качество retrieval снижается (~5–8% NDCG@10):
- Для поисковых запросов: `"query: {text}"`
- Для индексируемых документов: `"passage: {text}"`

### 1.3 Ключевые решения

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| HTTP клиент | `httpx.AsyncClient` | Нативный async, connection pool, совместим с FastAPI event loop |
| Connection pool | Один клиент на инстанс (не per-request) | Pool overhead значительный при >10 RPS; инстанс через `lru_cache` в deps.py |
| Timeout embedding | `30.0s` для batch, `10.0s` для single | Batch 100 doc × 1024-dim на GPU ≈ 50–200ms; 30s — запас при cold start TEI |
| Timeout reranker | `30.0s` | rerank 80 passages ≈ 100–300ms; запас на cold start |
| Normalize | `True` (передаётся в TEI) | Qdrant cosine similarity требует L2-normalized векторов |
| Instruction prefix | В клиенте, не в вызывающем коде | Инкапсуляция: вызывающий код не должен знать о деталях модели |
| Reranker output order | Восстановить исходный порядок | TEI `/rerank` возвращает результаты отсортированными по score; нужен порядок по `index` |
| Error handling | `httpx.HTTPStatusError`, `httpx.ConnectError`, `asyncio.TimeoutError` | TEI может быть не запущен при старте контейнера; нужен graceful log + re-raise |

### 1.4 Что НЕ делать

- Не создавать новый `httpx.AsyncClient` на каждый запрос — будет утечка соединений
- Не делать retry внутри клиента — retry-логика на уровне orchestration (agent_service)
- Не хардкодить URL — только через `settings.embedding_tei_url` / `settings.reranker_tei_url`
- Не добавлять instruction prefix для reranker (bge-reranker-v2-m3 не требует)
- Не обрезать/обрабатывать ответ в клиенте — возвращать raw vectors/scores, нормализация на стороне вызывающего кода для reranker

---

## 2. src/adapters/tei/embedding_client.py

```python
"""
HTTP-клиент для TEI embedding service (intfloat/multilingual-e5-large).

Обёртка над TEI REST API:
  POST /embed  → list[list[float]]  (normalize=True, 1024-dim)

Instruction prefix для e5-large (обязателен для корректного retrieval):
  query-текст:    "query: {text}"
  document-текст: "passage: {text}"
"""

from __future__ import annotations

import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)

# Instruction prefixes для intfloat/multilingual-e5-large.
# Без префикса качество retrieval снижается ~5–8% NDCG@10.
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class TEIEmbeddingClient:
    """
    Async HTTP-клиент для TEI embedding service.

    Используется для:
    - embed_query: встраивание поискового запроса (с prefix "query: ")
    - embed_documents: батчевое встраивание документов при ingest (с prefix "passage: ")

    Connection pool переиспользуется между вызовами — инстанс должен быть singleton.
    Создаётся через deps.get_tei_embedding_client().
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """
        Args:
            base_url: URL TEI service, например "http://host.docker.internal:8082"
            timeout: таймаут HTTP запроса в секундах (default 30s)
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            # Лимит соединений: embedding вызывается последовательно в pipeline,
            # но при параллельных subquery-запросах может быть несколько concurrent
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        logger.info("TEIEmbeddingClient инициализирован: %s", self.base_url)

    async def embed_query(self, text: str) -> List[float]:
        """
        Встраивает один поисковый запрос.

        Применяет prefix "query: " согласно спецификации multilingual-e5-large.
        Возвращает L2-нормализованный вектор 1024-dim.

        Args:
            text: поисковый запрос на естественном языке

        Returns:
            list[float] длиной 1024

        Raises:
            httpx.ConnectError: TEI service недоступен
            httpx.HTTPStatusError: TEI вернул ошибку (4xx/5xx)
        """
        prefixed = _QUERY_PREFIX + text
        vectors = await self._embed_batch([prefixed], normalize=True)
        return vectors[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Батчевое встраивание документов для ingest.

        Применяет prefix "passage: " к каждому тексту.
        Возвращает L2-нормализованные векторы 1024-dim.

        Args:
            texts: список текстов документов

        Returns:
            list[list[float]], каждый вектор длиной 1024

        Raises:
            httpx.ConnectError: TEI service недоступен
            httpx.HTTPStatusError: TEI вернул ошибку (4xx/5xx)
        """
        prefixed = [_PASSAGE_PREFIX + t for t in texts]
        return await self._embed_batch(prefixed, normalize=True)

    async def _embed_batch(
        self, texts: List[str], normalize: bool = True
    ) -> List[List[float]]:
        """Внутренний метод: POST /embed с батчем текстов."""
        try:
            response = await self._client.post(
                "/embed",
                json={"inputs": texts, "normalize": normalize},
            )
            response.raise_for_status()
            vectors: List[List[float]] = response.json()
            logger.debug(
                "TEI embed: %d текстов → %d векторов (dim=%d)",
                len(texts),
                len(vectors),
                len(vectors[0]) if vectors else 0,
            )
            return vectors
        except httpx.ConnectError as exc:
            logger.error("TEI embedding недоступен (%s): %s", self.base_url, exc)
            raise
        except httpx.HTTPStatusError as exc:
            logger.error(
                "TEI embedding вернул ошибку %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise

    async def healthcheck(self) -> bool:
        """
        Проверяет доступность TEI service.

        Returns:
            True если service отвечает на GET /health, False иначе
        """
        try:
            response = await self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("TEI embedding healthcheck failed: %s", exc)
            return False

    async def aclose(self) -> None:
        """Закрывает HTTP connection pool. Вызывать при shutdown приложения."""
        await self._client.aclose()
```

---

## 3. src/adapters/tei/reranker_client.py

```python
"""
HTTP-клиент для TEI reranker service (BAAI/bge-reranker-v2-m3).

Обёртка над TEI REST API:
  POST /rerank → list[{index: int, score: float}]  (отсортировано по score desc)

bge-reranker-v2-m3 НЕ требует instruction prefix.
Возвращаем scores в исходном порядке passages (по index), не по score.
"""

from __future__ import annotations

import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)


class TEIRerankerClient:
    """
    Async HTTP-клиент для TEI reranker service.

    Принимает query + список passages, возвращает relevance scores
    в том же порядке, что и входные passages.

    Создаётся через deps.get_tei_reranker_client().
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """
        Args:
            base_url: URL TEI service, например "http://host.docker.internal:8083"
            timeout: таймаут HTTP запроса в секундах
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        logger.info("TEIRerankerClient инициализирован: %s", self.base_url)

    async def rerank(self, query: str, passages: List[str]) -> List[float]:
        """
        Переранжирует passages по релевантности к query.

        TEI /rerank возвращает результаты отсортированными по убыванию score.
        Этот метод восстанавливает исходный порядок passages: scores[i]
        соответствует passages[i].

        Args:
            query: поисковый запрос
            passages: список текстов для ранжирования (обычно 20–80 штук)

        Returns:
            list[float] длиной len(passages): score[i] для passages[i].
            Score в диапазоне (обычно) -10..10, не нормализован.
            Для нормализации в [0,1] вызывающий код использует sigmoid или min-max.

        Raises:
            httpx.ConnectError: TEI service недоступен
            httpx.HTTPStatusError: TEI вернул ошибку
        """
        if not passages:
            return []

        try:
            response = await self._client.post(
                "/rerank",
                json={
                    "query": query,
                    "texts": passages,
                    "raw_scores": True,   # получаем logit scores до sigmoid
                    "truncate": True,     # автообрезка длинных текстов
                },
            )
            response.raise_for_status()

            # TEI возвращает [{"index": i, "score": f}, ...] sorted by score desc.
            # Восстанавливаем порядок по index, чтобы scores[i] ↔ passages[i].
            results = response.json()  # list[{index, score}]
            scores = [0.0] * len(passages)
            for item in results:
                scores[item["index"]] = item["score"]

            logger.debug(
                "TEI rerank: query=%r, %d passages → scores [%.3f..%.3f]",
                query[:50],
                len(passages),
                min(scores),
                max(scores),
            )
            return scores

        except httpx.ConnectError as exc:
            logger.error("TEI reranker недоступен (%s): %s", self.base_url, exc)
            raise
        except httpx.HTTPStatusError as exc:
            logger.error(
                "TEI reranker вернул ошибку %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise

    async def healthcheck(self) -> bool:
        """Проверяет доступность TEI reranker service."""
        try:
            response = await self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("TEI reranker healthcheck failed: %s", exc)
            return False

    async def aclose(self) -> None:
        """Закрывает HTTP connection pool."""
        await self._client.aclose()
```

---

## 4. src/adapters/tei/\_\_init\_\_.py

```python
"""TEI HTTP адаптеры для embedding и reranking."""

from .embedding_client import TEIEmbeddingClient
from .reranker_client import TEIRerankerClient

__all__ = ["TEIEmbeddingClient", "TEIRerankerClient"]
```

---

## 5. Интеграция — добавить в src/core/deps.py

> Полная переработка `deps.py` — в SPEC-RAG-04. Здесь только два новых фрагмента,
> которые нужно добавить к существующему файлу до SPEC-RAG-04.

```python
# --- добавить в импорты deps.py ---
from adapters.tei import TEIEmbeddingClient, TEIRerankerClient

# --- добавить новые фабрики ---

@lru_cache()
def get_tei_embedding_client() -> TEIEmbeddingClient:
    """Singleton TEI embedding клиента. URL берётся из settings."""
    settings = get_settings()
    return TEIEmbeddingClient(base_url=settings.embedding_tei_url)


@lru_cache()
def get_tei_reranker_client() -> TEIRerankerClient:
    """Singleton TEI reranker клиента. URL берётся из settings."""
    settings = get_settings()
    return TEIRerankerClient(base_url=settings.reranker_tei_url)
```

**Важно:** `httpx.AsyncClient` внутри клиентов не закрывается при `cache_clear()`.
Для production shutdown нужен lifespan hook в `main.py`:

```python
# src/main.py — добавить в lifespan или shutdown event
from core.deps import get_tei_embedding_client, get_tei_reranker_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Graceful shutdown: закрыть HTTP connection pools
    emb = get_tei_embedding_client()
    await emb.aclose()
    rer = get_tei_reranker_client()
    await rer.aclose()
```

Если в `main.py` уже есть lifespan — добавить `aclose()` вызовы в shutdown секцию.

---

## 6. Тесты

Файл: `src/tests/test_tei_clients.py`

```python
"""
Тесты для TEI HTTP клиентов.

Unit-тесты используют httpx.MockTransport для изоляции от реального TEI.
Integration-тест (помечен @pytest.mark.integration) требует запущенного TEI.
"""

import pytest
import httpx
from adapters.tei import TEIEmbeddingClient, TEIRerankerClient


# ─── Fixtures ────────────────────────────────────────────────────────────────

class MockEmbedTransport(httpx.MockTransport):
    """Mock TEI embedding: возвращает фиксированный вектор 1024-dim."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embed":
            import json
            body = json.loads(request.content)
            n = len(body["inputs"])
            vectors = [[0.1] * 1024 for _ in range(n)]
            return httpx.Response(200, json=vectors)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(404)


class MockRerankTransport(httpx.MockTransport):
    """Mock TEI reranker: возвращает убывающие scores."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rerank":
            import json
            body = json.loads(request.content)
            n = len(body["texts"])
            # Возвращаем в обратном порядке (имитируем TEI sort-by-score)
            results = [{"index": i, "score": float(n - i)} for i in range(n)]
            results.sort(key=lambda x: x["score"], reverse=True)
            return httpx.Response(200, json=results)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(404)


# ─── Embedding tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_query_returns_vector():
    """embed_query возвращает вектор правильной размерности."""
    client = TEIEmbeddingClient.__new__(TEIEmbeddingClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(
        transport=MockEmbedTransport(), base_url="http://mock"
    )
    vector = await client.embed_query("тест запрос")
    assert len(vector) == 1024
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.asyncio
async def test_embed_query_adds_prefix():
    """embed_query добавляет prefix 'query: ' к тексту."""
    captured_inputs = []

    class CapturingTransport(httpx.MockTransport):
        def handle_request(self, request):
            import json
            body = json.loads(request.content)
            captured_inputs.extend(body["inputs"])
            return httpx.Response(200, json=[[0.1] * 1024])

    client = TEIEmbeddingClient.__new__(TEIEmbeddingClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(
        transport=CapturingTransport(), base_url="http://mock"
    )
    await client.embed_query("новости крипто")
    assert captured_inputs[0] == "query: новости крипто"


@pytest.mark.asyncio
async def test_embed_documents_adds_passage_prefix():
    """embed_documents добавляет prefix 'passage: ' к каждому документу."""
    captured_inputs = []

    class CapturingTransport(httpx.MockTransport):
        def handle_request(self, request):
            import json
            body = json.loads(request.content)
            captured_inputs.extend(body["inputs"])
            return httpx.Response(200, json=[[0.1] * 1024, [0.2] * 1024])

    client = TEIEmbeddingClient.__new__(TEIEmbeddingClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(
        transport=CapturingTransport(), base_url="http://mock"
    )
    await client.embed_documents(["текст 1", "текст 2"])
    assert captured_inputs[0] == "passage: текст 1"
    assert captured_inputs[1] == "passage: текст 2"


@pytest.mark.asyncio
async def test_embed_connect_error_propagates():
    """ConnectError пробрасывается наружу без подавления."""
    class FailTransport(httpx.MockTransport):
        def handle_request(self, request):
            raise httpx.ConnectError("connection refused")

    client = TEIEmbeddingClient.__new__(TEIEmbeddingClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(
        transport=FailTransport(), base_url="http://mock"
    )
    with pytest.raises(httpx.ConnectError):
        await client.embed_query("test")


# ─── Reranker tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rerank_restores_original_order():
    """rerank() возвращает scores в порядке входных passages, не по убыванию score."""
    client = TEIRerankerClient.__new__(TEIRerankerClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(
        transport=MockRerankTransport(), base_url="http://mock"
    )
    passages = ["doc_a", "doc_b", "doc_c"]
    scores = await client.rerank("query", passages)

    assert len(scores) == 3
    # MockRerankTransport присваивает score = n - index, т.е. [3.0, 2.0, 1.0]
    assert scores[0] == 3.0
    assert scores[1] == 2.0
    assert scores[2] == 1.0


@pytest.mark.asyncio
async def test_rerank_empty_passages():
    """rerank() с пустым списком возвращает пустой список без HTTP запроса."""
    client = TEIRerankerClient.__new__(TEIRerankerClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(base_url="http://mock")
    scores = await client.rerank("query", [])
    assert scores == []


@pytest.mark.asyncio
async def test_healthcheck_returns_true_on_200():
    client = TEIEmbeddingClient.__new__(TEIEmbeddingClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(
        transport=MockEmbedTransport(), base_url="http://mock"
    )
    assert await client.healthcheck() is True
```

---

## 7. Чеклист реализации

- [ ] `src/adapters/tei/__init__.py` создан, экспортирует `TEIEmbeddingClient`, `TEIRerankerClient`
- [ ] `src/adapters/tei/embedding_client.py` создан
- [ ] `TEIEmbeddingClient.__init__` принимает `base_url`, `timeout=30.0`
- [ ] `embed_query` добавляет prefix `"query: "`, возвращает `list[float]` длиной 1024
- [ ] `embed_documents` добавляет prefix `"passage: "` к каждому тексту
- [ ] `_embed_batch` передаёт `{"inputs": [...], "normalize": True}` в `POST /embed`
- [ ] `src/adapters/tei/reranker_client.py` создан
- [ ] `TEIRerankerClient.rerank` восстанавливает порядок по `index` (не по убыванию score)
- [ ] `TEIRerankerClient.rerank` передаёт `{"raw_scores": True, "truncate": True}`
- [ ] `rerank(query, [])` возвращает `[]` без HTTP запроса
- [ ] Оба клиента: `httpx.ConnectError` логируется и пробрасывается
- [ ] Оба клиента: `httpx.HTTPStatusError` логируется и пробрасывается
- [ ] Оба клиента: `healthcheck()` возвращает `bool`, не бросает исключений
- [ ] Оба клиента: `aclose()` закрывает `AsyncClient`
- [ ] `deps.py`: добавлены `get_tei_embedding_client()`, `get_tei_reranker_client()` с `@lru_cache`
- [ ] `main.py`: lifespan вызывает `aclose()` на обоих клиентах при shutdown
- [ ] `src/tests/test_tei_clients.py` создан, все тесты проходят: `pytest src/tests/test_tei_clients.py`
