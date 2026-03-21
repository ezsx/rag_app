# SPEC-RAG-05: RerankerService Migration

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Draft
> **Цель:** Заменить локальный CrossEncoder (CPU, sentence-transformers) на синхронный HTTP-клиент к TEI reranker service (BAAI/bge-reranker-v2-m3, WSL2 native, RTX 5060 Ti).
> **Источники:** DEC-0017 (`docs/architecture/11-decisions/decision-log.md`),
>               `docs/architecture/04-system/overview.md`,
>               SPEC-RAG-02 (TEI adapter layer — `TEIRerankerClient` async интерфейс),
>               SPEC-RAG-04 (deps.py `get_reranker()` factory)

---

## 0. Implementation Pointers

### 0.1 Текущие файлы

| Файл | Текущее поведение | После SPEC-RAG-05 |
|------|-------------------|-------------------|
| `src/services/reranker_service.py` | `CrossEncoder` (sentence-transformers, CPU); `__init__` принимает `model_name: str`; `rerank()` → `List[int]` | `httpx.Client` (sync HTTP к TEI); `__init__` принимает `base_url: str`; то же API |
| `src/services/tools/rerank.py` | Вызывает `reranker.rerank()`, возвращает dummy scores `[1.0, 0.9, ...]` | Вызывает `reranker.rerank_with_scores()`, возвращает реальные sigmoid scores |
| `src/core/deps.py` | `get_reranker()` → `RerankerService(reranker_client)` (SPEC-RAG-04 заглушка) | `get_reranker()` → `RerankerService(base_url=settings.reranker_tei_url)` |
| `src/core/settings.py` | Поле `reranker_model_key` (CrossEncoder model path) | Поле удаляется — больше не нужно |
| `src/utils/model_downloader.py` | Утилита скачивания HF-моделей (LLM, embedding, reranker) | **Удаляется полностью** |

### 0.2 Новые файлы

Новых файлов нет. Только рефакторинг существующих.

### 0.3 Что удалить

- `src/utils/model_downloader.py` — полностью. Все локальные HF-модели выведены за пределы Docker (DEC-0016, DEC-0017, DEC-0024). TEI управляет своим кэшем самостоятельно.
- `sentence_transformers` зависимость — из `requirements.txt` / `pyproject.toml` (если не нужна другим модулям).
- Поле `reranker_model_key` из `src/core/settings.py`.

---

## 1. Обзор

### 1.1 Задача

1. Переписать `src/services/reranker_service.py`: убрать `CrossEncoder`, заменить на sync `httpx.Client` к TEI reranker (`http://host.docker.internal:8083`).
2. Добавить метод `rerank_with_scores()` → возвращает индексы + sigmoid-нормализованные scores. Нужен для SPEC-RAG-07 (`passage_relevance` сигнал composite coverage).
3. Обновить `src/services/tools/rerank.py`: использовать `rerank_with_scores()` вместо dummy scores.
4. Обновить `get_reranker()` в `src/core/deps.py`: передавать `base_url` напрямую вместо `TEIRerankerClient`.
5. Удалить `src/utils/model_downloader.py`.
6. Удалить поле `reranker_model_key` из `src/core/settings.py`.

### 1.2 Контекст — почему именно так

**Проблема с async TEIRerankerClient:**
SPEC-RAG-02 определил `TEIRerankerClient` с `async def rerank()`. Однако `RerankerService.rerank()` вызывается из синхронных методов `qa_service.py` (`_fetch_context`, `answer_v2`), которые в свою очередь могут вызываться из `async def stream_answer`. При этом asyncio event loop уже запущен → `asyncio.run()` внутри sync метода вызовет `RuntimeError: This event loop is already running`.

**Решение:**
`RerankerService` использует sync `httpx.Client` напрямую, **не** через `TEIRerankerClient`. Это безопасно как из async FastAPI контекста, так и из ThreadPoolExecutor ToolRunner. Sync HTTP-блокировка приемлема — rerank-запрос занимает 20–80 мс на GPU.

**Отличие от HybridRetriever (SPEC-RAG-04):**
HybridRetriever использует `asyncio.run()` через `search_with_plan()`, потому что search-инструмент вызывается исключительно через ToolRunner (ThreadPoolExecutor). QAService вызывает `_fetch_context()` синхронно, включая rerank. Использование `asyncio.run()` здесь невозможно.

**`TEIRerankerClient` (async) — не удаляется:**
Остаётся в `src/adapters/tei/reranker_client.py` как async-вариант для возможного будущего использования (Phase 2: vLLM + Hermes). В текущем Phase 1 реально используется только sync `RerankerService`.

### 1.3 Ключевые решения

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| Sync vs async HTTP | sync `httpx.Client` | qa_service.py вызывает rerank из sync контекстов; `asyncio.run()` недоступен при запущенном loop |
| Сохранить `batch_size` параметр | Да, ignored | qa_service.py передаёт `batch_size=settings.reranker_batch_size` явно; нельзя менять qa_service.py |
| Normalisation | sigmoid(logit) | TEI возвращает raw logit scores; sigmoid → [0,1] монотонна (порядок ранжирования не меняется) |
| `rerank_with_scores()` | Новый метод | `rerank()` (backward compat) + отдельный метод для SPEC-RAG-07; не ломать qa_service.py |
| `model_downloader.py` | Удалить целиком | Весь смысл файла — скачивать локальные модели; в Phase 1 все модели вне Docker |

### 1.4 Что НЕ делать

- **НЕ** делать `rerank()` async — qa_service.py использует его синхронно и помечен "без изменений".
- **НЕ** использовать `asyncio.run()` в `rerank()` — event loop может быть запущен при вызове из async контекста.
- **НЕ** оборачивать `TEIRerankerClient` — создать sync клиент напрямую в `RerankerService`.
- **НЕ** менять сигнатуру `rerank(query, docs, top_n, batch_size)` — qa_service.py вызывает с именованными аргументами.
- **НЕ** трогать `src/api/`, `src/services/agent_service.py`, `qa_service.py`, `src/schemas/` — вне scope.
- **НЕ** нормализовать scores через min-max — sigmoid стабильнее при экстремальных значениях logit.

---

## 2. src/services/reranker_service.py

Полный rewrite. Удаляем CrossEncoder, добавляем sync httpx.

```python
"""
ReAct-сервис переранжирования на основе TEI HTTP (bge-reranker-v2-m3).

Phase 1 migration: CrossEncoder (local CPU, sentence-transformers)
→ sync HTTP-клиент к TEI reranker service (WSL2 native, RTX 5060 Ti, :8083).

Используется:
  - src/services/tools/rerank.py  — через ToolRunner (ThreadPoolExecutor)
  - src/services/qa_service.py    — sync методы _fetch_context(), answer_v2()
  - src/services/agent_service.py — SPEC-RAG-07 через rerank_with_scores()

Намеренно sync (httpx.Client), не async — qa_service.py вызывает rerank() из
синхронных методов, которые могут находиться внутри async контекста FastAPI.
asyncio.run() в этом случае недоступен (RuntimeError: event loop already running).
"""

from __future__ import annotations

import logging
import math
from typing import List

import httpx

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Sync HTTP-сервис переранжирования на основе TEI reranker (bge-reranker-v2-m3).

    Принимает query + список текстов, возвращает индексы отсортированных документов.
    Внутренне вызывает TEI POST /rerank с raw_scores=True, восстанавливает порядок
    по index из ответа, сортирует по убыванию logit score.

    Создаётся через deps.get_reranker().
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """
        Args:
            base_url: URL TEI reranker service, например "http://host.docker.internal:8083"
            timeout:  HTTP таймаут в секундах (rerank ~20–80 мс на GPU, запас для деградации)
        """
        self.base_url = base_url.rstrip("/")
        # Connection pool: rerank вызывается последовательно, 2 keepalive соединений достаточно
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        logger.info("RerankerService инициализирован (TEI HTTP sync): %s", self.base_url)

    # ------------------------------------------------------------------
    # Публичный API (backward compatible с Phase 0 CrossEncoder)
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_n: int,
        batch_size: int = 16,
    ) -> List[int]:
        """
        Переранжирует документы по релевантности к запросу.

        Сигнатура совместима с Phase 0 CrossEncoder-версией.
        Параметр batch_size сохранён для совместимости с qa_service.py, игнорируется —
        TEI управляет батчингом внутренне.

        Args:
            query:      поисковый запрос на естественном языке
            docs:       список текстов документов (обычно 20–80 штук)
            top_n:      максимальное число возвращаемых индексов
            batch_size: игнорируется (оставлен для compat с qa_service.py)

        Returns:
            list[int] — индексы docs в порядке убывания релевантности, длиной <= top_n.
            При ошибке возвращает список [0, 1, 2, ...] (исходный порядок).

        Raises:
            Не поднимает исключения — при ошибке логирует и возвращает fallback.
        """
        if not docs:
            return []
        try:
            raw_scores = self._get_raw_scores(query, docs)
            order = sorted(range(len(docs)), key=lambda i: raw_scores[i], reverse=True)
            if top_n and top_n > 0:
                order = order[: min(top_n, len(order))]
            return order
        except Exception as exc:
            logger.error("Ошибка ререйкера: %s", exc)
            # Fallback: исходный порядок без переранжирования
            return list(range(min(len(docs), top_n or len(docs))))

    def rerank_with_scores(
        self,
        query: str,
        docs: List[str],
        top_n: int,
        batch_size: int = 16,
    ) -> tuple[List[int], List[float]]:
        """
        Переранжирует документы и возвращает нормализованные scores.

        Используется в SPEC-RAG-07 composite coverage metric (passage_relevance сигнал):
        top_score = scores[0] — sigmoid-нормализованный score лучшего документа ∈ [0, 1].

        TEI возвращает raw logit scores (обычно -10..10).
        Нормализация: sigmoid(x) = 1/(1+exp(-x)).
        Нормализация монотонна → порядок ранжирования не меняется.

        Args:
            query:      поисковый запрос
            docs:       список текстов документов
            top_n:      максимальное число возвращаемых пар
            batch_size: игнорируется

        Returns:
            tuple(indices, scores):
              - indices: list[int] индексы docs по убыванию релевантности
              - scores:  list[float] sigmoid(logit) ∈ [0, 1] для каждого индекса

        При ошибке возвращает (fallback_indices, []) — пустой список scores.
        """
        if not docs:
            return [], []
        try:
            raw_scores = self._get_raw_scores(query, docs)
            order = sorted(range(len(docs)), key=lambda i: raw_scores[i], reverse=True)
            if top_n and top_n > 0:
                order = order[: min(top_n, len(order))]
            norm_scores = [self._sigmoid(raw_scores[i]) for i in order]
            return order, norm_scores
        except Exception as exc:
            logger.error("Ошибка ререйкера (with_scores): %s", exc)
            return list(range(min(len(docs), top_n or len(docs)))), []

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _get_raw_scores(self, query: str, passages: List[str]) -> List[float]:
        """
        POST /rerank к TEI service.

        TEI возвращает [{"index": i, "score": f}, ...] отсортированно по score desc.
        Восстанавливаем порядок по index → scores[i] соответствует passages[i].

        Args:
            query:    поисковый запрос
            passages: список текстов для ранжирования

        Returns:
            list[float] длиной len(passages) — logit scores (не нормализованы)

        Raises:
            httpx.ConnectError:    TEI service недоступен
            httpx.HTTPStatusError: TEI вернул 4xx/5xx
        """
        response = self._client.post(
            "/rerank",
            json={
                "query": query,
                "texts": passages,
                "raw_scores": True,  # логит scores до sigmoid
                "truncate": True,    # автообрезка длинных текстов по max_length модели
            },
        )
        response.raise_for_status()

        # results: [{"index": int, "score": float}, ...] — отсортировано по score desc
        results: list[dict] = response.json()

        # Восстанавливаем в исходный порядок passages
        scores = [0.0] * len(passages)
        for item in results:
            scores[item["index"]] = float(item["score"])

        logger.debug(
            "TEI rerank: query=%r, %d passages → logit scores [%.3f..%.3f]",
            query[:50],
            len(passages),
            min(scores),
            max(scores),
        )
        return scores

    def healthcheck(self) -> bool:
        """
        Проверяет доступность TEI reranker service.

        Returns:
            True если GET /health отвечает 200, False в любом другом случае
        """
        try:
            response = self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("TEI reranker healthcheck failed: %s", exc)
            return False

    def close(self) -> None:
        """Закрывает HTTP connection pool. Вызывать при shutdown приложения."""
        self._client.close()

    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Sigmoid-нормализация logit-скора в [0, 1].

        Для экстремальных значений (|x| > 20) точность float достаточна,
        math.exp не переполняется (Python использует C double).
        """
        return 1.0 / (1.0 + math.exp(-x))
```

---

## 3. src/services/tools/rerank.py

Минимальное изменение: заменить dummy scores на реальные из `rerank_with_scores()`.

```python
"""
Инструмент rerank для переранжирования результатов поиска
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from services.reranker_service import RerankerService

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    docs: List[str],
    top_n: Optional[int] = None,
    reranker: Optional[RerankerService] = None,
    hits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Переранжирует документы по релевантности к запросу.

    Args:
        query:   поисковый запрос
        docs:    список текстов документов для ранжирования
        top_n:   максимальное число результатов (None = все)
        reranker: RerankerService (передаётся через ToolRunner)
        hits:    альтернативный источник docs (если docs пуст)

    Returns:
        {
            "indices": [2, 0, 1],         # индексы по убыванию релевантности
            "scores":  [0.95, 0.87, 0.72], # sigmoid-нормализованные scores ∈ [0,1]
            "top_n":   3
        }
    """
    if not reranker:
        return {"indices": [], "scores": [], "error": "RerankerService not provided"}

    if not query or not query.strip():
        return {"indices": [], "scores": [], "error": "Empty query"}

    # Если docs не передан, извлекаем из hits
    if not docs and hits:
        docs = [item.get("text", "") for item in hits if item]

    if not docs:
        return {"indices": [], "scores": [], "error": "No documents provided"}

    try:
        if top_n is None:
            top_n = len(docs)

        # rerank_with_scores → реальные sigmoid-нормализованные scores (Phase 1)
        # Phase 0 использовал dummy scores [1.0 - i*0.1]
        indices, scores = reranker.rerank_with_scores(
            query=query,
            docs=docs,
            top_n=top_n,
            batch_size=16,
        )

        return {"indices": indices, "scores": scores, "top_n": len(indices)}

    except Exception as exc:
        logger.error("Error in rerank tool: %s", exc)
        return {"indices": [], "scores": [], "error": str(exc)}
```

---

## 4. Интеграция — deps.py

**Контекст:** SPEC-RAG-04 определил `get_reranker()` как `RerankerService(reranker_client)`,
где `reranker_client = get_tei_reranker_client()`. SPEC-RAG-05 уточняет: `RerankerService`
использует sync httpx.Client и принимает `base_url` напрямую, не `TEIRerankerClient`.

Изменить только функцию `get_reranker()`. Остальные функции из SPEC-RAG-04 не трогать.

```python
# src/core/deps.py — изменить только get_reranker()
# Заменить блок из SPEC-RAG-04:
#
#   @lru_cache
#   def get_reranker() -> Optional[RerankerService]:
#       ...
#       reranker_client = get_tei_reranker_client()
#       return RerankerService(reranker_client)
#
# На:

@lru_cache
def get_reranker() -> Optional[RerankerService]:
    """
    Singleton RerankerService (TEI HTTP sync, bge-reranker-v2-m3).

    Использует sync httpx.Client → безопасен как из async FastAPI контекста,
    так и из ThreadPoolExecutor (ToolRunner).
    Возвращает None если enable_reranker=False.
    """
    settings = get_settings()
    if not settings.enable_reranker:
        logger.info("RerankerService отключён (enable_reranker=False)")
        return None
    logger.info("RerankerService: TEI reranker → %s", settings.reranker_tei_url)
    return RerankerService(base_url=settings.reranker_tei_url)
```

**Примечание по `get_tei_reranker_client()`:** функция остаётся в deps.py как определена
в SPEC-RAG-04 — async `TEIRerankerClient` не используется в Phase 1, но не удаляется
(может понадобиться в Phase 2 при переходе на async reranking).

### 4.1 Lifespan shutdown

В `src/api/main.py` в блоке `finally` lifespan-хендлера добавить:

```python
# В lifespan shutdown (после QdrantStore.aclose() и TEI clients):
reranker = get_reranker()
if reranker is not None:
    reranker.close()
    logger.info("RerankerService: sync httpx.Client закрыт")
```

**Важно:** Реальный путь — `src/main.py` (не `src/api/main.py`). Lifespan — инфраструктурный
код, его можно обновить. Если lifespan уже обновлён в SPEC-RAG-03/04, добавить только
`reranker.close()` в существующий блок.

### 4.2 settings.py — удалить reranker_model_key

Из `src/core/settings.py` удалить:

```python
# Удалить эти строки:
self.reranker_model_key: str = os.getenv(
    "RERANKER_MODEL_KEY", "BAAI/bge-reranker-v2-m3"
)
```

`reranker_tei_url` — уже присутствует после SPEC-RAG-01. `reranker_batch_size` и
`reranker_top_n` — оставить (используются в qa_service.py, `batch_size` передаётся
в `rerank()` как backward compat параметр).

---

## 5. Удаление src/utils/model_downloader.py

Файл содержит утилиты для скачивания HF-моделей локально:
- `download_embedding_model()` — заменено TEI (SPEC-RAG-02)
- `download_reranker_model()` — заменено TEI (SPEC-RAG-05)
- `download_llm_model_from_hf()` — llama-server запускается на Windows Host, не в Docker
- `auto_download_models()` — обёртка, тоже неактуальна

**Действие:** `git rm src/utils/model_downloader.py`

**Проверить зависимости перед удалением:**
```bash
grep -r "model_downloader" src/ scripts/ --include="*.py"
```
Все найденные импорты удалить.

---

## 6. Тесты

Файл: `src/tests/test_reranker_service.py`

Тесты используют `httpx.MockTransport` для изоляции от реального TEI service.
TEI `/rerank` возвращает `[{"index": i, "score": f}, ...]` отсортировано по score desc.

```python
"""
Unit-тесты для RerankerService (Phase 1: TEI HTTP sync).
"""

from __future__ import annotations

import math
import json
import pytest
import httpx

from services.reranker_service import RerankerService


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_rerank_response(scores_by_index: dict[int, float]) -> bytes:
    """Формирует TEI /rerank ответ: [{index, score}, ...] sorted by score desc."""
    items = [{"index": i, "score": s} for i, s in scores_by_index.items()]
    items.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(items).encode()


def _make_transport(rerank_body: bytes, status_code: int = 200) -> httpx.MockTransport:
    """httpx.MockTransport, отвечающий фиксированным телом на /rerank."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b"OK")
        if request.url.path == "/rerank":
            return httpx.Response(status_code, content=rerank_body)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _make_service(rerank_body: bytes, status_code: int = 200) -> RerankerService:
    """Создаёт RerankerService с подменённым транспортом."""
    svc = RerankerService(base_url="http://mock-reranker")
    svc._client = httpx.Client(
        base_url="http://mock-reranker",
        transport=_make_transport(rerank_body, status_code),
    )
    return svc


# ------------------------------------------------------------------
# Тесты rerank()
# ------------------------------------------------------------------

class TestRerank:
    def test_rerank_returns_sorted_indices(self):
        """rerank() сортирует индексы по убыванию relevance score."""
        # docs[0] score=1.0, docs[1] score=5.0, docs[2] score=3.0
        # Ожидаем: [1, 2, 0]
        body = _make_rerank_response({0: 1.0, 1: 5.0, 2: 3.0})
        svc = _make_service(body)

        indices = svc.rerank("query", ["doc0", "doc1", "doc2"], top_n=3)

        assert indices == [1, 2, 0]

    def test_rerank_respects_top_n(self):
        """rerank() усекает результат до top_n."""
        body = _make_rerank_response({0: 1.0, 1: 5.0, 2: 3.0})
        svc = _make_service(body)

        indices = svc.rerank("query", ["doc0", "doc1", "doc2"], top_n=2)

        assert len(indices) == 2
        assert indices[0] == 1  # лучший

    def test_rerank_empty_docs_returns_empty(self):
        """rerank() с пустым списком → []."""
        svc = _make_service(b"[]")
        result = svc.rerank("query", [], top_n=5)
        assert result == []

    def test_rerank_fallback_on_connection_error(self):
        """При ConnectError rerank() возвращает исходный порядок, не поднимает исключение."""

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        svc = RerankerService(base_url="http://mock-reranker")
        svc._client = httpx.Client(
            base_url="http://mock-reranker",
            transport=httpx.MockTransport(fail_handler),
        )

        docs = ["doc0", "doc1", "doc2"]
        result = svc.rerank("query", docs, top_n=3)

        assert result == [0, 1, 2]  # исходный порядок

    def test_rerank_ignores_batch_size(self):
        """batch_size не влияет на результат (compat параметр)."""
        body = _make_rerank_response({0: 2.0, 1: 1.0})
        svc = _make_service(body)

        r1 = svc.rerank("q", ["a", "b"], top_n=2, batch_size=1)
        r2 = svc.rerank("q", ["a", "b"], top_n=2, batch_size=64)

        assert r1 == r2 == [0, 1]


# ------------------------------------------------------------------
# Тесты rerank_with_scores()
# ------------------------------------------------------------------

class TestRerankWithScores:
    def test_returns_correct_indices_and_scores(self):
        """rerank_with_scores() возвращает (indices, sigmoid_scores) по убыванию."""
        body = _make_rerank_response({0: 0.0, 1: 2.0, 2: -1.0})
        svc = _make_service(body)

        indices, scores = svc.rerank_with_scores(
            "query", ["doc0", "doc1", "doc2"], top_n=3
        )

        # Порядок: doc1 (2.0) > doc0 (0.0) > doc2 (-1.0)
        assert indices == [1, 0, 2]
        # Проверяем sigmoid нормализацию
        assert len(scores) == 3
        assert 0.0 < scores[0] < 1.0  # sigmoid(2.0) ≈ 0.88
        assert scores[0] > scores[1] > scores[2]  # монотонность

    def test_sigmoid_values_correct(self):
        """Проверяем конкретные значения sigmoid."""
        body = _make_rerank_response({0: 0.0})
        svc = _make_service(body)

        _, scores = svc.rerank_with_scores("query", ["doc"], top_n=1)

        assert len(scores) == 1
        assert abs(scores[0] - 0.5) < 1e-6  # sigmoid(0) = 0.5

    def test_returns_empty_on_connection_error(self):
        """При ошибке rerank_with_scores() возвращает fallback_indices + [] scores."""

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        svc = RerankerService(base_url="http://mock-reranker")
        svc._client = httpx.Client(
            base_url="http://mock-reranker",
            transport=httpx.MockTransport(fail_handler),
        )

        indices, scores = svc.rerank_with_scores("query", ["a", "b", "c"], top_n=3)

        assert indices == [0, 1, 2]  # fallback
        assert scores == []


# ------------------------------------------------------------------
# Тест healthcheck
# ------------------------------------------------------------------

class TestHealthcheck:
    def test_healthcheck_true_on_200(self):
        """healthcheck() возвращает True при ответе 200."""
        body = _make_rerank_response({})
        svc = _make_service(body)
        assert svc.healthcheck() is True

    def test_healthcheck_false_on_error(self):
        """healthcheck() возвращает False при ConnectError."""

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        svc = RerankerService(base_url="http://mock-reranker")
        svc._client = httpx.Client(
            base_url="http://mock-reranker",
            transport=httpx.MockTransport(fail_handler),
        )
        assert svc.healthcheck() is False


# ------------------------------------------------------------------
# Тест _sigmoid
# ------------------------------------------------------------------

class TestSigmoid:
    def test_sigmoid_zero(self):
        assert abs(RerankerService._sigmoid(0.0) - 0.5) < 1e-9

    def test_sigmoid_large_positive(self):
        assert RerankerService._sigmoid(10.0) > 0.99

    def test_sigmoid_large_negative(self):
        assert RerankerService._sigmoid(-10.0) < 0.01

    def test_sigmoid_monotone(self):
        values = [-5.0, -1.0, 0.0, 1.0, 5.0]
        results = [RerankerService._sigmoid(v) for v in values]
        assert results == sorted(results)
```

---

## 7. Чеклист реализации

- [ ] `src/services/reranker_service.py` переписан: CrossEncoder удалён, `httpx.Client` добавлен
- [ ] `RerankerService.__init__(base_url, timeout)` — конструктор принимает URL
- [ ] `rerank(query, docs, top_n, batch_size)` → `List[int]` — backward compatible
- [ ] `rerank_with_scores(query, docs, top_n, batch_size)` → `tuple[List[int], List[float]]` — новый метод
- [ ] `_get_raw_scores()` — POST /rerank, восстановление order по index из TEI ответа
- [ ] `_sigmoid()` — статический метод, нормализация logit → [0, 1]
- [ ] `healthcheck()` — sync GET /health
- [ ] `close()` — закрытие httpx.Client
- [ ] `src/services/tools/rerank.py` обновлён: `rerank_with_scores()` вместо dummy scores
- [ ] `src/core/deps.py` — `get_reranker()` обновлён: `RerankerService(base_url=settings.reranker_tei_url)`
- [ ] `src/core/settings.py` — поле `reranker_model_key` удалено
- [ ] `src/api/main.py` — `reranker.close()` добавлен в lifespan shutdown
- [ ] `src/utils/model_downloader.py` — файл удалён (`git rm`)
- [ ] Импорты `model_downloader` во всём репозитории удалены
- [ ] `sentence_transformers` удалён из зависимостей (если не используется нигде ещё)
- [ ] `pytest src/tests/test_reranker_service.py` — все тесты проходят (no network)
