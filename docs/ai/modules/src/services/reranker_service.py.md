### Модуль: `src/services/reranker_service.py`

Назначение: Синхронная обёртка (sync bridge) над async `TEIRerankerClient` для TEI reranker (BAAI/bge-reranker-v2-m3).

Phase 1: CrossEncoder удалён, `httpx.Client` заменён на bridge через выделенный event loop
в фоновом потоке (`asyncio.run_coroutine_threadsafe`). Безопасен из sync и async контекстов.

#### Класс `RerankerService`
- `__init__(reranker_client: TEIRerankerClient)` — принимает общий async клиент из DI.
- `rerank(query, docs, top_n, batch_size)` → `List[int]` — индексы по убыванию релевантности. `batch_size` игнорируется (compat).
- `rerank_with_scores(query, docs, top_n, batch_size)` → `(List[int], List[float])` — индексы + sigmoid-нормализованные scores ∈ [0,1].
- `healthcheck()` → `bool` — проксирует TEI `/health`.
- `close()` — останавливает фоновый event loop (TEIRerankerClient не закрывает — его lifecycle в main.py).
- `_sigmoid(x)` — двухветочная sigmoid для numerical stability.

#### Зависимости
- `adapters.tei.reranker_client.TEIRerankerClient` — общий async клиент.
- Создаётся через `deps.get_reranker()` → `RerankerService(get_tei_reranker_client())`.
