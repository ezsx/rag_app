### Модуль: `src/api/v1/endpoints/search.py`

Назначение: эндпоинты для построения плана и выполнения семантического поиска с RRF/MMR и опциональным ререйком/гибридом.

#### Эндпоинты
- `POST /v1/search/plan` → `SearchPlan` (LLM‑планировщик).
- `POST /v1/search` → `SearchResponse`.
  - Если `ENABLE_QUERY_PLANNER`: строит план, dense или hybrid (если доступен), RRF‑слияние; опционально MMR и CrossEncoder rerank; вернёт ограниченный Top‑N.
  - Иначе — прямой dense‑поиск.

#### Кеш
- Redis по ключу `search:<hash>` при `settings.redis_enabled`.

#### Зависимости
- `get_retriever`, `get_query_planner`, `get_reranker`, `get_hybrid_retriever`, `get_settings`, `get_redis_client`.





