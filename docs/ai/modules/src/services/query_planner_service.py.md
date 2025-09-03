### Модуль: `src/services/query_planner_service.py`

Назначение: построение плана поиска (под‑запросы, фильтры, k_per_query, стратегия слияния) с использованием LLM и последующее кеширование планов/результатов RRF‑слияния.

#### Ключевые элементы
- `_TTLCache` — простой in‑memory TTL‑кеш.
- `QueryPlannerService`:
  - `make_plan(query)` — пытается получить структурированный JSON через chat‑completion с `response_format=json_schema`, при неудаче — free‑completion с пост‑парсингом; затем нормализация и валидация полей, ограничение числа под‑запросов.
  - `_build_prompt(query)` — системные правила на русском + пример целевого JSON.
  - `get_cached_fusion/set_cached_fusion` — доступ к кешу результатов слияния (для повторного использования в QA/поиске).

#### Выход
- `schemas.search.SearchPlan` (normalized_queries, must/should, MetadataFilters, k_per_query, fusion).





