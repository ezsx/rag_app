### Модуль: `src/adapters/search/bm25_retriever.py`

Назначение: обёртка над `BM25IndexManager` для выполнения BM25‑поиска с учётом плана `SearchPlan`.

#### Класс `BM25Retriever`
- `search(query_text, plan, k)` — собирает must/should термы и фильтры из `SearchPlan` (включая преобразование дат в ordinal days) и вызывает `BM25IndexManager.search` по коллекции из `Settings`.

#### Выходной формат
- Список `schemas.search.Candidate` с полями: `id`, `text`, `metadata`, `bm25_score`, `source='bm25'`.





