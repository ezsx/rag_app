### Модуль: `src/services/reranker_service.py`

Назначение: CPU‑ререйкер на базе `sentence_transformers.CrossEncoder` (BAAI/bge‑reranker‑v2‑m3).

#### Класс `RerankerService`
- `__init__(model_name)` — загрузка CrossEncoder на CPU, модель берётся из `Settings` по умолчанию.
- `rerank(query, docs, top_n, batch_size)` — возвращает список индексов документов, отсортированных по убыванию релевантности.





