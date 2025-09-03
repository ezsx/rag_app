### Модуль: `src/adapters/search/hybrid_retriever.py`

Назначение: гибридный ретривер, объединяющий BM25 и dense‑поиск с двухступенчатым RRF‑слиянием и дедупликацией.

#### Класс `HybridRetriever`
- `search_with_plan(query_text, plan)`:
  - BM25: `bm25.search` → кандидаты.
  - Dense: для каждого под‑запроса `normalized_queries` вызывает `dense.search`, сливает списки через `rrf_merge` (формат: `(doc, distance, meta)`).
  - Слияние BM25 + Dense: повторный `rrf_merge` по подготовленным тройкам.
  - Дедуп по устойчивому `id` (channel_id:msg_id | hash).
  - Возвращает список `Candidate` с `source='hybrid'` или `source='dense'`.





