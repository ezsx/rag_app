### Модуль: `src/adapters/chroma/retriever.py`

Назначение: dense‑retriever поверх ChromaDB. Инкапсулирует подключение к коллекции, вычисление эмбеддингов (SentenceTransformer), поиск и выдачу результатов в унифицированном формате.

#### Классы и функции
- `Retriever`:
  - `__init__(client, collection_name, embedding_model)` — коннект к коллекции (или создание), инициализация `SentenceTransformerEmbeddingFunction`, локальный энкодер для `embed_texts`.
  - `get_context(query, k)` — возврат только документов.
  - `get_context_with_metadata(query, k)` — документы + метаданные + дистанции.
  - `search(query, k, filters)` — поиск с поддержкой Chroma `where` (часть фильтров пост‑обрабатывается в Python: `date_from/date_to`). Возвращает список items `{id,text,metadata,distance}` с устойчивым `id` (channel_id:msg_id, либо hash).
  - `embed_texts(texts)` — эмбеддинг через локальный `SentenceTransformer` (np.ndarray).
  - `_build_where(filters)` — трансляция бизнес‑фильтров в Chroma where (`$and/$or`, `$eq/$gte/$lte`).

#### Особенности
- Для старых версий Chroma предусмотрен fallback без `where` + лог‑предупреждение.
- E5‑префикс `query: ` добавляется перед эмбеддингом запроса.
- Пост‑фильтрация по датам (ISO YYYY‑MM‑DD) выполняется на Python‑стороне.





