### Модуль `src/adapters/chroma/retriever.py`

Назначение: поиск релевантных документов в ChromaDB и вспомогательные операции.

Ключевые методы:
- `search(query, k, filters?) -> List[Item]` — E5‑префикс, include метаданных, пост‑фильтры по датам (ISO) на Python‑стороне
- `get_context(query, k) -> List[str]`, `get_context_with_metadata(query, k) -> List[{document, metadata, distance}]`
- `get_by_ids(ids) -> List[Dict]` — best‑effort по парам (channel_id, msg_id)
- `embed_texts(texts) -> List[np.ndarray]` — CPU‑векторизация через SentenceTransformer

Примечания:
- Инициализация коллекции с `SentenceTransformerEmbeddingFunction`
- Косинусная метрика (`metadata: hnsw:space=cosine`)





