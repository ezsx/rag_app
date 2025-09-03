### Модуль: `src/adapters/search/bm25_index.py`

Назначение: управление офлайновым BM25 индексом на базе `python-tantivy`. Создание/открытие индекса, добавление документов, перезагрузка searcher, поиск.

#### Основные структуры
- `BM25Doc` — документ индекса (идентификаторы, текст, канал, даты, просмотры, reply_to, msg_id).
- `BM25Query` — must/should термы + фильтры (channel_usernames, channel_ids, reply_to, min_views, date_days диапазоны).
- `BM25Hit` — результат поиска: `doc_id`, `text`, `metadata`, `bm25_score`.
- `IndexHandle` — дескриптор индекса с `schema/index/writer/reader/searcher/paths`.

#### Класс `BM25IndexManager`
- `get_or_create(collection, for_write=False)` — создаёт/открывает индекс в `bm25-index/<collection>`; совместим с разными версиями API tantivy.
- `reload(collection)` — контроли́руемый по времени перезапуск `reader/searcher`.
- `add_documents(collection, docs, commit_every=1000)` — потоковая запись с промежуточными commit.
- `search(collection, query, top_k)` — сбор строкового запроса по полям/фильтрам, парсинг и поиск; преобразование документа в `BM25Hit`.

#### Совместимость
- Обёртки `_add_u64/_add_i64` и альтернативные конструкторы полей для разных версий `python-tantivy`.





