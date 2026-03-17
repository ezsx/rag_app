## scripts/ingest_telegram.py — Telegram → Qdrant инжестор

### Назначение
- CLI-инжестор сообщений Telegram-каналов в Qdrant.
- Dense-векторы получает через `TEIEmbeddingClient.embed_documents()`.
- Sparse-векторы строит через `fastembed.SparseTextEmbedding.embed()`.
- Загружает точки в Qdrant через `QdrantStore.upsert()`.

### Запуск
```bash
docker compose --profile ingest run --rm ingest \
  --channel @some_channel \
  --since 2024-06-01 \
  --until 2024-07-01
```

`--collection` опционален. Если не указан, используется `QDRANT_COLLECTION` или значение из `settings`.

### Переменные окружения
- `TG_API_ID`, `TG_API_HASH` — обязательны
- `TG_PHONE` или `BOT_TOKEN` — для авторизации
- `TG_SESSION` — путь к файлу сессии Telethon
- `QDRANT_URL` — адрес Qdrant
- `QDRANT_COLLECTION` — имя коллекции по умолчанию
- `EMBEDDING_TEI_URL` — адрес TEI embedding service

### CLI параметры
- `--channel`/`--channels`: один канал (`@username` или id) или CSV-список
- `--since`/`--until`: ISO даты (включительно/исключительно)
- `--collection`: имя Qdrant-коллекции
- `--batch-size`: сколько сообщений отдавать в один TEI embed запрос
- `--max-messages`: ограничение по числу последних сообщений
- `--chunk-size`: разбиение длинных сообщений на чанки N символов
- `--log-level`: уровень логирования

### Ключевые функции
- `create_telegram_client()` — авторизация и создание клиента Telethon.
- `gather_messages()` и `_gather_with_retries()` — сбор сообщений по диапазону дат с ретраями.
- `_build_point_docs_flat()` — маппинг `Message[] + dense + sparse` в `PointDocument`.
- `ingest_batches()` — TEI dense embedding, fastembed sparse encoding и `QdrantStore.upsert()`.
- `main()` — инициализация `TEIEmbeddingClient`, `SparseTextEmbedding`, `QdrantStore`, `ensure_collection()` и цикл по каналам.

### Инварианты данных
- Point ID:
  - без chunking: `"{channel_name}:{message_id}"`
  - с chunking: `"{channel_name}:{message_id}:{chunk_idx}"`
- `channel_name`:
  - `channel_hint.lstrip("@")`, если канал задан как `@username`
  - иначе `str(chat_id)`
- Обязательный payload:
  - `text`
  - `channel`
  - `channel_id`
  - `message_id`
  - `date`

### Что удалено относительно Phase 0
- ChromaDB
- BM25IndexManager
- локальный `SentenceTransformer`
- GPU/MPS autodetect для локального embedding
- автоскачивание embedding-моделей внутри ingest-контейнера
