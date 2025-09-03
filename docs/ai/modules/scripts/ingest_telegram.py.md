## scripts/ingest_telegram.py — Telegram → Chroma/BM25 инжестор

### Назначение
- **CLI‑инжестор** сообщений Telegram каналов в ChromaDB с параллельным добавлением документов в BM25 индекс.
- Может работать автономно (как отдельный загрузчик коллекций) и как часть проекта (функции импортируемы из `scripts/ingest_telegram.py`).

### Запуск
```bash
python -m scripts.ingest_telegram \
  --channel @some_channel \
  --since 2024-06-01 \
  --until 2024-07-01 \
  --collection tg_some_channel
```

Поддерживается множественный ввод каналов через `--channels` (CSV).

### Переменные окружения (.env)
- `TG_API_ID`, `TG_API_HASH` — обязательны
- `TG_PHONE` или `BOT_TOKEN` — для авторизации
- `TG_SESSION` — путь к файлу сессии Telethon (по умолчанию `telegram.session` в корне)
- `CHROMA_HOST`, `CHROMA_PORT` — адрес Chroma HTTP
- `EMBEDDING_MODEL_KEY` или `EMBEDDING_MODEL` — выбор embedding‑модели
- `HUGGINGFACE_API_KEY`/`CHROMA_HUGGINGFACE_API_KEY` — ключ для новых версий chromadb/эмбеддингов

### CLI параметры
- `--channel`/`--channels`: один канал (`@username` или id) или CSV‑список
- `--since`/`--until`: ISO даты (включительно/исключительно)
- `--collection`: имя коллекции Chroma
- `--batch-size`: размер батча (по умолчанию авто‑детект по устройству)
- `--embed-model-key`/`--embed-model`: выбор модели эмбеддинга
- `--device`: `auto|cpu|cuda|mps`
- `--max-messages`: ограничение по числу последних сообщений (debug)
- `--chunk-size`: разбиение длинных сообщений на чанки N символов (0 = без разбиения)
- `--log-level`: уровень логирования
- `--gpu-batch-multiplier`: множитель батча на GPU

### Основные функции и точки интеграции
- `create_telegram_client()` — авторизация/создание клиента Telethon.
- `detect_optimal_device()` и `get_optimal_batch_size()` — подбор устройства и размера батча.
- `resolve_embedding_model()` — выбор модели эмбеддинга из конфигурации проекта или явного имени.
- `FastEmbeddingFunction` — ускоренная функция эмбеддинга на базе `SentenceTransformer` (GPU при наличии).
- `create_chroma_collection(name, embed_model, device)` — подключение к Chroma и получение/создание коллекции с прикреплённой embedding‑функцией.
- `gather_messages(...)` и `_gather_with_retries(...)` — сбор сообщений по диапазону дат с устойчивостью к `FloodWait`/сетевым ошибкам.
- `ingest_batches(collection_name, collection, messages, batch_size, ...)` — батчовая запись:
  - В Chroma: `upsert`/`add` документов, метаданных и детерминированных id (`chat_id:msg_id:chunk`).
  - В BM25: формирование `BM25Doc` и добавление в индекс через `BM25IndexManager` (если доступен).
- `main()` — парсинг аргументов, автоскачивание модели при необходимости, цикл по каналам, прогресс/оценка времени, финальные логи.

### Поведение и устойчивость
- Ретраи сбора сообщений с бэкоффом и обработкой `FloodWaitError`.
- Оценка времени обработки в зависимости от устройства (`cuda`/`cpu`).
- Детерминированные идентификаторы и расширенные метаданные (`channel_id`, `msg_id`, `date`, `reply_to`, `views`, `channel_username`).
- Совместимость с изменениями API Chroma (проверка наличия `HttpClient`, параметров `SentenceTransformerEmbeddingFunction`).

### Использование внутри проекта
- Скрипт расположён вне `src`, но экспортируемые функции можно импортировать в сервисы пайплайна (напр., для повторного использования `create_chroma_collection` или `ingest_batches`).
- Путь добавляется динамически при необходимости (`sys.path`), либо используйте абсолютные импорты при вызове как модуля.

### Связь с пайплайном
- См. раздел Ingest в `docs/ai/pipelines.md` — этот скрипт реализует шаги: сбор Telegram → эмбеддинг → запись в Chroma + BM25.


