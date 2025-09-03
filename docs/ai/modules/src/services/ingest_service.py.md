## src/services/ingest_service.py — менеджер задач инжеста Telegram

### Назначение
- Управление асинхронными задачами инжеста Telegram: постановка, запуск, отмена, мониторинг прогресса.
- Инкапсулирует логику работы с `scripts/ingest_telegram.py` и агрегирует метрики выполнения.

### Ключевые сущности
- `IngestJob` (dataclass): состояние задачи (status, progress, counts, timestamps, logs).
- `IngestJobManager`:
  - `create_job(request)` → `job_id`
  - `get_job(job_id)` → состояние
  - `cancel_job(job_id)` → bool
  - внутренние: очередь, параллелизм (max 2), запуск `_run_ingestion`.

### Взаимодействия
- Импортирует функции из `scripts/ingest_telegram.py`:
  - `create_telegram_client`, `create_chroma_collection`, `_gather_with_retries`, `ingest_batches`, и утилиты (device/batch/date).
- Освобождает VRAM LLM на время инжеста через `core.deps.release_llm_vram_temporarily()`.

### Поток выполнения (_упрощённо_)
1. Собирает список каналов из `TelegramIngestRequest`.
2. Определяет `device`, `embed_model`, `batch_size`, парсит даты.
3. Создаёт Telegram client, подключается к Chroma, для каждого канала:
   - собирает сообщения (с ретраями) → оценивает время → пишет батчами в Chroma + BM25.
4. Обновляет прогресс/логи, выставляет финальный статус и счётчики.

### Зависимости
- `schemas.qa` (IngestJobStatus, TelegramIngestRequest, IngestJobStatusResponse)
- `core.deps.release_llm_vram_temporarily`
- `scripts/ingest_telegram` (основная ingest‑логика)

### Примечания
- Логи задачи хранятся в кольцевом буфере (последние 50 сообщений).
- Параллелизм ограничен до 2 задач; очередь свободных слотов обрабатывается автоматически.


