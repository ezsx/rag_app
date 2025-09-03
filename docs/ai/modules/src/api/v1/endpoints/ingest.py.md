### Модуль: `src/api/v1/endpoints/ingest.py`

Назначение: запуск и управление задачами Telegram ingestion.

#### Эндпоинты
- `POST /v1/ingest/telegram` → создаёт ingestion‑задачу через `services.ingest_service.job_manager.create_job`.
- `GET /v1/ingest/{job_id}` → статус/прогресс задачи.
- `DELETE /v1/ingest/{job_id}` → отмена выполняющейся задачи.
- `GET /v1/ingest` → список задач (сортированы по времени).

#### Валидация
- Даты `since/until` валидируются через `dateutil.parser.isoparse`; проверяется порядок дат.





