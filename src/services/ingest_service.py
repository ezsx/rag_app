"""
Сервис для управления задачами Telegram ingestion
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from schemas.qa import IngestJobStatus, IngestJobStatusResponse, TelegramIngestRequest

logger = logging.getLogger(__name__)


@dataclass
class IngestJob:
    """Класс для хранения информации о задаче ingestion"""

    job_id: str
    request: TelegramIngestRequest
    status: IngestJobStatus = IngestJobStatus.QUEUED
    progress: float = 0.0
    messages_processed: int = 0
    total_messages: int | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    log_messages: list[str] = field(default_factory=list)
    task: asyncio.Task | None = None

    def add_log(self, message: str):
        """Добавляет сообщение в лог задачи"""
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        # Оставляем только последние 50 сообщений
        if len(self.log_messages) > 50:
            self.log_messages = self.log_messages[-50:]
        logger.info("Job %s: %s", self.job_id, message)

    def to_response(self) -> IngestJobStatusResponse:
        """Конвертирует в response schema"""
        return IngestJobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            messages_processed=self.messages_processed,
            total_messages=self.total_messages,
            error_message=self.error_message,
            started_at=self.started_at,
            completed_at=self.completed_at,
            log_messages=self.log_messages.copy(),
        )


class IngestJobManager:
    """Менеджер для управления задачами ingestion"""

    def __init__(self):
        self.jobs: dict[str, IngestJob] = {}
        self.max_concurrent_jobs = 2  # Максимум параллельных задач

    def create_job(self, request: TelegramIngestRequest) -> str:
        """Создает новую задачу ingestion"""
        job_id = str(uuid.uuid4())
        job = IngestJob(job_id=job_id, request=request)
        # Список каналов для логов
        channels: list[str] = []
        if request.channel:
            channels.append(request.channel)
        if request.channels:
            channels.extend([c for c in request.channels if c])
        seen: set[str] = set()
        channels = [c for c in channels if c not in seen and not seen.add(c)]  # type: ignore[func-returns-value]
        if channels:
            job.add_log(f"Задача создана для каналов: {channels}")
        else:
            job.add_log("Задача создана: каналы не указаны")

        self.jobs[job_id] = job

        # Запускаем задачу если есть свободные слоты
        if self._get_running_jobs_count() < self.max_concurrent_jobs:
            self._start_job(job)
        else:
            job.add_log("Задача добавлена в очередь ожидания")

        return job_id

    def get_job(self, job_id: str) -> IngestJob | None:
        """Получает задачу по ID"""
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Отменяет задачу"""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in [
            IngestJobStatus.COMPLETED,
            IngestJobStatus.FAILED,
            IngestJobStatus.CANCELLED,
        ]:
            return False

        job.status = IngestJobStatus.CANCELLED
        job.add_log("Задача отменена пользователем")

        if job.task and not job.task.done():
            job.task.cancel()

        return True

    def _get_running_jobs_count(self) -> int:
        """Подсчитывает количество выполняющихся задач"""
        return sum(
            1 for job in self.jobs.values() if job.status == IngestJobStatus.RUNNING
        )

    def _start_job(self, job: IngestJob):
        """Запускает выполнение задачи"""
        job.status = IngestJobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        job.add_log("Задача запущена")

        # Создаем асинхронную задачу
        job.task = asyncio.create_task(self._run_ingestion(job))

    async def _run_ingestion(self, job: IngestJob):
        """Run ingestion job.

        LEGACY: this code path is broken since the ChromaDB → Qdrant migration.
        Real ingest goes through: docker compose run --rm ingest
        """
        raise NotImplementedError(
            "API-based ingest is disabled.  "
            "Use: docker compose -f deploy/compose/compose.dev.yml run --rm ingest "
            "--channel @name --since YYYY-MM-DD --until YYYY-MM-DD"
        )

    def _check_queue(self):
        """Проверяет очередь и запускает следующую задачу если возможно"""
        if self._get_running_jobs_count() >= self.max_concurrent_jobs:
            return

        # Ищем задачи в очереди
        for job in self.jobs.values():
            if job.status == IngestJobStatus.QUEUED:
                job.add_log("Задача извлечена из очереди")
                self._start_job(job)
                break


# Глобальный экземпляр менеджера задач
job_manager = IngestJobManager()
