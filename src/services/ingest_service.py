"""
Сервис для управления задачами Telegram ingestion
"""

import asyncio
import logging
import uuid
import traceback
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from schemas.qa import IngestJobStatus, TelegramIngestRequest, IngestJobStatusResponse

logger = logging.getLogger(__name__)


@dataclass
class IngestJob:
    """Класс для хранения информации о задаче ingestion"""

    job_id: str
    request: TelegramIngestRequest
    status: IngestJobStatus = IngestJobStatus.QUEUED
    progress: float = 0.0
    messages_processed: int = 0
    total_messages: Optional[int] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    log_messages: List[str] = field(default_factory=list)
    task: Optional[asyncio.Task] = None

    def add_log(self, message: str):
        """Добавляет сообщение в лог задачи"""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        # Оставляем только последние 50 сообщений
        if len(self.log_messages) > 50:
            self.log_messages = self.log_messages[-50:]
        logger.info(f"Job {self.job_id}: {message}")

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
        self.jobs: Dict[str, IngestJob] = {}
        self.max_concurrent_jobs = 2  # Максимум параллельных задач

    def create_job(self, request: TelegramIngestRequest) -> str:
        """Создает новую задачу ingestion"""
        job_id = str(uuid.uuid4())
        job = IngestJob(job_id=job_id, request=request)
        job.add_log(f"Задача создана для канала {request.channel}")

        self.jobs[job_id] = job

        # Запускаем задачу если есть свободные слоты
        if self._get_running_jobs_count() < self.max_concurrent_jobs:
            self._start_job(job)
        else:
            job.add_log("Задача добавлена в очередь ожидания")

        return job_id

    def get_job(self, job_id: str) -> Optional[IngestJob]:
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
        job.started_at = datetime.now(timezone.utc)
        job.add_log("Задача запущена")

        # Создаем асинхронную задачу
        job.task = asyncio.create_task(self._run_ingestion(job))

    async def _run_ingestion(self, job: IngestJob):
        """Основная функция выполнения ingestion"""
        try:
            from scripts.ingest_telegram import (
                create_telegram_client,
                create_chroma_collection,
                gather_messages,
                ingest_batches,
                _to_utc_naive,
                detect_optimal_device,
                get_optimal_batch_size,
                resolve_embedding_model,
                estimate_processing_time,
            )
            from dateutil import parser as date_parser

            request = job.request

            # Определяем устройство
            if request.device == "auto":
                device = detect_optimal_device()
            else:
                device = request.device
            job.add_log(f"Используем устройство: {device}")

            # Определяем модель embedding
            embed_model = resolve_embedding_model(None, None)
            job.add_log(f"Embedding модель: {embed_model}")

            # Определяем batch size
            batch_size = get_optimal_batch_size(device)
            job.add_log(f"Размер batch: {batch_size}")

            # Парсим даты
            start_date = _to_utc_naive(date_parser.isoparse(request.since))
            end_date = _to_utc_naive(date_parser.isoparse(request.until))
            job.add_log(f"Период: {start_date.date()} - {end_date.date()}")

            # Подключаемся к Telegram
            job.add_log("Подключение к Telegram...")
            client = await create_telegram_client()

            try:
                # Получаем сообщения
                job.add_log(f"Получение сообщений из канала {request.channel}...")
                messages = await gather_messages(
                    client, request.channel, start_date, end_date, request.max_messages
                )

                if not messages:
                    job.add_log("Сообщения не найдены")
                    job.status = IngestJobStatus.COMPLETED
                    job.completed_at = datetime.now(timezone.utc)
                    return

                job.total_messages = len(messages)
                job.add_log(f"Получено {len(messages)} сообщений")

                # Оценка времени
                estimated_time = estimate_processing_time(
                    len(messages), batch_size, device
                )
                job.add_log(f"Примерное время обработки: {estimated_time}")

                # Подключаемся к ChromaDB
                job.add_log("Подключение к ChromaDB...")
                collection = create_chroma_collection(
                    request.collection, embed_model, device
                )

                # Запускаем обработку с прогрессом
                job.add_log("Начинаем обработку сообщений...")
                await self._ingest_with_progress(job, collection, messages, batch_size)

                # Завершаем успешно
                final_count = collection.count()
                job.add_log(
                    f"Обработка завершена. Всего в коллекции: {final_count} документов"
                )
                job.status = IngestJobStatus.COMPLETED
                job.progress = 1.0
                job.completed_at = datetime.now(timezone.utc)

            finally:
                await client.disconnect()

        except asyncio.CancelledError:
            job.add_log("Задача была отменена")
            job.status = IngestJobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
        except Exception as e:
            error_msg = f"Ошибка выполнения: {str(e)}"
            job.add_log(error_msg)
            job.add_log(f"Детали ошибки:\n{traceback.format_exc()}")
            job.error_message = error_msg
            job.status = IngestJobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            logger.error(f"Job {job.job_id} failed: {e}")

        finally:
            # Проверяем очередь и запускаем следующую задачу
            self._check_queue()

    async def _ingest_with_progress(
        self, job: IngestJob, collection, messages, batch_size
    ):
        """Выполняет ingestion с обновлением прогресса"""
        total_messages = len(messages)
        processed = 0

        # Разбиваем на батчи
        for i in range(0, total_messages, batch_size):
            if job.status == IngestJobStatus.CANCELLED:
                break

            batch = messages[i : i + batch_size]

            try:
                # Подготавливаем данные для batch
                docs = [m.message for m in batch]
                ids = [f"{m.id}_{uuid.uuid4().hex[:6]}" for m in batch]
                metas = []
                for m in batch:
                    meta = {
                        "channel_id": m.chat_id,
                        "msg_id": m.id,
                        "date": m.date.isoformat(),
                    }
                    if m.reply_to_msg_id is not None:
                        meta["reply_to"] = m.reply_to_msg_id
                    views = getattr(m, "views", None)
                    if views is not None:
                        meta["views"] = views
                    metas.append(meta)

                # Добавляем в коллекцию
                collection.add(documents=docs, metadatas=metas, ids=ids)

                processed += len(batch)
                job.messages_processed = processed
                job.progress = processed / total_messages

                # Добавляем лог каждые 10 батчей
                if (i // batch_size + 1) % 10 == 0:
                    job.add_log(
                        f"Обработано {processed}/{total_messages} сообщений ({job.progress:.1%})"
                    )

            except Exception as e:
                job.add_log(f"Ошибка обработки batch {i//batch_size + 1}: {e}")
                # Продолжаем обработку следующих батчей

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
