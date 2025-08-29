"""
Ingest endpoints для управления задачами Telegram ingestion
"""

import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from services.ingest_service import job_manager
from schemas.qa import (
    TelegramIngestRequest,
    IngestJobResponse,
    IngestJobStatusResponse,
    IngestJobStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest/telegram", response_model=IngestJobResponse, tags=["ingest"])
async def start_telegram_ingestion(request: TelegramIngestRequest) -> IngestJobResponse:
    """
    Запускает задачу ingestion сообщений из Telegram канала

    - **channel**: Telegram канал (@username или ID)
    - **since**: Дата начала в формате ISO (YYYY-MM-DD)
    - **until**: Дата окончания в формате ISO (YYYY-MM-DD)
    - **collection**: Название коллекции ChromaDB для сохранения
    - **device**: Устройство для обработки (auto, cpu, cuda, mps)
    - **max_messages**: Максимум сообщений для обработки (опционально)

    Возвращает job_id для отслеживания прогресса выполнения.
    """
    try:
        # Собираем итоговый список каналов для логов
        channels: List[str] = []
        if request.channel:
            channels.append(request.channel)
        if request.channels:
            channels.extend([c for c in request.channels if c])
        # Уникализуем, сохраняя порядок
        seen = set()
        channels = [c for c in channels if not (c in seen or seen.add(c))]

        logger.info(
            f"Запуск Telegram ingestion: каналы={channels or [request.channel]}, "
            f"период={request.since} - {request.until}, коллекция={request.collection}"
        )

        # Валидация дат
        try:
            from dateutil import parser as date_parser

            since_date = date_parser.isoparse(request.since)
            until_date = date_parser.isoparse(request.until)

            if since_date >= until_date:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Дата начала должна быть меньше даты окончания",
                )

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неверный формат даты: {e}",
            )

        # Создаем задачу
        job_id = job_manager.create_job(request)

        # Примерная оценка времени (будет уточнена при выполнении)
        estimated_time = "Определяется при выполнении"
        if request.device == "cpu":
            estimated_time = "Может занять несколько часов"
        elif request.device in ["cuda", "mps"]:
            estimated_time = "Ожидается быстрое выполнение"

        logger.info(f"Создана задача ingestion с ID: {job_id}")

        return IngestJobResponse(
            job_id=job_id,
            status=IngestJobStatus.QUEUED,
            message="Задача создана и добавлена в очередь выполнения",
            estimated_time=estimated_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при создании задачи ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось создать задачу: {str(e)}",
        )


@router.get("/ingest/{job_id}", response_model=IngestJobStatusResponse, tags=["ingest"])
async def get_ingestion_status(job_id: str) -> IngestJobStatusResponse:
    """
    Получает статус и прогресс выполнения задачи ingestion

    - **job_id**: Уникальный идентификатор задачи

    Возвращает детальную информацию о состоянии задачи, включая:
    - Текущий статус (queued, running, completed, failed, cancelled)
    - Прогресс выполнения (0.0-1.0)
    - Количество обработанных сообщений
    - Сообщения об ошибках (если есть)
    - Логи выполнения
    """
    try:
        job = job_manager.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Задача с ID '{job_id}' не найдена",
            )

        return job.to_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении статуса задачи: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить статус задачи: {str(e)}",
        )


@router.delete("/ingest/{job_id}", tags=["ingest"])
async def cancel_ingestion(job_id: str) -> dict:
    """
    Отменяет выполнение задачи ingestion

    - **job_id**: Уникальный идентификатор задачи

    Задачу можно отменить только если она находится в состоянии
    queued или running. Завершенные задачи отменить нельзя.
    """
    try:
        job = job_manager.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Задача с ID '{job_id}' не найдена",
            )

        if job.status in [
            IngestJobStatus.COMPLETED,
            IngestJobStatus.FAILED,
            IngestJobStatus.CANCELLED,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Задача уже завершена со статусом: {job.status.value}",
            )

        success = job_manager.cancel_job(job_id)

        if success:
            logger.info(f"Задача {job_id} успешно отменена")
            return {
                "job_id": job_id,
                "message": "Задача успешно отменена",
                "status": "cancelled",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Не удалось отменить задачу",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при отмене задачи: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось отменить задачу: {str(e)}",
        )


@router.get("/ingest", response_model=List[IngestJobStatusResponse], tags=["ingest"])
async def list_ingestion_jobs() -> List[IngestJobStatusResponse]:
    """
    Получает список всех задач ingestion

    Возвращает информацию о всех созданных задачах, включая
    завершенные, выполняющиеся и ожидающие выполнения.
    """
    try:
        jobs = []
        for job in job_manager.jobs.values():
            jobs.append(job.to_response())

        # Сортируем по времени создания (новые первыми)
        jobs.sort(key=lambda x: x.started_at or datetime.min, reverse=True)

        logger.info(f"Возвращаем информацию о {len(jobs)} задачах")
        return jobs

    except Exception as e:
        logger.error(f"Ошибка при получении списка задач: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить список задач: {str(e)}",
        )
