"""
Rate limiting middleware для защиты от спама и DDoS
"""

import time
import logging
from typing import Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, Response
from starlette.middleware.base import BaseHTTPMiddleware
from threading import Lock
import hashlib

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware с поддержкой:
    - Per-IP limiting
    - Per-endpoint limiting
    - Sliding window algorithm
    - Exponential backoff для повторных нарушений
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
        enable_exponential_backoff: bool = True,
    ):
        super().__init__(app)

        # Конфигурация
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.enable_exponential_backoff = enable_exponential_backoff

        # Хранилище для отслеживания запросов
        # {client_id: [(timestamp, endpoint), ...]}
        self.requests: Dict[str, list] = defaultdict(list)

        # Счетчик нарушений для exponential backoff
        # {client_id: (violation_count, last_violation_time)}
        self.violations: Dict[str, Tuple[int, float]] = {}

        # Thread-safe lock
        self.lock = Lock()

        # Cleanup interval
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 минут

    def _get_client_id(self, request: Request) -> str:
        """Получает уникальный идентификатор клиента"""
        # Приоритет: API key -> Auth token -> IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{hashlib.md5(api_key.encode()).hexdigest()[:16]}"

        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return f"jwt:{hashlib.md5(token.encode()).hexdigest()[:16]}"

        # Fallback to IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip:{client_ip}"

    def _cleanup_old_requests(self):
        """Удаляет устаревшие записи"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return

        cutoff_time = now - 3600  # Храним данные за последний час

        with self.lock:
            # Очистка запросов
            for client_id in list(self.requests.keys()):
                self.requests[client_id] = [
                    (ts, ep) for ts, ep in self.requests[client_id] if ts > cutoff_time
                ]
                if not self.requests[client_id]:
                    del self.requests[client_id]

            # Очистка нарушений
            for client_id in list(self.violations.keys()):
                _, last_violation = self.violations[client_id]
                if last_violation < cutoff_time:
                    del self.violations[client_id]

        self.last_cleanup = now

    def _check_rate_limit(self, client_id: str, endpoint: str) -> Optional[int]:
        """
        Проверяет rate limit для клиента.
        Возвращает None если OK, или количество секунд до разблокировки.
        """
        now = time.time()

        with self.lock:
            # Получаем историю запросов
            request_history = self.requests[client_id]

            # Фильтруем по временным окнам
            minute_ago = now - 60
            hour_ago = now - 3600

            requests_last_minute = [
                (ts, ep) for ts, ep in request_history if ts > minute_ago
            ]

            requests_last_hour = [
                (ts, ep) for ts, ep in request_history if ts > hour_ago
            ]

            # Проверка burst (последние 10 секунд)
            burst_window = now - 10
            burst_requests = len(
                [ts for ts, _ in requests_last_minute if ts > burst_window]
            )

            # Проверка лимитов
            if burst_requests >= self.burst_size:
                return 10  # Подождите 10 секунд

            if len(requests_last_minute) >= self.requests_per_minute:
                return 60  # Подождите минуту

            if len(requests_last_hour) >= self.requests_per_hour:
                return 3600  # Подождите час

            # Проверка exponential backoff при повторных нарушениях
            if self.enable_exponential_backoff and client_id in self.violations:
                violation_count, last_violation = self.violations[client_id]

                # Exponential backoff: 2^n минут
                backoff_minutes = min(2**violation_count, 60)  # Max 1 час
                backoff_seconds = backoff_minutes * 60

                if now - last_violation < backoff_seconds:
                    remaining = int(backoff_seconds - (now - last_violation))
                    return remaining

            # Добавляем текущий запрос
            request_history.append((now, endpoint))
            self.requests[client_id] = requests_last_hour  # Сохраняем только за час

            return None

    def _record_violation(self, client_id: str):
        """Записывает нарушение rate limit"""
        now = time.time()

        with self.lock:
            if client_id in self.violations:
                count, _ = self.violations[client_id]
                self.violations[client_id] = (count + 1, now)
            else:
                self.violations[client_id] = (1, now)

    async def dispatch(self, request: Request, call_next):
        """Обрабатывает запрос с проверкой rate limit"""
        # Периодическая очистка
        self._cleanup_old_requests()

        # Пропускаем health check и системные endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)

        client_id = self._get_client_id(request)
        endpoint = f"{request.method}:{request.url.path}"

        # Проверяем rate limit
        wait_seconds = self._check_rate_limit(client_id, endpoint)

        if wait_seconds is not None:
            # Записываем нарушение
            self._record_violation(client_id)

            # Логируем
            logger.warning(
                f"Rate limit exceeded for {client_id} on {endpoint}. "
                f"Wait {wait_seconds} seconds."
            )

            # Возвращаем 429 с заголовками
            return Response(
                content=f"Rate limit exceeded. Please wait {wait_seconds} seconds.",
                status_code=429,
                headers={
                    "Retry-After": str(wait_seconds),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + wait_seconds)),
                },
            )

        # Выполняем запрос
        response = await call_next(request)

        # Добавляем заголовки rate limit в ответ
        with self.lock:
            requests_last_minute = len(
                [ts for ts, _ in self.requests[client_id] if ts > time.time() - 60]
            )
            remaining = max(0, self.requests_per_minute - requests_last_minute)

        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

        return response


# Конфигурируемые presets для разных endpoints
RATE_LIMIT_PRESETS = {
    "default": {"requests_per_minute": 60, "requests_per_hour": 1000, "burst_size": 10},
    "search": {"requests_per_minute": 30, "requests_per_hour": 500, "burst_size": 5},
    "agent": {"requests_per_minute": 10, "requests_per_hour": 100, "burst_size": 2},
    "ingest": {"requests_per_minute": 5, "requests_per_hour": 20, "burst_size": 1},
}
