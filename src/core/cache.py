"""Обёртка Redis кеша для API endpoints."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def cache_get(redis_client, key: str) -> dict[str, Any] | None:
    """Читает JSON из Redis. None если кеш недоступен или промах."""
    if not redis_client:
        return None
    try:
        raw = redis_client.get(key)
        if raw:
            logger.info("Cache hit: %s", key[:50])
            return json.loads(raw)
    except Exception:
        logger.warning("Cache read failed: %s", key[:50])
    return None


def cache_set(redis_client, key: str, data: dict[str, Any], ttl: int) -> None:
    """Пишет JSON в Redis с TTL. Молча пропускает при ошибке."""
    if not redis_client:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(data, ensure_ascii=False, default=str))
    except Exception:
        logger.warning("Cache write failed: %s", key[:50])
