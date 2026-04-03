"""Redis cache wrapper for API endpoints."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def cache_get(redis_client, key: str) -> dict[str, Any] | None:
    """Read JSON from Redis. Returns None on miss or if cache is unavailable."""
    if not redis_client:
        return None
    try:
        raw = redis_client.get(key)
        if raw:
            logger.info("Cache hit: %s", key[:50])
            return json.loads(raw)
    except Exception:  # broad: redis adapter boundary
        logger.warning("Cache read failed: %s", key[:50])
    return None


def cache_set(redis_client, key: str, data: dict[str, Any], ttl: int) -> None:
    """Write JSON to Redis with TTL. Silently ignores errors."""
    if not redis_client:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(data, ensure_ascii=False, default=str))
    except Exception:  # broad: redis adapter boundary
        logger.warning("Cache write failed: %s", key[:50])
