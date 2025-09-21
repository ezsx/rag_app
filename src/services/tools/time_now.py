from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


def time_now(format: str = "iso", timezone_name: str = "UTC") -> Dict[str, Any]:
    """Возвращает текущее время в указанном формате.

    Args:
        format: Формат времени ("iso", "timestamp", "readable", "date", "time")
        timezone_name: Временная зона (по умолчанию UTC)

    Returns:
        {current_time: str, timezone: str, timestamp: float, format_used: str}
    """
    try:
        # Получаем текущее время в UTC
        now_utc = datetime.now(timezone.utc)

        # Базовые данные
        result = {
            "timestamp": now_utc.timestamp(),
            "timezone": timezone_name,
            "format_used": format,
        }

        # Форматирование в зависимости от запрошенного формата
        if format == "iso":
            result["current_time"] = now_utc.isoformat()
        elif format == "timestamp":
            result["current_time"] = str(int(now_utc.timestamp()))
        elif format == "readable":
            result["current_time"] = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif format == "date":
            result["current_time"] = now_utc.strftime("%Y-%m-%d")
        elif format == "time":
            result["current_time"] = now_utc.strftime("%H:%M:%S")
        elif format == "datetime":
            result["current_time"] = now_utc.strftime("%Y-%m-%d %H:%M:%S")
        elif format == "rfc3339":
            result["current_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            # Пользовательский формат через strftime
            try:
                result["current_time"] = now_utc.strftime(format)
            except ValueError:
                # Если формат невалидный, возвращаем ISO
                result["current_time"] = now_utc.isoformat()
                result["format_used"] = "iso (fallback)"
                result["warning"] = f"Неизвестный формат '{format}', использован ISO"

        # Дополнительная информация
        result.update(
            {
                "year": now_utc.year,
                "month": now_utc.month,
                "day": now_utc.day,
                "hour": now_utc.hour,
                "minute": now_utc.minute,
                "second": now_utc.second,
                "weekday": now_utc.strftime("%A"),
                "month_name": now_utc.strftime("%B"),
            }
        )

        return result

    except Exception as e:
        logger.error(f"Ошибка при получении текущего времени: {e}")
        return {
            "current_time": None,
            "timezone": timezone_name,
            "format_used": format,
            "error": f"Ошибка: {str(e)}",
        }


# Доступные форматы для документации
SUPPORTED_FORMATS = {
    "iso": "ISO 8601 формат (2023-12-01T15:30:45.123456+00:00)",
    "timestamp": "Unix timestamp (1701443445)",
    "readable": "Читаемый формат (2023-12-01 15:30:45 UTC)",
    "date": "Только дата (2023-12-01)",
    "time": "Только время (15:30:45)",
    "datetime": "Дата и время (2023-12-01 15:30:45)",
    "rfc3339": "RFC 3339 формат (2023-12-01T15:30:45Z)",
    "custom": "Пользовательский формат strftime",
}
