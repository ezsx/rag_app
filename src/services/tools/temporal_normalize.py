"""
Temporal normalization tool.
Парсит и нормализует временные выражения в ISO формат.
"""

import re
from datetime import datetime, timezone
from typing import Dict, Optional

# Будем использовать простые паттерны, так как dateparser может не быть установлен
# В продакшене рекомендуется использовать dateparser библиотеку


class TemporalNormalizer:
    """Нормализует временные выражения."""

    # Относительные выражения
    RELATIVE_PATTERNS = {
        r"сегодня|today": 0,
        r"вчера|yesterday": -1,
        r"позавчера|day before yesterday": -2,
        r"завтра|tomorrow": 1,
        r"послезавтра|day after tomorrow": 2,
    }

    # Периоды
    PERIOD_PATTERNS = {
        r"(\d+)\s*(?:дн[ейя]|days?)\s*назад": lambda m: -int(m.group(1)),
        r"(\d+)\s*(?:недел[ьи]|weeks?)\s*назад": lambda m: -int(m.group(1)) * 7,
        r"(\d+)\s*(?:месяц[аев]?|months?)\s*назад": lambda m: -int(m.group(1)) * 30,
        r"через\s*(\d+)\s*(?:дн[ейя]|days?)": lambda m: int(m.group(1)),
        r"через\s*(\d+)\s*(?:недел[ьи]|weeks?)": lambda m: int(m.group(1)) * 7,
    }

    def normalize(self, text: str) -> Dict[str, any]:
        """
        Нормализует временные выражения в тексте.

        Args:
            text: Текст для анализа

        Returns:
            Dict с найденными датами и нормализованным текстом
        """
        try:
            # Импортируем dateparser если доступен
            import dateparser

            use_dateparser = True
        except ImportError:
            use_dateparser = False

        found_dates = []
        normalized_text = text

        if use_dateparser:
            # Используем dateparser для продвинутого парсинга
            settings = {
                "PREFER_DATES_FROM": "past",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "LANGUAGES": ["ru", "en"],
            }

            # Ищем временные выражения
            # Простой подход - ищем известные паттерны
            temporal_keywords = [
                r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b",  # даты
                r"\b\d{4}-\d{2}-\d{2}\b",  # ISO даты
                r"\b(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\b",
                r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
                r"\b(?:понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)\b",
                r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                r"\b(?:сегодня|вчера|завтра|позавчера|послезавтра)\b",
                r"\b(?:today|yesterday|tomorrow)\b",
                r"\b\d+\s*(?:дн[ейя]|недел[ьи]|месяц[аев]?|год[а]?|лет)\s*(?:назад|вперед)\b",
                r"\b\d+\s*(?:days?|weeks?|months?|years?)\s*(?:ago|later)\b",
            ]

            for pattern in temporal_keywords:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    expr = match.group()
                    parsed = dateparser.parse(expr, settings=settings)
                    if parsed:
                        iso_date = parsed.isoformat()
                        found_dates.append(
                            {
                                "original": expr,
                                "normalized": iso_date,
                                "position": match.span(),
                            }
                        )
                        # Заменяем в тексте
                        normalized_text = normalized_text.replace(
                            expr, f"{expr} [{iso_date}]"
                        )
        else:
            # Fallback к простым паттернам
            now = datetime.now(timezone.utc)

            # Относительные даты
            for pattern, days_offset in self.RELATIVE_PATTERNS.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    expr = match.group()
                    target_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    if days_offset != 0:
                        from datetime import timedelta

                        target_date += timedelta(days=days_offset)

                    iso_date = target_date.isoformat()
                    found_dates.append(
                        {
                            "original": expr,
                            "normalized": iso_date,
                            "position": match.span(),
                        }
                    )
                    normalized_text = normalized_text.replace(
                        expr, f"{expr} [{iso_date}]"
                    )

            # Периоды
            for pattern, calc_offset in self.PERIOD_PATTERNS.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    expr = match.group()
                    days_offset = calc_offset(match)

                    from datetime import timedelta

                    target_date = now + timedelta(days=days_offset)

                    iso_date = target_date.isoformat()
                    found_dates.append(
                        {
                            "original": expr,
                            "normalized": iso_date,
                            "position": match.span(),
                        }
                    )
                    normalized_text = normalized_text.replace(
                        expr, f"{expr} [{iso_date}]"
                    )

        return {
            "found_dates": found_dates,
            "normalized_text": normalized_text,
            "dateparser_available": use_dateparser,
        }


def temporal_normalize(text: str) -> Dict[str, any]:
    """
    Нормализует временные выражения в тексте.

    Args:
        text: Текст для анализа

    Returns:
        Dict с результатами:
        - found_dates: список найденных дат
        - normalized_text: текст с нормализованными датами
        - dateparser_available: доступен ли dateparser
    """
    normalizer = TemporalNormalizer()
    return normalizer.normalize(text)
