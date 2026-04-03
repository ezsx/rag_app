"""
Rule-based pre-validator for adaptive retrieval.

Deterministic signal extraction from query text (<1ms).
Called before LLM query_plan -- results are injected as hints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# 36 каналов из нашей коллекции
KNOWN_CHANNELS = [
    "protechietich", "data_secrets", "ai_machinelearning_big_data", "data_easy",
    "xor_journal", "ai_newz", "neurohive", "denissexy", "gonzo_ml",
    "seeallochnaya", "dendi_math_ai", "complete_ai", "llm_under_hood",
    "varim_ml", "boris_again", "cryptovalerii", "scientific_opensource",
    "ruadaptnaya", "techsparks", "addmeto", "aioftheday", "singularityfm",
    "oulenspiegel_channel", "rybolos_channel", "stuffynlp", "MLunderhood",
    "deep_school", "CVML_team", "smalldatascience", "inforetriever",
    "theworldisnoteasy", "aihappens", "AIgobrr", "toBeAnMLspecialist",
    "ml_product", "techno_yandex", "atmyre_channell",
]

# Известные авторы → канал
AUTHOR_TO_CHANNEL = {
    "сапунов": "gonzo_ml",
    "себрант": "techsparks",
    "бакунов": "addmeto",
    "bobuk": "addmeto",
    "котенков": "seeallochnaya",
    "абдуллин": "llm_under_hood",
    "цейтлин": "boris_again",
    "санакоев": "ai_newz",
    "горный": "aioftheday",
    "марков": "oulenspiegel_channel",
    "шаврина": "rybolos_channel",
    "димитров": "dendi_math_ai",
    "кузнецов": "complete_ai",
}

# AI/ML entity patterns — компании, продукты, технологии
ENTITY_PATTERNS = [
    "nvidia", "openai", "google", "deepmind", "meta", "anthropic", "microsoft",
    "apple", "yandex", "яндекс", "sber", "сбер",
    "gpt-5", "gpt-4o", "gpt-4", "gpt-3", "gpt5", "gpt4",
    "claude", "gemini", "llama", "qwen", "mistral", "deepseek",
    "manus", "vera rubin", "sora", "dall-e", "midjourney",
    "kandinsky", "gigachat", "yandexgpt",
    "transformer", "трансформер", "attention", "moe", "rag",
    "colbert", "bge", "bert",
]

# Русские месяцы → номер
_RU_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4,
    "ма": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


@dataclass
class QuerySignals:
    """Result of rule-based query analysis."""
    strategy_hint: str | None = None  # "temporal" | "channel" | "entity" | None
    confidence: float = 0.0
    date_from: str | None = None      # ISO YYYY-MM-DD
    date_to: str | None = None        # ISO YYYY-MM-DD
    channels: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


def extract_query_signals(query: str) -> QuerySignals:
    """Extract signals from query text via regex. <1ms."""
    signals = QuerySignals()
    q_lower = query.lower().strip()

    # --- Channel detection ---
    _detect_channels(q_lower, signals)

    # --- Temporal detection ---
    _detect_temporal(q_lower, signals)

    # --- Entity detection ---
    _detect_entities(q_lower, query, signals)

    # --- Выбор strategy_hint по приоритету ---
    # Channel > Temporal > Entity (channel самый конкретный)
    if signals.channels:
        signals.strategy_hint = "channel"
        signals.confidence = 0.9
    elif signals.date_from or signals.date_to:
        signals.strategy_hint = "temporal"
        signals.confidence = 0.85
    elif signals.entities:
        signals.strategy_hint = "entity"
        signals.confidence = 0.6  # entity менее надёжен через regex

    # Если есть и channel и temporal — оба фильтра сработают
    # (combined filter), но strategy hint = channel (более специфичный)

    return signals


def _detect_channels(q_lower: str, signals: QuerySignals) -> None:
    """Detect channel @mentions and author references."""
    # @mentions
    at_mentions = re.findall(r"@(\w+)", q_lower)
    for mention in at_mentions:
        if mention in (ch.lower() for ch in KNOWN_CHANNELS):
            # Найти оригинальное имя канала (с регистром)
            for ch in KNOWN_CHANNELS:
                if ch.lower() == mention:
                    signals.channels.append(ch)
                    break

    # Exact channel name в тексте
    for ch in KNOWN_CHANNELS:
        ch_lower = ch.lower()
        # Ищем как слово (не часть другого слова)
        if re.search(rf"\b{re.escape(ch_lower)}\b", q_lower):
            if ch not in signals.channels:
                signals.channels.append(ch)

    # Автор → канал
    for author, channel in AUTHOR_TO_CHANNEL.items():
        if author in q_lower and channel not in signals.channels:
            signals.channels.append(channel)


def _detect_temporal(q_lower: str, signals: QuerySignals) -> None:
    """Detect temporal markers and extract date ranges."""
    now = datetime.now()

    # Русские месяцы + год: "в январе 2026", "декабрь 2025"
    for stem, month_num in _RU_MONTHS.items():
        match = re.search(rf"{stem}\w*\s+(\d{{4}})", q_lower)
        if match:
            year = int(match.group(1))
            _set_month_range(signals, year, month_num)
            return

    # Английские месяцы: "january 2026"
    for name, month_num in _EN_MONTHS.items():
        match = re.search(rf"{name}\s+(\d{{4}})", q_lower)
        if match:
            year = int(match.group(1))
            _set_month_range(signals, year, month_num)
            return

    # "начало/конец 2026", "в 2026 году"
    year_match = re.search(r"\b(202[4-9])\b", q_lower)
    if year_match:
        year = int(year_match.group(1))
        if "начал" in q_lower:
            signals.date_from = f"{year}-01-01"
            signals.date_to = f"{year}-03-31"
            return
        elif "конц" in q_lower or "конец" in q_lower:
            signals.date_from = f"{year}-10-01"
            signals.date_to = f"{year}-12-31"
            return

    # Relative: "последняя неделя", "за последний месяц", "недавно"
    if any(w in q_lower for w in ["последн", "прошл", "недавн", "latest", "recent", "last week"]):
        if "недел" in q_lower or "week" in q_lower:
            signals.date_from = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            signals.date_to = now.strftime("%Y-%m-%d")
        elif "месяц" in q_lower or "month" in q_lower:
            signals.date_from = (now - timedelta(days=30)).strftime("%Y-%m-%d")
            signals.date_to = now.strftime("%Y-%m-%d")
        else:
            # "недавно", "последнее" — последние 2 недели
            signals.date_from = (now - timedelta(days=14)).strftime("%Y-%m-%d")
            signals.date_to = now.strftime("%Y-%m-%d")
        return

    # ISO даты: "2026-01-15"
    iso_dates = re.findall(r"(\d{4}-\d{2}-\d{2})", q_lower)
    if len(iso_dates) >= 2:
        signals.date_from = iso_dates[0]
        signals.date_to = iso_dates[1]
    elif len(iso_dates) == 1:
        signals.date_from = iso_dates[0]
        signals.date_to = iso_dates[0]


def _detect_entities(q_lower: str, q_original: str, signals: QuerySignals) -> None:
    """Detect entity mentions (companies, products, technologies)."""
    for pattern in ENTITY_PATTERNS:
        if re.search(rf"\b{re.escape(pattern)}\b", q_lower):
            # Используем оригинальный регистр из запроса если найдём
            match = re.search(rf"\b{re.escape(pattern)}\b", q_original, re.IGNORECASE)
            entity_text = match.group(0) if match else pattern
            if entity_text not in signals.entities:
                signals.entities.append(entity_text)


def _set_month_range(signals: QuerySignals, year: int, month: int) -> None:
    """Set date_from/date_to to span the full month."""
    signals.date_from = f"{year}-{month:02d}-01"
    # Последний день месяца
    if month == 12:
        signals.date_to = f"{year}-12-31"
    else:
        next_month_first = datetime(year, month + 1, 1)
        last_day = next_month_first - timedelta(days=1)
        signals.date_to = last_day.strftime("%Y-%m-%d")
