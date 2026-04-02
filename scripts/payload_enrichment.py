"""
Модуль обогащения payload для Qdrant.
Извлекает entities, URLs, arxiv IDs, hashtags, язык из текста и Telethon Message.

Используется в ingest_telegram.py при построении payload для каждого point.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# ─── Regex patterns ──────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_ARXIV_RE = re.compile(r"(?:arxiv\.org/abs/|arxiv:\s?)(\d{4}\.\d{4,5})", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+)", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#(\w{2,})")
_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
_DOMAIN_RE = re.compile(r"https?://([^/\s]+)")

# ─── Entity dictionary ───────────────────────────────────────────

_ENTITY_DICT_PATH = Path(__file__).parent.parent / "datasets" / "entity_dictionary.json"

_entity_patterns: list[tuple] | None = None  # (canonical, compiled_re, category)


def _load_entity_patterns() -> list[tuple]:
    """Загрузить и скомпилировать словарь entities."""
    global _entity_patterns
    if _entity_patterns is not None:
        return _entity_patterns

    with open(_ENTITY_DICT_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    patterns = []
    for canonical, info in raw.items():
        aliases = info.get("aliases", [])
        category = info.get("category", "other")
        case_sensitive = info.get("case_sensitive", False)
        # Собираем regex: canonical + все aliases, сортируем по длине (longest first)
        all_forms = [canonical, *aliases]
        all_forms.sort(key=len, reverse=True)
        # Escape и join
        escaped = [re.escape(form) for form in all_forms]
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags)
        patterns.append((canonical, pattern, category))

    _entity_patterns = patterns
    return _entity_patterns


# ─── Text extraction ─────────────────────────────────────────────


def extract_from_text(text: str) -> dict[str, Any]:
    """Regex-based extraction из текста поста.

    Возвращает dict с полями для payload:
    entities, entity_orgs, entity_models, urls, url_domains,
    arxiv_ids, github_repos, hashtags, lang, text_length,
    has_arxiv, has_links, year_week, year_month (последние два — заглушка,
    заполняются из даты в build_enriched_payload).
    """
    result: dict[str, Any] = {}

    # URLs
    urls = _URL_RE.findall(text)
    result["urls"] = urls[:20]  # cap
    result["has_links"] = bool(urls)

    # URL domains
    domains = []
    for url in urls:
        m = _DOMAIN_RE.match(url)
        if m:
            domains.append(m.group(1).lower())
    result["url_domains"] = list(set(domains))

    # Arxiv IDs
    arxiv_ids = list(set(_ARXIV_RE.findall(text)))
    result["arxiv_ids"] = arxiv_ids
    result["has_arxiv"] = bool(arxiv_ids)

    # GitHub repos
    result["github_repos"] = list(set(_GITHUB_RE.findall(text)))

    # Hashtags — lowercase, dedup
    raw_tags = _HASHTAG_RE.findall(text)
    result["hashtags"] = list({t.lower() for t in raw_tags})

    # Language detection — простая эвристика по доле кириллицы
    if text:
        cyrillic_count = len(_CYRILLIC_RE.findall(text))
        ratio = cyrillic_count / max(len(text), 1)
        result["lang"] = "ru" if ratio > 0.15 else "en"
    else:
        result["lang"] = "unknown"

    result["text_length"] = len(text)

    # Entities
    entities = set()
    entity_orgs = set()
    entity_models = set()

    for canonical, pattern, category in _load_entity_patterns():
        if pattern.search(text):
            entities.add(canonical)
            if category == "org":
                entity_orgs.add(canonical)
            elif category == "model":
                entity_models.add(canonical)

    result["entities"] = sorted(entities)
    result["entity_orgs"] = sorted(entity_orgs)
    result["entity_models"] = sorted(entity_models)

    return result


# ─── Message metadata extraction ─────────────────────────────────


def extract_from_message(msg: Any) -> dict[str, Any]:
    """Извлечь metadata из Telethon Message object.

    Поля: is_forward, forwarded_from_id (str), forwarded_from_name,
    reply_to_msg_id, media_types.
    """
    result: dict[str, Any] = {}

    # Forward info
    fwd = getattr(msg, "fwd_from", None)
    result["is_forward"] = bool(fwd)
    if fwd:
        from_id = getattr(fwd, "from_id", None)
        if from_id and hasattr(from_id, "channel_id"):
            result["forwarded_from_id"] = str(from_id.channel_id)
        from_name = getattr(fwd, "from_name", None)
        if from_name:
            result["forwarded_from_name"] = from_name

    # Reply
    reply = getattr(msg, "reply_to", None)
    if reply:
        reply_id = getattr(reply, "reply_to_msg_id", None)
        if reply_id is not None:
            result["reply_to_msg_id"] = int(reply_id)

    # Media types
    media_types = []
    if getattr(msg, "photo", None):
        media_types.append("photo")
    if getattr(msg, "video", None):
        media_types.append("video")
    if getattr(msg, "document", None) and not getattr(msg, "video", None):
        media_types.append("document")
    if getattr(msg, "audio", None):
        media_types.append("audio")
    if media_types:
        result["media_types"] = media_types

    return result


# ─── Derived temporal fields ─────────────────────────────────────


def compute_temporal_fields(date_str: str) -> dict[str, str]:
    """Вычислить year_week и year_month из ISO date string."""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        iso_cal = dt.isocalendar()
        return {
            "year_week": f"{iso_cal[0]}-W{iso_cal[1]:02d}",
            "year_month": f"{dt.year}-{dt.month:02d}",
        }
    except (ValueError, AttributeError):
        return {"year_week": "unknown", "year_month": "unknown"}


# ─── Public API ──────────────────────────────────────────────────


def build_enriched_payload(
    text: str,
    message: Any,
    channel_name: str,
    date_iso: str,
    point_id: str,
    author: str | None = None,
) -> dict[str, Any]:
    """Полный enriched payload для одного Qdrant point.

    Объединяет base fields + text extraction + message metadata + temporal.
    """
    payload: dict[str, Any] = {
        "text": text,
        "channel": channel_name,
        "channel_id": int(message.chat_id),
        "message_id": int(message.id),
        "date": date_iso,
        "url": f"https://t.me/{channel_name}/{message.id}" if channel_name else None,
        "point_id": point_id,
        # root_message_id на стабильной основе channel_id (не channel_name,
        # который может измениться при rename канала или при разном формате CLI hint)
        "root_message_id": f"{int(message.chat_id)}:{int(message.id)}",
    }

    if author:
        payload["author"] = author

    # Text-based extraction
    text_meta = extract_from_text(text)
    payload.update(text_meta)

    # Telethon Message metadata
    msg_meta = extract_from_message(message)
    payload.update(msg_meta)

    # Temporal derived fields
    temporal = compute_temporal_fields(date_iso)
    payload.update(temporal)

    return payload
