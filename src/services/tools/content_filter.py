"""
Content filter tool.
Базовая модерация контента: PII/URLs/токсичность/брань, с подсказкой дейстий.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.security import sanitize_for_logging


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}"
)
_URL_RE = re.compile(r"https?://[^\s]+")

_BAD_WORDS = {"shit", "fuck", "сука", "блять", "хуй", "идиот", "мразь"}
_HATE_WORDS = {"nazi", "нацист", "расист", "racist", "hate"}
_SEX_WORDS = {"porn", "sex", "xxx", "порн", "секс"}


def content_filter(text: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
    cats = set([c.lower() for c in (categories or [])])
    flags: Dict[str, List[str]] = {}

    def _flag(name: str, values: List[str]) -> None:
        if values:
            flags[name] = list(dict.fromkeys(values))

    emails = _EMAIL_RE.findall(text)
    phones = _PHONE_RE.findall(text)
    urls = _URL_RE.findall(text)

    words = re.findall(r"\w+", text.lower())
    bad = [w for w in words if w in _BAD_WORDS]
    hate = [w for w in words if w in _HATE_WORDS]
    sex = [w for w in words if w in _SEX_WORDS]

    if not cats or "pii" in cats:
        _flag("pii_email", emails)
        _flag("pii_phone", phones)
        _flag("url", urls)
    if not cats or "toxicity" in cats:
        _flag("toxicity", bad)
    if not cats or "hate" in cats:
        _flag("hate", hate)
    if not cats or "sexual" in cats:
        _flag("sexual", sex)

    # Решение
    total_hits = sum(len(v) for v in flags.values())
    if "hate" in flags or len(flags.get("sexual", [])) > 3:
        action = "reject"
        allowed = False
    elif total_hits > 0:
        action = "sanitize"
        allowed = True
    else:
        action = "allow"
        allowed = True

    return {
        "allowed": allowed,
        "violations": flags,
        "suggested_action": action,
        "sanitized_preview": sanitize_for_logging(text, max_length=200),
    }
