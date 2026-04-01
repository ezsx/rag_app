"""
SPEC-RAG-20c Step 2: Data-driven routing и policies из tool_keywords.json.

Вынесено из agent_service.py. Lazy load + global cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ROUTING_DATA: Optional[Dict[str, Any]] = None


def _load_routing_data() -> Dict[str, Any]:
    """Загрузить tool_keywords.json (routing + policies).

    Lazy load + global cache. Структура: {tool_keywords: {...}, agent_policies: {...}}.
    """
    global _ROUTING_DATA
    if _ROUTING_DATA is not None:
        return _ROUTING_DATA
    # Ищем datasets/tool_keywords.json поднимаясь от текущего файла
    # agent/routing.py → services/ → src/ → repo_root/datasets/
    base = Path(__file__).resolve().parent
    for _ in range(6):  # +1 уровень т.к. теперь в agent/ subdirectory
        candidate = base / "datasets" / "tool_keywords.json"
        if candidate.exists():
            path = candidate
            break
        base = base.parent
    else:
        path = Path("datasets/tool_keywords.json")
    try:
        with open(path, encoding="utf-8") as f:
            _ROUTING_DATA = json.load(f)
    except Exception:
        logger.warning("tool_keywords.json not found, routing/policies disabled")
        _ROUTING_DATA = {}
    return _ROUTING_DATA


def load_tool_keywords() -> Dict[str, List[str]]:
    """Keyword routing: {tool_name: [keywords]}."""
    data = _load_routing_data()
    section = data.get("tool_keywords", {})
    return {
        tool: entry["keywords"]
        for tool, entry in section.items()
        if isinstance(entry, dict) and "keywords" in entry
    }


def load_policy(name: str) -> List[str]:
    """Загрузить список values из agent_policies.{name}.values."""
    data = _load_routing_data()
    return data.get("agent_policies", {}).get(name, {}).get("values", [])
