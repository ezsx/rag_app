"""
Export-to-formats tool.
Поддержка лёгкого экспорта: md/txt/json. PDF/DOCX возвращает статус not_supported.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def export_to_formats(
    content: str,
    fmt: str,
    filename_base: str = "export",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fmt_l = (fmt or "md").lower()
    name = filename_base or "export"

    if fmt_l == "md":
        return {
            "filename": f"{name}.md",
            "mime_type": "text/markdown; charset=utf-8",
            "data": content,
            "encoding": "utf-8",
            "meta": metadata or {},
        }
    if fmt_l == "txt":
        return {
            "filename": f"{name}.txt",
            "mime_type": "text/plain; charset=utf-8",
            "data": content,
            "encoding": "utf-8",
            "meta": metadata or {},
        }
    if fmt_l == "json":
        try:
            obj = json.loads(content)
        except Exception:
            obj = {"content": content}
        return {
            "filename": f"{name}.json",
            "mime_type": "application/json",
            "data": obj,
            "encoding": "utf-8",
            "meta": metadata or {},
        }

    # Не поддерживаемые форматы в MVP
    if fmt_l in {"pdf", "doc", "docx"}:
        return {
            "status": "not_supported",
            "reason": "PDF/DOCX экспорт не включён в MVP",
            "suggested": ["md", "txt", "json"],
        }

    return {
        "status": "unknown_format",
        "format": fmt_l,
        "suggested": ["md", "txt", "json"],
    }
