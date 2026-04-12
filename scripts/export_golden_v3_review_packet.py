#!/usr/bin/env python3
"""Экспорт semi-manual review packet для golden v3 draft.

Пакет содержит query, expected answer, required claims и source snippet из
Qdrant. Это защита от ошибки golden v2: слишком узкие expected answers для
open-ended вопросов.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    """POST JSON helper."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def fetch_source_text(qdrant_url: str, collection: str, source_id: str) -> str:
    """Возвращает полный текст source post по `channel:message_id`."""
    if ":" not in source_id:
        return ""
    channel, raw_message_id = source_id.split(":", 1)
    if not raw_message_id.isdigit():
        return ""
    payload = {
        "limit": 32,
        "with_payload": ["channel", "message_id", "date", "point_id", "text"],
        "with_vector": False,
        "filter": {
            "must": [
                {"key": "channel", "match": {"value": channel}},
                {"key": "message_id", "match": {"value": int(raw_message_id)}},
            ],
        },
    }
    result = post_json(
        f"{qdrant_url}/collections/{collection}/points/scroll",
        payload,
        timeout=30,
    )["result"]
    points = result.get("points") or []
    if not points:
        return ""
    points.sort(key=_chunk_sort_key)
    source_payload = points[0].get("payload", {})
    date = str(source_payload.get("date") or "")[:10]
    text = "\n\n".join(
        str((point.get("payload") or {}).get("text") or "").strip()
        for point in points
        if str((point.get("payload") or {}).get("text") or "").strip()
    )
    return f"{source_id} | {date}\n{text}"


def _chunk_sort_key(point: dict[str, Any]) -> tuple[int, str]:
    """Сортирует chunks одного Telegram post по suffix из payload.point_id."""
    payload = point.get("payload") or {}
    point_id = str(payload.get("point_id") or "")
    try:
        chunk_index = int(point_id.rsplit(":", 1)[1])
    except (ValueError, IndexError):
        chunk_index = 0
    return chunk_index, str(point.get("id") or "")


def wrap_block(text: str, width: int = 110) -> str:
    """Нормализует длинный текст для markdown review."""
    lines = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(lines).strip()


def select_items(
    items: list[dict[str, Any]],
    start_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Выбирает новые v3 items по numeric suffix."""
    selected = []
    for item in items:
        item_id = str(item.get("id") or "")
        if not item_id.startswith("golden_v3_q"):
            continue
        try:
            number = int(item_id.rsplit("q", 1)[1])
        except ValueError:
            continue
        if number >= start_id:
            selected.append(item)
    selected.sort(key=lambda item: int(str(item["id"]).rsplit("q", 1)[1]))
    return selected[:limit]


def render_packet(
    items: list[dict[str, Any]],
    *,
    qdrant_url: str,
    collection: str,
    source_chars: int,
) -> str:
    """Рендерит markdown review packet."""
    lines = [
        "# Golden v3 Review Packet",
        "",
        "> Status values: `accept`, `edit`, `reject`.",
        "> Review goal: catch too-narrow expected answers, weak/non-core questions, and broken anchors.",
        "",
    ]
    for item in items:
        source_ids = item.get("source_post_ids") or []
        source_text = ""
        if source_ids:
            source_text = fetch_source_text(qdrant_url, collection, source_ids[0])
        source_text = wrap_block(source_text[:source_chars])
        metadata = item.get("metadata") or {}
        review_status = metadata.get("review_status", "TODO")
        review_notes = metadata.get("review_notes", "")

        lines.extend([
            f"## {item['id']} — {item.get('category')} / {item.get('eval_mode')}",
            "",
            f"- Status: {review_status}",
            "- Action: accept / edit / reject" if review_status == "TODO" else f"- Action: {review_status}",
            f"- Query: {item.get('query')}",
            f"- Source IDs: {', '.join(source_ids) if source_ids else '-'}",
            f"- Expected channels: {', '.join(item.get('expected_channels') or []) or '-'}",
            "",
            "**Expected answer**",
            "",
            wrap_block(str(item.get("expected_answer") or "")),
            "",
            "**Required claims**",
            "",
        ])
        for claim in item.get("required_claims") or []:
            lines.append(f"- {claim}")
        lines.extend([
            "",
            "**Source snippet**",
            "",
            "```text",
            source_text or "(no source snippet; static/analytics item)",
            "```",
            "",
            "**Reviewer notes**",
            "",
            f"- {review_notes}" if review_notes else "- ",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export golden v3 review packet")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/golden_v3/eval_golden_v3_draft.json"))
    parser.add_argument("--output", type=Path, default=Path("datasets/golden_v3/golden_v3_review_packet_001.md"))
    parser.add_argument("--start-id", type=int, default=37)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--qdrant-url", default="http://localhost:16333")
    parser.add_argument("--collection", default="news_colbert_v2")
    parser.add_argument("--source-chars", type=int, default=1800)
    args = parser.parse_args()

    items = json.loads(args.dataset.read_text(encoding="utf-8"))
    selected = select_items(items, args.start_id, args.limit)
    packet = render_packet(
        selected,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        source_chars=args.source_chars,
    )
    args.output.write_text(packet, encoding="utf-8")
    print(f"saved {args.output} ({len(selected)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
