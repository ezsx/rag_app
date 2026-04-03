"""Custom agent adapter — HTTP call to /v1/agent/stream with SSE parsing.

Подключается к нашему running API (port 8001).
Парсит SSE events: tool_invoked, final.
"""

from __future__ import annotations

import json
import time
import urllib.request

from benchmarks.config import AGENT_URL
from benchmarks.protocols import AgentResult, RetrievalResult


def _iter_sse_lines(response) -> list[tuple[str, str]]:
    """Парсит SSE stream из urllib response в список (event_name, data)."""
    events = []
    current_event = ""
    current_data = ""

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")

        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "":
            if current_event and current_data:
                events.append((current_event, current_data))
            current_event = ""
            current_data = ""

    # Last event if no trailing blank line
    if current_event and current_data:
        events.append((current_event, current_data))

    return events


class CustomAgent:
    """Calls our /v1/agent/stream endpoint, parses SSE events."""

    def run(self, query: str) -> AgentResult:
        t0 = time.time()

        payload = json.dumps({
            "query": query,
            "max_steps": 8,
        }).encode()

        req = urllib.request.Request(
            f"{AGENT_URL}/v1/agent/stream",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test",
            },
        )

        tools_invoked: list[str] = []
        citations: list[dict] = []
        observations: list[dict] = []
        answer = ""

        try:
            resp = urllib.request.urlopen(req, timeout=180)
            events = _iter_sse_lines(resp)

            for event_name, event_data in events:
                try:
                    decoded = json.loads(event_data)
                except json.JSONDecodeError:
                    continue

                if event_name == "tool_invoked":
                    tool_name = decoded.get("tool") or decoded.get("name", "")
                    if tool_name:
                        tools_invoked.append(tool_name)

                elif event_name == "observation":
                    observations.append({
                        "tool": decoded.get("tool", ""),
                        "content": str(decoded.get("content", ""))[:1000],
                        "success": decoded.get("success", False),
                    })

                elif event_name == "citations":
                    for cit in decoded.get("citations", []):
                        meta = cit.get("metadata", {})
                        citations.append({
                            "id": cit.get("id"),
                            "channel": meta.get("channel", ""),
                            "message_id": meta.get("message_id", ""),
                            "text": str(cit.get("text", meta.get("text", "")))[:500],
                        })

                elif event_name == "final":
                    answer = decoded.get("answer", "")
                    break

        except Exception as e:
            answer = f"ERROR: {e}"

        latency = time.time() - t0

        # Конвертируем citations в RetrievalResult для docs
        docs = []
        for cit in citations:
            docs.append(RetrievalResult(
                doc_id=f"{cit['channel']}:{cit['message_id']}",
                score=0.0,
                channel=cit["channel"],
                message_id=int(cit["message_id"]) if cit["message_id"] else 0,
                text=cit.get("text"),
            ))

        return AgentResult(
            answer=answer,
            docs=docs,
            tool_calls=tools_invoked,
            latency=latency,
        )
