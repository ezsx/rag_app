"""
SPEC-RAG-20c Step 6: Tool execution — normalize params, temporal guard, execute.

Runtime-coupled: execute вызывает normalize, normalize для compose_context
дёргает fetch_docs через tool_runner.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.security import security_manager
from schemas.agent import AgentAction, ToolMeta, ToolRequest, ToolResponse
from services.agent.formatting import tool_error_action
from services.agent.state import RequestContext

logger = logging.getLogger(__name__)


async def execute_action(
    tool_name: str,
    params: dict[str, Any],
    request_id: str,
    step: int,
    ctx: RequestContext,
    tool_runner,
    settings,
) -> AgentAction | None:
    """Нормализует параметры и выполняет инструмент через ToolRunner."""
    try:
        safe_params = dict(params or {})
        normalized = normalize_tool_params(tool_name, safe_params, ctx, tool_runner, settings)

        # Temporal guard: если запрос содержит даты полностью
        # вне корпуса (июль 2025 — март 2026), возвращаем refusal.
        # Проверяем для всех tools — даты могут прийти из query_plan/signals.
        _guard_params = dict(safe_params)
        # Для search-like tools даты могут быть в filters
        if isinstance(normalized.get("filters"), dict):
            _guard_params.update(normalized["filters"])
        if _guard_params.get("date_from") or _guard_params.get("date_to"):
            refusal = _temporal_guard(tool_name, _guard_params, step=step)
            if refusal is not None:
                return refusal

        # Pipeline-internal tools: payload содержит LLM output или Qdrant docs,
        # которые могут содержать HTML, кавычки, перечисления через ";".
        # Security validation только для user-facing input, не internal data.
        _skip_security = {"rerank", "compose_context", "final_answer", "verify", "fetch_docs"}
        if tool_name not in _skip_security:
            serialized = json.dumps(normalized, ensure_ascii=False, default=str)
            is_valid, violations = security_manager.validate_input(serialized)
            if not is_valid:
                logger.warning(
                    "Security violations in tool params for %s: %s",
                    tool_name, violations,
                )
                return tool_error_action(
                    tool_name=tool_name,
                    params=safe_params,
                    step=step,
                    error="security_violation",
                )

        # Специализированные search tools маппятся на "search" в ToolRunner
        actual_tool = tool_name
        if tool_name in ("temporal_search", "channel_search"):
            actual_tool = "search"

        tool_request = ToolRequest(tool=actual_tool, input=normalized)
        return tool_runner.run(request_id, step, tool_request, deadline=ctx.deadline)
    except Exception as exc:
        logger.error("Ошибка выполнения инструмента %s: %s", tool_name, exc)
        return tool_error_action(
            tool_name=tool_name, params=params, step=step, error=str(exc),
        )


def _temporal_guard(tool_name: str, safe_params: dict[str, Any], step: int = 0) -> AgentAction | None:
    """Temporal guard: refusal если даты полностью вне корпуса."""
    _corpus_min = "2025-07-01"
    _corpus_max = "2026-03-31"
    _date_to = str(safe_params.get("date_to", ""))
    _date_from = str(safe_params.get("date_from", ""))
    if _date_to and _date_to < _corpus_min:
        return AgentAction(
            step=step, tool=tool_name, input=safe_params,
            output=ToolResponse(
                ok=True,
                data={"hits": [], "refusal": "Запрошенный период вне диапазона данных базы (июль 2025 — март 2026)."},
                meta=ToolMeta(took_ms=0),
            ),
        )
    if _date_from and _date_from > _corpus_max:
        return AgentAction(
            step=step, tool=tool_name, input=safe_params,
            output=ToolResponse(
                ok=True,
                data={"hits": [], "refusal": "Запрошенный период вне диапазона данных базы (июль 2025 — март 2026)."},
                meta=ToolMeta(took_ms=0),
            ),
        )
    return None


def normalize_tool_params(
    tool_name: str,
    params: dict[str, Any],
    ctx: RequestContext,
    tool_runner,
    settings,
) -> dict[str, Any]:
    """Нормализует параметры инструментов для совместимости и системных вызовов."""
    normalized = dict(params or {})

    if tool_name == "query_plan":
        normalized.setdefault("query", ctx.query or "")
        return normalized

    # temporal_search и channel_search маппятся на search() с фильтрами
    if tool_name == "temporal_search":
        filters = {}
        if normalized.get("date_from"):
            filters["date_from"] = normalized.pop("date_from")
        if normalized.get("date_to"):
            filters["date_to"] = normalized.pop("date_to")
        normalized["filters"] = filters

    if tool_name == "channel_search":
        filters = {}
        if normalized.get("channel"):
            filters["channel"] = normalized.pop("channel")
        normalized["filters"] = filters

    if tool_name in ("search", "temporal_search", "channel_search"):
        if not normalized.get("queries"):
            if normalized.get("query"):
                normalized["queries"] = [str(normalized.pop("query"))]
            elif ctx.plan_summary and ctx.plan_summary.get("normalized_queries"):
                normalized["queries"] = list(ctx.plan_summary["normalized_queries"])
            elif ctx.query:
                normalized["queries"] = [ctx.query]

        # Оригинальный запрос пользователя в subqueries для BM25 keyword match
        if ctx.query and normalized.get("queries"):
            orig = ctx.query.strip()
            if orig and orig not in normalized["queries"]:
                normalized["queries"].insert(0, orig)

        normalized.setdefault(
            "k",
            (ctx.plan_summary or {}).get("k_per_query", settings.search_k_per_query_default),
        )
        normalized.setdefault("route", "hybrid")

        # Прокидываем metadata_filters из query_plan
        if not normalized.get("filters") and ctx.plan_summary:
            plan_filters = ctx.plan_summary.get("metadata_filters")
            if isinstance(plan_filters, dict):
                clean_filters = {k: v for k, v in plan_filters.items() if v is not None}
                if clean_filters:
                    normalized["filters"] = clean_filters

        return normalized

    # hot_topics: нормализация period из query_signals
    if tool_name == "hot_topics":
        period = normalized.get("period", "this_week")
        if ctx.query_signals and period in ("this_week", "last_week", None, ""):
            sig = ctx.query_signals
            if sig.date_from:
                try:
                    from datetime import datetime as _dt
                    d = _dt.fromisoformat(sig.date_from)
                    iso = d.isocalendar()
                    normalized["period"] = f"{iso[0]}-W{iso[1]:02d}"
                except (ValueError, TypeError):
                    pass
        return normalized

    if tool_name == "rerank":
        normalized.setdefault("query", ctx.query or "")
        hits = normalized.pop("hits", None) or ctx.search_hits
        if not normalized.get("docs") or not any(normalized["docs"]):
            normalized["docs"] = [
                str(hit.get("text") or hit.get("snippet") or "")
                for hit in hits
                if isinstance(hit, dict)
                and (hit.get("text") or hit.get("snippet"))
            ]
        if normalized.get("docs") and not normalized.get("top_n"):
            normalized["top_n"] = min(
                len(normalized["docs"]),
                max(1, int(settings.reranker_top_n)),
            )
        # CRAG-style: cross-encoder как confidence filter, не reranker (DEC-0045).
        # Документы с score < 0 (negative logit = "not relevant") отсекаются.
        # Calibration: keep 92% relevant, remove 55% irrelevant at t=0.0.
        normalized.setdefault("filter_threshold", 0.0)
        return normalized

    if tool_name == "compose_context":
        normalized.pop("hits", None)
        normalized.pop("raw_input", None)
        normalized.pop("hit_ids", None)
        normalized["query"] = ctx.query or ""

        last_hits = ctx.search_hits
        selected_hits: list[dict[str, Any]] = [
            hit for hit in last_hits if isinstance(hit, dict)
        ]

        normalized_docs: list[dict[str, Any]] = []
        missing_ids: list[str] = []
        for doc in selected_hits:
            doc_id = doc.get("id")
            text_value = (
                doc.get("text") or doc.get("snippet")
                or doc.get("meta", {}).get("text")
                or doc.get("metadata", {}).get("text")
                or ""
            )
            if not text_value and doc_id:
                missing_ids.append(str(doc_id))
            normalized_docs.append({
                "id": doc_id,
                "text": text_value,
                "metadata": doc.get("metadata") or doc.get("meta", {}),
                "dense_score": doc.get("dense_score"),
            })

        if missing_ids:
            try:
                fetch_result = tool_runner.run(
                    ctx.request_id or "fetch-docs", ctx.step,
                    ToolRequest(tool="fetch_docs", input={"ids": missing_ids}),
                    deadline=ctx.deadline,
                )
                if fetch_result.output.ok:
                    fetched_docs = {
                        item.get("id"): item
                        for item in fetch_result.output.data.get("docs", [])
                        if isinstance(item, dict) and item.get("id")
                    }
                    for doc in normalized_docs:
                        if not doc.get("text") and doc.get("id") in fetched_docs:
                            fetched = fetched_docs[doc["id"]]
                            doc["text"] = fetched.get("text", "")
                            doc["metadata"] = fetched.get("metadata", doc.get("metadata", {}))
            except Exception as exc:
                logger.warning("fetch_docs during compose_context failed: %s", exc)

        normalized["docs"] = normalized_docs
        normalized.setdefault("max_tokens_ctx", 4000)
        return normalized

    if tool_name == "fetch_docs":
        if "doc_ids" in normalized and "ids" not in normalized:
            normalized["ids"] = normalized.pop("doc_ids")
        return normalized

    if tool_name == "final_answer":
        raw = normalized.pop("raw_input", None)
        if raw and not normalized.get("answer"):
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, dict):
                    normalized.update(parsed)
            except (json.JSONDecodeError, TypeError):
                normalized["answer"] = str(raw)
        return normalized

    if tool_name == "verify":
        if "k" in normalized and "top_k" not in normalized:
            normalized["top_k"] = normalized.pop("k")
        normalized.setdefault("query", ctx.query or "")
        return normalized

    return normalized
