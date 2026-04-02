"""
SPEC-RAG-20c Step 5: Чистые formatting/serialization helpers.

Без side effects, без state mutation. Только преобразование данных.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from schemas.agent import AgentAction, ToolMeta, ToolResponse

logger = logging.getLogger(__name__)


def _fmt_search(d: dict) -> str:
    hits = d.get("hits", [])
    hit_ids = [h.get("id", "?") for h in hits if isinstance(h, dict)]
    return (
        f"Found {len(hits)} documents (total: {d.get('total_found', len(hits))}). "
        f"Route: {d.get('route_used', '?')}. Use these IDs for compose_context: {hit_ids}"
    )


def _fmt_rerank(d: dict) -> str:
    scores = d.get("scores", [])
    score_str = f", scores: [{', '.join(f'{s:.3f}' for s in scores[:5])}]" if scores else ""
    return (
        f"Reranked {d.get('top_n', len(d.get('indices', [])))} documents by relevance{score_str}. "
        f"Call compose_context() to build context from reranked results."
    )


def _fmt_compose(d: dict) -> str:
    coverage = float(d.get("citation_coverage", 0.0) or 0.0)
    return (
        f"Composed context with {len(d.get('citations', []))} citations, "
        f"coverage: {coverage:.2f}, contexts: {len(d.get('contexts', []))}"
    )


def _fmt_verify(d: dict) -> str:
    return (
        f"Verification: {d.get('verified', False)} "
        f"(confidence: {float(d.get('confidence', 0) or 0):.3f}, "
        f"threshold: {d.get('threshold', 0.6)}, docs: {d.get('documents_found', 0)})"
    )


def _fmt_query_plan(d: dict) -> str:
    plan = d.get("plan", {})
    queries = plan.get("normalized_queries", [])
    return f"Plan: {len(queries)} queries, k={plan.get('k_per_query', 0)}, fusion={plan.get('fusion', '?')}"


def _fmt_final_answer(d: dict) -> str:
    return f"Final answer prepared ({len(str(d.get('answer', '')).strip())} chars)"


def _fmt_hot_topics(d: dict) -> str:
    parts = [f"period: {d.get('period', '?')}"]
    if d.get("fallback_used"):
        parts.append(f"(fallback: запрошен {d.get('requested_period')}, показан {d.get('resolved_period')})")
    parts.append(f"posts: {d.get('post_count', 0)}")
    summary = (d.get("summary") or "")[:200]
    if summary:
        parts.append(f"summary: {summary}")
    for t in (d.get("topics") or [])[:5]:
        chs = ",".join((t.get("channels") or [])[:3])
        parts.append(f"- {t.get('label', '?')} (score={t.get('hot_score', 0)}, {t.get('post_count', 0)} posts, channels: {chs})")
    ents = d.get("top_entities") or []
    if ents:
        parts.append("top entities: " + ", ".join(f"{e['entity']}({e['count']})" for e in ents[:5]))
    return "; ".join(parts[:3]) + "\n" + "\n".join(parts[3:])


def _fmt_channel_expertise(d: dict) -> str:
    if d.get("channel"):
        return f"Channel {d['channel']}: authority={d.get('authority_score', 0)}, summary: {(d.get('profile_summary') or '')[:200]}"
    channels = d.get("channels") or []
    metric = d.get("metric", "authority")
    return (
        f"Found {len(channels)} channels for topic='{d.get('topic', '')}', metric={metric}: "
        + ", ".join(f"{c.get('channel', '?')}({c.get(metric + '_score', 0)})" for c in channels[:5])
    )


def _fmt_cross_channel(d: dict) -> str:
    groups = d.get("groups", [])
    channels = [g.get("channel", "?") for g in groups]
    return (
        f"Compared {len(groups)} channels on topic='{d.get('topic', '?')}': "
        f"{', '.join(channels[:5])}. Total hits: {d.get('total_found', 0)}"
    )


def _fmt_summarize_channel(d: dict) -> str:
    return (
        f"Channel {d.get('channel', '?')} ({d.get('period', '?')}): "
        f"{d.get('post_count', 0)} posts"
    )


def _fmt_list_channels(d: dict) -> str:
    channels = d.get("channels", [])
    total = d.get("total", len(channels))
    top = ", ".join(f"{c.get('channel', '?')}({c.get('count', 0)})" for c in channels[:5])
    return f"{total} channels: {top}"


def _fmt_related_posts(d: dict) -> str:
    hits = d.get("hits", [])
    return f"Found {len(hits)} related posts for source={d.get('source_id', '?')}"


def _fmt_entity_tracker(d: dict) -> str:
    mode = d.get("mode", "?")
    summary = d.get("summary", "")
    return f"entity_tracker({mode}): {summary[:200]}" if summary else f"entity_tracker({mode}): {len(d.get('data', []))} items"


def _fmt_arxiv_tracker(d: dict) -> str:
    mode = d.get("mode", "?")
    summary = d.get("summary", "")
    return f"arxiv_tracker({mode}): {summary[:200]}" if summary else f"arxiv_tracker({mode}): {len(d.get('data', []))} items"


def _fmt_default(d: dict) -> str:
    parts = []
    for key, value in d.items():
        if key in {"error", "result", "answer", "route", "prompt"}:
            parts.append(f"{key}: {value}")
        elif isinstance(value, list):
            parts.append(f"{key}: {len(value)}")
        elif isinstance(value, dict):
            parts.append(f"{key}: object")
        else:
            parts.append(f"{key}: {str(value)[:100]}")
    return "; ".join(parts) if parts else str(d)


_FORMATTERS: dict[str, Any] = {
    "search": _fmt_search,
    "temporal_search": _fmt_search,
    "channel_search": _fmt_search,
    "cross_channel_compare": _fmt_cross_channel,
    "summarize_channel": _fmt_summarize_channel,
    "list_channels": _fmt_list_channels,
    "related_posts": _fmt_related_posts,
    "entity_tracker": _fmt_entity_tracker,
    "arxiv_tracker": _fmt_arxiv_tracker,
    "rerank": _fmt_rerank,
    "compose_context": _fmt_compose,
    "verify": _fmt_verify,
    "query_plan": _fmt_query_plan,
    "final_answer": _fmt_final_answer,
    "hot_topics": _fmt_hot_topics,
    "channel_expertise": _fmt_channel_expertise,
}


def format_observation(tool_response: ToolResponse, tool_name: str = "") -> str:
    """Форматирует результат инструмента для observation SSE."""
    if not tool_response.ok:
        return f"Ошибка: {tool_response.meta.error or 'Неизвестная ошибка'}"
    if not tool_response.data:
        return "Результат получен (пустые данные)"
    try:
        formatter = _FORMATTERS.get(tool_name, _fmt_default)
        return formatter(tool_response.data)
    except Exception as exc:
        logger.warning("Failed to format observation for %s: %s", tool_name, exc)
        return str(tool_response.data)[:500]


def extract_tool_calls(
    assistant_message: dict[str, Any],
    visible_tools: set | None = None,
) -> list[dict[str, Any]]:
    """Приводит tool_calls llama-server к единому внутреннему формату.

    FIX-04: если visible_tools задан, отбрасывает tool names вне visible set.
    """
    raw_tool_calls = assistant_message.get("tool_calls") or []
    normalized_calls: list[dict[str, Any]] = []

    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue

        fn = item.get("function")
        function_block: dict = fn if isinstance(fn, dict) else item
        tool_name = function_block.get("name")

        # FIX-04: whitelist по visible set
        if visible_tools and tool_name and tool_name not in visible_tools:
            logger.warning(
                "LLM вызвала tool '%s' вне visible set %s — пропускаем",
                tool_name, visible_tools,
            )
            continue
        raw_arguments = function_block.get("arguments", {})

        if isinstance(raw_arguments, str):
            try:
                loaded = json.loads(raw_arguments)
                parsed_arguments = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError as jde:
                logger.warning(
                    "JSON parse failed for %s args (len=%d): %s | first 200: %s",
                    tool_name, len(raw_arguments), jde, raw_arguments[:200],
                )
                parsed_arguments = {"raw_input": raw_arguments}
        elif isinstance(raw_arguments, dict):
            parsed_arguments = raw_arguments
        else:
            parsed_arguments = {}

        if tool_name:
            normalized_calls.append(
                {
                    "id": item.get("id"),
                    "name": tool_name,
                    "arguments": parsed_arguments,
                }
            )

    return normalized_calls


def assistant_message_for_history(assistant_message: dict[str, Any]) -> dict[str, Any]:
    """Сохраняет assistant message в chat history без искажения tool_calls.

    Qwen3 jinja template с --jinja трактует пустой content в assistant
    message с tool_calls как "response prefill", что конфликтует с
    enable_thinking. Поэтому content="" не добавляем.
    """
    message: dict[str, Any] = {"role": "assistant"}
    content = assistant_message.get("content")
    if content:
        message["content"] = content
    if "tool_calls" in assistant_message:
        message["tool_calls"] = assistant_message.get("tool_calls")
    return message


def serialize_tool_payload(
    payload: dict[str, Any], max_chars: int = 8000, tool_name: str = "",
) -> str:
    """Безопасно сериализует payload инструмента для chat history.

    SPEC-RAG-20d: три принципа сериализации:
    1. compose_context.prompt — source of truth для LLM, НЕ обрезается
       (уже ограничен max_tokens_ctx=4000 в compose_context tool).
    2. search hits — убираем полные тексты (они в ctx.search_hits для compose),
       оставляем id + score + snippet[:200] + meta.
    3. contexts/docs — дубликаты prompt, убираем всегда.
    """
    _SEARCH_TOOLS = {"search", "temporal_search", "channel_search",
                     "cross_channel_compare", "summarize_channel"}
    try:
        trimmed = dict(payload)
        # Убираем поля-дубликаты prompt
        trimmed.pop("contexts", None)
        trimmed.pop("docs", None)

        # Search-like tools: strip full texts из hits, оставляем lightweight metadata.
        # Полные тексты остаются в ctx.search_hits → compose_context их использует.
        if tool_name in _SEARCH_TOOLS and isinstance(trimmed.get("hits"), list):
            trimmed["hits"] = [
                {
                    "id": h.get("id"),
                    "score": h.get("score"),
                    "dense_score": h.get("dense_score"),
                    "snippet": (h.get("snippet") or (h.get("text") or "")[:200]),
                    "meta": h.get("meta", {}),
                }
                for h in trimmed["hits"]
                if isinstance(h, dict)
            ]

        # compose_context: prompt уже budget-controlled (max_tokens_ctx=4000),
        # не обрезаем. Лимит на финальный JSON поднимаем до 24000.
        if tool_name == "compose_context":
            effective_limit = max(max_chars, 24000)
        else:
            effective_limit = max_chars

        if "prompt" in trimmed:
            prompt_len = len(str(trimmed["prompt"]))
            logger.info(
                "tool_payload tool=%s prompt_len=%d effective_limit=%d",
                tool_name, prompt_len, effective_limit,
            )
            # Обрезаем prompt только для НЕ-compose tools
            if tool_name != "compose_context" and prompt_len > effective_limit:
                trimmed["prompt"] = str(trimmed["prompt"])[:effective_limit] + "… [ОБРЕЗАНО]"

        serialized = json.dumps(trimmed, ensure_ascii=False, default=str)
        if len(serialized) > effective_limit:
            serialized = serialized[:effective_limit] + "… [ОБРЕЗАНО]}"
        return serialized
    except Exception:
        return json.dumps({"error": "serialization_failed"}, ensure_ascii=False)


def tool_message_for_history(
    tool_call: dict[str, Any],
    tool_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Сериализует результат инструмента в стандартный `role=tool` message."""
    message: dict[str, Any] = {
        "role": "tool",
        "name": tool_name,
        "content": serialize_tool_payload(payload, tool_name=tool_name),
    }
    if tool_call.get("id"):
        message["tool_call_id"] = tool_call["id"]
    return message


def trim_messages(
    messages: list[dict[str, Any]], max_chars: int = 30000
) -> list[dict[str, Any]]:
    """Обрезает messages чтобы уложиться в context window LLM.

    SPEC-RAG-20d: trim по atomic blocks (assistant+tool пары),
    всегда pin-ит последний compose_context block.
    Стратегия:
    1. Сохраняем system + user (первые 2 сообщения)
    2. Группируем остальные в atomic blocks (assistant(tool_calls) + tool(responses))
    3. Всегда сохраняем последний block с compose_context
    4. Из остальных оставляем последние N blocks по бюджету
    """
    total = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in messages)
    if total <= max_chars:
        return messages

    head = messages[:2]
    tail = messages[2:]

    # Группируем tail в atomic blocks: assistant(tool_calls) + следующие tool messages
    blocks: list[list[dict[str, Any]]] = []
    current_block: list[dict[str, Any]] = []
    for msg in tail:
        role = msg.get("role", "")
        if role == "assistant" and current_block:
            blocks.append(current_block)
            current_block = [msg]
        else:
            current_block.append(msg)
    if current_block:
        blocks.append(current_block)

    # Найти последний block с compose_context (pin его)
    pinned_idx: int = -1
    for i in range(len(blocks) - 1, -1, -1):
        for msg in blocks[i]:
            if msg.get("role") == "tool" and msg.get("name") == "compose_context":
                pinned_idx = i
                break
        if pinned_idx >= 0:
            break

    head_size = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in head)
    budget = max_chars - head_size

    # Если есть pinned block — резервируем место для него
    pinned_block: list[dict[str, Any]] = []
    if pinned_idx >= 0:
        pinned_block = blocks[pinned_idx]
        pinned_size = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in pinned_block)
        budget -= pinned_size

    # Оставляем последние blocks по бюджету (исключая pinned)
    kept_blocks: list[list[dict[str, Any]]] = []
    for block in reversed(blocks):
        if block is pinned_block:
            continue
        block_size = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in block)
        if budget - block_size < 0 and kept_blocks:
            break
        kept_blocks.append(block)
        budget -= block_size
    kept_blocks.reverse()

    # Собираем: head + kept_blocks (в правильном порядке) + pinned_block (если не в kept)
    result = list(head)
    for block in kept_blocks:
        result.extend(block)
    if pinned_block and pinned_block not in kept_blocks:
        result.extend(pinned_block)

    logger.debug(
        "Trimmed messages: %d → %d (%d blocks, pinned=%s, from %d chars to ~%d)",
        len(messages), len(result), len(kept_blocks),
        pinned_idx >= 0, total, max_chars - budget,
    )
    return result


def tool_error_action(
    tool_name: str,
    params: dict[str, Any],
    step: int,
    error: str,
) -> AgentAction:
    """Строит псевдо-результат инструмента при локальной ошибке вызова."""
    return AgentAction(
        step=step,
        tool=tool_name,
        input=dict(params or {}),
        output=ToolResponse(
            ok=False,
            data={},
            meta=ToolMeta(took_ms=0, error=error),
        ),
    )
