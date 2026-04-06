"""
SPEC-RAG-20c Step 3: Agent state — per-request isolation via ContextVar.

AgentState: dynamic agent state across steps.
RequestContext: per-request context (ContextVar isolation, SPEC-RAG-17 FIX-01).
apply_action_state: state update after tool execution.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class AgentState:
    """Tracks dynamic state of the agent between steps."""

    def __init__(self) -> None:
        self.coverage: float = 0.0
        self.refinement_count: int = 0
        self.max_refinements: int = 2
        self.low_coverage_disclaimer: bool = False
        self.search_count: int = 0
        self.navigation_answered: bool = False  # list_channels дал ответ
        self.analytics_done: bool = False  # entity_tracker/arxiv_tracker ответили
        # Adaptive retrieval state
        self.strategy: str = "broad"
        self.applied_filters: dict[str, Any] = {}
        self.routing_source: str = "default"


@dataclass
class RequestContext:
    """Per-request state — изолирован от других запросов через ContextVar.

    FIX-01 (SPEC-RAG-17): вместо хранения state в self AgentService (singleton),
    каждый запрос получает свой RequestContext.
    """

    request_id: str
    query: str
    original_query: str
    query_signals: Any = None
    agent_state: AgentState = field(default_factory=AgentState)
    step: int = 1
    search_hits: list[dict[str, Any]] = field(default_factory=list)
    search_route: str | None = None
    plan_summary: dict[str, Any] | None = None
    compose_citations: list[dict[str, Any]] = field(default_factory=list)
    coverage_score: float = 0.0
    deadline: float | None = None  # FIX-08: wall-clock deadline (monotonic)
    final_answer_text: str | None = None  # SPEC-RAG-20b: для trace output
    # SPEC-RAG-20d: token usage aggregation для observability
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # LANCER-style: nuggets не покрытые документами — для targeted refinement
    uncovered_nuggets: list[str] = field(default_factory=list)


_request_ctx: ContextVar[RequestContext | None] = ContextVar(
    "agent_request_ctx", default=None
)


def _adaptive_ce_filter(
    hits: list[dict[str, Any]],
    gap_threshold: float = 2.0,
    min_docs: int = 5,
    floor_score: float = -2.0,
) -> list[dict[str, Any]]:
    """Адаптивный CE фильтр: gap detection + top-K guarantee + floor.

    Гибридная эвристика (ablation phase 3):
    1. Gap detection: ищем резкое падение CE score (> gap_threshold).
       Обрезаем после gap — отделяем "cluster релевантных" от "хвост шума".
    2. Top-K guarantee: если после gap < min_docs, расширяем до min_docs.
       LLM всегда получает достаточно контекста для ответа.
    3. Adaptive floor: если < 3 docs с score > 0, берём до floor_score.
       Спасает edge cases где мало релевантных docs.

    Вход: hits отсортированы по rerank_score desc (CE re-sort уже применён).
    """
    if not hits:
        return hits

    scores = [h.get("rerank_score", 0.0) for h in hits]

    # Шаг 1: Gap detection — ищем резкий обрыв CE score
    gap_cut = len(hits)
    for i in range(len(scores) - 1):
        if scores[i] - scores[i + 1] > gap_threshold:
            gap_cut = i + 1
            break

    # Шаг 2: Positive count — сколько docs с CE > 0
    positive_count = sum(1 for s in scores if s > 0)

    # Шаг 3: Adaptive cut
    if positive_count >= min_docs:
        # Достаточно хороших docs — берём до gap или все positive, что больше
        cut = max(gap_cut, positive_count)
    elif positive_count > 0:
        # Мало хороших — берём все positive + top-K guarantee до min_docs
        cut = max(positive_count, min(min_docs, gap_cut))
    else:
        # Нет ни одного positive — floor: берём до floor_score
        floor_cut = sum(1 for s in scores if s >= floor_score)
        cut = max(min(min_docs, len(hits)), floor_cut)

    return hits[:cut]


def apply_action_state(ctx: RequestContext, action) -> None:
    """Сохраняет детерминированное состояние после успешных tool calls.

    Принимает ctx явно (не через self._ctx) — тестируемо без AgentService.
    action: AgentAction (не импортируем напрямую чтобы избежать circular).
    """
    if not action.output.ok:
        return

    if action.tool == "query_plan":
        ctx.plan_summary = action.output.data.get("plan") or {}
        return

    if action.tool == "list_channels":
        ctx.agent_state.navigation_answered = True
        return

    # SPEC-RAG-15/16: analytics tools (все устанавливают analytics_done)
    if action.tool in ("entity_tracker", "arxiv_tracker", "hot_topics", "channel_expertise"):
        ctx.agent_state.analytics_done = True
        # arxiv_tracker(lookup) возвращает hits — search-like, нужен rerank/compose
        if action.tool == "arxiv_tracker" and action.output.data.get("hits"):
            ctx.search_hits = list(action.output.data.get("hits", []))
            ctx.agent_state.search_count += 1
        return

    if action.tool in ("search", "temporal_search", "channel_search",
                       "cross_channel_compare", "summarize_channel"):
        ctx.search_hits = list(action.output.data.get("hits", []) or [])
        ctx.search_route = action.output.data.get("route_used")
        ctx.agent_state.search_count += 1
        # Adaptive retrieval state tracking
        ctx.agent_state.strategy = action.output.data.get("strategy", "broad")
        ctx.agent_state.routing_source = action.output.data.get("routing_source", "default")
        # LANCER: если query_plan не вызывался, подхватываем search subqueries как implicit nuggets
        if not ctx.plan_summary:
            search_queries = action.input.get("queries") if isinstance(action.input, dict) else None
            if isinstance(search_queries, list) and len(search_queries) > 1:
                ctx.plan_summary = {"normalized_queries": search_queries}
        logger.info(
            "Agent search | tool=%s | strategy=%s | routing_source=%s | hits=%d",
            action.tool,
            ctx.agent_state.strategy,
            ctx.agent_state.routing_source,
            len(ctx.search_hits),
        )
        return

    if action.tool == "rerank":
        indices = action.output.data.get("indices") or []
        scores = action.output.data.get("scores") or []
        if not isinstance(indices, list) or not ctx.search_hits:
            return

        # CE re-sort + adaptive filter.
        # 1. Помечаем docs rerank_score
        # 2. Re-sort по CE score desc (ablation: expected doc 4→1)
        # 3. Adaptive filter: gap detection + top-K guarantee + floor
        passed_indices: set[int] = set()
        score_map: dict[int, float] = {}
        for position, raw_idx in enumerate(indices):
            if isinstance(raw_idx, int) and 0 <= raw_idx < len(ctx.search_hits):
                passed_indices.add(raw_idx)
                if position < len(scores):
                    score_map[raw_idx] = scores[position]

        kept_hits: list[dict[str, Any]] = []
        for idx, hit in enumerate(ctx.search_hits):
            if idx in passed_indices:
                h = dict(hit)
                h["rerank_score"] = score_map.get(idx, 0.0)
                kept_hits.append(h)

        # Re-sort по CE score desc
        kept_hits.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)

        # Adaptive CE filter: gap detection + top-K guarantee + floor
        kept_hits = _adaptive_ce_filter(kept_hits)

        if kept_hits:
            original_count = len(ctx.search_hits)
            logger.info(
                "CE adaptive filter: %d → %d docs, re-sorted by CE score",
                original_count, len(kept_hits),
            )
            ctx.search_hits = kept_hits
        return

    if action.tool == "compose_context":
        ctx.compose_citations = list(
            action.output.data.get("citations", []) or []
        )
        ctx.coverage_score = float(
            action.output.data.get("citation_coverage", 0.0) or 0.0
        )
