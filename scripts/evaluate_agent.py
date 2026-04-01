#!/usr/bin/env python3
"""
Evaluation pipeline v2 для ReAct-агента (SPEC-RAG-14).

Поддерживает:
- Legacy формат (v1/v2/v3) и golden формат (key_tools, forbidden_tools, judge)
- Tool selection tracking из SSE events (tool_invoked + step_started)
- LLM judge через Claude API (factual correctness + usefulness)
- Failure attribution (tool_hidden, tool_wrong, retrieval_empty, generation_wrong)
- JSON + Markdown отчёты с failure breakdown
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import httpx

DEFAULT_DATASET = Path("datasets/eval_golden_v1.json")
FALLBACK_DATASETS = [
    Path("datasets/eval_dataset_v3.json"),
    Path("datasets/eval_dataset.json"),
    Path("datasets/eval_questions.json"),
]
DEFAULT_AGENT_URL = "http://localhost:8001/v1/agent/stream"
DEFAULT_QA_URL = "http://localhost:8001/v1/qa"

logger = logging.getLogger(__name__)

SEARCH_TYPE_TOOLS = {"search", "temporal_search", "channel_search", "related_posts"}
ANALYTICS_TOOLS = {"entity_tracker", "arxiv_tracker", "hot_topics", "channel_expertise"}


# ─── Failure types (SPEC-RAG-14 §3.4) ────────────────────────────


class FailureType(str, Enum):
    TOOL_HIDDEN = "tool_hidden"
    TOOL_WRONG = "tool_selected_wrong"
    TOOL_FAILED = "tool_execution_failed"
    RETRIEVAL_EMPTY = "retrieval_empty"
    GENERATION_WRONG = "generation_wrong"
    REFUSAL_WRONG = "refusal_wrong"
    JUDGE_UNCERTAIN = "judge_uncertain"


# ─── Dataset model ────────────────────────────────────────────────


@dataclass
class EvalItem:
    """Единица датасета — поддерживает и legacy и golden формат."""

    id: str
    version: str
    query: str
    category: str
    answerable: bool
    expected_answer: Optional[str] = None
    notes: Optional[str] = None

    # Legacy
    expected_documents: List[str] = field(default_factory=list)

    # Golden format (SPEC-RAG-14)
    key_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    acceptable_alternatives: List[str] = field(default_factory=list)
    expected_refusal: bool = False
    refusal_reason: Optional[str] = None
    source_post_ids: List[str] = field(default_factory=list)
    calibration: bool = False
    difficulty: str = "medium"
    eval_mode: str = "retrieval_evidence"
    required_claims: List[str] = field(default_factory=list)
    expected_entities: List[str] = field(default_factory=list)
    expected_topics: List[str] = field(default_factory=list)
    expected_channels: List[str] = field(default_factory=list)
    acceptable_evidence_sets: List[List[str]] = field(default_factory=list)
    strict_anchor_recall_eligible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_golden(self) -> bool:
        """Auto-detect golden vs legacy формат."""
        return bool(self.key_tools) or self.expected_refusal or self.eval_mode != "retrieval_evidence"

    def dataset_contract(self) -> Dict[str, Any]:
        """Нормализованный контракт вопроса для offline judge packet."""
        return {
            "id": self.id,
            "version": self.version,
            "eval_mode": self.eval_mode,
            "answerable": self.answerable,
            "expected_refusal": self.expected_refusal,
            "refusal_reason": self.refusal_reason,
            "key_tools": self.key_tools,
            "forbidden_tools": self.forbidden_tools,
            "acceptable_alternatives": self.acceptable_alternatives,
            "required_claims": self.required_claims,
            "expected_answer": self.expected_answer,
            "expected_entities": self.expected_entities,
            "expected_topics": self.expected_topics,
            "expected_channels": self.expected_channels,
            "source_post_ids": self.source_post_ids,
            "acceptable_evidence_sets": self.acceptable_evidence_sets,
            "strict_anchor_recall_eligible": self.strict_anchor_recall_eligible,
            "calibration": self.calibration,
            "difficulty": self.difficulty,
            "notes": self.notes,
            "metadata": self.metadata,
        }


def infer_eval_mode(record: Dict[str, Any]) -> str:
    """Пытается вывести eval_mode для legacy/golden_v1 записей."""
    explicit = record.get("eval_mode")
    if explicit:
        return str(explicit)

    category = str(record.get("category", ""))
    if category == "navigation":
        return "navigation"
    if category == "negative_refusal":
        return "refusal"

    key_tools = set(record.get("key_tools") or [])
    if key_tools & ANALYTICS_TOOLS:
        return "analytics"

    return "retrieval_evidence"


def load_dataset(path: Path) -> List[EvalItem]:
    """Загрузка датасета с auto-detect формата (golden vs legacy)."""
    source_path = path
    if not path.exists():
        for fallback in FALLBACK_DATASETS:
            if fallback.exists():
                logger.warning("Датасет %s не найден, fallback → %s", path, fallback)
                source_path = fallback
                break
        else:
            raise FileNotFoundError(f"Датасет не найден: {path}")

    with source_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, dict) and "questions" in payload:
        records = payload["questions"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Неизвестный формат: {source_path}")

    items: List[EvalItem] = []
    for record in records:
        eval_mode = infer_eval_mode(record)
        strict_recall_eligible = record.get("strict_anchor_recall_eligible")
        if strict_recall_eligible is None:
            strict_recall_eligible = bool(record.get("source_post_ids")) and eval_mode == "retrieval_evidence"

        item = EvalItem(
            id=str(record.get("id") or len(items) + 1),
            version=str(record.get("version", "1.0")),
            query=record["query"],
            category=record.get("category", "unknown"),
            answerable=bool(record.get("answerable", True)),
            expected_answer=record.get("expected_answer"),
            notes=record.get("notes"),
            expected_documents=list(record.get("expected_documents") or record.get("source_post_ids") or []),
            key_tools=list(record.get("key_tools", [])),
            forbidden_tools=list(record.get("forbidden_tools", [])),
            acceptable_alternatives=list(record.get("acceptable_alternatives", [])),
            expected_refusal=bool(record.get("expected_refusal", False)),
            refusal_reason=record.get("refusal_reason"),
            source_post_ids=list(record.get("source_post_ids", [])),
            calibration=bool(record.get("calibration", False)),
            difficulty=record.get("difficulty", "medium"),
            eval_mode=eval_mode,
            required_claims=list(record.get("required_claims", [])),
            expected_entities=list(record.get("expected_entities", [])),
            expected_topics=list(record.get("expected_topics", [])),
            expected_channels=list(record.get("expected_channels") or record.get("source_channels") or []),
            acceptable_evidence_sets=[
                list(evidence_set) for evidence_set in record.get("acceptable_evidence_sets", [])
            ],
            strict_anchor_recall_eligible=bool(strict_recall_eligible),
            metadata=dict(record.get("metadata", {})),
        )
        items.append(item)

    golden_count = sum(1 for i in items if i.is_golden)
    logger.info(
        "Загружено %d вопросов из %s (%d golden, %d legacy)",
        len(items), source_path, golden_count, len(items) - golden_count,
    )
    return items


def compact_thought(text: str, limit: int = 200) -> Optional[str]:
    """Очищает thought event до компактного judge-friendly вида."""
    cleaned = (text or "").replace("</think>", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return None
    return cleaned[:limit]


def normalize_citation(citation: Dict[str, Any]) -> Dict[str, Any]:
    """Приводит citation к компактному стабильному формату."""
    metadata = citation.get("metadata", {}) if isinstance(citation, dict) else {}
    channel = (
        citation.get("channel")
        or metadata.get("channel")
        or metadata.get("source")
        or citation.get("source")
    )
    message_id = citation.get("message_id") or metadata.get("message_id")
    url = citation.get("url") or metadata.get("url")
    return {
        "id": citation.get("id") if isinstance(citation, dict) else None,
        "channel": channel,
        "message_id": message_id,
        "url": url,
        "score": citation.get("score") if isinstance(citation, dict) else None,
        "text_excerpt": citation.get("text_excerpt") if isinstance(citation, dict) else None,
        "source": citation.get("source") if isinstance(citation, dict) else None,
    }


def extract_doc_ids_from_observation(content: str) -> List[str]:
    """Пытается извлечь doc IDs из search observation строки."""
    match = re.search(r"Use these IDs for compose_context:\s*(\[[^\]]*\])", content or "")
    if not match:
        return []
    try:
        ids = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(ids, list):
        return []
    return [str(doc_id) for doc_id in ids if isinstance(doc_id, (str, int))]


def extract_search_docs_from_observation(tool_name: str, content: str) -> List[Dict[str, Any]]:
    """Best-effort извлечение retrieved docs из observation без новых SSE событий."""
    if tool_name not in SEARCH_TYPE_TOOLS:
        return []
    doc_ids = extract_doc_ids_from_observation(content)
    docs: List[Dict[str, Any]] = []
    for doc_id in doc_ids[:20]:
        channel = None
        message_id = None
        if ":" in doc_id:
            channel, message_id = doc_id.split(":", 1)
        docs.append(
            {
                "id": doc_id,
                "channel": channel,
                "message_id": message_id,
                "text_excerpt": None,
                "score": None,
                "source": "sse_observation",
            }
        )
    return docs


# ─── SSE parsing ──────────────────────────────────────────────────


def iter_sse_events(response: httpx.Response) -> Iterator[Tuple[str, str]]:
    """Парсинг SSE потока."""
    event_type: Optional[str] = None
    data_lines: List[str] = []

    for raw_line in response.iter_lines():
        if raw_line is None:
            continue
        line = raw_line.strip("\r")
        if not line:
            if event_type and data_lines:
                yield event_type, "\n".join(data_lines)
            event_type = None
            data_lines = []
            continue
        if line.startswith(":") or line.startswith("retry:"):
            continue
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
        else:
            data_lines.append(line.strip())

    if event_type and data_lines:
        yield event_type, "\n".join(data_lines)


# ─── Tool selection metrics (SPEC-RAG-14 §3.3) ───────────────────


def compute_key_tool_accuracy(
    predicted_tools: List[str],
    item: EvalItem,
) -> Optional[float]:
    """
    Binary whitelist: key_tools ∪ acceptable_alternatives.
    Forbidden = hard 0.
    Возвращает None для legacy вопросов (без key_tools).
    """
    if not item.key_tools:
        return None

    key_set = set(item.key_tools)
    alternatives = set(item.acceptable_alternatives)
    forbidden = set(item.forbidden_tools)
    predicted_set = set(predicted_tools)

    # Forbidden → 0
    if forbidden & predicted_set:
        return 0.0

    # Hit = agent вызвал хотя бы один из whitelist
    whitelist = key_set | alternatives
    return 1.0 if (whitelist & predicted_set) else 0.0


# ─── Failure attribution (SPEC-RAG-14 §3.4) ──────────────────────


def classify_failure(
    item: EvalItem,
    agent_result: Dict[str, Any],
    factual_score: Optional[float],
    usefulness_score: Optional[float],
    key_tool_acc: Optional[float],
) -> Optional[str]:
    """
    Классификация причины ошибки.
    Trigger: factual < 0.5 or usefulness == 0 or key_tool == 0.
    """
    if agent_result.get("error"):
        return FailureType.TOOL_FAILED

    # Operational error surfacing — runtime error замаскирован в final answer
    answer_text = (agent_result.get("answer") or "").lower()
    _error_markers = [
        "client error", "http error", "traceback", "exception",
        "произошла ошибка", "ошибка при обработке", "error 4", "error 5",
        "connectionerror", "timeout",
    ]
    if any(marker in answer_text for marker in _error_markers):
        return FailureType.TOOL_FAILED

    # Judge uncertain — judge вернул error/None на answerable вопросе
    if (
        item.answerable
        and item.is_golden
        and factual_score is None
        and usefulness_score is None
        and key_tool_acc is None
    ):
        return FailureType.JUDGE_UNCERTAIN

    # Refusal check — выполняется ВСЕГДА для refusal вопросов, независимо от judge
    if item.expected_refusal:
        answer = agent_result.get("answer", "")
        refusal_markers = ["не найд", "нет информации", "не содержит", "отсутствует", "не могу", "нет данных", "не обнаружен", "обнаружено в базе", "нет в базе", "вне периода", "вне диапазона", "не упоминается", "не существует", "не было"]
        is_refusal = any(m in answer.lower() for m in refusal_markers) if answer else True
        if not is_refusal:
            return FailureType.REFUSAL_WRONG
        return None  # правильно отказался

    if not item.answerable:
        answer = agent_result.get("answer", "")
        if answer and len(answer) > 50:  # содержательный ответ на unanswerable
            return FailureType.REFUSAL_WRONG
        return None

    should_trigger = (
        (factual_score is not None and factual_score < 0.5)
        or (usefulness_score is not None and usefulness_score == 0)
        or (key_tool_acc is not None and key_tool_acc == 0.0)
    )
    if not should_trigger:
        return None

    tools_invoked = agent_result.get("tools_invoked", [])
    visible_tools_all = agent_result.get("visible_tools_history", [])

    # Tool hidden — key tool не был в visible set
    if item.key_tools and visible_tools_all:
        all_visible = set()
        for vis in visible_tools_all:
            all_visible.update(vis)
        key_set = set(item.key_tools)
        if key_set and not (key_set & all_visible):
            return FailureType.TOOL_HIDDEN

    # 3. Tool wrong — key tool был видим, но agent не вызвал
    if item.key_tools and key_tool_acc == 0.0:
        return FailureType.TOOL_WRONG

    # 4. Retrieval empty
    citation_hits = agent_result.get("citation_hits", [])
    if not citation_hits and item.answerable:
        return FailureType.RETRIEVAL_EMPTY

    # 5. Generation wrong — docs найдены, но ответ плохой
    if factual_score is not None and factual_score < 0.5:
        return FailureType.GENERATION_WRONG

    return None


# ─── LLM Judge (SPEC-RAG-14 §3.3) ───────────────────────────────


class LangfuseTraceExporter:
    """Экспорт traces из Langfuse для offline judge review."""

    def __init__(
        self,
        host: str = "http://localhost:3100",
        public_key: str = "pk-lf-rag-app-dev",
        secret_key: str = "sk-lf-rag-app-dev",
    ):
        self.host = host.rstrip("/")
        self._client = httpx.Client(
            base_url=self.host,
            auth=(public_key, secret_key),
            timeout=30.0,
        )

    def fetch_recent_traces(self, limit: int = 36, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Получает последние traces из Langfuse."""
        params = {"limit": limit, "orderBy": "timestamp.desc"}
        try:
            resp = self._client.get("/api/public/traces", params=params)
            resp.raise_for_status()
            traces = resp.json().get("data", [])
            if tags:
                traces = [t for t in traces if any(tag in (t.get("tags") or []) for tag in tags)]
            return traces
        except Exception as exc:
            logger.error("Langfuse fetch_recent_traces failed: %s", exc)
            return []

    def fetch_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Получает полный trace с observations."""
        try:
            resp = self._client.get(f"/api/public/traces/{trace_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Langfuse fetch_trace(%s) failed: %s", trace_id, exc)
            return None

    def extract_trace_data(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Извлекает ключевые данные из trace для offline judge."""
        trace_input = trace.get("input") or "{}"
        trace_output = trace.get("output") or "{}"

        # Parse escaped JSON
        if isinstance(trace_input, str):
            try:
                trace_input = json.loads(trace_input)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(trace_output, str):
            try:
                trace_output = json.loads(trace_output)
            except (json.JSONDecodeError, TypeError):
                pass

        query = trace_input.get("query", "") if isinstance(trace_input, dict) else ""
        answer = trace_output.get("answer", "") if isinstance(trace_output, dict) else ""

        # Извлекаем tool calls и observations из trace observations
        observations = trace.get("observations") or []
        tool_calls = []
        llm_generations = []

        for obs in observations:
            obs_type = obs.get("type", "")
            name = obs.get("name", "")

            if obs_type == "TOOL":
                tool_calls.append({
                    "name": name,
                    "latency": obs.get("latency"),
                    "toolCalls": obs.get("toolCalls", []),
                })
            elif obs_type == "GENERATION":
                gen_data = {
                    "name": name,
                    "model": obs.get("model"),
                    "latency": obs.get("latency"),
                    "toolCalls": obs.get("toolCalls", []),
                    "toolCallNames": obs.get("toolCallNames", []),
                    "usageDetails": obs.get("usageDetails", {}),
                }
                llm_generations.append(gen_data)

        # Извлекаем final_answer toolCall (содержит полный ответ с sources)
        final_answer_raw = ""
        for gen in llm_generations:
            if "final_answer" in gen.get("toolCallNames", []):
                for tc in gen.get("toolCalls", []):
                    if isinstance(tc, str):
                        try:
                            tc_parsed = json.loads(tc)
                        except json.JSONDecodeError:
                            continue
                    else:
                        tc_parsed = tc
                    args = tc_parsed.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(args, dict) and args.get("answer"):
                        final_answer_raw = args["answer"]

        return {
            "trace_id": trace.get("id"),
            "query": query,
            "answer": answer or final_answer_raw,
            "latency": trace.get("latency"),
            "tool_calls": tool_calls,
            "llm_generations": llm_generations,
            "total_input_tokens": sum(
                g.get("usageDetails", {}).get("input", 0) or 0
                for g in llm_generations
            ),
            "total_output_tokens": sum(
                g.get("usageDetails", {}).get("output", 0) or 0
                for g in llm_generations
            ),
        }

    def close(self):
        self._client.close()


# ─── Utility ──────────────────────────────────────────────────────


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return fmean(values) if values else None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_vals[int(rank)]
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * (rank - lower)


def fmt(v: Optional[float], decimals: int = 3) -> str:
    """Форматирование float для отчёта."""
    return f"{v:.{decimals}f}" if v is not None else "N/A"


# ─── Runner ───────────────────────────────────────────────────────


class AgentEvaluationRunner:
    """Оркестрация eval pipeline v2 (SPEC-RAG-14)."""

    def __init__(
        self,
        dataset: List[EvalItem],
        *,
        agent_url: str,
        qa_url: str,
        collection: Optional[str],
        max_steps: int,
        planner_enabled: bool,
        agent_timeout: float,
        baseline_timeout: float,
        api_key: Optional[str],
        agent_retries: int,
        baseline_retries: int,
        judge: Optional[Any] = None,  # deprecated, kept for backward compat
        dry_run: bool = False,
        limit: int = 0,
        run_baseline: bool = False,
    ) -> None:
        self.dataset = dataset
        self.agent_url = agent_url
        self.qa_url = qa_url
        self.collection = collection
        self.max_steps = max_steps
        self.planner_enabled = planner_enabled
        self.agent_timeout = agent_timeout
        self.baseline_timeout = baseline_timeout
        self.api_key = api_key
        self.agent_retries = max(1, agent_retries)
        self.baseline_retries = max(1, baseline_retries)
        self.judge = judge
        self.dry_run = dry_run
        self.limit = min(limit, len(dataset)) if limit > 0 else len(dataset)
        self.run_baseline = run_baseline

        self._headers: Dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def run(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(self.dataset[: self.limit]):
            logger.info(
                "[%d/%d] %s (%s, %s)",
                idx + 1, self.limit, item.id, item.category, item.difficulty,
            )
            agent_result = self._fake_agent_result() if self.dry_run else self._call_agent(item)
            baseline_result = (
                self._fake_baseline_result() if (self.dry_run or not self.run_baseline)
                else self._call_baseline(item)
            )

            # Judge scores — offline only (через Langfuse trace review)
            judge_scores = self._run_judge(item, agent_result)

            # Metrics
            metrics = self._compute_metrics(item, agent_result, baseline_result, judge_scores)

            # Failure attribution
            failure = classify_failure(
                item, agent_result,
                factual_score=judge_scores.get("factual_correctness", {}).get("score"),
                usefulness_score=judge_scores.get("usefulness", {}).get("score"),
                key_tool_acc=metrics.get("key_tool_accuracy"),
            )

            offline_judge_packet = self._build_offline_judge_packet(
                item=item,
                agent_result=agent_result,
                metrics=metrics,
                qdrant_url=os.environ.get("QDRANT_EVAL_URL", "http://localhost:16333"),
                qdrant_collection=os.environ.get("QDRANT_COLLECTION", "news_colbert_v2"),
            )

            results.append({
                "query_id": item.id,
                "query": item.query,
                "version": item.version,
                "category": item.category,
                "eval_mode": item.eval_mode,
                "difficulty": item.difficulty,
                "answerable": item.answerable,
                "expected_answer": item.expected_answer,
                "calibration": item.calibration,
                "dataset_contract": item.dataset_contract(),
                "required_claims": item.required_claims,
                "agent": agent_result,
                "baseline": baseline_result,
                "judge": judge_scores,
                "metrics": metrics,
                "failure_type": failure,
                "status": self._status(agent_result, baseline_result),
                "offline_judge_packet": offline_judge_packet,
            })
        return results

    @staticmethod
    def _build_offline_judge_packet(
        *,
        item: EvalItem,
        agent_result: Dict[str, Any],
        metrics: Dict[str, Any],
        qdrant_url: Optional[str] = None,
        qdrant_collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Строит самодостаточный packet для offline judge review.

        Если qdrant_url указан — дополучает тексты документов по citation IDs.
        """
        # Enrich citations with document texts from Qdrant (REST API, без qdrant_client)
        citations = agent_result.get("citations_detailed", [])
        if qdrant_url and qdrant_collection and citations:
            try:
                point_ids = [c.get("id") for c in citations if c.get("id")]
                if point_ids:
                    resp = httpx.post(
                        f"{qdrant_url}/collections/{qdrant_collection}/points",
                        json={"ids": point_ids, "with_payload": True},
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        points = resp.json().get("result", [])
                        text_map = {
                            str(p["id"]): (p.get("payload") or {}).get("text", "")
                            for p in points
                            if isinstance(p, dict) and p.get("id")
                        }
                        for cit in citations:
                            cid = cit.get("id")
                            if cid and str(cid) in text_map:
                                # Полный текст без обрезки — NLI нужен весь док��мент.
                                # Markdown артефакт обрезает отдельно при генерации.
                                cit["text"] = text_map[str(cid)]
                        logger.info("Enriched %d/%d citations with Qdrant texts", len(text_map), len(point_ids))
            except Exception as exc:
                logger.warning("Qdrant citation enrichment failed: %s", exc)

        return {
            "query_id": item.id,
            "query": item.query,
            "eval_mode": item.eval_mode,
            "category": item.category,
            "answer": agent_result.get("answer"),
            "status": "error" if agent_result.get("error") else "ok",
            "latency_sec": agent_result.get("latency_sec"),
            "coverage": agent_result.get("coverage"),
            "tools_invoked": agent_result.get("tools_invoked", []),
            "visible_tools_history": agent_result.get("visible_tools_history", []),
            "agent_thoughts": agent_result.get("agent_thoughts", []),
            "tool_observations": agent_result.get("tool_observations", []),
            "citations": agent_result.get("citations_detailed", []),
            "citation_hits": agent_result.get("citation_hits", []),
            "retrieved_docs": agent_result.get("retrieved_docs", []),
            "dataset_contract": item.dataset_contract(),
            "diagnostic_metrics": {
                "strict_anchor_recall": metrics.get("strict_anchor_recall"),
                "acceptable_set_hit": metrics.get("acceptable_set_hit"),
                "key_tool_accuracy": metrics.get("key_tool_accuracy"),
            },
        }

    def _call_agent(self, item: EvalItem) -> Dict[str, Any]:
        """Вызов агента через SSE. Собирает tools_invoked и visible_tools."""
        payload: Dict[str, Any] = {"query": item.query, "max_steps": self.max_steps}
        if self.collection:
            payload["collection"] = self.collection
        if not self.planner_enabled:
            payload["planner"] = False
        # Langfuse tagging: eval session + question ID + trace name
        payload["session_id"] = f"eval-{time.strftime('%Y%m%d-%H%M')}"
        payload["tags"] = [item.id, "eval"]
        payload["trace_name"] = f"agent_request_{item.id.replace('golden_', '')}"

        last_error: Optional[str] = None

        for attempt in range(1, self.agent_retries + 1):
            citation_hits: List[str] = []
            tools_invoked: List[str] = []
            visible_tools_history: List[List[str]] = []
            citations_detailed: List[Dict[str, Any]] = []
            tool_observations: List[Dict[str, Any]] = []
            agent_thoughts: List[str] = []
            retrieved_docs: List[Dict[str, Any]] = []
            final_payload: Optional[Dict[str, Any]] = None
            coverage: Optional[float] = None
            refinements: int = 0
            pending_observation_tool: Optional[str] = None

            try:
                start = time.perf_counter()
                with httpx.Client(timeout=self.agent_timeout, headers=self._headers) as client:
                    with client.stream("POST", self.agent_url, json=payload) as response:
                        response.raise_for_status()
                        for event_name, event_data in iter_sse_events(response):
                            try:
                                decoded = json.loads(event_data)
                            except json.JSONDecodeError:
                                continue

                            if event_name == "step_started":
                                vis = decoded.get("visible_tools", [])
                                visible_tools_history.append(vis)

                            elif event_name == "tool_invoked":
                                tool_name = decoded.get("tool") or decoded.get("name", "")
                                if tool_name:
                                    tools_invoked.append(tool_name)
                                    pending_observation_tool = tool_name

                            elif event_name == "observation":
                                observation_text = str(decoded.get("content", "") or "")
                                observation_tool = pending_observation_tool or "unknown"
                                tool_observations.append(
                                    {
                                        "tool": observation_tool,
                                        "summary": observation_text[:500],
                                        "success": bool(decoded.get("success", False)),
                                        "took_ms": decoded.get("took_ms"),
                                        "system_generated": bool(decoded.get("system_generated", False)),
                                        "refinement": bool(decoded.get("refinement", False)),
                                        "verification_refinement": bool(decoded.get("verification_refinement", False)),
                                    }
                                )
                                if observation_tool in SEARCH_TYPE_TOOLS:
                                    for doc in extract_search_docs_from_observation(observation_tool, observation_text):
                                        if doc["id"] not in {d["id"] for d in retrieved_docs}:
                                            retrieved_docs.append(doc)
                                pending_observation_tool = None

                            elif event_name == "citations":
                                for cit in decoded.get("citations", []):
                                    normalized = normalize_citation(cit)
                                    if normalized["id"] and normalized["id"] not in {c["id"] for c in citations_detailed if c.get("id")}:
                                        citations_detailed.append(normalized)
                                    meta = cit.get("metadata", {})
                                    ch = meta.get("channel", "")
                                    msg = meta.get("message_id", "")
                                    if ch and msg:
                                        key = f"{ch}:{msg}"
                                        if key not in citation_hits:
                                            citation_hits.append(key)
                                cov = decoded.get("coverage")
                                if cov is not None:
                                    coverage = cov

                            elif event_name == "thought":
                                content = decoded.get("content", "") or ""
                                compact = compact_thought(content)
                                if compact:
                                    agent_thoughts.append(compact)
                                if "недостаточно" in content.lower() or "дополнительный" in content.lower():
                                    refinements += 1

                            elif event_name == "final":
                                final_payload = decoded
                                break

                latency = time.perf_counter() - start
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Agent error (attempt %d/%d): %s", attempt, self.agent_retries, exc)
                continue

            if final_payload:
                return {
                    "answer": final_payload.get("answer", ""),
                    "citations": final_payload.get("citations", []),
                    "coverage": coverage or final_payload.get("coverage"),
                    "refinements": refinements,
                    "verification": final_payload.get("verification"),
                    "fallback": final_payload.get("fallback", False),
                    "request_id": final_payload.get("request_id"),
                    "citation_hits": citation_hits,
                    "tools_invoked": tools_invoked,
                    "visible_tools_history": visible_tools_history,
                    "citations_detailed": citations_detailed,
                    "tool_observations": tool_observations,
                    "agent_thoughts": agent_thoughts,
                    "retrieved_docs": retrieved_docs,
                    "latency_sec": latency,
                    "error": False,
                }

            last_error = "Agent stream завершён без события final"
            logger.warning("No final event (attempt %d/%d)", attempt, self.agent_retries)

        return {
            "error": True, "error_message": last_error,
            "latency_sec": None, "citation_hits": [], "tools_invoked": [],
            "visible_tools_history": [], "citations_detailed": [],
            "tool_observations": [], "agent_thoughts": [], "retrieved_docs": [],
        }

    def _call_baseline(self, item: EvalItem) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": item.query}
        if self.collection:
            payload["collection"] = self.collection
        last_error: Optional[str] = None

        for attempt in range(1, self.baseline_retries + 1):
            start = time.perf_counter()
            try:
                resp = httpx.post(
                    self.qa_url, json=payload,
                    headers=self._headers, timeout=self.baseline_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return {"answer": data.get("answer"), "latency_sec": time.perf_counter() - start, "error": False}
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Baseline error (attempt %d/%d): %s", attempt, self.baseline_retries, exc)
        return {"error": True, "error_message": last_error, "latency_sec": None}

    def _run_judge(self, item: EvalItem, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Judge scores — заполняются offline через Langfuse trace review.

        Все score поля = None при eval прогоне. Заполняются вручную
        (Claude/Codex в чате) на основе exported offline judge artifacts.
        """
        return {}

    def _compute_metrics(
        self,
        item: EvalItem,
        agent_result: Dict[str, Any],
        baseline_result: Dict[str, Any],
        judge_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Primary + grounding + diagnostic metrics."""
        citation_hits = agent_result.get("citation_hits") or []
        expected = item.expected_documents or item.source_post_ids or []
        strict_anchor_recall = None

        # Legacy strict/fuzzy recall — diagnostic only.
        broad_categories = {"temporal", "multi_hop", "constrained_search", "future_baseline"}
        fuzzy_tolerance = 50 if item.category in broad_categories else 5

        if item.answerable and expected and item.strict_anchor_recall_eligible:
            matched = 0
            for exp_doc in expected:
                parts = exp_doc.split(":", 1)
                if len(parts) != 2:
                    continue
                exp_ch, exp_msg = parts[0].lower(), int(parts[1])
                for hit in citation_hits:
                    h_parts = hit.split(":", 1)
                    if len(h_parts) != 2:
                        continue
                    try:
                        h_ch, h_msg = h_parts[0].lower(), int(h_parts[1])
                    except ValueError:
                        continue
                    if h_ch == exp_ch and abs(h_msg - exp_msg) <= fuzzy_tolerance:
                        matched += 1
                        break
            strict_anchor_recall = matched / len(expected) if expected else None

        acceptable_set_hit = None
        if item.acceptable_evidence_sets:
            hit_set = set(citation_hits)
            acceptable_set_hit = 1.0 if any(set(evidence_set).issubset(hit_set) for evidence_set in item.acceptable_evidence_sets) else 0.0

        # Key tool accuracy
        tools_invoked = agent_result.get("tools_invoked", [])
        key_tool_acc = compute_key_tool_accuracy(tools_invoked, item)

        # Judge
        factual = judge_scores.get("factual_correctness", {}).get("score")
        usefulness = judge_scores.get("usefulness", {}).get("score")

        return {
            "agent_latency_sec": agent_result.get("latency_sec"),
            "baseline_latency_sec": baseline_result.get("latency_sec"),
            "agent_coverage": agent_result.get("coverage"),
            "strict_anchor_recall": strict_anchor_recall,
            "strict_anchor_recall_eligible": item.strict_anchor_recall_eligible,
            "acceptable_set_hit": acceptable_set_hit,
            "retrieval_sufficiency_score": judge_scores.get("retrieval_sufficiency", {}).get("score"),
            "evidence_support_score": judge_scores.get("evidence_support", {}).get("score"),
            "key_tool_accuracy": key_tool_acc,
            "factual_correctness": factual,
            "usefulness": usefulness,
            "tools_invoked": tools_invoked,
            "citation_hits": citation_hits,
            "expected_documents": expected,
        }

    @staticmethod
    def _fake_agent_result() -> Dict[str, Any]:
        return {
            "answer": "[dry-run]", "citations": [], "coverage": None,
            "refinements": 0, "verification": None, "fallback": False,
            "citation_hits": [], "tools_invoked": [], "visible_tools_history": [],
            "citations_detailed": [], "tool_observations": [], "agent_thoughts": [],
            "retrieved_docs": [],
            "latency_sec": None, "error": True, "error_message": "dry_run",
        }

    @staticmethod
    def _fake_baseline_result() -> Dict[str, Any]:
        return {"answer": "[dry-run]", "latency_sec": None, "error": True, "error_message": "dry_run"}

    @staticmethod
    def _status(agent_result: Dict[str, Any], baseline_result: Dict[str, Any]) -> str:
        a_err = bool(agent_result.get("error"))
        b_err = bool(baseline_result.get("error"))
        if a_err and b_err:
            return "agent_and_baseline_error"
        if a_err:
            return "agent_error"
        if b_err:
            return "baseline_error"
        return "ok"


# ─── Aggregation ──────────────────────────────────────────────────


def aggregate_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Агрегация метрик + failure breakdown."""
    agent_latencies = [r["metrics"]["agent_latency_sec"] for r in results if r["metrics"].get("agent_latency_sec") is not None]
    baseline_latencies = [r["metrics"]["baseline_latency_sec"] for r in results if r["metrics"].get("baseline_latency_sec") is not None]
    coverages = [r["metrics"]["agent_coverage"] for r in results if r["metrics"].get("agent_coverage") is not None]
    strict_recalls = [r["metrics"]["strict_anchor_recall"] for r in results if r["metrics"].get("strict_anchor_recall") is not None]
    acceptable_hits = [r["metrics"]["acceptable_set_hit"] for r in results if r["metrics"].get("acceptable_set_hit") is not None]
    retrieval_sufficiency_scores = [r["metrics"]["retrieval_sufficiency_score"] for r in results if r["metrics"].get("retrieval_sufficiency_score") is not None]
    evidence_support_scores = [r["metrics"]["evidence_support_score"] for r in results if r["metrics"].get("evidence_support_score") is not None]
    key_tool_accs = [r["metrics"]["key_tool_accuracy"] for r in results if r["metrics"].get("key_tool_accuracy") is not None]
    factual_scores = [r["metrics"]["factual_correctness"] for r in results if r["metrics"].get("factual_correctness") is not None]
    usefulness_scores = [r["metrics"]["usefulness"] for r in results if r["metrics"].get("usefulness") is not None]

    # Failure breakdown
    failure_counts: Dict[str, int] = {}
    for r in results:
        ft = r.get("failure_type")
        if ft:
            failure_counts[ft] = failure_counts.get(ft, 0) + 1

    # By category
    by_category: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {
                "queries": 0,
                "strict_recall": [],
                "acceptable_set_hit": [],
                "key_tool": [],
                "factual": [],
                "useful": [],
                "latency": [],
            }
        by_category[cat]["queries"] += 1
        m = r["metrics"]
        if m.get("strict_anchor_recall") is not None:
            by_category[cat]["strict_recall"].append(m["strict_anchor_recall"])
        if m.get("acceptable_set_hit") is not None:
            by_category[cat]["acceptable_set_hit"].append(m["acceptable_set_hit"])
        if m.get("key_tool_accuracy") is not None:
            by_category[cat]["key_tool"].append(m["key_tool_accuracy"])
        if m.get("factual_correctness") is not None:
            by_category[cat]["factual"].append(m["factual_correctness"])
        if m.get("usefulness") is not None:
            by_category[cat]["useful"].append(m["usefulness"])
        if m.get("agent_latency_sec") is not None:
            by_category[cat]["latency"].append(m["agent_latency_sec"])

    by_cat_summary = {}
    for cat, vals in by_category.items():
        by_cat_summary[cat] = {
            "queries": vals["queries"],
            "strict_anchor_recall_mean": safe_mean(vals["strict_recall"]),
            "acceptable_set_hit_mean": safe_mean(vals["acceptable_set_hit"]),
            "key_tool_accuracy_mean": safe_mean(vals["key_tool"]),
            "factual_correctness_mean": safe_mean(vals["factual"]),
            "usefulness_mean": safe_mean(vals["useful"]),
            "agent_latency_mean": safe_mean(vals["latency"]),
        }

    statuses = [r["status"] for r in results]

    return {
        "total_queries": len(results),
        "answerable_queries": sum(1 for r in results if r["answerable"]),
        "negative_queries": sum(1 for r in results if not r["answerable"]),
        "golden_queries": sum(1 for r in results if r["metrics"].get("key_tool_accuracy") is not None),
        "errors": {
            "agent": sum(1 for s in statuses if "agent" in s),
            "baseline": sum(1 for s in statuses if "baseline" in s),
        },
        "primary": {
            "key_tool_accuracy": {
                "mean": safe_mean(key_tool_accs),
                "total_evaluated": len(key_tool_accs),
            },
            "factual_correctness": {
                "mean": safe_mean(factual_scores),
                "total_evaluated": len(factual_scores),
            },
            "usefulness": {
                "mean": safe_mean(usefulness_scores),
                "total_evaluated": len(usefulness_scores),
            },
            "failure_breakdown": failure_counts,
        },
        "retrieval_grounding": {
            "acceptable_set_hit": {
                "mean": safe_mean(acceptable_hits),
                "total_evaluated": len(acceptable_hits),
            },
            "retrieval_sufficiency_score": {
                "mean": safe_mean(retrieval_sufficiency_scores),
                "total_evaluated": len(retrieval_sufficiency_scores),
                "pending_offline_judge": len(retrieval_sufficiency_scores) == 0,
            },
            "evidence_support_score": {
                "mean": safe_mean(evidence_support_scores),
                "total_evaluated": len(evidence_support_scores),
                "pending_offline_judge": len(evidence_support_scores) == 0,
            },
        },
        "diagnostic": {
            "strict_anchor_recall": {
                "mean": safe_mean(strict_recalls),
                "full": sum(1 for r in strict_recalls if r == 1.0),
                "zero": sum(1 for r in strict_recalls if r == 0.0),
                "total_evaluated": len(strict_recalls),
            },
            "coverage": {"mean": safe_mean(coverages)},
            "latency": {
                "agent": {"mean": safe_mean(agent_latencies), "p95": percentile(agent_latencies, 95)},
                "baseline": {"mean": safe_mean(baseline_latencies), "p95": percentile(baseline_latencies, 95)},
            },
        },
        "by_category": by_cat_summary,
    }


# ─── Reports ──────────────────────────────────────────────────────


def build_markdown_report(
    agg: Dict[str, Any],
    timestamp: datetime,
    dataset_path: Path,
    judge_model: Optional[str],
) -> str:
    """Markdown отчёт с failure breakdown и per-category."""
    primary = agg["primary"]
    grounding = agg["retrieval_grounding"]
    diagnostic = agg["diagnostic"]
    lines = [
        "# Agent Evaluation Report (v2)",
        f"**Date:** {timestamp.isoformat()}",
        f"**Dataset:** {dataset_path} ({agg['total_queries']} queries)",
        f"**Judge:** {judge_model or 'disabled'}",
        "",
        "## Primary Metrics",
        f"- Key Tool Accuracy: **{fmt(primary['key_tool_accuracy']['mean'])}** ({primary['key_tool_accuracy']['total_evaluated']} evaluated)",
        f"- Factual Correctness: **{fmt(primary['factual_correctness']['mean'])}** ({primary['factual_correctness']['total_evaluated']} evaluated)",
        f"- Usefulness: **{fmt(primary['usefulness']['mean'])}** ({primary['usefulness']['total_evaluated']} evaluated)",
        "",
        "## Retrieval Grounding",
        f"- Acceptable Set Hit: **{fmt(grounding['acceptable_set_hit']['mean'])}** ({grounding['acceptable_set_hit']['total_evaluated']} evaluated)",
        f"- Retrieval Sufficiency Score: {fmt(grounding['retrieval_sufficiency_score']['mean'])} ({grounding['retrieval_sufficiency_score']['total_evaluated']} evaluated)",
        f"- Evidence Support Score: {fmt(grounding['evidence_support_score']['mean'])} ({grounding['evidence_support_score']['total_evaluated']} evaluated)",
        "",
        "## Diagnostic Metrics",
        f"- Strict Anchor Recall: **{fmt(diagnostic['strict_anchor_recall']['mean'])}** (full={diagnostic['strict_anchor_recall']['full']}, zero={diagnostic['strict_anchor_recall']['zero']})",
        f"- Coverage: {fmt(diagnostic['coverage']['mean'])}",
        f"- Agent Latency: mean={fmt(diagnostic['latency']['agent']['mean'], 1)}s, p95={fmt(diagnostic['latency']['agent']['p95'], 1)}s",
        "",
    ]

    # Failure breakdown
    failures = primary.get("failure_breakdown", {})
    if failures:
        lines.extend(["## Failure Breakdown", "| Type | Count |", "|------|-------|"])
        for ft, count in sorted(failures.items(), key=lambda x: -x[1]):
            lines.append(f"| {ft} | {count} |")
        lines.append("")

    # By category
    lines.extend([
        "## By Category",
        "| Category | Qs | StrictRecall | AcceptableSet | KeyTool | Factual | Useful | Latency |",
        "|----------|-----|--------------|---------------|---------|---------|--------|---------|",
    ])
    for cat, stats in sorted(agg["by_category"].items()):
        lines.append(
            f"| {cat} | {stats['queries']} | "
            f"{fmt(stats['strict_anchor_recall_mean'])} | "
            f"{fmt(stats['acceptable_set_hit_mean'])} | "
            f"{fmt(stats['key_tool_accuracy_mean'])} | "
            f"{fmt(stats['factual_correctness_mean'])} | "
            f"{fmt(stats['usefulness_mean'])} | "
            f"{fmt(stats['agent_latency_mean'], 1)}s |"
        )

    return "\n".join(lines)


def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Разбивает последовательность на батчи фиксированного размера."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_offline_judge_markdown(batch: Sequence[Dict[str, Any]], batch_no: int) -> str:
    """Строит markdown packet для offline judge review."""
    lines = [
        f"# Offline Judge Batch {batch_no:02d}",
        "",
        f"Questions: {len(batch)}",
        "",
    ]
    for item in batch:
        packet = item["offline_judge_packet"]
        contract = packet["dataset_contract"]
        lines.extend(
            [
                f"## {packet['query_id']} — {packet['query']}",
                f"- Eval mode: `{packet['eval_mode']}`",
                f"- Category: `{packet['category']}`",
                f"- Answerable: `{contract['answerable']}`",
                f"- Key tools: `{', '.join(contract['key_tools']) or '-'}`",
                f"- Forbidden tools: `{', '.join(contract['forbidden_tools']) or '-'}`",
                f"- Tools invoked: `{', '.join(packet['tools_invoked']) or '-'}`",
                f"- Coverage: `{packet.get('coverage')}`",
                "",
                "**Expected answer**",
                "",
                contract.get("expected_answer") or "_N/A_",
                "",
                "**Required claims**",
                "",
            ]
        )
        claims = contract.get("required_claims") or []
        if claims:
            lines.extend([f"- {claim}" for claim in claims])
        else:
            lines.append("- _none_")
        lines.extend(["", "**Agent answer**", "", packet.get("answer") or "_empty_", ""])

        lines.extend(["**Agent thoughts**", ""])
        thoughts = packet.get("agent_thoughts") or []
        if thoughts:
            lines.extend([f"- {thought}" for thought in thoughts])
        else:
            lines.append("- _none_")
        lines.append("")

        lines.extend(["**Tool observations**", ""])
        observations = packet.get("tool_observations") or []
        if observations:
            for obs in observations:
                lines.append(f"- `{obs.get('tool', 'unknown')}`: {obs.get('summary', '')}")
        else:
            lines.append("- _none_")
        lines.append("")

        lines.extend(["**Citations (документы использованные в ответе)**", ""])
        citations = packet.get("citations") or []
        if citations:
            for i, cit in enumerate(citations, 1):
                lines.append(
                    f"**[{i}]** `{cit.get('channel') or '-'}:{cit.get('message_id') or '-'}` "
                    f"({cit.get('id') or '-'})"
                )
                text = cit.get("text") or ""
                if text:
                    if len(text) > 1000:
                        lines.append(f"  > {text[:1000]}... [ОБРЕЗАНО]")
                    else:
                        lines.append(f"  > {text}")
                else:
                    lines.append("  > _(текст не загружен)_")
                lines.append("")
        else:
            lines.append("- _none_")
        lines.append("")

        # Retrieved docs секция убрана — дублирует Citations но без текстов.
        # Тексты документов доступны в Citations (enriched из Qdrant).

        # Langfuse trace data (если enriched)
        langfuse_trace = packet.get("langfuse_trace")
        if langfuse_trace:
            lines.extend(["**Langfuse Trace**", ""])
            lines.append(f"- Trace ID: `{langfuse_trace.get('trace_id', 'N/A')}`")
            lines.append(f"- Latency: {langfuse_trace.get('latency', 'N/A')}s")
            lines.append(f"- Tokens: input={langfuse_trace.get('total_input_tokens', 0)}, output={langfuse_trace.get('total_output_tokens', 0)}")
            lines.append("")
            for gen in langfuse_trace.get("llm_generations", []):
                tool_names = gen.get("toolCallNames", [])
                lines.append(f"- `{gen['name']}`: {gen.get('latency', '?')}s, tools={tool_names}, usage={gen.get('usageDetails', {})}")
            lines.append("")

        # Offline judge scoring section
        if packet["eval_mode"] == "retrieval_evidence":
            lines.extend([
                "**Offline Judge Scoring**", "",
                "| Metric | Score | Reasoning |",
                "|--------|-------|-----------|",
                "| factual_correctness (0/0.5/1) | ___ | |",
                "| usefulness (0/1/2) | ___ | |",
                "| evidence_support (0/0.5/1): подтверждается ли ответ документами? | ___ | |",
                "| retrieval_sufficiency (0/0.5/1): достаточно ли документов для ответа? | ___ | |",
                "",
            ])
        elif packet["eval_mode"] == "analytics":
            lines.extend([
                "**Offline Judge Scoring**", "",
                "| Metric | Score | Reasoning |",
                "|--------|-------|-----------|",
                "| factual_correctness (0/0.5/1) | ___ | |",
                "| usefulness (0/1/2) | ___ | |",
                "",
            ])
        elif packet["eval_mode"] == "refusal":
            lines.extend([
                "**Offline Judge Scoring**", "",
                "| Metric | Score | Reasoning |",
                "|--------|-------|-----------|",
                "| correct_refusal (0/1) | ___ | |",
                "| usefulness (0/1/2) | ___ | |",
                "",
            ])
        lines.extend(["---", ""])

    return "\n".join(lines)


def export_offline_judge_batches(
    *,
    results: Sequence[Dict[str, Any]],
    output_dir: Path,
    eval_id: str,
    batch_size: int,
) -> List[Path]:
    """Экспортирует judging packets батчами в JSON и Markdown."""
    judge_dir = output_dir / "judge_batches" / eval_id
    ensure_dirs(judge_dir)
    written: List[Path] = []
    for batch_index, batch in enumerate(chunked(results, batch_size), start=1):
        batch_name = f"judge_batch_{batch_index:02d}"
        batch_json = judge_dir / f"{batch_name}.json"
        batch_md = judge_dir / f"{batch_name}.md"
        batch_payload = [item["offline_judge_packet"] for item in batch]
        write_json(batch_json, batch_payload)
        batch_md.write_text(build_offline_judge_markdown(batch, batch_index), encoding="utf-8")
        written.extend([batch_json, batch_md])
    return written


# ─── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval pipeline v2 для ReAct-агента (SPEC-RAG-14)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Путь к JSON-датасету")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Директория для отчётов")
    parser.add_argument("--agent-url", default=DEFAULT_AGENT_URL, help="URL /v1/agent/stream")
    parser.add_argument("--qa-url", default=DEFAULT_QA_URL, help="URL /v1/qa")
    parser.add_argument("--collection", default=None, help="Коллекция Qdrant")
    parser.add_argument("--max-steps", type=int, default=8, help="max_steps для агента")
    parser.add_argument("--disable-planner", action="store_true", help="Отключить query planner")
    parser.add_argument("--agent-timeout", type=float, default=90.0, help="Таймаут агента (сек)")
    parser.add_argument("--agent-retries", type=int, default=2, help="Повторы при ошибке")
    parser.add_argument("--run-baseline", action="store_true", help="Запустить baseline /v1/qa (по умолчанию выключен)")
    parser.add_argument("--baseline-timeout", type=float, default=30.0, help="Таймаут baseline")
    parser.add_argument("--baseline-retries", type=int, default=1, help="Повторы baseline")
    parser.add_argument("--limit", type=int, default=0, help="Макс. число запросов (0 = все)")
    parser.add_argument("--api-key", default=None, help="(deprecated, auth removed) API ключ для агента")

    # Judge (offline only — через Langfuse trace review)
    parser.add_argument("--skip-judge", action="store_true", help="(deprecated, judge is always offline)")
    parser.add_argument("--judge-model", default=None, help="(deprecated, judge is offline)")

    # Other
    parser.add_argument("--export-offline-judge", action="store_true", help="Экспортировать offline judge packets батчами")
    parser.add_argument("--judge-batch-size", type=int, default=30, help="Размер judge batch для offline review")
    parser.add_argument("--skip-markdown", action="store_true", help="Не сохранять Markdown")
    parser.add_argument("--dry-run", action="store_true", help="Пропустить API вызовы")
    parser.add_argument("--verbose", action="store_true", help="Подробное логирование")
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        dataset = load_dataset(args.dataset)
    except Exception as exc:
        logger.error("Не удалось загрузить датасет: %s", exc)
        return 1

    # Judge is offline-only (через Langfuse trace review)
    judge_model: Optional[str] = None

    runner = AgentEvaluationRunner(
        dataset,
        agent_url=args.agent_url,
        qa_url=args.qa_url,
        collection=args.collection,
        max_steps=args.max_steps,
        planner_enabled=not args.disable_planner,
        agent_timeout=args.agent_timeout,
        baseline_timeout=args.baseline_timeout,
        api_key=args.api_key,
        agent_retries=args.agent_retries,
        baseline_retries=args.baseline_retries,
        dry_run=args.dry_run,
        limit=args.limit,
        run_baseline=args.run_baseline,
    )

    try:
        raw_results = runner.run()
    except KeyboardInterrupt:
        logger.warning("Оценка прервана пользователем")
        return 130

    aggregated = aggregate_results(raw_results)

    timestamp = datetime.now(tz=None)
    raw_dir = args.output_dir / "raw"
    reports_dir = args.output_dir / "reports"
    ensure_dirs(args.output_dir, raw_dir, reports_dir)

    ts_str = f"{timestamp:%Y%m%d-%H%M%S}"
    eval_id = f"eval_{ts_str}"

    # Единый JSON report (SPEC-RAG-14 §3.8)
    unified_report = {
        "eval_metadata": {
            "eval_id": eval_id,
            "timestamp": timestamp.isoformat(),
            "dataset": str(args.dataset),
            "judge_model": judge_model,
            "total_questions": len(raw_results),
            "duration_sec": sum(
                r["metrics"].get("agent_latency_sec") or 0 for r in raw_results
            ),
        },
        "aggregate": aggregated,
        "per_question": raw_results,
    }

    raw_path = raw_dir / f"eval_results_{ts_str}.json"
    report_path = reports_dir / f"eval_report_{ts_str}.json"

    write_json(raw_path, unified_report)
    write_json(report_path, aggregated)

    markdown_path = None
    if not args.skip_markdown:
        markdown_path = reports_dir / f"eval_report_{ts_str}.md"
        md = build_markdown_report(aggregated, timestamp, args.dataset, judge_model)
        markdown_path.write_text(md, encoding="utf-8")

    exported_batches: List[Path] = []
    if args.export_offline_judge:
        # Enrich offline judge packets with Langfuse trace data
        langfuse_host = os.environ.get("LANGFUSE_EXPORT_HOST", "http://localhost:3100")
        langfuse_pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-rag-app-dev")
        langfuse_sk = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-rag-app-dev")
        try:
            exporter = LangfuseTraceExporter(
                host=langfuse_host, public_key=langfuse_pk, secret_key=langfuse_sk,
            )
            recent_traces = exporter.fetch_recent_traces(limit=len(raw_results) * 2, tags=["eval"])
            # Match traces to results by query text
            for result in raw_results:
                query = result.get("query", "")
                packet = result.get("offline_judge_packet", {})
                for trace in recent_traces:
                    trace_input = trace.get("input") or ""
                    if isinstance(trace_input, str) and query and query in trace_input:
                        full_trace = exporter.fetch_trace(trace["id"])
                        if full_trace:
                            trace_data = exporter.extract_trace_data(full_trace)
                            packet["langfuse_trace"] = trace_data
                            logger.debug("Matched trace %s to %s", trace["id"], result.get("query_id"))
                        break
            exporter.close()
        except Exception as exc:
            logger.warning("Langfuse trace enrichment failed (non-blocking): %s", exc)

        exported_batches = export_offline_judge_batches(
            results=raw_results,
            output_dir=args.output_dir,
            eval_id=eval_id,
            batch_size=max(1, args.judge_batch_size),
        )

    logger.info("Raw: %s", raw_path)
    logger.info("Report: %s", report_path)
    if markdown_path:
        logger.info("Markdown: %s", markdown_path)
    for batch_path in exported_batches:
        logger.info("Offline judge artifact: %s", batch_path)

    print(json.dumps(aggregated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
