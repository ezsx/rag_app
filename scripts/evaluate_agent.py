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
import json
import logging
import math
import os
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
    future_tool_flag: bool = False
    future_key_tools: Optional[List[str]] = None
    calibration: bool = False
    difficulty: str = "medium"

    @property
    def is_golden(self) -> bool:
        """Auto-detect golden vs legacy формат."""
        return bool(self.key_tools) or self.expected_refusal


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
        item = EvalItem(
            id=str(record.get("id") or len(items) + 1),
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
            future_tool_flag=bool(record.get("future_tool_flag", False)),
            future_key_tools=record.get("future_key_tools"),
            calibration=bool(record.get("calibration", False)),
            difficulty=record.get("difficulty", "medium"),
        )
        items.append(item)

    golden_count = sum(1 for i in items if i.is_golden)
    logger.info(
        "Загружено %d вопросов из %s (%d golden, %d legacy)",
        len(items), source_path, golden_count, len(items) - golden_count,
    )
    return items


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
        refusal_markers = ["не найд", "нет информации", "не содержит", "отсутствует", "не могу", "нет данных"]
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


class ClaudeJudge:
    """LLM judge через Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6-20250514",
        timeout: float = 30.0,
        max_retries: int = 2,
        rate_limit_delay: float = 2.0,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self._client = httpx.Client(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=timeout,
        )

    def judge_factual(self, question: str, answer: str, expected: str) -> Dict[str, Any]:
        """Factual correctness: 0.0 / 0.5 / 1.0"""
        prompt = (
            f"Вопрос: {question}\n"
            f"Ответ системы: {answer}\n"
            f"Эталонный ответ: {expected}\n\n"
            "Оцени фактическую корректность ответа относительно эталона:\n"
            "0.0 — Содержит фактические ошибки или противоречит эталону\n"
            "0.5 — Частично корректен, но упускает важные факты или содержит неточности\n"
            "1.0 — Фактически корректен, соответствует эталону\n\n"
            'JSON: {"reasoning": "...", "score": 0.0|0.5|1.0}'
        )
        return self._call(prompt, "factual_correctness")

    def judge_usefulness(self, question: str, answer: str) -> Dict[str, Any]:
        """Usefulness: 0 / 1 / 2"""
        prompt = (
            f"Вопрос: {question}\n"
            f"Ответ системы: {answer}\n\n"
            "Оцени полезность ответа:\n"
            "0 — Бесполезный: не содержит релевантной информации\n"
            "1 — Частично полезный: некоторая информация есть, но неполный\n"
            "2 — Полезный: полностью отвечает, конкретен, хорошо структурирован\n\n"
            'JSON: {"reasoning": "...", "score": 0|1|2}'
        )
        return self._call(prompt, "usefulness")

    def _call(self, prompt: str, criterion: str) -> Dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(
                    "/v1/messages",
                    json={
                        "model": self.model,
                        "max_tokens": 512,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if resp.status_code == 429:
                    logger.warning("Judge rate limited, sleeping %ss", self.rate_limit_delay)
                    time.sleep(self.rate_limit_delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["content"][0]["text"]
                # Извлекаем JSON из ответа
                parsed = self._extract_json(text)
                return {
                    "criterion": criterion,
                    "score": parsed.get("score"),
                    "reasoning": parsed.get("reasoning", ""),
                    "raw": text,
                    "error": False,
                }
            except Exception as exc:
                logger.warning("Judge error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.rate_limit_delay)
                    continue
                return {
                    "criterion": criterion,
                    "score": None,
                    "reasoning": str(exc),
                    "error": True,
                }
        return {"criterion": criterion, "score": None, "error": True}

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Извлекает JSON из текста ответа judge."""
        # Ищем {...} в тексте
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {"score": None, "reasoning": text}

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
        judge: Optional[ClaudeJudge] = None,
        dry_run: bool = False,
        limit: int = 0,
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
            baseline_result = self._fake_baseline_result() if self.dry_run else self._call_baseline(item)

            # Judge scores
            judge_scores = self._run_judge(item, agent_result) if self.judge and not self.dry_run else {}

            # Metrics
            metrics = self._compute_metrics(item, agent_result, baseline_result, judge_scores)

            # Failure attribution
            failure = classify_failure(
                item, agent_result,
                factual_score=judge_scores.get("factual_correctness", {}).get("score"),
                usefulness_score=judge_scores.get("usefulness", {}).get("score"),
                key_tool_acc=metrics.get("key_tool_accuracy"),
            )

            results.append({
                "query_id": item.id,
                "query": item.query,
                "category": item.category,
                "difficulty": item.difficulty,
                "answerable": item.answerable,
                "expected_answer": item.expected_answer,
                "calibration": item.calibration,
                "future_tool_flag": item.future_tool_flag,
                "agent": agent_result,
                "baseline": baseline_result,
                "judge": judge_scores,
                "metrics": metrics,
                "failure_type": failure,
                "status": self._status(agent_result, baseline_result),
            })
        return results

    def _call_agent(self, item: EvalItem) -> Dict[str, Any]:
        """Вызов агента через SSE. Собирает tools_invoked и visible_tools."""
        payload: Dict[str, Any] = {"query": item.query, "max_steps": self.max_steps}
        if self.collection:
            payload["collection"] = self.collection
        if not self.planner_enabled:
            payload["planner"] = False

        last_error: Optional[str] = None

        for attempt in range(1, self.agent_retries + 1):
            citation_hits: List[str] = []
            tools_invoked: List[str] = []
            visible_tools_history: List[List[str]] = []
            final_payload: Optional[Dict[str, Any]] = None
            coverage: Optional[float] = None
            refinements: int = 0

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

                            elif event_name == "citations":
                                for cit in decoded.get("citations", []):
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
                                content = decoded.get("content", "")
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
                    "latency_sec": latency,
                    "error": False,
                }

            last_error = "Agent stream завершён без события final"
            logger.warning("No final event (attempt %d/%d)", attempt, self.agent_retries)

        return {
            "error": True, "error_message": last_error,
            "latency_sec": None, "citation_hits": [], "tools_invoked": [],
            "visible_tools_history": [],
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
        """Запуск LLM judge для factual + usefulness."""
        if not self.judge or agent_result.get("error"):
            return {}

        answer = agent_result.get("answer", "")
        scores: Dict[str, Any] = {}

        # Factual correctness (только для answerable с expected_answer)
        if item.answerable and item.expected_answer:
            scores["factual_correctness"] = self.judge.judge_factual(
                item.query, answer, item.expected_answer,
            )

        # Usefulness (для всех)
        scores["usefulness"] = self.judge.judge_usefulness(item.query, answer)

        return scores

    def _compute_metrics(
        self,
        item: EvalItem,
        agent_result: Dict[str, Any],
        baseline_result: Dict[str, Any],
        judge_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recall@5 + key_tool_accuracy + judge scores."""
        citation_hits = agent_result.get("citation_hits") or []
        expected = item.expected_documents or item.source_post_ids or []
        recall = None

        # Fuzzy recall (совместим с legacy)
        broad_categories = {"temporal", "multi_hop", "constrained_search", "future_baseline"}
        fuzzy_tolerance = 50 if item.category in broad_categories else 5

        if item.answerable and expected:
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
            recall = matched / len(expected) if expected else None

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
            "recall_at_5": recall,
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
    recalls = [r["metrics"]["recall_at_5"] for r in results if r["metrics"].get("recall_at_5") is not None]
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
            by_category[cat] = {"queries": 0, "recall": [], "key_tool": [], "factual": [], "useful": [], "latency": []}
        by_category[cat]["queries"] += 1
        m = r["metrics"]
        if m.get("recall_at_5") is not None:
            by_category[cat]["recall"].append(m["recall_at_5"])
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
            "recall_at_5_mean": safe_mean(vals["recall"]),
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
        "golden_queries": sum(1 for r in results if any(r["metrics"].get("key_tool_accuracy") is not None for _ in [1])),
        "errors": {
            "agent": sum(1 for s in statuses if "agent" in s),
            "baseline": sum(1 for s in statuses if "baseline" in s),
        },
        "recall_at_5": {
            "mean": safe_mean(recalls),
            "full": sum(1 for r in recalls if r == 1.0),
            "zero": sum(1 for r in recalls if r == 0.0),
        },
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
        "coverage": {"mean": safe_mean(coverages)},
        "latency": {
            "agent": {"mean": safe_mean(agent_latencies), "p95": percentile(agent_latencies, 95)},
            "baseline": {"mean": safe_mean(baseline_latencies), "p95": percentile(baseline_latencies, 95)},
        },
        "failure_breakdown": failure_counts,
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
    lines = [
        "# Agent Evaluation Report (v2)",
        f"**Date:** {timestamp.isoformat()}",
        f"**Dataset:** {dataset_path} ({agg['total_queries']} queries)",
        f"**Judge:** {judge_model or 'disabled'}",
        "",
        "## Overall Metrics",
        f"- Recall@5: **{fmt(agg['recall_at_5']['mean'])}** (full={agg['recall_at_5']['full']}, zero={agg['recall_at_5']['zero']})",
        f"- Key Tool Accuracy: **{fmt(agg['key_tool_accuracy']['mean'])}** ({agg['key_tool_accuracy']['total_evaluated']} evaluated)",
        f"- Factual Correctness: **{fmt(agg['factual_correctness']['mean'])}** ({agg['factual_correctness']['total_evaluated']} evaluated)",
        f"- Usefulness: **{fmt(agg['usefulness']['mean'])}** ({agg['usefulness']['total_evaluated']} evaluated)",
        f"- Coverage: {fmt(agg['coverage']['mean'])}",
        f"- Agent Latency: mean={fmt(agg['latency']['agent']['mean'], 1)}s, p95={fmt(agg['latency']['agent']['p95'], 1)}s",
        "",
    ]

    # Failure breakdown
    failures = agg.get("failure_breakdown", {})
    if failures:
        lines.extend(["## Failure Breakdown", "| Type | Count |", "|------|-------|"])
        for ft, count in sorted(failures.items(), key=lambda x: -x[1]):
            lines.append(f"| {ft} | {count} |")
        lines.append("")

    # By category
    lines.extend([
        "## By Category",
        "| Category | Qs | Recall@5 | KeyTool | Factual | Useful | Latency |",
        "|----------|-----|----------|---------|---------|--------|---------|",
    ])
    for cat, stats in sorted(agg["by_category"].items()):
        lines.append(
            f"| {cat} | {stats['queries']} | "
            f"{fmt(stats['recall_at_5_mean'])} | "
            f"{fmt(stats['key_tool_accuracy_mean'])} | "
            f"{fmt(stats['factual_correctness_mean'])} | "
            f"{fmt(stats['usefulness_mean'])} | "
            f"{fmt(stats['agent_latency_mean'], 1)}s |"
        )

    return "\n".join(lines)


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
    parser.add_argument("--baseline-timeout", type=float, default=30.0, help="Таймаут baseline")
    parser.add_argument("--baseline-retries", type=int, default=1, help="Повторы baseline")
    parser.add_argument("--limit", type=int, default=0, help="Макс. число запросов (0 = все)")
    parser.add_argument("--api-key", default=None, help="API ключ для агента (Bearer)")

    # Judge
    parser.add_argument(
        "--judge", choices=["claude", "skip"], default="skip",
        help="LLM judge: claude (Anthropic API) или skip",
    )
    parser.add_argument("--skip-judge", action="store_true", help="Алиас для --judge skip")
    parser.add_argument("--judge-model", default=None, help="Модель judge (default: claude-sonnet-4-6-20250514)")

    # Other
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

    # Judge setup (--skip-judge алиас для --judge skip)
    if args.skip_judge:
        args.judge = "skip"

    judge: Optional[ClaudeJudge] = None
    judge_model: Optional[str] = None
    if args.judge == "claude":
        judge_api_key = os.environ.get("EVAL_JUDGE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not judge_api_key:
            logger.warning("Judge=claude, но EVAL_JUDGE_API_KEY не задан → fallback на skip")
            args.judge = "skip"
    if args.judge == "claude":
        judge_api_key = os.environ.get("EVAL_JUDGE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        judge_model = args.judge_model or os.environ.get("EVAL_JUDGE_MODEL", "claude-sonnet-4-6-20250514")
        judge = ClaudeJudge(
            api_key=judge_api_key,
            model=judge_model,
            timeout=float(os.environ.get("EVAL_JUDGE_TIMEOUT", "30")),
            max_retries=int(os.environ.get("EVAL_JUDGE_MAX_RETRIES", "2")),
            rate_limit_delay=float(os.environ.get("EVAL_JUDGE_RATE_LIMIT_DELAY", "2")),
        )
        logger.info("Judge: %s", judge_model)

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
        judge=judge,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    try:
        raw_results = runner.run()
    except KeyboardInterrupt:
        logger.warning("Оценка прервана пользователем")
        return 130
    finally:
        if judge:
            judge.close()

    aggregated = aggregate_results(raw_results)

    timestamp = datetime.now(tz=None)
    raw_dir = args.output_dir / "raw"
    reports_dir = args.output_dir / "reports"
    ensure_dirs(args.output_dir, raw_dir, reports_dir)

    ts_str = f"{timestamp:%Y%m%d-%H%M%S}"

    # Единый JSON report (SPEC-RAG-14 §3.8)
    unified_report = {
        "eval_metadata": {
            "eval_id": f"eval_{ts_str}",
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

    logger.info("Raw: %s", raw_path)
    logger.info("Report: %s", report_path)
    if markdown_path:
        logger.info("Markdown: %s", markdown_path)

    print(json.dumps(aggregated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
