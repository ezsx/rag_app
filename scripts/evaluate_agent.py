#!/usr/bin/env python3
"""
Инструмент оценки ReAct-агента согласно спецификации
`docs/ai/planning/agent_evaluation_spec.md`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import httpx

DEFAULT_DATASET = Path("datasets/eval_dataset.json")
FALLBACK_DATASET = Path("datasets/eval_questions.json")
DEFAULT_AGENT_URL = "http://localhost:8000/v1/agent/stream"
DEFAULT_QA_URL = "http://localhost:8000/v1/qa"


@dataclass
class EvalItem:
    """Единица датасета для оценки агента."""

    id: str
    query: str
    category: str
    expected_documents: List[str]
    answerable: bool
    expected_answer: Optional[str] = None
    notes: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка Agentic ReAct-RAG по спецификации MVP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Путь к JSON-датасету (см. спецификацию §2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Базовая директория для сохранения отчетов",
    )
    parser.add_argument(
        "--agent-url",
        default=DEFAULT_AGENT_URL,
        help="Полный URL эндпойнта /v1/agent/stream",
    )
    parser.add_argument(
        "--qa-url",
        default=DEFAULT_QA_URL,
        help="Полный URL эндпойнта /v1/qa",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Необязательная коллекция для обоих вызовов",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Параметр max_steps для агента",
    )
    parser.add_argument(
        "--disable-planner",
        action="store_true",
        help="Отключить query planner (спецификация включает его по умолчанию)",
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=60.0,
        help="Таймаут ожидания агента (сек.)",
    )
    parser.add_argument(
        "--agent-retries",
        type=int,
        default=2,
        help="Кол-во повторов вызова агента при ошибке/таймауте",
    )
    parser.add_argument(
        "--baseline-timeout",
        type=float,
        default=30.0,
        help="Таймаут ожидания baseline QA (сек.)",
    )
    parser.add_argument(
        "--baseline-retries",
        type=int,
        default=1,
        help="Кол-во повторов вызова baseline при ошибке/таймауте",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Максимальное число запросов (0 = весь датасет)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Опциональный API ключ (Bearer)",
    )
    parser.add_argument(
        "--skip-markdown",
        action="store_true",
        help="Не сохранять Markdown-отчет",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Пропустить вызовы API (для проверки пайплайна без сервера)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Включить подробное логирование",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[EvalItem]:
    """Загрузка датасета согласно §2 спецификации."""
    source_path = path
    if not path.exists() and FALLBACK_DATASET.exists():
        logging.warning("Датасет %s не найден, fallback на %s", path, FALLBACK_DATASET)
        source_path = FALLBACK_DATASET

    with source_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, dict) and "questions" in payload:
        records = payload["questions"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(
            f"Неизвестный формат датасета {source_path}: ожидается список объектов"
        )

    items: List[EvalItem] = []
    for record in records:
        item = EvalItem(
            id=str(record.get("id") or record.get("query") or len(items) + 1),
            query=record["query"],
            category=record.get("category", "unknown"),
            expected_documents=list(record.get("expected_documents", [])),
            answerable=bool(record.get("answerable", True)),
            expected_answer=record.get("expected_answer"),
            notes=record.get("notes"),
        )
        items.append(item)
    logging.info("Загружено %d запросов из %s", len(items), source_path)
    return items


def iter_sse_events(response: httpx.Response) -> Iterator[Tuple[str, str]]:
    """Парсинг SSE потока (см. §3.1 спецификации)."""
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

        if line.startswith(":"):
            continue

        if line.startswith("retry:"):
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


def extract_hit_ids(hits: Sequence[Any], limit: int = 5) -> List[str]:
    """Извлечь первые N идентификаторов документов для расчета recall@5."""
    ids: List[str] = []
    for hit in hits:
        if isinstance(hit, dict):
            hit_id = hit.get("id") or hit.get("doc_id")
        else:
            hit_id = str(hit)
        if hit_id:
            ids.append(str(hit_id))
        if len(ids) >= limit:
            break
    return ids


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
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * (
        rank - lower
    )


class AgentEvaluationRunner:
    """Оркестрация шага 2-5 из спецификации (§5)."""

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
        self.dry_run = dry_run
        self.limit = min(limit, len(dataset)) if limit > 0 else len(dataset)

        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def run(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(self.dataset[: self.limit]):
            logging.info(
                "[%d/%d] Обработка запроса %s (%s)",
                idx + 1,
                self.limit,
                item.id,
                item.category,
            )
            agent_result = (
                self._fake_agent_result() if self.dry_run else self._call_agent(item)
            )
            baseline_result = (
                self._fake_baseline_result()
                if self.dry_run
                else self._call_baseline(item)
            )
            metrics = self._compute_metrics(item, agent_result, baseline_result)
            status = self._status(agent_result, baseline_result)
            results.append(
                {
                    "query_id": item.id,
                    "query": item.query,
                    "category": item.category,
                    "expected_documents": item.expected_documents,
                    "answerable": item.answerable,
                    "expected_answer": item.expected_answer,
                    "notes": item.notes,
                    "agent": agent_result,
                    "baseline": baseline_result,
                    "metrics": metrics,
                    "status": status,
                }
            )
        return results

    def _call_agent(self, item: EvalItem) -> Dict[str, Any]:
        """
        Вызов агента через SSE endpoint.
        Парсит event: header (не type внутри JSON).
        Собирает citations из event=citations для recall.
        """
        payload: Dict[str, Any] = {
            "query": item.query,
            "max_steps": self.max_steps,
        }
        if self.collection:
            payload["collection"] = self.collection
        if not self.planner_enabled:
            payload["planner"] = False

        last_error: Optional[str] = None

        for attempt in range(1, self.agent_retries + 1):
            citation_hits: List[str] = []  # "channel:message_id"
            final_payload: Optional[Dict[str, Any]] = None
            coverage: Optional[float] = None
            refinements: int = 0
            latency: Optional[float] = None

            try:
                start = time.perf_counter()
                with httpx.Client(
                    timeout=self.agent_timeout, headers=self._headers
                ) as client:
                    with client.stream(
                        "POST", self.agent_url, json=payload
                    ) as response:
                        response.raise_for_status()
                        for event_name, event_data in iter_sse_events(response):
                            try:
                                decoded = json.loads(event_data)
                            except json.JSONDecodeError:
                                logging.debug(
                                    "SSE %s: не JSON: %s",
                                    event_name,
                                    event_data[:100],
                                )
                                continue

                            if event_name == "citations":
                                # Извлекаем channel:message_id из citations
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
                                # Считаем refinements
                                content = decoded.get("content", "")
                                if "недостаточно" in content.lower() or "дополнительный" in content.lower():
                                    refinements += 1

                            elif event_name == "final":
                                final_payload = decoded
                                break

                latency = time.perf_counter() - start
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logging.warning(
                    "Ошибка при вызове агента (попытка %d/%d): %s",
                    attempt,
                    self.agent_retries,
                    exc,
                )
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
                    "latency_sec": latency,
                    "error": False,
                }

            last_error = "Agent stream завершён без события final"
            logging.warning(
                "Агент завершился без final (попытка %d/%d)",
                attempt,
                self.agent_retries,
            )

        return {
            "error": True,
            "error_message": last_error,
            "latency_sec": None,
            "citation_hits": [],
        }

    def _call_baseline(self, item: EvalItem) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": item.query}
        if self.collection:
            payload["collection"] = self.collection
        last_error: Optional[str] = None

        for attempt in range(1, self.baseline_retries + 1):
            start = time.perf_counter()
            try:
                response = httpx.post(
                    self.qa_url,
                    json=payload,
                    headers=self._headers,
                    timeout=self.baseline_timeout,
                )
                response.raise_for_status()
                data = response.json()
                latency = time.perf_counter() - start
                return {
                    "answer": data.get("answer"),
                    "latency_sec": latency,
                    "error": False,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logging.warning(
                    "Ошибка baseline QA (попытка %d/%d): %s",
                    attempt,
                    self.baseline_retries,
                    exc,
                )
                continue

        return {
            "error": True,
            "error_message": last_error,
            "latency_sec": None,
        }

    @staticmethod
    def _fake_agent_result() -> Dict[str, Any]:
        return {
            "answer": "[dry-run] agent answer",
            "citations": [],
            "coverage": None,
            "refinements": None,
            "verification": None,
            "fallback": False,
            "top5_hits": [],
            "latency_sec": None,
            "error": True,
            "error_message": "dry_run=true",
        }

    @staticmethod
    def _fake_baseline_result() -> Dict[str, Any]:
        return {
            "answer": "[dry-run] baseline answer",
            "latency_sec": None,
            "error": True,
            "error_message": "dry_run=true",
        }

    @staticmethod
    def _compute_metrics(
        item: EvalItem,
        agent_result: Dict[str, Any],
        baseline_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Recall считается по citation_hits (формат channel:message_id).
        Стратегия matching зависит от категории:
        - factual, channel_specific, comparative: точный match (channel + msg_id ±5)
        - temporal, multi_hop: мягкий match (channel совпадает + msg_id ±50)
          Для broad queries (temporal, multi_hop) нет единственно правильного поста —
          любой релевантный пост из нужного канала считается валидным.
        """
        citation_hits = agent_result.get("citation_hits") or []
        expected = item.expected_documents or []
        recall = None

        # Мягкие категории: temporal и multi_hop — более широкий fuzzy
        broad_categories = {"temporal", "multi_hop"}
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

        return {
            "agent_latency_sec": agent_result.get("latency_sec"),
            "baseline_latency_sec": baseline_result.get("latency_sec"),
            "agent_coverage": agent_result.get("coverage"),
            "recall_at_5": recall,
            "citation_hits": citation_hits,
            "expected_documents": expected,
            "agent_correct": None,
            "baseline_correct": None,
        }

    @staticmethod
    def _status(agent_result: Dict[str, Any], baseline_result: Dict[str, Any]) -> str:
        agent_err = bool(agent_result.get("error"))
        base_err = bool(baseline_result.get("error"))
        if agent_err and base_err:
            return "agent_and_baseline_error"
        if agent_err:
            return "agent_error"
        if base_err:
            return "baseline_error"
        return "ok"


def aggregate_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    agent_latencies = [
        r["metrics"]["agent_latency_sec"]
        for r in results
        if r["metrics"].get("agent_latency_sec") is not None
    ]
    baseline_latencies = [
        r["metrics"]["baseline_latency_sec"]
        for r in results
        if r["metrics"].get("baseline_latency_sec") is not None
    ]
    coverages = [
        r["metrics"]["agent_coverage"]
        for r in results
        if r["metrics"].get("agent_coverage") is not None
    ]
    recalls = [
        r["metrics"]["recall_at_5"]
        for r in results
        if r["metrics"].get("recall_at_5") is not None
    ]
    statuses = [r.get("status", "ok") for r in results]

    categories: Dict[str, Dict[str, List[float]]] = {}
    for record in results:
        cat = record["category"] or "uncategorized"
        categories.setdefault(
            cat,
            {"agent_latency_sec": [], "agent_coverage": [], "recall_at_5": []},
        )
        metrics = record["metrics"]
        if metrics.get("agent_latency_sec") is not None:
            categories[cat]["agent_latency_sec"].append(metrics["agent_latency_sec"])
        if metrics.get("agent_coverage") is not None:
            categories[cat]["agent_coverage"].append(metrics["agent_coverage"])
        if metrics.get("recall_at_5") is not None:
            categories[cat]["recall_at_5"].append(metrics["recall_at_5"])

    summary = {
        "total_queries": len(results),
        "answerable_queries": sum(1 for r in results if r["answerable"]),
        "negative_queries": sum(1 for r in results if not r["answerable"]),
        "errors": {
            "agent": sum(
                1 for s in statuses if s in {"agent_error", "agent_and_baseline_error"}
            ),
            "baseline": sum(
                1
                for s in statuses
                if s in {"baseline_error", "agent_and_baseline_error"}
            ),
            "both": sum(1 for s in statuses if s == "agent_and_baseline_error"),
        },
        "latency": {
            "agent": {
                "mean": safe_mean(agent_latencies),
                "p95": percentile(agent_latencies, 95),
                "max": max(agent_latencies) if agent_latencies else None,
            },
            "baseline": {
                "mean": safe_mean(baseline_latencies),
                "p95": percentile(baseline_latencies, 95),
                "max": max(baseline_latencies) if baseline_latencies else None,
            },
        },
        "coverage": {
            "mean": safe_mean(coverages),
            "min": min(coverages) if coverages else None,
            "max": max(coverages) if coverages else None,
        },
        "recall_at_5": {
            "mean": safe_mean(recalls),
            "queries_with_full_recall": sum(1 for r in recalls if r == 1.0),
            "queries_with_partial_recall": sum(
                1 for r in recalls if r not in (None, 0.0, 1.0)
            ),
        },
        "correctness": {
            "agent_validated": sum(
                1 for r in results if r["metrics"]["agent_correct"] is not None
            ),
            "baseline_validated": sum(
                1 for r in results if r["metrics"]["baseline_correct"] is not None
            ),
        },
    }

    by_category = {}
    for cat, values in categories.items():
        by_category[cat] = {
            "queries": sum(1 for r in results if r["category"] == cat),
            "agent_latency_mean": safe_mean(values["agent_latency_sec"]),
            "agent_coverage_mean": safe_mean(values["agent_coverage"]),
            "recall_at_5_mean": safe_mean(values["recall_at_5"]),
        }

    summary["by_category"] = by_category
    return summary


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def build_markdown_report(
    aggregated: Dict[str, Any],
    timestamp: datetime,
    dataset_path: Path,
) -> str:
    lines = [
        "# Agent Evaluation Report",
        f"**Date:** {timestamp.isoformat()}",
        f"**Dataset:** {dataset_path} ({aggregated['total_queries']} queries)",
        "",
        "## Overall Metrics",
    ]

    latency = aggregated["latency"]
    coverage = aggregated["coverage"]
    recall = aggregated["recall_at_5"]

    lines.extend(
        [
            f"- Agent Latency: mean={latency['agent']['mean']}, "
            f"p95={latency['agent']['p95']}, max={latency['agent']['max']}",
            f"- Baseline Latency: mean={latency['baseline']['mean']}, "
            f"p95={latency['baseline']['p95']}, max={latency['baseline']['max']}",
            f"- Agent Coverage: mean={coverage['mean']} "
            f"(min={coverage['min']}, max={coverage['max']})",
            f"- Recall@5: mean={recall['mean']} "
            f"(full={recall['queries_with_full_recall']}, "
            f"partial={recall['queries_with_partial_recall']})",
            "",
            "## By Category",
            "| Category | Queries | Avg Latency | Avg Coverage | Recall@5 |",
            "|----------|---------|-------------|--------------|----------|",
        ]
    )

    for category, stats in aggregated["by_category"].items():
        lines.append(
            f"| {category} | {stats['queries']} | "
            f"{stats['agent_latency_mean']} | "
            f"{stats['agent_coverage_mean']} | "
            f"{stats['recall_at_5_mean']} |"
        )

    lines.extend(
        [
            "",
            "## Next Steps",
            "- Провести ручную валидацию correctness (agent_correct/baseline_correct)",
            "- Запланировать Phase 2 метрики (LLM-judge, faithfulness)",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        dataset = load_dataset(args.dataset)
    except Exception as exc:  # noqa: BLE001
        logging.error("Не удалось загрузить датасет: %s", exc)
        return 1

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
    )

    try:
        raw_results = runner.run()
    except KeyboardInterrupt:
        logging.warning("Оценка прервана пользователем")
        return 130

    aggregated = aggregate_results(raw_results)

    timestamp = datetime.utcnow()
    raw_dir = args.output_dir / "raw"
    reports_dir = args.output_dir / "reports"
    ensure_dirs(args.output_dir, raw_dir, reports_dir)

    raw_path = raw_dir / f"eval_results_{timestamp:%Y%m%d-%H%M%S}.json"
    report_path = reports_dir / f"eval_report_{timestamp:%Y%m%d-%H%M%S}.json"

    write_json(raw_path, raw_results)
    write_json(report_path, aggregated)

    markdown_path = None
    if not args.skip_markdown:
        markdown_path = reports_dir / f"eval_report_{timestamp:%Y%m%d-%H%M%S}.md"
        markdown = build_markdown_report(aggregated, timestamp, args.dataset)
        markdown_path.write_text(markdown, encoding="utf-8")

    logging.info("Raw results сохранены: %s", raw_path)
    logging.info("Aggregated report сохранен: %s", report_path)
    if markdown_path:
        logging.info("Markdown summary сохранен: %s", markdown_path)

    print(json.dumps(aggregated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
