#!/usr/bin/env python3
"""Обогащает benchmark artifacts документами custom pipeline из Langfuse traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

if __package__ in (None, ""):
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    sys.path.insert(0, str(_root / "src"))

from benchmarks.config import RESULTS_DIR  # noqa: E402


DEFAULT_HOST = os.environ.get("LANGFUSE_EXPORT_HOST", "http://localhost:3100")
DEFAULT_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-rag-app-dev")
DEFAULT_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-rag-app-dev")
MAX_PAGE_SIZE = 100


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover custom pipeline docs from Langfuse traces."
    )
    parser.add_argument(
        "--answers-path",
        default=str(Path(RESULTS_DIR) / "agent_answers.json"),
        help="Path to benchmark agent answers JSON",
    )
    parser.add_argument(
        "--judge-path",
        default=str(Path(RESULTS_DIR) / "judge_artifact.json"),
        help="Path to judge artifact JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help="Directory for enriched output files",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Langfuse host URL",
    )
    parser.add_argument(
        "--public-key",
        default=DEFAULT_PUBLIC_KEY,
        help="Langfuse public key",
    )
    parser.add_argument(
        "--secret-key",
        default=DEFAULT_SECRET_KEY,
        help="Langfuse secret key",
    )
    parser.add_argument(
        "--pipeline",
        default="custom",
        help="Pipeline name to enrich",
    )
    parser.add_argument(
        "--trace-cache-path",
        default=str(Path(RESULTS_DIR) / "langfuse_custom_traces.jsonl"),
        help="Optional local JSONL cache with full Langfuse traces",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="How many Langfuse list pages to scan (100 traces per page)",
    )
    return parser.parse_args()


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _maybe_json(value: Any) -> Any:
    """Пытается декодировать JSON-строку, иначе возвращает value как есть."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _trace_root(trace: dict[str, Any]) -> dict[str, Any]:
    """Langfuse export бывает в двух формах: trace-root или {trace, observations}."""
    nested = trace.get("trace")
    return nested if isinstance(nested, dict) else trace


def _trace_observations(trace: dict[str, Any]) -> list[dict[str, Any]]:
    root = _trace_root(trace)
    observations = trace.get("observations")
    if isinstance(observations, list) and observations:
        return observations
    nested = root.get("observations")
    return nested if isinstance(nested, list) else []


def _extract_trace_query(trace: dict[str, Any]) -> str:
    root = _trace_root(trace)
    trace_input = _maybe_json(root.get("input"))
    if isinstance(trace_input, dict):
        query = trace_input.get("query")
        return str(query).strip() if query else ""
    return ""


def _extract_trace_answer(trace: dict[str, Any]) -> str:
    """Берёт полный финальный answer из tool:final_answer, root output — только fallback."""
    observations = _trace_observations(trace)
    final_obs = next(
        (obs for obs in observations if obs.get("name") == "tool:final_answer"),
        None,
    )
    if final_obs is not None:
        final_input = _maybe_json(final_obs.get("input"))
        if isinstance(final_input, dict):
            answer = final_input.get("answer")
            if answer:
                return str(answer).strip()
        final_output = _maybe_json(final_obs.get("output"))
        if isinstance(final_output, dict):
            answer = final_output.get("answer")
            if answer:
                return str(answer).strip()

    root = _trace_root(trace)
    trace_output = _maybe_json(root.get("output"))
    if isinstance(trace_output, dict):
        answer = trace_output.get("answer")
        if answer:
            return str(answer).strip()

    return ""


def _iter_recent_traces(
    client: httpx.Client,
    *,
    max_pages: int,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        resp = client.get(
            "/api/public/traces",
            params={"limit": MAX_PAGE_SIZE, "page": page, "orderBy": "timestamp.desc"},
        )
        resp.raise_for_status()
        batch = resp.json().get("data", [])
        if not batch:
            break
        traces.extend(batch)
        if len(batch) < MAX_PAGE_SIZE:
            break
    return traces


def _load_trace_cache(path: str) -> list[dict[str, Any]]:
    trace_path = Path(path)
    if not trace_path.exists():
        return []

    traces: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return traces


def _normalize_trace_docs(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Берёт финальные документы из compose_context input."""
    observations = _trace_observations(trace)
    compose_obs = next(
        (obs for obs in observations if obs.get("name") == "tool:compose_context"),
        None,
    )
    if compose_obs is None:
        return []

    raw_docs = (compose_obs.get("input") or {}).get("docs") or []
    docs: list[dict[str, Any]] = []
    for doc in raw_docs:
        metadata = doc.get("metadata") or {}
        channel = metadata.get("channel") or ""
        message_id = metadata.get("message_id") or 0
        docs.append(
            {
                "doc_id": f"{channel}:{message_id}" if channel and message_id else "",
                "channel": channel,
                "message_id": int(message_id) if message_id else 0,
                "text": doc.get("text") or "",
            }
        )
    return docs


def _normalize_tool_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for hit in hits:
        meta = hit.get("meta") or {}
        channel = meta.get("channel") or ""
        message_id = meta.get("message_id") or 0
        docs.append(
            {
                "doc_id": f"{channel}:{message_id}" if channel and message_id else "",
                "channel": channel,
                "message_id": int(message_id) if message_id else 0,
                "text": hit.get("text") or "",
            }
        )
    return docs


def _recover_summarize_channel_docs(trace: dict[str, Any]) -> list[dict[str, Any]]:
    observations = _trace_observations(trace)
    summarize_obs = next(
        (obs for obs in observations if obs.get("name") == "tool:summarize_channel"),
        None,
    )
    if summarize_obs is None:
        return []

    params = summarize_obs.get("input") or {}
    channel = params.get("channel") or ""
    time_range = params.get("time_range") or "week"
    if not channel:
        return []

    from core.deps import get_hybrid_retriever
    from services.tools.summarize_channel import summarize_channel

    hybrid_retriever = get_hybrid_retriever()
    if hybrid_retriever is None:
        return []

    result = summarize_channel(
        channel=channel,
        time_range=time_range,
        hybrid_retriever=hybrid_retriever,
    )
    return _normalize_tool_hits(result.get("hits") or [])


def _build_query_map(agent_answers: list[dict[str, Any]], pipeline: str) -> dict[str, dict[str, Any]]:
    missing: dict[str, dict[str, Any]] = {}
    for item in agent_answers:
        answer = item.get("answers", {}).get(pipeline, {})
        if answer.get("docs") and answer.get("answer"):
            continue
        query = item.get("query")
        if query:
            missing[query] = item
    return missing


def _choose_latest_traces_by_query(
    traces: list[dict[str, Any]],
    *,
    queries: set[str],
) -> dict[str, dict[str, Any]]:
    matched: dict[str, dict[str, Any]] = {}
    for trace in traces:
        query = _extract_trace_query(trace)
        if query not in queries or query in matched:
            continue
        docs = _normalize_trace_docs(trace)
        if not docs:
            docs = _recover_summarize_channel_docs(trace)
        answer = _extract_trace_answer(trace)
        if not docs and not answer:
            continue
        matched[query] = {
            "trace_id": _trace_root(trace).get("id") or trace.get("id"),
            "timestamp": _trace_root(trace).get("timestamp") or trace.get("timestamp"),
            "docs": docs,
            "answer": answer,
        }
        if len(matched) == len(queries):
            break
    return matched


def _enrich_agent_answers(
    agent_answers: list[dict[str, Any]],
    *,
    pipeline: str,
    matched: dict[str, dict[str, Any]],
) -> int:
    updated = 0
    for item in agent_answers:
        query = item.get("query")
        trace_info = matched.get(query)
        if not trace_info:
            continue
        answer = item.setdefault("answers", {}).setdefault(pipeline, {})
        changed = False
        if not answer.get("docs") and trace_info.get("docs"):
            answer["docs"] = trace_info["docs"]
            changed = True
        current_answer = str(answer.get("answer") or "")
        trace_answer = str(trace_info.get("answer") or "")
        if trace_answer and len(trace_answer) > len(current_answer):
            answer["answer"] = trace_info["answer"]
            changed = True
        if changed or trace_info.get("trace_id"):
            answer["langfuse_trace_id"] = trace_info["trace_id"]
        if changed:
            updated += 1
    return updated


def _enrich_judge_artifact(
    judge_artifact: list[dict[str, Any]],
    *,
    pipeline: str,
    matched: dict[str, dict[str, Any]],
) -> int:
    updated = 0
    for item in judge_artifact:
        query = item.get("query")
        trace_info = matched.get(query)
        if not trace_info:
            continue
        answer = item.setdefault("answers", {}).setdefault(pipeline, {})
        changed = False
        if not answer.get("source_documents") and trace_info.get("docs"):
            answer["source_documents"] = trace_info["docs"]
            changed = True
        current_answer = str(answer.get("answer") or "")
        trace_answer = str(trace_info.get("answer") or "")
        if trace_answer and len(trace_answer) > len(current_answer):
            answer["answer"] = trace_info["answer"]
            changed = True
        if changed or trace_info.get("trace_id"):
            answer["langfuse_trace_id"] = trace_info["trace_id"]
        if changed:
            updated += 1
    return updated


def main() -> int:
    args = _parse_args()
    agent_answers = _load_json(args.answers_path)
    judge_artifact = _load_json(args.judge_path)

    missing_by_query = _build_query_map(agent_answers, args.pipeline)
    if not missing_by_query:
        print(f"No missing docs for pipeline '{args.pipeline}'.")
        return 0

    matched: dict[str, dict[str, Any]] = {}
    cached_traces = _load_trace_cache(args.trace_cache_path)
    if cached_traces:
        matched.update(
            _choose_latest_traces_by_query(
                cached_traces,
                queries=set(missing_by_query),
            )
        )

    unresolved_queries = set(missing_by_query) - set(matched)
    if unresolved_queries:
        with httpx.Client(
            base_url=args.host.rstrip("/"),
            auth=(args.public_key, args.secret_key),
            timeout=30.0,
        ) as client:
            recent = _iter_recent_traces(client, max_pages=args.max_pages)
            for short_trace in recent:
                trace_id = short_trace.get("id")
                if not trace_id:
                    continue
                full = client.get(f"/api/public/traces/{trace_id}")
                full.raise_for_status()
                trace = full.json()
                query = _extract_trace_query(trace)
                if query not in unresolved_queries or query in matched:
                    continue
                docs = _normalize_trace_docs(trace)
                if not docs:
                    docs = _recover_summarize_channel_docs(trace)
                answer = _extract_trace_answer(trace)
                if not docs and not answer:
                    continue
                matched[query] = {
                    "trace_id": _trace_root(trace).get("id") or trace.get("id"),
                    "timestamp": _trace_root(trace).get("timestamp") or trace.get("timestamp"),
                    "docs": docs,
                    "answer": answer,
                }
                if len(matched) == len(missing_by_query):
                    break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updated_answers = _enrich_agent_answers(
        agent_answers,
        pipeline=args.pipeline,
        matched=matched,
    )
    updated_judge = _enrich_judge_artifact(
        judge_artifact,
        pipeline=args.pipeline,
        matched=matched,
    )

    answers_out = output_dir / "agent_answers.langfuse_enriched.json"
    judge_out = output_dir / "judge_artifact.langfuse_enriched.json"
    answers_out.write_text(
        json.dumps(agent_answers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    judge_out.write_text(
        json.dumps(judge_artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    unresolved = sorted(set(missing_by_query) - set(matched))
    print(f"Recovered docs for {updated_answers} agent answers and {updated_judge} judge entries.")
    print(f"Agent answers: {answers_out}")
    print(f"Judge artifact: {judge_out}")
    if unresolved:
        print(f"Unresolved queries: {len(unresolved)}")
        for query in unresolved:
            print(f"  - {query}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
