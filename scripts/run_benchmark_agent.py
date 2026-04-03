#!/usr/bin/env python3
"""
Agent E2E benchmark runner — 4 pipelines × 17 retrieval questions.

Запуск:
    docker compose -f deploy/compose/compose.benchmark.yml run --rm benchmark \
        python scripts/run_benchmark_agent.py [--dataset PATH] [--pipelines LIST]

Выход: benchmarks/results/agent_answers.json + judge artifact + markdown таблица.

После запуска: export judge_artifact.json → отправить в чат Claude/Codex для offline judge.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from benchmarks.config import AGENT_DATASET, AGENT_RETRIEVAL_IDS, RESULTS_DIR
from benchmarks.export_for_judge import export_for_judge


def load_dataset(path: str) -> list[dict]:
    """Загружает golden v2 и фильтрует по exact ID list."""
    with open(path, encoding="utf-8") as f:
        all_items = json.load(f)

    filtered = [item for item in all_items if item["id"] in AGENT_RETRIEVAL_IDS]
    print(f"Dataset: {len(all_items)} total, {len(filtered)} retrieval questions selected")

    missing = set(AGENT_RETRIEVAL_IDS) - {item["id"] for item in filtered}
    if missing:
        print(f"WARNING: {len(missing)} IDs not found in dataset: {missing}")

    return filtered


def run_pipeline(pipeline_name: str, agent, items: list[dict]) -> list[dict]:
    """Прогоняет один agent pipeline по всем вопросам."""
    results = []

    for i, item in enumerate(items):
        query = item["query"]
        qid = item["id"]
        print(f"  [{pipeline_name}] [{i+1}/{len(items)}] {qid}: {query[:60]}...")

        try:
            result = agent.run(query)
            docs = [
                {"doc_id": d.doc_id, "channel": d.channel, "message_id": d.message_id, "text": (d.text or "")[:500]}
                for d in result.docs
            ]
            results.append({
                "question_id": qid,
                "answer": result.answer,
                "tool_calls": result.tool_calls,
                "latency": result.latency,
                "docs": docs,
                "error": None,
            })
            answer_preview = result.answer[:100].replace("\n", " ")
            print(f"    → {result.latency:.1f}s, tools={result.tool_calls}, answer={answer_preview}...")
        except Exception as e:
            print(f"    → ERROR: {e}")
            results.append({
                "question_id": qid,
                "answer": "",
                "tool_calls": [],
                "latency": 0,
                "error": str(e),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Agent E2E benchmark — 4 pipelines")
    parser.add_argument("--dataset", default=AGENT_DATASET, help="Path to golden dataset JSON")
    parser.add_argument("--pipelines", default="naive,li_stock,li_maxed,custom",
                        help="Comma-separated pipeline names")
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    if not items:
        print("No questions found. Exiting.")
        return

    pipelines_to_run = [p.strip() for p in args.pipelines.split(",")]
    all_answers: dict[str, list[dict]] = {}

    for name in pipelines_to_run:
        print(f"\n{'='*60}")
        print(f"Running agent pipeline: {name}")
        print(f"{'='*60}")

        if name == "naive":
            from benchmarks.naive.agent import NaiveAgent
            agent = NaiveAgent()
        elif name == "li_stock":
            from benchmarks.llamaindex_pipeline.agent import LlamaIndexAgentStock
            agent = LlamaIndexAgentStock()
        elif name == "li_maxed":
            from benchmarks.llamaindex_pipeline.agent import LlamaIndexAgentMaxed
            agent = LlamaIndexAgentMaxed()
        elif name == "custom":
            from benchmarks.custom_adapter.agent import CustomAgent
            agent = CustomAgent()
        else:
            print(f"Unknown pipeline: {name}, skipping")
            continue

        results = run_pipeline(name, agent, items)
        all_answers[name] = results

    # Собираем в единый формат: вопрос → {pipeline: answer}
    os.makedirs(RESULTS_DIR, exist_ok=True)

    combined = []
    for i, item in enumerate(items):
        entry = {
            "question_id": item["id"],
            "query": item["query"],
            "expected_answer": item.get("expected_answer", ""),
            "category": item.get("category", ""),
            "answers": {},
        }
        for pipeline_name, results in all_answers.items():
            if i < len(results):
                entry["answers"][pipeline_name] = results[i]
        combined.append(entry)

    # Сохраняем ответы
    answers_path = Path(RESULTS_DIR) / "agent_answers.json"
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nAnswers saved to {answers_path}")

    # Генерируем judge artifact
    judge_path = export_for_judge(str(answers_path))
    print(f"Judge artifact saved to {judge_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("AGENT E2E BENCHMARK SUMMARY")
    print(f"{'='*80}")
    header = f"{'Pipeline':<12} | {'Answered':>8} | {'Errors':>6} | {'Avg Latency':>11} | {'Avg Tools':>9}"
    print(header)
    print("-" * len(header))

    for pipeline_name, results in all_answers.items():
        answered = sum(1 for r in results if r["answer"] and not r.get("error"))
        errors = sum(1 for r in results if r.get("error"))
        latencies = [r["latency"] for r in results if r["latency"] > 0]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        avg_tools = sum(len(r["tool_calls"]) for r in results) / len(results) if results else 0
        print(f"{pipeline_name:<12} | {answered:>8} | {errors:>6} | {avg_lat:>10.1f}s | {avg_tools:>9.1f}")

    print(f"\nNext step: send {judge_path} to Claude/Codex chat for offline factual + usefulness scoring.")


if __name__ == "__main__":
    main()
