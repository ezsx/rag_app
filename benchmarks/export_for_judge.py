"""Export agent benchmark results as JSON artifact for offline judge.

Генерирует JSON файл с 17 вопросами × 4 ответами,
готовый для отправки в чат Claude/Codex как judge artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.config import RESULTS_DIR


def export_for_judge(results_path: str | None = None, output_path: str | None = None) -> str:
    """Читает agent_answers.json и генерирует judge-friendly artifact."""
    if results_path is None:
        results_path = str(Path(RESULTS_DIR) / "agent_answers.json")
    if output_path is None:
        output_path = str(Path(RESULTS_DIR) / "judge_artifact.json")

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    artifact = []
    for item in data:
        entry = {
            "question_id": item["question_id"],
            "query": item["query"],
            "expected_answer": item.get("expected_answer", ""),
            "category": item.get("category", ""),
            "answers": {},
        }
        for pipeline_name, result in item.get("answers", {}).items():
            entry["answers"][pipeline_name] = {
                "answer": result.get("answer", ""),
                "tool_calls": result.get("tool_calls", []),
                "latency": result.get("latency", 0),
                "source_documents": result.get("docs", []),
            }
        artifact.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    path = export_for_judge()
    print(f"Judge artifact exported to: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Questions: {len(data)}")
    print(f"Pipelines per question: {list(data[0]['answers'].keys()) if data else 'none'}")
