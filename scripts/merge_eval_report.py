#!/usr/bin/env python3
"""Merge eval sources → final report (SPEC-RAG-21).

Объединяет:
  --eval-results    eval_results_YYYYMMDD.json (agent performance)
  --judge-verdicts  judge_verdicts_YYYYMMDD.json (Claude judge)
  --claims          claims_YYYYMMDD.json (Claude decomposition)
  --nli-scores      nli_scores_YYYYMMDD.json (XLM-RoBERTa NLI)
Выход:
  results/reports/final_eval_YYYYMMDD.json + .md
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("merge_eval")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_reports(
    eval_results: dict,
    judge_verdicts: dict,
    claims_data: dict,
    nli_scores: dict,
) -> dict:
    """Объединяет все источники в единый отчёт."""

    questions = eval_results.get("per_question", [])

    # Index judge verdicts by id
    verdicts_map: Dict[str, dict] = {}
    vlist = judge_verdicts.get("verdicts", judge_verdicts if isinstance(judge_verdicts, list) else [])
    for v in vlist:
        vid = v.get("id", v.get("query_id", ""))
        verdicts_map[vid] = v

    # Index claims by id
    claims_map: Dict[str, list] = {}
    if isinstance(claims_data, dict) and "questions" in claims_data:
        for q in claims_data["questions"]:
            claims_map[q["id"]] = q.get("claims", [])
    elif isinstance(claims_data, list):
        for q in claims_data:
            claims_map[q.get("id", "")] = q.get("claims", [])

    # Index NLI scores by query_id
    nli_map: Dict[str, dict] = {}
    for nq in nli_scores.get("per_question", []):
        nli_map[nq["query_id"]] = nq

    # Merge per question
    merged_questions: List[dict] = []
    for q in questions:
        qid = q.get("query_id", "")
        verdict = verdicts_map.get(qid, {})
        nli = nli_map.get(qid, {})
        q_claims = claims_map.get(qid, [])

        merged = {
            "query_id": qid,
            "query": q.get("query", q.get("offline_judge_packet", {}).get("query", "")),
            "eval_mode": q.get("eval_mode", ""),
            "category": q.get("category", ""),
            # Agent performance
            "latency_sec": q.get("metrics", {}).get("agent_latency_sec"),
            "tools_invoked": q.get("agent", {}).get("tools_invoked", []),
            "coverage": q.get("agent", {}).get("coverage"),
            "key_tool_accuracy": q.get("metrics", {}).get("key_tool_accuracy"),
            # Judge
            "factual": verdict.get("factual"),
            "useful": verdict.get("useful"),
            "reasoning": verdict.get("reasoning", ""),
            # Claims
            "claims": q_claims,
            "claims_count": len(q_claims),
            "claims_verifiable": sum(1 for c in q_claims if c.get("type") == "verifiable"),
            # NLI faithfulness
            "faithfulness": nli.get("faithfulness"),
            "faithfulness_strict": nli.get("faithfulness_strict"),
            "claims_supported": nli.get("claims_supported", 0),
            "claims_contradicted": nli.get("claims_contradicted", 0),
            "claims_neutral": nli.get("claims_neutral", 0),
            "contradictions": nli.get("contradictions", []),
            "per_claim_nli": nli.get("per_claim", []),
        }
        merged_questions.append(merged)

    # Aggregate metrics
    all_factual = [q["factual"] for q in merged_questions if q["factual"] is not None]
    all_useful = [q["useful"] for q in merged_questions if q["useful"] is not None]
    all_kta = [q["key_tool_accuracy"] for q in merged_questions if q["key_tool_accuracy"] is not None]
    all_latency = [q["latency_sec"] for q in merged_questions if q["latency_sec"] is not None]

    retrieval = [q for q in merged_questions if q["faithfulness"] is not None]
    all_faith = [q["faithfulness"] for q in retrieval]
    all_faith_strict = [q["faithfulness_strict"] for q in retrieval if q["faithfulness_strict"] is not None]

    nli_agg = nli_scores.get("aggregate", {})

    aggregate = {
        "factual": round(sum(all_factual) / len(all_factual), 4) if all_factual else None,
        "useful": round(sum(all_useful) / len(all_useful), 4) if all_useful else None,
        "kta": round(sum(all_kta) / len(all_kta), 4) if all_kta else None,
        "faithfulness": round(sum(all_faith) / len(all_faith), 4) if all_faith else None,
        "faithfulness_strict": round(sum(all_faith_strict) / len(all_faith_strict), 4) if all_faith_strict else None,
        "citation_precision": nli_agg.get("citation_precision"),
        "latency_mean": round(sum(all_latency) / len(all_latency), 1) if all_latency else None,
        "total_questions": len(merged_questions),
        "retrieval_questions": len(retrieval),
        "total_contradictions": sum(len(q["contradictions"]) for q in merged_questions),
    }

    all_contradictions = []
    for q in merged_questions:
        for c in q.get("contradictions", []):
            c["query_id"] = q["query_id"]
            all_contradictions.append(c)

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "judge_prompt_version": "v1",
            "nli_metadata": nli_scores.get("metadata", {}),
        },
        "aggregate": aggregate,
        "contradictions": all_contradictions,
        "per_question": merged_questions,
    }


def build_markdown(report: dict) -> str:
    """Строит человекочитаемый markdown отчёт."""
    agg = report["aggregate"]
    lines = [
        "# Final Eval Report",
        "",
        f"Date: {report['metadata']['timestamp'][:10]}",
        f"Judge prompt: {report['metadata']['judge_prompt_version']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value | Scope |",
        "|--------|-------|-------|",
        f"| **Factual** | **{agg['factual']}** | All {agg['total_questions']} Qs |",
        f"| **Useful** | **{agg['useful']}** | All {agg['total_questions']} Qs |",
        f"| **KTA** | **{agg['kta']}** | All {agg['total_questions']} Qs |",
        f"| **Faithfulness** | **{agg['faithfulness']}** | {agg['retrieval_questions']} retrieval Qs |",
        f"| **Faithfulness (strict)** | **{agg['faithfulness_strict']}** | {agg['retrieval_questions']} retrieval Qs |",
        f"| **Citation Precision** | **{agg['citation_precision']}** | {agg['retrieval_questions']} retrieval Qs |",
        f"| **Latency** | **{agg['latency_mean']}s** | All {agg['total_questions']} Qs |",
        "",
    ]

    # Contradictions
    contradictions = report.get("contradictions", [])
    if contradictions:
        lines.extend([
            "## Contradictions (hallucination signals)",
            "",
            "| Question | Claim | Document | Score |",
            "|----------|-------|----------|-------|",
        ])
        for c in contradictions:
            lines.append(
                f"| {c.get('query_id', '?')} | {c.get('claim', '?')[:80]} | {c.get('document_id', '?')} | {c.get('contradiction_score', '?')} |"
            )
        lines.extend(["", ""])

    # Per question summary
    lines.extend([
        "## Per Question",
        "",
        "| ID | Mode | Factual | Useful | Faithfulness | Claims | Supported | Contradicted |",
        "|----|------|---------|--------|-------------|--------|-----------|-------------|",
    ])
    for q in report["per_question"]:
        faith = q.get("faithfulness")
        faith_str = f"{faith:.3f}" if faith is not None else "N/A"
        lines.append(
            f"| {q['query_id']} | {q['eval_mode']} | {q.get('factual', '?')} | {q.get('useful', '?')} "
            f"| {faith_str} | {q.get('claims_verifiable', 0)} | {q.get('claims_supported', 0)} | {q.get('claims_contradicted', 0)} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Merge eval sources → final report (SPEC-RAG-21)")
    parser.add_argument("--eval-results", required=True, help="eval_results JSON")
    parser.add_argument("--judge-verdicts", required=True, help="judge_verdicts JSON")
    parser.add_argument("--claims", required=True, help="claims JSON")
    parser.add_argument("--nli-scores", required=True, help="nli_scores JSON")
    parser.add_argument("--output-dir", default="results/reports", help="Output directory")
    args = parser.parse_args()

    eval_results = load_json(Path(args.eval_results))
    judge_verdicts = load_json(Path(args.judge_verdicts))
    claims_data = load_json(Path(args.claims))
    nli_scores = load_json(Path(args.nli_scores))

    report = merge_reports(eval_results, judge_verdicts, claims_data, nli_scores)

    ts = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"final_eval_{ts}.json"
    md_path = out_dir / f"final_eval_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path.write_text(build_markdown(report), encoding="utf-8")

    agg = report["aggregate"]
    logger.info("=== Final Report ===")
    logger.info("Factual: %s | Useful: %s | KTA: %s", agg["factual"], agg["useful"], agg["kta"])
    logger.info("Faithfulness: %s (strict: %s)", agg["faithfulness"], agg["faithfulness_strict"])
    logger.info("Citation Precision: %s", agg["citation_precision"])
    logger.info("Contradictions: %d", agg["total_contradictions"])
    logger.info("JSON: %s", json_path)
    logger.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
