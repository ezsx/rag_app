#!/usr/bin/env python3
"""Статистические confidence intervals для eval метрик (SPEC-RAG-22 Layer 3).

Поддерживает два формата входа:
- raw eval JSON из `scripts/evaluate_agent.py`
- aggregated `results.yaml` из `experiments/runs/*`

Вычисляет:
- Bootstrap CIs для continuous metrics
- Wilson intervals для binary metrics
- Paired bootstrap test для A/B comparisons

Использование:
    python scripts/compute_confidence.py results/raw/eval_results_YYYYMMDD.json
    python scripts/compute_confidence.py experiments/runs/RUN-008/results.yaml --dataset datasets/eval_golden_v2_fixed.json --portfolio
    python scripts/compute_confidence.py a.json --compare b.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ─── Bootstrap CI ─────────────────────────────────────────────────


def bootstrap_ci(
    scores: list[float],
    B: int = 1000,
    alpha: float = 0.05,
    statistic=np.mean,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval (percentile method).

    Returns: (mean, ci_lower, ci_upper).
    """
    if not scores:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.array(scores)
    observed = float(statistic(arr))
    n = len(arr)

    boot_stats = np.array([
        float(statistic(np.random.choice(arr, n, replace=True)))
        for _ in range(B)
    ])

    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (observed, lo, hi)


# ─── Wilson interval для binary ───────────────────────────────────


def wilson_ci(
    successes: int,
    total: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Wilson score interval для пропорций (лучше normal approx для малых n).

    Returns: (proportion, ci_lower, ci_upper).
    """
    if total == 0:
        return (float("nan"), float("nan"), float("nan"))

    p = successes / total
    z = 1.96  # z_{alpha/2} для 95% CI
    if alpha != 0.05:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)

    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom

    return (p, max(0.0, centre - margin), min(1.0, centre + margin))


# ─── Paired bootstrap test для A/B ───────────────────────────────


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    B: int = 10000,
) -> tuple[float, float, float]:
    """Paired bootstrap: is system B better than A?

    Returns: (mean_diff, p_value, ci_diff_95).
    Positive mean_diff = B is better.
    """
    assert len(scores_a) == len(scores_b), "Paired test requires same questions"

    diffs = np.array(scores_b) - np.array(scores_a)
    observed = float(np.mean(diffs))

    boot_means = np.array([
        float(np.mean(np.random.choice(diffs, len(diffs), replace=True)))
        for _ in range(B)
    ])

    # One-sided p-value: P(diff ≤ 0) under bootstrap
    p_value = float(np.mean(boot_means <= 0))
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))

    return (observed, p_value, ci)


# ─── Extract metrics from eval results ────────────────────────────


CONTINUOUS_METRICS = [
    "factual_correctness", "usefulness", "bertscore_f1", "summac_faithfulness",
    "precision_at_5", "mrr", "ndcg_at_5", "agent_latency_sec", "agent_coverage",
    "strict_anchor_recall", "tool_call_f1",
]

BINARY_METRICS = [
    "key_tool_accuracy", "acceptable_set_hit",
]

RETRIEVAL_MODES = {"retrieval_evidence", "retrieval"}


def normalize_query_id(query_id: str | None) -> str | None:
    """Нормализует qid между dataset (`golden_q01`) и results (`q01`)."""
    if not query_id:
        return None
    return query_id.removeprefix("golden_")


def load_structured_file(path: Path) -> Any:
    """Загружает JSON/YAML по расширению файла."""
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        return json.load(f)


def load_dataset_index(path: Path | None) -> dict[str, dict[str, Any]]:
    """Строит индекс dataset item по query id для mode/category lookup."""
    if path is None:
        return {}

    data = load_structured_file(path)
    items = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Unsupported dataset format: {path}")

    index: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        qid = item.get("id") or item.get("query_id")
        if qid:
            index[qid] = item
            normalized_qid = normalize_query_id(qid)
            if normalized_qid:
                index[normalized_qid] = item
    return index


def normalize_results(data: Any, dataset_index: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Нормализует raw eval JSON и aggregated results.yaml к единому виду."""
    dataset_index = dataset_index or {}

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("Unsupported results payload")

    per_question = data.get("per_question", data)

    if isinstance(per_question, list):
        return per_question

    if isinstance(per_question, dict):
        normalized = []
        for qid, values in per_question.items():
            values = values or {}
            dataset_item = dataset_index.get(qid, {})
            normalized.append(
                {
                    "query_id": qid,
                    "eval_mode": values.get("mode") or dataset_item.get("eval_mode"),
                    "category": values.get("category") or dataset_item.get("category"),
                    "metrics": {
                        "factual_correctness": values.get("factual"),
                        "usefulness": values.get("useful"),
                        "evidence": values.get("evidence"),
                        "sufficiency": values.get("sufficiency"),
                    },
                }
            )
        return normalized

    raise ValueError("Unsupported per_question format")


def extract_metric(results: list[dict], metric: str) -> list[float]:
    """Извлечь per-question metric values из eval results."""
    values = []
    for r in results:
        m = r.get("metrics", {})
        v = m.get(metric)
        if v is not None:
            values.append(float(v))
    return values


def extract_metric_with_filter(
    results: list[dict],
    metric: str,
    predicate,
) -> list[float]:
    """Извлекает metric только для подмножества вопросов."""
    values = []
    for r in results:
        if not predicate(r):
            continue
        m = r.get("metrics", {})
        v = m.get(metric)
        if v is not None:
            values.append(float(v))
    return values


def compute_all_cis(results: list[dict], B: int = 1000) -> dict[str, Any]:
    """Вычислить CIs для всех метрик."""
    cis = {}

    for metric in CONTINUOUS_METRICS:
        values = extract_metric(results, metric)
        if values:
            mean, lo, hi = bootstrap_ci(values, B=B)
            cis[metric] = {
                "mean": round(mean, 4),
                "ci_95_lower": round(lo, 4),
                "ci_95_upper": round(hi, 4),
                "ci_width": round(hi - lo, 4),
                "n": len(values),
                "method": "bootstrap_percentile",
            }

    for metric in BINARY_METRICS:
        values = extract_metric(results, metric)
        if values:
            successes = sum(1 for v in values if v >= 0.5)
            total = len(values)
            prop, lo, hi = wilson_ci(successes, total)
            cis[metric] = {
                "proportion": round(prop, 4),
                "ci_95_lower": round(lo, 4),
                "ci_95_upper": round(hi, 4),
                "ci_width": round(hi - lo, 4),
                "n": total,
                "method": "wilson",
            }

    return cis


def build_ci_report(cis: dict[str, Any]) -> str:
    """Markdown отчёт с confidence intervals."""
    lines = [
        "# Confidence Intervals Report",
        "",
        "| Metric | Value | 95% CI | Width | N | Method |",
        "|--------|-------|--------|-------|---|--------|",
    ]
    for metric, data in sorted(cis.items()):
        val = data.get("mean", data.get("proportion", ""))
        lo = data["ci_95_lower"]
        hi = data["ci_95_upper"]
        lines.append(
            f"| {metric} | {val:.3f} | [{lo:.3f}, {hi:.3f}] | {data['ci_width']:.3f} | {data['n']} | {data['method']} |"
        )
    return "\n".join(lines)


def build_portfolio_report(results: list[dict], B: int = 10000) -> str:
    """Короткий отчёт для README/roadmap по golden dataset."""
    slices = [
        (
            "Factual (all)",
            extract_metric(results, "factual_correctness"),
        ),
        (
            "Factual (retrieval)",
            extract_metric_with_filter(
                results,
                "factual_correctness",
                lambda r: r.get("eval_mode") in RETRIEVAL_MODES,
            ),
        ),
        (
            "Factual (analytics)",
            extract_metric_with_filter(
                results,
                "factual_correctness",
                lambda r: r.get("eval_mode") == "analytics",
            ),
        ),
        (
            "Useful (all)",
            extract_metric(results, "usefulness"),
        ),
    ]

    lines = [
        "=" * 60,
        "Bootstrap Confidence Intervals (95%)",
        "=" * 60,
    ]

    for name, scores in slices:
        if not scores:
            continue
        mean, lo, hi = bootstrap_ci(scores, B=B)
        margin = (hi - lo) / 2
        lines.extend(
            [
                "",
                f"{name}:",
                f"  {mean:.3f} ± {margin:.3f} (95% CI [{lo:.3f}, {hi:.3f}], n={len(scores)})",
            ]
        )

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compute confidence intervals for eval metrics")
    parser.add_argument("results", type=Path, help="Path to eval_results JSON")
    parser.add_argument("--compare", type=Path, default=None, help="Second results file for A/B test")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap resamples (B)")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset JSON/YAML for eval_mode/category join")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--portfolio", action="store_true", help="Print compact portfolio report for results.yaml/golden set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible bootstrap")
    args = parser.parse_args()

    np.random.seed(args.seed)
    dataset_index = load_dataset_index(args.dataset)
    data = load_structured_file(args.results)
    results = normalize_results(data, dataset_index=dataset_index)

    print(f"Computing CIs for {len(results)} questions (B={args.bootstrap_samples})...")
    cis = compute_all_cis(results, B=args.bootstrap_samples)

    # A/B comparison
    ab_results = None
    if args.compare:
        data_b = load_structured_file(args.compare)
        results_b = normalize_results(data_b, dataset_index=dataset_index)
        ab_results = {}
        for metric in CONTINUOUS_METRICS:
            va = extract_metric(results, metric)
            vb = extract_metric(results_b, metric)
            if va and vb and len(va) == len(vb):
                diff, p_val, ci_diff = paired_bootstrap_test(va, vb, B=args.bootstrap_samples)
                ab_results[metric] = {
                    "mean_diff": round(diff, 4),
                    "p_value": round(p_val, 4),
                    "significant_at_005": p_val < 0.05,
                    "ci_diff_95": [round(ci_diff[0], 4), round(ci_diff[1], 4)],
                }

    # Output
    output = {"confidence_intervals": cis}
    if ab_results:
        output["ab_comparison"] = ab_results

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Written to {args.output}")

    # Print report
    print()
    if args.portfolio:
        print(build_portfolio_report(results, B=args.bootstrap_samples))
    else:
        print(build_ci_report(cis))

    if ab_results:
        print("\n## A/B Comparison")
        print("| Metric | Δ | p-value | Significant | 95% CI Δ |")
        print("|--------|---|---------|-------------|----------|")
        for metric, data in sorted(ab_results.items()):
            sig = "✓" if data["significant_at_005"] else "✗"
            ci = data["ci_diff_95"]
            print(f"| {metric} | {data['mean_diff']:+.3f} | {data['p_value']:.3f} | {sig} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")


if __name__ == "__main__":
    main()
