#!/usr/bin/env python3
"""Статистические confidence intervals для eval метрик (SPEC-RAG-22 Layer 3).

Читает eval_results JSON, вычисляет:
- Bootstrap CIs (BCa, B=1000) для всех continuous metrics
- Wilson intervals для binary metrics
- Paired bootstrap test для A/B comparisons

Использование:
    python scripts/compute_confidence.py results/raw/eval_results_YYYYMMDD.json
    python scripts/compute_confidence.py --compare results/a.json results/b.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Bootstrap CI ─────────────────────────────────────────────────


def bootstrap_ci(
    scores: List[float],
    B: int = 1000,
    alpha: float = 0.05,
    statistic=np.mean,
) -> Tuple[float, float, float]:
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
) -> Tuple[float, float, float]:
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
    scores_a: List[float],
    scores_b: List[float],
    B: int = 10000,
) -> Tuple[float, float, float]:
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


def extract_metric(results: List[Dict], metric: str) -> List[float]:
    """Извлечь per-question metric values из eval results."""
    values = []
    for r in results:
        m = r.get("metrics", {})
        v = m.get(metric)
        if v is not None:
            values.append(float(v))
    return values


def compute_all_cis(results: List[Dict], B: int = 1000) -> Dict[str, Any]:
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


def build_ci_report(cis: Dict[str, Any]) -> str:
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


# ─── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compute confidence intervals for eval metrics")
    parser.add_argument("results", type=Path, help="Path to eval_results JSON")
    parser.add_argument("--compare", type=Path, default=None, help="Second results file for A/B test")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap resamples (B)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    with args.results.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Поддержка unified report format (per_question внутри)
    results = data.get("per_question", data) if isinstance(data, dict) else data

    print(f"Computing CIs for {len(results)} questions (B={args.bootstrap_samples})...")
    cis = compute_all_cis(results, B=args.bootstrap_samples)

    # A/B comparison
    ab_results = None
    if args.compare:
        with args.compare.open("r", encoding="utf-8") as f:
            data_b = json.load(f)
        results_b = data_b.get("per_question", data_b) if isinstance(data_b, dict) else data_b
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
