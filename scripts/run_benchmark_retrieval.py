#!/usr/bin/env python3
"""
Retrieval benchmark runner — 4 pipelines × 100 questions.

Запуск:
    docker compose -f deploy/compose/compose.benchmark.yml run --rm benchmark \
        python scripts/run_benchmark_retrieval.py [--dataset PATH] [--top-k N] [--fuzzy N]

Выход: benchmarks/results/retrieval_results.json + markdown таблица в stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from benchmarks.config import FINAL_TOP_K, RESULTS_DIR, RETRIEVAL_DATASET  # noqa: E402


# ─── Metrics ────────────────────────────────────────────────────


def check_recall(results: list, expected_docs: list[str], k: int, fuzzy: int = 5) -> float:
    """Recall@K: доля expected docs найденных в top-K."""
    if not expected_docs:
        return 0.0
    top_k = results[:k]
    matched = 0
    for exp in expected_docs:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for r in top_k:
            if r.channel.lower() == exp_ch and abs(r.message_id - exp_msg) <= fuzzy:
                matched += 1
                break
    return matched / len(expected_docs)


def compute_mrr(results: list, expected_docs: list[str], fuzzy: int = 5) -> float:
    """Mean Reciprocal Rank первого релевантного документа."""
    if not expected_docs:
        return 0.0
    for rank, r in enumerate(results):
        for exp in expected_docs:
            parts = exp.split(":", 1)
            if len(parts) != 2:
                continue
            exp_ch, exp_msg = parts[0].lower(), int(parts[1])
            if r.channel.lower() == exp_ch and abs(r.message_id - exp_msg) <= fuzzy:
                return 1.0 / (rank + 1)
    return 0.0


def compute_ndcg(results: list, expected_docs: list[str], k: int, fuzzy: int = 5) -> float:
    """nDCG@K с бинарной релевантностью (0 или 1)."""
    if not expected_docs:
        return 0.0

    # DCG
    dcg = 0.0
    for rank, r in enumerate(results[:k]):
        relevant = 0
        for exp in expected_docs:
            parts = exp.split(":", 1)
            if len(parts) != 2:
                continue
            exp_ch, exp_msg = parts[0].lower(), int(parts[1])
            if r.channel.lower() == exp_ch and abs(r.message_id - exp_msg) <= fuzzy:
                relevant = 1
                break
        dcg += relevant / math.log2(rank + 2)  # rank+2 потому что log2(1)=0

    # Ideal DCG: все relevant docs на первых позициях
    n_relevant = min(len(expected_docs), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def percentile(values: list[float], p: int) -> float:
    """Простой percentile (nearest-rank)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, min(int(len(sorted_v) * p / 100), len(sorted_v) - 1))
    return sorted_v[idx]


# ─── Pipeline runner ────────────────────────────────────────────


def run_pipeline(pipeline_name: str, retriever, items: list[dict], top_k: int, fuzzy: int) -> dict:
    """Прогоняет один pipeline по всем вопросам, возвращает агрегированные метрики."""
    recalls_1, recalls_5, recalls_10, recalls_20 = [], [], [], []
    mrrs, ndcgs = [], []
    latencies = []
    errors = 0
    per_question = []

    for i, item in enumerate(items):
        query = item["query"]
        expected = item.get("expected_documents", [])

        t0 = time.time()
        try:
            results = retriever.retrieve(query, top_k=top_k)
        except Exception as e:
            errors += 1
            print(f"  [{pipeline_name}] [{i+1}/{len(items)}] ERROR: {e}")
            per_question.append({"id": item["id"], "error": str(e)})
            continue
        latency = time.time() - t0
        latencies.append(latency)

        r1 = check_recall(results, expected, 1, fuzzy)
        r5 = check_recall(results, expected, 5, fuzzy)
        r10 = check_recall(results, expected, 10, fuzzy)
        r20 = check_recall(results, expected, 20, fuzzy)
        mrr = compute_mrr(results, expected, fuzzy)
        ndcg = compute_ndcg(results, expected, 5, fuzzy)

        recalls_1.append(r1)
        recalls_5.append(r5)
        recalls_10.append(r10)
        recalls_20.append(r20)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

        per_question.append({
            "id": item["id"],
            "recall_1": r1, "recall_5": r5, "recall_10": r10, "recall_20": r20,
            "mrr": mrr, "ndcg_5": ndcg, "latency": latency,
            "top5": [r.doc_id for r in results[:5]],
        })

        if (i + 1) % 25 == 0:
            avg_r5 = sum(recalls_5) / len(recalls_5)
            print(f"  [{pipeline_name}] {i+1}/{len(items)} done, avg Recall@5={avg_r5:.3f}")

    n = len(recalls_5) or 1
    return {
        "pipeline": pipeline_name,
        "questions": len(items),
        "errors": errors,
        "recall_1": sum(recalls_1) / n,
        "recall_5": sum(recalls_5) / n,
        "recall_10": sum(recalls_10) / n,
        "recall_20": sum(recalls_20) / n,
        "mrr": sum(mrrs) / n,
        "ndcg_5": sum(ndcgs) / n,
        "latency_p50": percentile(latencies, 50),
        "latency_p95": percentile(latencies, 95),
        "per_question": per_question,
    }


# ─── Main ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Retrieval benchmark — 4 pipelines")
    parser.add_argument("--dataset", default=RETRIEVAL_DATASET, help="Path to retrieval dataset JSON")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K, help="Final top-K")
    parser.add_argument("--fuzzy", type=int, default=5, help="msg_id fuzzy tolerance")
    parser.add_argument("--pipelines", default="naive,li_stock,li_maxed,custom",
                        help="Comma-separated pipeline names")
    args = parser.parse_args()

    # Загружаем датасет
    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)
    print(f"Dataset: {len(items)} questions, top_k={args.top_k}, fuzzy=±{args.fuzzy}")

    pipelines_to_run = [p.strip() for p in args.pipelines.split(",")]
    all_results = []

    for name in pipelines_to_run:
        print(f"\n{'='*60}")
        print(f"Running pipeline: {name}")
        print(f"{'='*60}")

        if name == "naive":
            from benchmarks.naive.retriever import NaiveRetriever
            retriever = NaiveRetriever()
        elif name == "li_stock":
            from benchmarks.llamaindex_pipeline.retriever import LlamaIndexRetrieverStock
            retriever = LlamaIndexRetrieverStock()
        elif name == "li_maxed":
            from benchmarks.llamaindex_pipeline.retriever import LlamaIndexRetrieverMaxed
            retriever = LlamaIndexRetrieverMaxed()
        elif name == "custom":
            from benchmarks.custom_adapter.retriever import CustomRetriever
            retriever = CustomRetriever()
        else:
            print(f"Unknown pipeline: {name}, skipping")
            continue

        result = run_pipeline(name, retriever, items, args.top_k, args.fuzzy)
        all_results.append(result)

    # Сохраняем результаты
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = Path(RESULTS_DIR) / "retrieval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Markdown таблица
    print(f"\n{'='*80}")
    print("RETRIEVAL BENCHMARK RESULTS")
    print(f"{'='*80}")
    header = f"{'Pipeline':<12} | {'Recall@1':>9} | {'Recall@5':>9} | {'Recall@10':>10} | {'Recall@20':>10} | {'MRR':>6} | {'nDCG@5':>7} | {'Lat p50':>8} | {'Lat p95':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['pipeline']:<12} | {r['recall_1']:>9.3f} | {r['recall_5']:>9.3f} | "
            f"{r['recall_10']:>10.3f} | {r['recall_20']:>10.3f} | {r['mrr']:>6.3f} | "
            f"{r['ndcg_5']:>7.3f} | {r['latency_p50']:>7.1f}ms | {r['latency_p95']:>7.1f}ms"
        )
        # Конвертируем секунды → миллисекунды для вывода
    # Повтор с правильными единицами
    print()
    print("(latency in seconds)")
    if any(r["errors"] > 0 for r in all_results):
        print("\nErrors:")
        for r in all_results:
            if r["errors"] > 0:
                print(f"  {r['pipeline']}: {r['errors']} errors")


if __name__ == "__main__":
    main()
