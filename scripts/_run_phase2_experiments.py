"""Phase 2 experiments runner — 10 agreed experiments (A3, A2, A1, R2, R1, R3, R6, R5, E1, D2)."""
import json
import os
import subprocess
import sys
import time

PYTHON = sys.executable
DATASET = "datasets/eval_retrieval_v3.json"
COLLECTION = "news_colbert_v2"
OUTPUT_DIR = "results/ablation/phase2"
LEXICON = "datasets/query_normalization_lexicon.json"

# Base config = phase 1 winning: no-prefix + colbert
BASE = ["--no-prefix", "--collection", COLLECTION]

# Эксперименты: (id, name, script, extra_args, dataset_override)
EXPERIMENTS = [
    # Блок A: Confirmed levers (evaluate_retrieval.py)
    ("A3", "combo_d40_rrf100_cb40",
     "scripts/evaluate_retrieval.py",
     [*BASE, "--dense-limit", "40", "--rrf-weights", "1.0,3.0", "--rrf-limit", "100", "--colbert-pool", "40"],
     None),

    ("A2", "colbert_pool_40",
     "scripts/evaluate_retrieval.py",
     [*BASE, "--dense-limit", "40", "--rrf-weights", "1.0,3.0", "--colbert-pool", "40"],
     None),

    ("A1", "rrf_limit_100",
     "scripts/evaluate_retrieval.py",
     [*BASE, "--dense-limit", "40", "--rrf-weights", "1.0,3.0", "--rrf-limit", "100"],
     None),

    # Блок R2: Sparse-only normalization
    ("R2", "normalize_sparse_only",
     "scripts/evaluate_retrieval.py",
     [*BASE, "--dense-limit", "40", "--rrf-weights", "1.0,3.0", "--rrf-limit", "100", "--colbert-pool", "40", "--normalize-sparse-only", "--lexicon", LEXICON],
     None),

    # Блок R1: Full normalization (both branches)
    ("R1", "normalize_all",
     "scripts/evaluate_retrieval.py",
     [*BASE, "--dense-limit", "40", "--rrf-weights", "1.0,3.0", "--rrf-limit", "100", "--colbert-pool", "40", "--normalize-query", "--lexicon", LEXICON],
     None),

    # Блок R3: LLM single rewrite + raw (48 hard Qs)
    ("R3", "single_rewrite",
     "scripts/evaluate_retrieval_full.py",
     ["--single-rewrite", "--inject-original-query",
      "--categories", "edge,temporal,channel_specific",
      "--save-traces"],
     None),

    # Блок R6: Rule-based filters (48 hard Qs)
    ("R6", "rule_based_filters",
     "scripts/evaluate_retrieval_full.py",
     ["--rule-based-filters", "--inject-original-query",
      "--categories", "edge,temporal,channel_specific",
      "--save-traces"],
     None),

    # Блок R5: BM25 PRF-lite
    ("R5", "prf_expand",
     "scripts/evaluate_retrieval.py",
     [*BASE, "--dense-limit", "40", "--rrf-weights", "1.0,3.0", "--rrf-limit", "100", "--colbert-pool", "40", "--prf-expand", "--prf-top-k", "5"],
     None),

    # Блок E1: Dense-only + ColBERT (BM25 off)
    ("E1", "dense_only_colbert",
     "scripts/evaluate_retrieval.py",
     [*BASE, "--dense-limit", "40", "--rrf-limit", "100", "--colbert-pool", "40", "--dense-only"],
     None),

    # Блок D2: HyDE + original (48 hard Qs)
    ("D2", "hyde_dual_branch",
     "scripts/evaluate_retrieval_full.py",
     ["--hyde", "--inject-original-query",
      "--categories", "edge,temporal,channel_specific",
      "--save-traces"],
     None),
]

sys.stdout.reconfigure(encoding="utf-8")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_experiment(exp_id, name, script, extra_args, dataset_override):
    output_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    if os.path.exists(output_path):
        print(f"  SKIP {exp_id} {name} (exists)")
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)

    cmd = [PYTHON, script, "--dataset", dataset_override or DATASET, "--output", output_path, *extra_args]

    print(f"\n{'='*70}")
    print(f"[{exp_id}] {name}")
    print(f"  {' '.join(cmd[2:])}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):")
        stderr = result.stderr[-300:] if result.stderr else ""
        stdout = result.stdout[-300:] if result.stdout else ""
        print(f"  stderr: {stderr}")
        print(f"  stdout: {stdout}")
        return None

    # Show key metrics
    for line in result.stdout.split("\n"):
        if any(k in line for k in ["Recall@", "MRR@", "Total time", "Phase 2:"]):
            print(f"  {line.strip()}")

    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            report = json.load(f)
        r5 = report.get("recall_at_5", 0)
        mrr = report.get("mrr_at_20", 0)
        print(f"\n  → R@5={r5:.3f} MRR={mrr:.3f} ({elapsed:.0f}s)")
        return report

    return None


def main():
    all_results = {}

    for exp_id, name, script, extra_args, ds in EXPERIMENTS:
        report = run_experiment(exp_id, name, script, extra_args, ds)
        if report:
            all_results[f"{exp_id}_{name}"] = report

    # Summary
    print(f"\n\n{'='*70}")
    print("PHASE 2 EXPERIMENTS — SUMMARY")
    print(f"{'='*70}")
    header = f"{'ID':5s} {'Name':30s} {'R@1':>6s} {'R@5':>6s} {'R@20':>6s} {'MRR':>6s} {'Lat':>6s} {'Qs':>4s}"
    print(header)
    print("-" * len(header))

    for exp_id, name, *_ in EXPERIMENTS:
        key = f"{exp_id}_{name}"
        if key in all_results:
            r = all_results[key]
            n = r.get("total_queries", "?")
            print(f"{exp_id:5s} {name:30s} {r.get('recall_at_1',0):6.3f} "
                  f"{r.get('recall_at_5',0):6.3f} {r.get('recall_at_20',0):6.3f} "
                  f"{r.get('mrr_at_20',0):6.3f} {r.get('avg_latency',0):5.1f}s {n:>4}")

    # Save summary
    summary = {}
    for exp_id, name, *_ in EXPERIMENTS:
        key = f"{exp_id}_{name}"
        if key in all_results:
            r = all_results[key]
            summary[exp_id] = {
                "name": name,
                "recall_at_1": r.get("recall_at_1", 0),
                "recall_at_5": r.get("recall_at_5", 0),
                "recall_at_20": r.get("recall_at_20", 0),
                "mrr_at_20": r.get("mrr_at_20", 0),
                "total_queries": r.get("total_queries", 0),
                "avg_latency": r.get("avg_latency", 0),
            }

    with open(os.path.join(OUTPUT_DIR, "_phase2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to {OUTPUT_DIR}/_phase2_summary.json")


if __name__ == "__main__":
    main()
