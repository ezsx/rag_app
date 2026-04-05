"""Phase 2 ablation: query-plan matrix on 48 Qs subset (edge + temporal + channel_specific)."""
import json
import os
import subprocess
import sys
import time

PYTHON = sys.executable
DATASET = "datasets/eval_retrieval_v3.json"
OUTPUT_DIR = "results/ablation/phase2"
CATEGORIES = "edge,temporal,channel_specific"

EXPERIMENTS = [
    # 1. Raw retriever (no query plan, no CE, no dedup) — single-query baseline
    ("p2_raw_retriever", []),

    # 2. Query plan only
    ("p2_query_plan", ["--use-query-plan"]),

    # 3. Query plan + original query injection
    ("p2_qplan_inject", ["--use-query-plan", "--inject-original-query"]),

    # 4. Query plan + inject + metadata filters
    ("p2_qplan_inject_filters", ["--use-query-plan", "--inject-original-query", "--use-metadata-filters"]),

    # 5. Full prod: query plan + inject + filters + CE(0.0) + dedup(2)
    ("p2_full_prod", ["--use-query-plan", "--inject-original-query", "--use-metadata-filters",
                       "--ce-filter", "--ce-threshold", "0.0", "--channel-dedup", "2"]),
]

sys.stdout.reconfigure(encoding="utf-8")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run(name, extra_args):
    output_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    if os.path.exists(output_path):
        print(f"SKIP {name} (exists)")
        return

    cmd = [PYTHON, "scripts/evaluate_retrieval_full.py", "--dataset", DATASET, "--categories", CATEGORIES, "--save-traces", "--output", output_path, *extra_args]

    print(f"\n{'='*60}")
    print(f">>> {name}: {' '.join(extra_args) or '(raw retriever)'}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "")
        print(result.stdout[-500:] if result.stdout else "")
        return

    # Key metrics from output
    for line in result.stdout.split("\n"):
        if any(k in line for k in ["Recall@", "MRR@", "Total time", "Pipeline:", "По категориям"]):
            print(f"  {line.strip()}")
        if "n=" in line and ("R@5" in line or "MRR" in line):
            print(f"  {line.strip()}")

    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            r = json.load(f)
        print(f"\n  R@1={r['recall_at_1']:.3f} R@5={r['recall_at_5']:.3f} "
              f"R@20={r['recall_at_20']:.3f} MRR={r['mrr_at_20']:.3f} ({elapsed:.0f}s)")


def main():
    for name, extra_args in EXPERIMENTS:
        run(name, extra_args)

    # Summary table
    print(f"\n\n{'='*60}")
    print("PHASE 2 SUMMARY — Query Plan Ablation (48 Qs)")
    print(f"{'='*60}")
    header = f"{'Experiment':30s} {'R@1':>6s} {'R@5':>6s} {'R@20':>6s} {'MRR':>6s} {'Lat':>6s}"
    print(header)
    print("-" * len(header))

    for name, _ in EXPERIMENTS:
        path = os.path.join(OUTPUT_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            print(f"{name:30s} {r.get('recall_at_1',0):6.3f} {r.get('recall_at_5',0):6.3f} "
                  f"{r.get('recall_at_20',0):6.3f} {r.get('mrr_at_20',0):6.3f} "
                  f"{r.get('avg_latency',0):5.1f}s")


if __name__ == "__main__":
    main()
