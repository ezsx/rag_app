"""Анализ результатов eval v3 — per-question и per-category breakdown."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/raw/eval_results_20260321-180112.json"

with open(path, encoding="utf-8") as f:
    results = json.load(f)

print(f"Total questions: {len(results)}")

total_recall = 0
categories = {}

for r in results:
    qid = r["query_id"]
    cat = r["category"]
    expected = set(r["expected_documents"])
    answerable = r["answerable"]

    hits = r["agent"].get("citation_hits", [])
    hits_set = set(hits)

    if answerable and expected:
        found = expected & hits_set
        recall = len(found) / len(expected)
    else:
        recall = 1.0 if not hits else 0.0

    total_recall += recall

    if cat not in categories:
        categories[cat] = {"total": 0, "recall_sum": 0, "questions": []}
    categories[cat]["total"] += 1
    categories[cat]["recall_sum"] += recall

    coverage = r["agent"].get("coverage", 0)
    refinements = r["agent"].get("refinements", 0)
    fallback = r["agent"].get("fallback", False)
    latency = r["agent"].get("latency_sec", 0)
    error = r["agent"].get("error", False)

    found_str = ", ".join(sorted(expected & hits_set)) if answerable else "-"
    missed_str = ", ".join(sorted(expected - hits_set)) if answerable else "-"

    categories[cat]["questions"].append({
        "id": qid,
        "recall": recall,
        "found": found_str,
        "missed": missed_str,
        "hits": hits[:5],
        "coverage": coverage,
        "refinements": refinements,
        "fallback": fallback,
        "latency": latency,
        "error": error,
    })

overall = total_recall / len(results)
print(f"Overall Recall@5: {overall:.3f} ({total_recall:.1f}/{len(results)})")
print()

print("=== PER-CATEGORY RECALL ===")
for cat in sorted(categories.keys()):
    c = categories[cat]
    avg = c["recall_sum"] / c["total"]
    print(f"  {cat:15s}: {avg:.3f} ({c['recall_sum']:.1f}/{c['total']})")

print()
print("=== PER-QUESTION DETAIL ===")
for cat in sorted(categories.keys()):
    print(f"\n--- {cat.upper()} ---")
    for q in categories[cat]["questions"]:
        status = "OK" if q["recall"] == 1.0 else ("PARTIAL" if q["recall"] > 0 else "MISS")
        print(f"  {q['id']:10s} recall={q['recall']:.2f} [{status:7s}] cov={q['coverage']:.2f} ref={q['refinements']} lat={q['latency']:.0f}s fb={q['fallback']} err={q['error']}")
        if q["found"] != "-":
            print(f"              found: {q['found']}")
        if q["missed"] != "-" and q["missed"]:
            print(f"              MISSED: {q['missed']}")
        print(f"              top-5: {q['hits']}")
