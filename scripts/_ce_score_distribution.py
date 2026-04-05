"""CE score distribution — снять raw CE scores для всех 120 Qs × top-20 docs.

Нужно для информированного выбора CE threshold в ablation phase 2.
Output: JSON с per-query CE scores + summary percentiles.
"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

QDRANT_URL = "http://localhost:16333"
COLLECTION = "news_colbert_v2"
EMBEDDING_URL = "http://localhost:8082"
DATASET = "datasets/eval_retrieval_v3.json"
FUZZY = 5


def embed_query(text: str) -> list[float]:
    body = json.dumps({"inputs": [text], "normalize": True}).encode()
    req = urllib.request.Request(
        f"{EMBEDDING_URL}/embed", data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())[0]


def colbert_encode(text: str) -> list[list[float]] | None:
    try:
        body = json.dumps({"texts": [text], "is_query": True}).encode()
        req = urllib.request.Request(
            f"{EMBEDDING_URL}/colbert-encode", data=body,
            headers={"Content-Type": "application/json"},
        )
        result = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return result[0] if result else None
    except Exception:
        return None


def ce_rerank(query: str, texts: list[str]) -> list[dict]:
    """Вернуть [{index, score}] sorted by score desc."""
    body = json.dumps({
        "query": query, "texts": texts,
        "raw_scores": True, "truncate": True,
    }).encode()
    req = urllib.request.Request(
        f"{EMBEDDING_URL}/rerank", data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def search_colbert(query: str, dense_vec, sparse_q, colbert_vecs, top_k=20):
    """Full pipeline: BM25+Dense → RRF → ColBERT, return points with text."""
    body = {
        "prefetch": [{
            "prefetch": [
                {"query": dense_vec, "using": "dense_vector", "limit": 40},
                {"query": sparse_q, "using": "sparse_vector", "limit": 100},
            ],
            "query": {"rrf": {"weights": [1.0, 3.0]}},
            "limit": 60,
        }],
        "query": colbert_vecs,
        "using": "colbert_vector",
        "limit": top_k,
        "with_payload": ["channel", "message_id", "text"],
    }
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    return resp["result"]["points"]


def main():
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    with open(DATASET, encoding="utf-8") as f:
        items = json.load(f)

    print(f"CE score distribution: {len(items)} queries × top-20")

    all_relevant_scores = []  # CE scores для expected docs
    all_irrelevant_scores = []  # CE scores для non-expected docs
    per_query = []

    t0 = time.time()
    for i, item in enumerate(items):
        query = item["query"]
        expected = item.get("expected_documents", [])

        # Search
        dense_vec = embed_query(query)
        sparse_result = next(iter(sparse_model.query_embed(query)))
        sparse_q = {
            "indices": sparse_result.indices.tolist(),
            "values": sparse_result.values.tolist(),
        }
        colbert_vecs = colbert_encode(query)
        if not colbert_vecs:
            continue

        points = search_colbert(query, dense_vec, sparse_q, colbert_vecs, top_k=20)
        if not points:
            continue

        texts = [p["payload"].get("text", "")[:512] for p in points]

        # CE rerank
        try:
            ce_results = ce_rerank(query, texts)
        except Exception as e:
            print(f"  [{i+1}] CE error: {e}")
            continue

        # Map scores back to original order
        scores = [0.0] * len(texts)
        for r in ce_results:
            scores[r["index"]] = r["score"]

        # Classify: relevant vs irrelevant
        for idx, p in enumerate(points):
            pay = p.get("payload", {})
            is_relevant = False
            for exp in expected:
                exp_parts = exp.split(":", 1)
                if len(exp_parts) == 2:
                    if (pay.get("channel", "").lower() == exp_parts[0].lower()
                            and abs(pay.get("message_id", 0) - int(exp_parts[1])) <= FUZZY):
                        is_relevant = True
                        break

            if is_relevant:
                all_relevant_scores.append(scores[idx])
            else:
                all_irrelevant_scores.append(scores[idx])

        per_query.append({
            "id": item["id"],
            "category": item.get("category", ""),
            "scores": scores,
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(items)} done ({elapsed:.0f}s)")

    elapsed = time.time() - t0

    # Percentiles
    def percentiles(scores, ps=None):
        if ps is None:
            ps = [5, 10, 25, 50, 75, 90, 95]
        if not scores:
            return {}
        s = sorted(scores)
        return {f"p{p}": s[int(len(s) * p / 100)] for p in ps}

    rel_pct = percentiles(all_relevant_scores)
    irr_pct = percentiles(all_irrelevant_scores)

    print(f"\n{'='*60}")
    print(f"CE Score Distribution ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"Relevant docs:   n={len(all_relevant_scores)}")
    if all_relevant_scores:
        print(f"  min={min(all_relevant_scores):.3f} max={max(all_relevant_scores):.3f} "
              f"mean={sum(all_relevant_scores)/len(all_relevant_scores):.3f}")
        print(f"  percentiles: {rel_pct}")
    print(f"Irrelevant docs: n={len(all_irrelevant_scores)}")
    if all_irrelevant_scores:
        print(f"  min={min(all_irrelevant_scores):.3f} max={max(all_irrelevant_scores):.3f} "
              f"mean={sum(all_irrelevant_scores)/len(all_irrelevant_scores):.3f}")
        print(f"  percentiles: {irr_pct}")

    # Threshold analysis
    print("\nThreshold analysis (what gets filtered at each threshold):")
    for t in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        rel_lost = sum(1 for s in all_relevant_scores if s < t)
        irr_removed = sum(1 for s in all_irrelevant_scores if s < t)
        print(f"  t={t:5.1f}: relevant lost={rel_lost}/{len(all_relevant_scores)} "
              f"({rel_lost/max(len(all_relevant_scores),1)*100:.1f}%), "
              f"irrelevant removed={irr_removed}/{len(all_irrelevant_scores)} "
              f"({irr_removed/max(len(all_irrelevant_scores),1)*100:.1f}%)")

    # Save
    report = {
        "total_queries": len(items),
        "relevant_scores": all_relevant_scores,
        "irrelevant_scores": all_irrelevant_scores,
        "relevant_percentiles": rel_pct,
        "irrelevant_percentiles": irr_pct,
        "per_query": per_query,
    }
    out = "results/ablation/ce_score_distribution.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
