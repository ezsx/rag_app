#!/usr/bin/env python3
"""
Калибровка coverage threshold для pplx-embed.

Прямые Qdrant queries (без LLM) — собирает dense_score распределение
и вычисляет _compute_coverage() для определения оптимального threshold.
Также меряет recall@k.

Запуск:
    python scripts/calibrate_coverage.py \
        --dataset datasets/eval_retrieval_100.json \
        --collection news_colbert_v2 \
        --qdrant-url http://localhost:16333 \
        --embedding-url http://localhost:8082
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List

sys.stdout.reconfigure(encoding="utf-8")

# ─── Coverage computation (копия из compose_context.py) ─────────────

_STOP_WORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "are", "was", "from",
    "что", "как", "это", "для", "или", "при", "его", "её", "они",
    "по", "на", "в", "к", "у", "о", "из", "за", "до", "со", "не",
})


def _query_term_coverage(query: str, docs: List[Dict[str, Any]]) -> float:
    if not query or not docs:
        return 0.0
    tokens = [t.lower() for t in re.findall(r"\w+", query, re.UNICODE)
              if len(t) >= 3 and t.lower() not in _STOP_WORDS]
    if not tokens:
        return 0.5
    all_text = " ".join(str(doc.get("text", "")).lower() for doc in docs)
    covered = sum(1 for t in tokens if t in all_text)
    return covered / len(tokens)


def compute_coverage(
    query: str,
    docs: List[Dict[str, Any]],
    relevance_threshold: float = 0.55,
    target_k: int = 5,
) -> float:
    """Composite coverage metric (DEC-0018). Копия из compose_context.py."""
    if not docs:
        return 0.0
    sims = sorted(
        [float(doc.get("dense_score") or doc.get("score") or 0.0) for doc in docs],
        reverse=True,
    )
    top_k = sims[:target_k]
    max_sim = sims[0]
    mean_top_k = sum(top_k) / len(top_k)
    relevant_count = sum(1 for s in sims if s >= relevance_threshold)
    doc_count_adequacy = min(1.0, relevant_count / target_k)
    if max_sim > 0.0 and len(top_k) > 1:
        score_gap = 1.0 - (top_k[0] - top_k[-1]) / max_sim
    else:
        score_gap = 0.0
    above_threshold_ratio = relevant_count / len(sims)
    term_cov = _query_term_coverage(query, docs)
    return min(
        1.0,
        0.25 * max_sim + 0.20 * mean_top_k + 0.20 * term_cov
        + 0.15 * doc_count_adequacy + 0.15 * score_gap + 0.05 * above_threshold_ratio,
    )


# ─── Embedding + Search ─────────────────────────────────────────────

def embed_query(text: str, url: str) -> list[float]:
    """pplx-embed: mean pooling, без instruction prefix (DEC-0042)."""
    body = json.dumps({"inputs": [text], "normalize": True}).encode()
    req = urllib.request.Request(
        f"{url}/embed", data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=15).read())[0]


def colbert_encode(text: str, url: str) -> list[list[float]] | None:
    try:
        body = json.dumps({"texts": [text], "is_query": True}).encode()
        req = urllib.request.Request(
            f"{url}/colbert-encode", data=body,
            headers={"Content-Type": "application/json"},
        )
        result = json.loads(urllib.request.urlopen(req, timeout=15).read())
        return result[0] if result else None
    except Exception:
        return None


def search_qdrant(
    query_text: str,
    embedding_url: str,
    qdrant_url: str,
    collection: str,
    sparse_model,
    use_colbert: bool = True,
    top_k: int = 20,
) -> list[dict]:
    """BM25+Dense → weighted RRF (3:1) → ColBERT MaxSim rerank.

    Точная копия pipeline из hybrid_retriever.py.
    Возвращает points с score и payload (включая text для coverage).
    """
    dense_vec = embed_query(query_text, embedding_url)
    sparse_result = next(iter(sparse_model.query_embed(query_text)))
    sparse_q = {
        "indices": sparse_result.indices.tolist(),
        "values": sparse_result.values.tolist(),
    }

    colbert_vecs = colbert_encode(query_text, embedding_url) if use_colbert else None

    if colbert_vecs:
        # 3-stage: BM25+Dense → weighted RRF (3:1) → ColBERT MaxSim
        body = {
            "prefetch": [
                {
                    "prefetch": [
                        {"query": dense_vec, "using": "dense_vector", "limit": 20},
                        {"query": sparse_q, "using": "sparse_vector", "limit": 100},
                    ],
                    "query": {"rrf": {"weights": [1.0, 3.0]}},
                    "limit": max(top_k * 3, 30),
                }
            ],
            "query": colbert_vecs,
            "using": "colbert_vector",
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
        }
    else:
        # 2-stage: BM25+Dense → weighted RRF (3:1)
        body = {
            "prefetch": [
                {"query": dense_vec, "using": "dense_vector", "limit": 20},
                {"query": sparse_q, "using": "sparse_vector", "limit": 100},
            ],
            "query": {"rrf": {"weights": [1.0, 3.0]}},
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
        }

    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points/query",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return resp["result"]["points"]


def search_pipeline_v2(
    query_text: str,
    embedding_url: str,
    qdrant_url: str,
    collection: str,
    sparse_model,
    top_k: int = 20,
) -> list[dict]:
    """Pipeline v2: BM25+Dense → RRF(60) → Cross-encoder(40) → ColBERT(20).

    Cross-encoder фильтрует мусор перед ColBERT, ColBERT финально ранжирует.
    """
    dense_vec = embed_query(query_text, embedding_url)
    sparse_result = next(iter(sparse_model.query_embed(query_text)))
    sparse_q = {
        "indices": sparse_result.indices.tolist(),
        "values": sparse_result.values.tolist(),
    }

    # Stage 1: BM25+Dense → RRF top-60 (без ColBERT)
    rrf_body = {
        "prefetch": [
            {"query": dense_vec, "using": "dense_vector", "limit": 40},
            {"query": sparse_q, "using": "sparse_vector", "limit": 100},
        ],
        "query": {"rrf": {"weights": [1.0, 3.0]}},
        "limit": 60,
        "with_payload": True,
        "with_vectors": False,
    }
    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points/query",
        data=json.dumps(rrf_body).encode(),
        headers={"Content-Type": "application/json"},
    )
    rrf_points = json.loads(urllib.request.urlopen(req, timeout=30).read())["result"]["points"]

    if not rrf_points:
        return []

    # Stage 2: Cross-encoder rerank 60 → keep top-40
    texts = [(p.get("payload") or {}).get("text", "")[:512] for p in rrf_points]
    try:
        rerank_results = rerank_docs(query_text, texts, embedding_url)
        # Сортируем по score, берём top-40
        ranked_indices = [
            item["index"] for item in sorted(rerank_results, key=lambda x: x.get("score", 0), reverse=True)
        ][:40]
        filtered_points = [rrf_points[i] for i in ranked_indices if i < len(rrf_points)]
    except Exception:
        # Fallback: берём top-40 от RRF
        filtered_points = rrf_points[:40]

    if not filtered_points:
        return rrf_points[:top_k]

    # Stage 3: ColBERT rerank filtered 40 → top-20
    colbert_vecs = colbert_encode(query_text, embedding_url)
    if not colbert_vecs:
        return filtered_points[:top_k]

    point_ids = [p["id"] for p in filtered_points]
    colbert_body = {
        "query": colbert_vecs,
        "using": "colbert_vector",
        "filter": {"must": [{"has_id": point_ids}]},
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
    }
    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points/query",
        data=json.dumps(colbert_body).encode(),
        headers={"Content-Type": "application/json"},
    )
    colbert_points = json.loads(urllib.request.urlopen(req, timeout=30).read())["result"]["points"]
    return colbert_points if colbert_points else filtered_points[:top_k]


# ─── Recall ──────────────────────────────────────────────────────────

def check_recall(points: list[dict], expected: list[str], k: int, fuzzy: int = 5) -> float:
    if not expected:
        return 0.0
    top_k = points[:k]
    matched = 0
    for exp in expected:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for p in top_k:
            pay = p.get("payload", {})
            if pay.get("channel", "").lower() == exp_ch and abs(int(pay.get("message_id", 0)) - exp_msg) <= fuzzy:
                matched += 1
                break
    return matched / len(expected)


# ─── Cosine similarity (для dense_score) ────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def fetch_dense_vectors(
    point_ids: list, qdrant_url: str, collection: str,
) -> dict[str, list[float]]:
    """Batch fetch dense_vector из Qdrant по point IDs."""
    body = json.dumps({
        "ids": point_ids,
        "with_payload": False,
        "with_vector": ["dense_vector"],
    }).encode()
    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        result = {}
        for p in resp.get("result", []):
            pid = str(p.get("id", ""))
            vec = (p.get("vector") or {}).get("dense_vector")
            if pid and vec:
                result[pid] = vec
        return result
    except Exception:
        return {}


def rerank_docs(
    query: str, texts: list[str], embedding_url: str,
) -> list[dict]:
    """Cross-encoder rerank через gpu_server /rerank."""
    body = json.dumps({
        "query": query,
        "texts": texts,
        "raw_scores": True,
        "truncate": True,
    }).encode()
    req = urllib.request.Request(
        f"{embedding_url}/rerank",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    # [{index: i, score: f}, ...] sorted by score desc
    return resp


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Coverage threshold calibration")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", default="news_colbert_v2")
    parser.add_argument("--qdrant-url", default="http://localhost:16333")
    parser.add_argument("--embedding-url", default="http://localhost:8082")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-colbert", action="store_true")
    parser.add_argument("--fuzzy", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Макс. вопросов (0=все)")
    parser.add_argument("--test-reranker", action="store_true", help="Прогнать cross-encoder rerank и сравнить recall")
    parser.add_argument("--pipeline-v2", action="store_true", help="Тест pipeline v2: RRF(60)→CE(40)→ColBERT(20)")
    parser.add_argument("--ce-scores", action="store_true", help="Собрать CE score distribution для калибровки filter_threshold")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)

    if args.limit > 0:
        items = items[:args.limit]

    print(f"Dataset: {len(items)} queries, collection: {args.collection}")
    print(f"ColBERT: {'OFF' if args.no_colbert else 'ON'}")

    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    use_colbert = not args.no_colbert
    _query_vec_cache: dict[str, list[float]] = {}

    all_dense_scores: list[float] = []
    all_ce_scores: list[float] = []  # CE scores для калибровки filter_threshold
    ce_scores_relevant: list[float] = []  # CE scores для docs которые = expected
    ce_scores_irrelevant: list[float] = []  # CE scores для остальных docs
    all_coverages: list[float] = []
    # Recall на каждом k от 1 до top_k
    all_recalls: dict[int, list[float]] = {k: [] for k in range(1, args.top_k + 1)}
    results: list[dict] = []

    t_total = time.time()

    for i, item in enumerate(items):
        query = item["query"]
        expected = item.get("expected_documents", [])

        t0 = time.time()
        try:
            points = search_qdrant(
                query, args.embedding_url, args.qdrant_url,
                args.collection, sparse_model,
                use_colbert=use_colbert, top_k=args.top_k,
            )
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            results.append({"id": item["id"], "error": str(e)})
            continue
        latency = time.time() - t0

        # Dense cosine similarity — как в agent pipeline (pplx-embed).
        # Берём stored dense_vector из Qdrant, считаем cosine с query embedding.
        query_vec = _query_vec_cache.get(query)
        if query_vec is None:
            query_vec = embed_query(query, args.embedding_url)
            _query_vec_cache[query] = query_vec

        # Fetch dense vectors для этих point IDs
        point_ids = [p["id"] for p in points if p.get("id")]
        doc_vectors = fetch_dense_vectors(
            point_ids, args.qdrant_url, args.collection,
        )

        docs_for_coverage = []
        cosine_scores: list[float] = []
        for p in points:
            pay = p.get("payload", {})
            pid = p.get("id")
            dvec = doc_vectors.get(str(pid))
            csim = cosine_sim(query_vec, dvec) if dvec else 0.0
            cosine_scores.append(csim)
            docs_for_coverage.append({
                "text": pay.get("text", ""),
                "dense_score": csim,  # cosine 0-1, как в agent pipeline
                "metadata": {"channel": pay.get("channel", "")},
            })
        all_dense_scores.extend(cosine_scores)

        coverage = compute_coverage(query, docs_for_coverage)
        all_coverages.append(coverage)

        # Recall на каждом k от 1 до top_k
        recall_per_k: dict[int, float] = {}
        for k in range(1, args.top_k + 1):
            rk = check_recall(points, expected, k, args.fuzzy)
            recall_per_k[k] = rk
            all_recalls[k].append(rk)

        # CE score distribution (для калибровки filter_threshold)
        if args.ce_scores and points:
            try:
                texts = [(p.get("payload") or {}).get("text", "")[:512] for p in points]
                ce_results = rerank_docs(query, texts, args.embedding_url)
                # Маппим scores на point indices
                for ce_item in ce_results:
                    ce_score = ce_item.get("score", 0.0)
                    idx = ce_item.get("index", 0)
                    all_ce_scores.append(ce_score)
                    # Проверяем — это expected doc или нет?
                    if idx < len(points):
                        pay = points[idx].get("payload", {})
                        doc_ref = f"{pay.get('channel','')}:{pay.get('message_id','')}"
                        if doc_ref in set(expected):
                            ce_scores_relevant.append(ce_score)
                        else:
                            ce_scores_irrelevant.append(ce_score)
            except Exception as ce_err:
                if (i + 1) <= 3:
                    print(f"    CE scores error: {ce_err}")

        # Pipeline v2: RRF(60) → Cross-encoder(40) → ColBERT(20)
        v2_recall_per_k: dict[int, float] = {}
        if args.pipeline_v2:
            try:
                v2_points = search_pipeline_v2(
                    query, args.embedding_url, args.qdrant_url,
                    args.collection, sparse_model, top_k=args.top_k,
                )
                for k in range(1, args.top_k + 1):
                    v2_recall_per_k[k] = check_recall(v2_points, expected, k, args.fuzzy)
            except Exception as v2_err:
                if (i + 1) <= 3:
                    print(f"    Pipeline v2 error: {v2_err}")

        # Cross-encoder rerank (optional)
        reranked_recall_per_k: dict[int, float] = {}
        if args.test_reranker and points:
            try:
                texts_for_rerank = [
                    (p.get("payload") or {}).get("text", "")[:512]
                    for p in points
                ]
                rerank_results = rerank_docs(query, texts_for_rerank, args.embedding_url)
                # Reorder points by reranker score
                reranked_points = []
                for item_r in sorted(rerank_results, key=lambda x: x.get("score", 0), reverse=True):
                    idx = item_r.get("index", 0)
                    if 0 <= idx < len(points):
                        reranked_points.append(points[idx])
                for k in range(1, args.top_k + 1):
                    reranked_recall_per_k[k] = check_recall(reranked_points, expected, k, args.fuzzy)
            except Exception as re_err:
                if (i + 1) <= 3:
                    print(f"    Reranker error: {re_err}")

        result = {
            "id": item["id"],
            "query": query[:80],
            "expected": expected,
            "recall_per_k": recall_per_k,
            "v2_recall_per_k": v2_recall_per_k if v2_recall_per_k else None,
            "reranked_recall_per_k": reranked_recall_per_k if reranked_recall_per_k else None,
            "coverage": round(coverage, 4),
            "dense_cosine_top": round(cosine_scores[0], 4) if cosine_scores else 0,
            "dense_cosine_mean5": round(statistics.mean(cosine_scores[:5]), 4) if len(cosine_scores) >= 5 else 0,
            "hits_count": len(points),
            "latency": round(latency, 3),
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            r5 = recall_per_k.get(5, 0.0)
            top_csim = cosine_scores[0] if cosine_scores else 0.0
            print(f"  [{i+1}/{len(items)}] cov={coverage:.3f} r@5={r5:.2f} csim={top_csim:.3f} {latency:.2f}s | {query[:50]}")

    elapsed = time.time() - t_total

    # ─── Агрегация ───────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]

    print(f"\n{'='*70}")
    print(f"Coverage Calibration Results ({args.collection}, {len(valid)} queries)")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/max(len(items),1):.2f}s/query)\n")

    # Dense score distribution
    if all_dense_scores:
        sorted_scores = sorted(all_dense_scores)
        n = len(sorted_scores)
        print("Dense Score Distribution:")
        print(f"  min:    {sorted_scores[0]:.4f}")
        print(f"  p10:    {sorted_scores[int(n*0.1)]:.4f}")
        print(f"  p25:    {sorted_scores[int(n*0.25)]:.4f}")
        print(f"  median: {sorted_scores[int(n*0.5)]:.4f}")
        print(f"  p75:    {sorted_scores[int(n*0.75)]:.4f}")
        print(f"  p90:    {sorted_scores[int(n*0.9)]:.4f}")
        print(f"  max:    {sorted_scores[-1]:.4f}")
        print(f"  mean:   {statistics.mean(sorted_scores):.4f}")
        print(f"  stdev:  {statistics.stdev(sorted_scores):.4f}" if n > 1 else "")

    # Coverage distribution
    if all_coverages:
        sorted_cov = sorted(all_coverages)
        n = len(sorted_cov)
        print(f"\nCoverage Distribution (current _compute_coverage):")
        print(f"  min:    {sorted_cov[0]:.4f}")
        print(f"  p10:    {sorted_cov[int(n*0.1)]:.4f}")
        print(f"  p25:    {sorted_cov[int(n*0.25)]:.4f}")
        print(f"  median: {sorted_cov[int(n*0.5)]:.4f}")
        print(f"  p75:    {sorted_cov[int(n*0.75)]:.4f}")
        print(f"  p90:    {sorted_cov[int(n*0.9)]:.4f}")
        print(f"  max:    {sorted_cov[-1]:.4f}")
        print(f"  mean:   {statistics.mean(sorted_cov):.4f}")

        # Threshold simulation
        print(f"\nThreshold Simulation (% queries that would trigger refinement):")
        for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            below = sum(1 for c in all_coverages if c < t)
            pct = 100 * below / len(all_coverages)
            print(f"  threshold={t:.2f}: {below}/{len(all_coverages)} ({pct:.0f}%) refinements")

    # Recall curve k=1..20 + monotonicity check
    print(f"\nRecall@k curve (k=1..{args.top_k}):")
    recall_means: list[float] = []
    for k in range(1, args.top_k + 1):
        vals = all_recalls[k]
        if vals:
            mean = statistics.mean(vals)
            full = sum(1 for v in vals if v >= 1.0)
            zero = sum(1 for v in vals if v == 0.0)
            recall_means.append(mean)
            bar = "#" * int(mean * 40)
            print(f"  r@{k:2d}: {mean:.3f} |{bar:40s}| full={full}, zero={zero}")

    # Monotonicity check
    violations = []
    for i in range(1, len(recall_means)):
        if recall_means[i] < recall_means[i - 1] - 0.001:  # tolerance
            violations.append((i, recall_means[i - 1], recall_means[i]))
    if violations:
        print(f"\n  MONOTONICITY VIOLATIONS ({len(violations)}):")
        for k_idx, prev, curr in violations:
            print(f"    r@{k_idx} ({prev:.3f}) → r@{k_idx+1} ({curr:.3f}) DROPPED by {prev - curr:.3f}")
        print(f"  WARNING: non-monotonic recall suggests ranking quality issues!")
    else:
        print(f"\n  Monotonicity: OK (recall never drops with increasing k)")

    # Reranker comparison (if --test-reranker)
    reranked_results = [r for r in valid if r.get("reranked_recall_per_k")]
    if reranked_results:
        print(f"\nCross-encoder Reranker Recall@k ({len(reranked_results)} queries):")
        reranked_means: list[float] = []
        for k in range(1, args.top_k + 1):
            vals = [r["reranked_recall_per_k"].get(k, 0.0) for r in reranked_results]
            mean = statistics.mean(vals) if vals else 0.0
            reranked_means.append(mean)
            orig_mean = recall_means[k - 1] if k - 1 < len(recall_means) else 0
            delta = mean - orig_mean
            arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            bar = "#" * int(mean * 40)
            print(f"  r@{k:2d}: {mean:.3f} |{bar:40s}| {arrow}{abs(delta):.3f} vs retrieval")

        # Reranker monotonicity
        re_violations = []
        for i_r in range(1, len(reranked_means)):
            if reranked_means[i_r] < reranked_means[i_r - 1] - 0.001:
                re_violations.append((i_r, reranked_means[i_r - 1], reranked_means[i_r]))
        if re_violations:
            print(f"\n  RERANKER MONOTONICITY VIOLATIONS ({len(re_violations)}):")
            for k_idx, prev, curr in re_violations:
                print(f"    r@{k_idx} ({prev:.3f}) → r@{k_idx+1} ({curr:.3f}) DROPPED by {prev - curr:.3f}")
        else:
            print(f"\n  Reranker monotonicity: OK")

    # Pipeline v2 comparison
    v2_results = [r for r in valid if r.get("v2_recall_per_k")]
    if v2_results:
        print(f"\nPipeline v2: RRF(60)→CE(40)→ColBERT(20) ({len(v2_results)} queries):")
        v2_means: list[float] = []
        for k in range(1, args.top_k + 1):
            vals = [r["v2_recall_per_k"].get(k, r["v2_recall_per_k"].get(str(k), 0.0)) for r in v2_results]
            mean = statistics.mean(vals) if vals else 0.0
            v2_means.append(mean)
            orig_mean = recall_means[k - 1] if k - 1 < len(recall_means) else 0
            delta = mean - orig_mean
            arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            bar = "#" * int(mean * 40)
            print(f"  r@{k:2d}: {mean:.3f} |{bar:40s}| {arrow}{abs(delta):.3f} vs v1")

        # V2 monotonicity
        v2_violations = []
        for i_v in range(1, len(v2_means)):
            if v2_means[i_v] < v2_means[i_v - 1] - 0.001:
                v2_violations.append((i_v, v2_means[i_v - 1], v2_means[i_v]))
        if v2_violations:
            print(f"\n  V2 MONOTONICITY VIOLATIONS ({len(v2_violations)}):")
            for k_idx, prev, curr in v2_violations:
                print(f"    r@{k_idx} ({prev:.3f}) → r@{k_idx+1} ({curr:.3f}) DROPPED by {prev - curr:.3f}")
        else:
            print(f"\n  V2 monotonicity: OK")

    # CE score distribution (для калибровки filter_threshold)
    if all_ce_scores:
        sorted_ce = sorted(all_ce_scores)
        n = len(sorted_ce)
        print(f"\nCross-encoder Score Distribution ({n} doc scores):")
        print(f"  min:    {sorted_ce[0]:.4f}")
        print(f"  p10:    {sorted_ce[int(n*0.1)]:.4f}")
        print(f"  p25:    {sorted_ce[int(n*0.25)]:.4f}")
        print(f"  median: {sorted_ce[int(n*0.5)]:.4f}")
        print(f"  p75:    {sorted_ce[int(n*0.75)]:.4f}")
        print(f"  p90:    {sorted_ce[int(n*0.9)]:.4f}")
        print(f"  max:    {sorted_ce[-1]:.4f}")
        if ce_scores_relevant:
            print(f"\n  RELEVANT docs (expected): n={len(ce_scores_relevant)}")
            print(f"    min={min(ce_scores_relevant):.4f}, median={sorted(ce_scores_relevant)[len(ce_scores_relevant)//2]:.4f}, max={max(ce_scores_relevant):.4f}")
        if ce_scores_irrelevant:
            print(f"  IRRELEVANT docs (rest): n={len(ce_scores_irrelevant)}")
            print(f"    min={min(ce_scores_irrelevant):.4f}, median={sorted(ce_scores_irrelevant)[len(ce_scores_irrelevant)//2]:.4f}, max={max(ce_scores_irrelevant):.4f}")
        if ce_scores_relevant and ce_scores_irrelevant:
            # Оптимальный threshold = между median relevant и median irrelevant
            med_rel = sorted(ce_scores_relevant)[len(ce_scores_relevant)//2]
            med_irr = sorted(ce_scores_irrelevant)[len(ce_scores_irrelevant)//2]
            suggested = (med_rel + med_irr) / 2
            print(f"\n  Suggested filter_threshold: {suggested:.2f} (midpoint of medians)")
            print(f"  Threshold simulation:")
            for t in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, suggested]:
                kept_rel = sum(1 for s in ce_scores_relevant if s >= t)
                lost_rel = len(ce_scores_relevant) - kept_rel
                removed_irr = sum(1 for s in ce_scores_irrelevant if s < t)
                print(f"    t={t:5.2f}: keep {kept_rel}/{len(ce_scores_relevant)} relevant, remove {removed_irr}/{len(ce_scores_irrelevant)} irrelevant, lose {lost_rel} relevant")

    # Recall@5 по категориям
    by_cat = defaultdict(list)
    for r in valid:
        cat = r.get("category", "unknown")
        by_cat[cat].append(r["recall_per_k"].get(5, 0.0))
    if len(by_cat) > 1:
        print(f"\nRecall@5 по категориям:")
        for cat, vals in sorted(by_cat.items()):
            print(f"  {cat:30s}: {statistics.mean(vals):.3f} ({len(vals)} qs)")

    # Сохраняем
    ts = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "timestamp": ts,
        "collection": args.collection,
        "colbert": use_colbert,
        "total_queries": len(items),
        "valid_queries": len(valid),
        "dense_score_stats": {
            "mean": round(statistics.mean(all_dense_scores), 4) if all_dense_scores else None,
            "median": round(statistics.median(all_dense_scores), 4) if all_dense_scores else None,
            "stdev": round(statistics.stdev(all_dense_scores), 4) if len(all_dense_scores) > 1 else None,
        },
        "coverage_stats": {
            "mean": round(statistics.mean(all_coverages), 4) if all_coverages else None,
            "median": round(statistics.median(all_coverages), 4) if all_coverages else None,
        },
        "recall_curve": {
            str(k): round(statistics.mean(vals), 4) if vals else None
            for k, vals in all_recalls.items()
        },
        "recall_monotonic": len(violations) == 0,
        "recall_violations": [
            {"k": k_idx + 1, "prev": round(prev, 4), "curr": round(curr, 4)}
            for k_idx, prev, curr in violations
        ],
        "results": results,
    }
    out_path = f"results/raw/calibration_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
