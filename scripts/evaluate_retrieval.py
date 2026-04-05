"""
Retrieval-only evaluation — прямые Qdrant queries без LLM.

Быстрый: ~0.5с на запрос (vs 40с через agent).
Тестирует качество поиска изолированно от LLM generation.

Запуск:
    python scripts/evaluate_retrieval.py \
        --dataset datasets/eval_retrieval_100.json \
        --collection news_colbert \
        --qdrant-url http://localhost:16333 \
        --embedding-url http://localhost:8082
"""

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")


def embed_query(text: str, embedding_url: str, use_prefix: bool = True) -> list[float]:
    """Dense embedding через gpu_server."""
    prefix = (
        "Instruct: Given a user question about ML, AI, LLM or tech news, "
        "retrieve relevant Telegram channel posts\nQuery: "
    )
    input_text = prefix + text if use_prefix else text
    body = json.dumps({"inputs": [input_text], "normalize": True}).encode()
    req = urllib.request.Request(
        f"{embedding_url}/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())[0]


def colbert_encode(text: str, embedding_url: str) -> list[list[float]] | None:
    """ColBERT per-token encoding через gpu_server."""
    try:
        body = json.dumps({"texts": [text], "is_query": True}).encode()
        req = urllib.request.Request(
            f"{embedding_url}/colbert-encode",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        result = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return result[0] if result else None
    except Exception:
        return None


def normalize_query(text: str, lexicon: dict) -> str:
    """Нормализация query через JSON lexicon: добавляет синонимы, не заменяет оригинал."""
    additions = []
    text_lower = text.lower()
    for category in lexicon.values():
        if not isinstance(category, dict):
            continue
        for key, replacements in category.items():
            if key.lower() in text_lower:
                additions.extend(replacements)
    if additions:
        return text + " " + " ".join(additions)
    return text


def prf_expand(
    query: str, sparse_model, qdrant_url: str, collection: str, top_k: int = 5,
) -> str:
    """Pseudo-Relevance Feedback: BM25 top-K → top terms → expanded query."""
    from collections import Counter

    sparse_result = next(iter(sparse_model.query_embed(query)))
    sparse_q = {
        "indices": sparse_result.indices.tolist(),
        "values": sparse_result.values.tolist(),
    }
    body = {
        "query": sparse_q,
        "using": "sparse_vector",
        "limit": top_k,
        "with_payload": ["text"],
    }
    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points/query",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    points = resp["result"]["points"]

    all_words: Counter = Counter()
    for p in points:
        text = p.get("payload", {}).get("text", "")
        words = [w.lower().strip(".,!?;:()\"'") for w in text.split() if len(w) > 3]
        all_words.update(words)

    query_words = {w.lower().strip(".,!?;:()\"'") for w in query.split()}
    new_terms = [w for w, _ in all_words.most_common(30) if w not in query_words][:5]
    return query + " " + " ".join(new_terms) if new_terms else query


def search_qdrant(
    query_text: str,
    embedding_url: str,
    qdrant_url: str,
    collection: str,
    sparse_model,
    use_colbert: bool = False,
    top_k: int = 20,
    fusion: str = "rrf",
    dense_limit: int = 20,
    bm25_limit: int = 100,
    rrf_weights: list[float] | None = None,
    use_prefix: bool = True,
    rrf_limit: int | None = None,
    colbert_pool: int | None = None,
    dense_only: bool = False,
    sparse_query_text: str | None = None,
) -> list[dict]:
    """Поиск в Qdrant: BM25+Dense → RRF (→ ColBERT rerank если доступен).

    sparse_query_text: если задан, BM25 использует его вместо query_text (для R2 normalize-sparse-only).
    """
    dense_vec = embed_query(query_text, embedding_url, use_prefix=use_prefix)

    bm25_text = sparse_query_text or query_text
    sparse_result = next(iter(sparse_model.query_embed(bm25_text)))
    sparse_q = {
        "indices": sparse_result.indices.tolist(),
        "values": sparse_result.values.tolist(),
    }

    colbert_vecs = None
    if use_colbert:
        colbert_vecs = colbert_encode(query_text, embedding_url)

    # Fusion query — подставляем weights если заданы
    if fusion == "rrf" and rrf_weights:
        fusion_query = {"rrf": {"weights": rrf_weights}}
    else:
        fusion_query = {"fusion": fusion}

    # Лимиты
    effective_rrf_limit = rrf_limit or max(top_k * 3, 30)
    effective_colbert_pool = colbert_pool or top_k

    if colbert_vecs:
        # Prefetch ветки
        if dense_only:
            prefetch_inner = [
                {"query": dense_vec, "using": "dense_vector", "limit": dense_limit},
            ]
        else:
            prefetch_inner = [
                {"query": dense_vec, "using": "dense_vector", "limit": dense_limit},
                {"query": sparse_q, "using": "sparse_vector", "limit": bm25_limit},
            ]

        # 3-stage: prefetch → RRF → ColBERT MaxSim
        body = {
            "prefetch": [
                {
                    "prefetch": prefetch_inner,
                    "query": fusion_query,
                    "limit": effective_rrf_limit,
                }
            ],
            "query": colbert_vecs,
            "using": "colbert_vector",
            "limit": effective_colbert_pool,
            "with_payload": ["channel", "message_id"],
        }
    else:
        # 2-stage: prefetch → fusion
        if dense_only:
            prefetch = [
                {"query": dense_vec, "using": "dense_vector", "limit": dense_limit},
            ]
        else:
            prefetch = [
                {"query": dense_vec, "using": "dense_vector", "limit": dense_limit},
                {"query": sparse_q, "using": "sparse_vector", "limit": bm25_limit},
            ]
        body = {
            "prefetch": prefetch,
            "query": fusion_query,
            "limit": top_k,
            "with_payload": ["channel", "message_id"],
        }

    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points/query",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    return resp["result"]["points"]


def check_recall(points: list[dict], expected_docs: list[str], k: int, fuzzy: int = 5) -> float:
    """Проверяет recall@k: сколько expected docs найдено в top-k."""
    if not expected_docs:
        return 0.0
    top_k_points = points[:k]
    matched = 0
    for exp in expected_docs:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for p in top_k_points:
            pay = p.get("payload", {})
            h_ch = pay.get("channel", "").lower()
            h_msg = pay.get("message_id", 0)
            if h_ch == exp_ch and abs(h_msg - exp_msg) <= fuzzy:
                matched += 1
                break
    return matched / len(expected_docs)


def compute_mrr(results: list[dict], k: int = 20) -> float:
    """Mean Reciprocal Rank@k."""
    rr_sum = 0.0
    count = 0
    for r in results:
        if "error" in r:
            continue
        rr = r.get("reciprocal_rank", 0.0)
        rr_sum += rr
        count += 1
    return rr_sum / count if count > 0 else 0.0


def find_reciprocal_rank(
    points: list[dict], expected_docs: list[str], k: int = 20, fuzzy: int = 5,
) -> float:
    """Reciprocal rank: 1/позиция первого найденного expected doc (0 если не найден)."""
    top_k_points = points[:k]
    for exp in expected_docs:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for rank, p in enumerate(top_k_points, 1):
            pay = p.get("payload", {})
            if pay.get("channel", "").lower() == exp_ch and abs(pay.get("message_id", 0) - exp_msg) <= fuzzy:
                return 1.0 / rank
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation")
    parser.add_argument("--dataset", required=True, help="Path to retrieval dataset JSON")
    parser.add_argument("--collection", default="news_colbert")
    parser.add_argument("--qdrant-url", default="http://localhost:16333")
    parser.add_argument("--embedding-url", default="http://localhost:8082")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-colbert", action="store_true", help="Disable ColBERT rerank")
    parser.add_argument("--fuzzy", type=int, default=5, help="msg_id fuzzy tolerance")
    parser.add_argument("--fusion", default="rrf", choices=["rrf", "dbsf"], help="Fusion method")
    # Ablation параметры (phase 1)
    parser.add_argument("--rrf-weights", default=None, help='RRF weights, e.g. "1.0,3.0"')
    parser.add_argument("--bm25-limit", type=int, default=100, help="BM25 prefetch limit")
    parser.add_argument("--dense-limit", type=int, default=20, help="Dense prefetch limit")
    parser.add_argument("--no-prefix", action="store_true", help="Disable instruction prefix")
    parser.add_argument("--output", default=None, help="Output JSON path (default: results/raw/)")
    # Phase 2 параметры
    parser.add_argument("--rrf-limit", type=int, default=None, help="RRF output limit (default: top_k*3)")
    parser.add_argument("--colbert-pool", type=int, default=None, help="ColBERT rerank pool (default: top_k)")
    parser.add_argument("--dense-only", action="store_true", help="Skip BM25, dense+ColBERT only")
    parser.add_argument("--normalize-query", action="store_true", help="Apply lexicon normalization to both branches")
    parser.add_argument("--normalize-sparse-only", action="store_true", help="Lexicon normalization for BM25 only")
    parser.add_argument("--lexicon", default=None, help="JSON lexicon file for normalization")
    parser.add_argument("--prf-expand", action="store_true", help="BM25 PRF: initial search → top terms → expanded query")
    parser.add_argument("--prf-top-k", type=int, default=5, help="PRF: top-K initial hits for term extraction")
    args = parser.parse_args()

    # Парсим RRF weights
    rrf_weights = None
    if args.rrf_weights:
        rrf_weights = [float(x) for x in args.rrf_weights.split(",")]

    # Загружаем lexicon если нужен
    lexicon = None
    if args.lexicon and (args.normalize_query or args.normalize_sparse_only):
        with open(args.lexicon, encoding="utf-8") as f:
            lexicon = json.load(f)
        print(f"Lexicon loaded: {sum(len(v) for v in lexicon.values() if isinstance(v, dict))} entries")

    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)

    extra_flags = []
    if args.dense_only:
        extra_flags.append("dense-only")
    if args.normalize_query:
        extra_flags.append("normalize-all")
    if args.normalize_sparse_only:
        extra_flags.append("normalize-sparse")
    if args.prf_expand:
        extra_flags.append(f"PRF(k={args.prf_top_k})")
    if args.rrf_limit:
        extra_flags.append(f"rrf_limit={args.rrf_limit}")
    if args.colbert_pool:
        extra_flags.append(f"colbert_pool={args.colbert_pool}")

    print(f"Dataset: {len(items)} queries, collection: {args.collection}")
    print(f"ColBERT: {'OFF' if args.no_colbert else 'ON'}, Fusion: {args.fusion}, fuzzy: ±{args.fuzzy}")
    print(f"Dense limit: {args.dense_limit}, BM25 limit: {args.bm25_limit}, "
          f"Prefix: {'OFF' if args.no_prefix else 'ON'}, "
          f"RRF weights: {rrf_weights or 'default'}")
    if extra_flags:
        print(f"Phase 2: {', '.join(extra_flags)}")

    # Загружаем BM25 sparse model
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    use_colbert = not args.no_colbert and "colbert" in args.collection

    results = []
    t_total = time.time()

    for i, item in enumerate(items):
        query = item["query"]
        expected = item.get("expected_documents", [])

        # Query preprocessing
        search_query = query
        sparse_query = None  # None = same as search_query

        if args.prf_expand:
            search_query = prf_expand(
                query, sparse_model, args.qdrant_url, args.collection, args.prf_top_k,
            )

        if lexicon and args.normalize_sparse_only:
            sparse_query = normalize_query(query, lexicon)
        elif lexicon and args.normalize_query:
            search_query = normalize_query(query, lexicon)

        t0 = time.time()
        try:
            points = search_qdrant(
                search_query, args.embedding_url, args.qdrant_url,
                args.collection, sparse_model,
                use_colbert=use_colbert, top_k=args.top_k, fusion=args.fusion,
                dense_limit=args.dense_limit, bm25_limit=args.bm25_limit,
                rrf_weights=rrf_weights, use_prefix=not args.no_prefix,
                rrf_limit=args.rrf_limit, colbert_pool=args.colbert_pool,
                dense_only=args.dense_only, sparse_query_text=sparse_query,
            )
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            results.append({"id": item["id"], "recall_1": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0, "reciprocal_rank": 0, "latency": 0, "error": str(e)})
            continue
        latency = time.time() - t0

        r1 = check_recall(points, expected, 1, args.fuzzy)
        r3 = check_recall(points, expected, 3, args.fuzzy)
        r5 = check_recall(points, expected, 5, args.fuzzy)
        r10 = check_recall(points, expected, 10, args.fuzzy)
        r20 = check_recall(points, expected, 20, args.fuzzy)
        rr = find_reciprocal_rank(points, expected, args.top_k, args.fuzzy)

        hit_docs = [f"{p['payload']['channel']}:{p['payload']['message_id']}" for p in points[:5]]

        results.append({
            "id": item["id"],
            "query": query[:80],
            "expected": expected,
            "category": item.get("category", ""),
            "recall_1": r1, "recall_3": r3, "recall_5": r5, "recall_10": r10, "recall_20": r20,
            "reciprocal_rank": rr,
            "latency": latency,
            "top5_hits": hit_docs,
        })

        status = "✅" if r5 > 0 else "❌"
        if (i + 1) % 10 == 0 or r5 == 0:
            print(f"  [{i+1}/{len(items)}] {status} r@5={r5:.2f} {latency:.2f}s | {query[:60]}")

    elapsed = time.time() - t_total

    # Агрегация
    valid = [r for r in results if "error" not in r]
    r1_list = [r["recall_1"] for r in valid]
    r3_list = [r["recall_3"] for r in valid]
    r5_list = [r["recall_5"] for r in valid]
    r10_list = [r["recall_10"] for r in valid]
    r20_list = [r["recall_20"] for r in valid]
    mrr = compute_mrr(results)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\n{'='*60}")
    print(f"Retrieval Evaluation Results ({args.collection})")
    print(f"{'='*60}")
    print(f"Queries: {len(items)}, Errors: {sum(1 for r in results if 'error' in r)}")
    print(f"ColBERT: {'ON' if use_colbert else 'OFF'}, Fusion: {args.fusion}, Fuzzy: ±{args.fuzzy}")
    print(f"Dense limit: {args.dense_limit}, BM25 limit: {args.bm25_limit}, "
          f"Prefix: {'OFF' if args.no_prefix else 'ON'}, "
          f"RRF weights: {rrf_weights or 'default'}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(items):.2f}s/query)")
    print()
    print(f"Recall@1:  {avg(r1_list):.3f}  ({sum(1 for r in r1_list if r==1.0)}/{len(r1_list)})")
    print(f"Recall@3:  {avg(r3_list):.3f}  ({sum(1 for r in r3_list if r==1.0)}/{len(r3_list)})")
    print(f"Recall@5:  {avg(r5_list):.3f}  ({sum(1 for r in r5_list if r==1.0)}/{len(r5_list)})")
    print(f"Recall@10: {avg(r10_list):.3f}  ({sum(1 for r in r10_list if r==1.0)}/{len(r10_list)})")
    print(f"Recall@20: {avg(r20_list):.3f}  ({sum(1 for r in r20_list if r==1.0)}/{len(r20_list)})")
    print(f"MRR@20:    {mrr:.3f}")

    # По категориям
    by_category = defaultdict(list)
    for r in valid:
        cat = r.get("category", "unknown")
        by_category[cat].append(r)

    if len(by_category) > 1:
        print("\nПо категориям:")
        for cat, cat_results in sorted(by_category.items()):
            cat_r1 = avg([r["recall_1"] for r in cat_results])
            cat_r5 = avg([r["recall_5"] for r in cat_results])
            cat_r20 = avg([r["recall_20"] for r in cat_results])
            cat_mrr = avg([r["reciprocal_rank"] for r in cat_results])
            print(f"  {cat:20s} n={len(cat_results):3d}  R@1={cat_r1:.2f}  R@5={cat_r5:.2f}  R@20={cat_r20:.2f}  MRR={cat_mrr:.3f}")

    # По каналам
    by_channel = defaultdict(list)
    for r in valid:
        ch = r["expected"][0].split(":")[0] if r.get("expected") else "?"
        by_channel[ch].append(r["recall_5"])
    print("\nRecall@5 по каналам:")
    for ch, vals in sorted(by_channel.items(), key=lambda x: sum(x[1])/len(x[1])):
        mean = sum(vals) / len(vals)
        print(f"  {ch:35s} {mean:.2f} ({sum(1 for v in vals if v==1.0)}/{len(vals)})")

    # Сохраняем
    ts = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "timestamp": ts,
        "collection": args.collection,
        "colbert": use_colbert,
        "fusion": args.fusion,
        "fuzzy": args.fuzzy,
        "dense_limit": args.dense_limit,
        "bm25_limit": args.bm25_limit,
        "prefix": not args.no_prefix,
        "rrf_weights": rrf_weights,
        "rrf_limit": args.rrf_limit,
        "colbert_pool": args.colbert_pool,
        "dense_only": args.dense_only,
        "normalize": "sparse" if args.normalize_sparse_only else ("all" if args.normalize_query else None),
        "prf_expand": args.prf_expand,
        "total_queries": len(items),
        "recall_at_1": avg(r1_list),
        "recall_at_3": avg(r3_list),
        "recall_at_5": avg(r5_list),
        "recall_at_10": avg(r10_list),
        "recall_at_20": avg(r20_list),
        "mrr_at_20": mrr,
        "avg_latency": elapsed / len(items),
        "results": results,
    }
    out_path = args.output or f"results/raw/retrieval_eval_{ts}.json"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
