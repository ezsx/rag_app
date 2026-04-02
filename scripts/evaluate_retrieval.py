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


def embed_query(text: str, embedding_url: str) -> list[float]:
    """Dense embedding через gpu_server."""
    prefix = (
        "Instruct: Given a user question about ML, AI, LLM or tech news, "
        "retrieve relevant Telegram channel posts\nQuery: "
    )
    body = json.dumps({"inputs": [prefix + text], "normalize": True}).encode()
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


def search_qdrant(
    query_text: str,
    embedding_url: str,
    qdrant_url: str,
    collection: str,
    sparse_model,
    use_colbert: bool = False,
    top_k: int = 20,
    fusion: str = "rrf",
) -> list[dict]:
    """Поиск в Qdrant: BM25+Dense → RRF (→ ColBERT rerank если доступен)."""
    dense_vec = embed_query(query_text, embedding_url)
    sparse_result = next(iter(sparse_model.query_embed(query_text)))

    sparse_q = {
        "indices": sparse_result.indices.tolist(),
        "values": sparse_result.values.tolist(),
    }

    colbert_vecs = None
    if use_colbert:
        colbert_vecs = colbert_encode(query_text, embedding_url)

    if colbert_vecs:
        # 3-stage: BM25+Dense → RRF → ColBERT MaxSim
        body = {
            "prefetch": [
                {
                    "prefetch": [
                        {"query": dense_vec, "using": "dense_vector", "limit": 20},
                        {"query": sparse_q, "using": "sparse_vector", "limit": 100},
                    ],
                    "query": {"fusion": fusion},
                    "limit": max(top_k * 3, 30),
                }
            ],
            "query": colbert_vecs,
            "using": "colbert_vector",
            "limit": top_k,
            "with_payload": ["channel", "message_id"],
        }
    else:
        # 2-stage: BM25+Dense → weighted RRF
        body = {
            "prefetch": [
                {"query": dense_vec, "using": "dense_vector", "limit": 20},
                {"query": sparse_q, "using": "sparse_vector", "limit": 100},
            ],
            "query": {"fusion": "rrf"},
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
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)

    print(f"Dataset: {len(items)} queries, collection: {args.collection}")
    print(f"ColBERT: {'OFF' if args.no_colbert else 'ON'}, Fusion: {args.fusion}, fuzzy: ±{args.fuzzy}")

    # Загружаем BM25 sparse model
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    use_colbert = not args.no_colbert and "colbert" in args.collection

    results = []
    t_total = time.time()

    for i, item in enumerate(items):
        query = item["query"]
        expected = item.get("expected_documents", [])

        t0 = time.time()
        try:
            points = search_qdrant(
                query, args.embedding_url, args.qdrant_url,
                args.collection, sparse_model,
                use_colbert=use_colbert, top_k=args.top_k, fusion=args.fusion,
            )
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            results.append({"id": item["id"], "recall_1": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0, "latency": 0, "error": str(e)})
            continue
        latency = time.time() - t0

        r1 = check_recall(points, expected, 1, args.fuzzy)
        r5 = check_recall(points, expected, 5, args.fuzzy)
        r10 = check_recall(points, expected, 10, args.fuzzy)
        r20 = check_recall(points, expected, 20, args.fuzzy)

        hit_docs = [f"{p['payload']['channel']}:{p['payload']['message_id']}" for p in points[:5]]

        results.append({
            "id": item["id"],
            "query": query[:80],
            "expected": expected,
            "recall_1": r1, "recall_5": r5, "recall_10": r10, "recall_20": r20,
            "latency": latency,
            "top5_hits": hit_docs,
        })

        status = "✅" if r5 > 0 else "❌"
        if (i + 1) % 10 == 0 or r5 == 0:
            print(f"  [{i+1}/{len(items)}] {status} r@5={r5:.2f} {latency:.2f}s | {query[:60]}")

    elapsed = time.time() - t_total

    # Агрегация
    r1_list = [r["recall_1"] for r in results if "error" not in r]
    r5_list = [r["recall_5"] for r in results if "error" not in r]
    r10_list = [r["recall_10"] for r in results if "error" not in r]
    r20_list = [r["recall_20"] for r in results if "error" not in r]

    print(f"\n{'='*60}")
    print(f"Retrieval Evaluation Results ({args.collection})")
    print(f"{'='*60}")
    print(f"Queries: {len(items)}, Errors: {sum(1 for r in results if 'error' in r)}")
    print(f"ColBERT: {'ON' if use_colbert else 'OFF'}, Fuzzy: ±{args.fuzzy}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(items):.2f}s/query)")
    print()
    print(f"Recall@1:  {sum(r1_list)/len(r1_list):.3f}  (full: {sum(1 for r in r1_list if r==1.0)}/{len(r1_list)})")
    print(f"Recall@5:  {sum(r5_list)/len(r5_list):.3f}  (full: {sum(1 for r in r5_list if r==1.0)}/{len(r5_list)})")
    print(f"Recall@10: {sum(r10_list)/len(r10_list):.3f}  (full: {sum(1 for r in r10_list if r==1.0)}/{len(r10_list)})")
    print(f"Recall@20: {sum(r20_list)/len(r20_list):.3f}  (full: {sum(1 for r in r20_list if r==1.0)}/{len(r20_list)})")

    # По каналам
    by_channel = defaultdict(list)
    for r in results:
        if "error" not in r:
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
        "fuzzy": args.fuzzy,
        "total_queries": len(items),
        "recall_at_1": sum(r1_list) / len(r1_list),
        "recall_at_5": sum(r5_list) / len(r5_list),
        "recall_at_10": sum(r10_list) / len(r10_list),
        "recall_at_20": sum(r20_list) / len(r20_list),
        "avg_latency": elapsed / len(items),
        "results": results,
    }
    out_path = f"results/raw/retrieval_eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
