"""
Retrieval-only evaluation — прямые Qdrant queries без LLM.

Быстрый: ~4с на запрос. Тестирует качество поиска изолированно от pipeline.

Артефакт: один .jsonl файл с live flush (tail -f для мониторинга).
  Строка 1: header (config, preflight, git_sha)
  Строки 2..N: per-query результаты
  Последняя строка: summary

Запуск:
    python scripts/evaluate_retrieval.py \
        --dataset datasets/eval_retrieval_v3.json \
        --collection news_colbert_v2 \
        --no-prefix --dense-limit 40 --rrf-weights "1.0,3.0" \
        --output experiments/runs/RUN-NNN/raw_ro.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")


# ── Helpers ───────────���────────────────────────────────────────────

def get_git_sha() -> str:
    """Текущий git commit SHA (short)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


class LiveWriter:
    """Пишет jsonl с flush после каждой строки."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")
        self._path = path

    def write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    @property
    def path(self) -> str:
        return self._path


def rerank_texts(query: str, texts: list[str], reranker_url: str) -> list[float] | None:
    """CE scoring через gpu_server /rerank. Возвращает scores в порядке input или None при ошибке."""
    if not texts:
        return []
    try:
        body = json.dumps({
            "query": query,
            "texts": [t[:512] for t in texts],
            "raw_scores": True,
            "truncate": True,
        }).encode()
        req = urllib.request.Request(
            f"{reranker_url.rstrip('/')}/rerank",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        results = json.loads(urllib.request.urlopen(req, timeout=30).read())
        scores = [0.0] * len(texts)
        for item in results:
            scores[item["index"]] = item["score"]
        return scores
    except Exception as e:
        return None


def preflight_check(qdrant_url: str, embedding_url: str, reranker_url: str | None = None) -> dict[str, dict]:
    """Проверить доступность сервисов."""
    checks = {
        "qdrant": (qdrant_url, "/collections"),
        "embedding": (embedding_url, "/health"),
    }
    if reranker_url:
        checks["reranker"] = (reranker_url, "/health")
    results = {}
    for svc, (base, endpoint) in checks.items():
        url = base.rstrip("/") + endpoint
        t0 = time.time()
        try:
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            ms = (time.time() - t0) * 1000
            results[svc] = {"ok": True, "url": base, "ms": round(ms)}
        except Exception as e:
            ms = (time.time() - t0) * 1000
            results[svc] = {"ok": False, "url": base, "error": str(e)[:120], "ms": round(ms)}
    return results


def find_expected_rank(points: list[dict], expected_docs: list[str], fuzzy: int = 5) -> int | None:
    """Позиция (1-based) первого expected doc. None если не найден."""
    for exp in expected_docs:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for rank, p in enumerate(points, 1):
            pay = p.get("payload", {})
            if (pay.get("channel", "").lower() == exp_ch
                    and abs(pay.get("message_id", 0) - exp_msg) <= fuzzy):
                return rank
    return None


# ── Search / Embedding ─────��───────────────────────────────────────

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
        if dense_only:
            prefetch_inner = [
                {"query": dense_vec, "using": "dense_vector", "limit": dense_limit},
            ]
        else:
            prefetch_inner = [
                {"query": dense_vec, "using": "dense_vector", "limit": dense_limit},
                {"query": sparse_q, "using": "sparse_vector", "limit": bm25_limit},
            ]

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
            "with_payload": ["channel", "message_id", "text"],
        }
    else:
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
            "with_payload": ["channel", "message_id", "text"],
        }

    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points/query",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    return resp["result"]["points"]


# ── Recall helpers ─────────────────────────────────────────────────

def check_recall(points: list[dict], expected_docs: list[str], k: int, fuzzy: int = 5) -> float:
    """Recall@k с fuzzy matching по message_id."""
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
            if (pay.get("channel", "").lower() == exp_ch
                    and abs(pay.get("message_id", 0) - exp_msg) <= fuzzy):
                matched += 1
                break
    return matched / len(expected_docs)


def find_reciprocal_rank(
    points: list[dict], expected_docs: list[str], k: int = 20, fuzzy: int = 5,
) -> float:
    """1/rank первого найденного expected doc."""
    top_k_points = points[:k]
    for exp in expected_docs:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for rank, p in enumerate(top_k_points, 1):
            pay = p.get("payload", {})
            if (pay.get("channel", "").lower() == exp_ch
                    and abs(pay.get("message_id", 0) - exp_msg) <= fuzzy):
                return 1.0 / rank
    return 0.0


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation")
    parser.add_argument("--dataset", required=True, help="Path to retrieval dataset JSON")
    parser.add_argument("--collection", default="news_colbert_v2")
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
    parser.add_argument("--output", default=None, help="Output path (.jsonl)")
    # Phase 2 параметры
    parser.add_argument("--rrf-limit", type=int, default=None, help="RRF output limit (default: top_k*3)")
    parser.add_argument("--colbert-pool", type=int, default=None, help="ColBERT rerank pool (default: top_k)")
    parser.add_argument("--dense-only", action="store_true", help="Skip BM25, dense+ColBERT only")
    parser.add_argument("--normalize-query", action="store_true", help="Apply lexicon normalization to both branches")
    parser.add_argument("--normalize-sparse-only", action="store_true", help="Lexicon normalization for BM25 only")
    parser.add_argument("--lexicon", default=None, help="JSON lexicon file for normalization")
    parser.add_argument("--prf-expand", action="store_true", help="BM25 PRF: initial search → top terms → expanded query")
    parser.add_argument("--prf-top-k", type=int, default=5, help="PRF: top-K initial hits for term extraction")
    parser.add_argument("--reranker-url", default=None, help="CE reranker URL for context quality metrics (e.g. http://localhost:8082)")
    args = parser.parse_args()

    # ── Preflight ──
    print("Preflight check...", file=sys.stderr)
    health = preflight_check(args.qdrant_url, args.embedding_url, args.reranker_url)
    for svc, info in health.items():
        status = "OK" if info["ok"] else "FAIL"
        detail = f"{info['ms']}ms" if info["ok"] else info.get("error", "?")[:60]
        print(f"  {svc:12s} {info['url']:40s} {status} ({detail})", file=sys.stderr)
        if not info["ok"]:
            print(f"\nPreflight FAILED — {svc} недоступен. Прерываю.", file=sys.stderr)
            sys.exit(1)
    print("Preflight OK\n", file=sys.stderr)

    # Парсим RRF weights
    rrf_weights = None
    if args.rrf_weights:
        rrf_weights = [float(x) for x in args.rrf_weights.split(",")]

    # Загружаем lexicon если нужен
    lexicon = None
    if args.lexicon and (args.normalize_query or args.normalize_sparse_only):
        with open(args.lexicon, encoding="utf-8") as f:
            lexicon = json.load(f)
        print(f"Lexicon loaded: {sum(len(v) for v in lexicon.values() if isinstance(v, dict))} entries", file=sys.stderr)

    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)

    # Загружаем BM25 sparse model
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    use_colbert = not args.no_colbert and "colbert" in args.collection

    # ── Output path ──
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.output or f"experiments/runs/retrieval_{ts}.jsonl"
    if not out_path.endswith(".jsonl"):
        out_path = out_path.rsplit(".", 1)[0] + ".jsonl"

    config = {
        "collection": args.collection,
        "colbert": use_colbert,
        "fusion": args.fusion,
        "dense_limit": args.dense_limit,
        "bm25_limit": args.bm25_limit,
        "prefix": not args.no_prefix,
        "rrf_weights": rrf_weights,
        "rrf_limit": args.rrf_limit,
        "colbert_pool": args.colbert_pool,
        "dense_only": args.dense_only,
        "normalize": "sparse" if args.normalize_sparse_only else ("all" if args.normalize_query else None),
        "prf_expand": args.prf_expand,
        "top_k": args.top_k,
        "fuzzy": args.fuzzy,
        "reranker_url": args.reranker_url,
    }

    writer = LiveWriter(out_path)

    # ── Header ──
    writer.write({
        "type": "header",
        "timestamp": ts,
        "git_sha": get_git_sha(),
        "total_queries": len(items),
        "dataset": args.dataset,
        "preflight": {svc: info["ok"] for svc, info in health.items()},
        "config": config,
    })

    print(f"Dataset: {len(items)} queries, collection: {args.collection}", file=sys.stderr)
    print(f"ColBERT: {'ON' if use_colbert else 'OFF'}, Fusion: {args.fusion}", file=sys.stderr)
    print(f"Output: {out_path}", file=sys.stderr)
    print(file=sys.stderr)

    # ── Eval loop ──
    t_total = time.time()
    running_r1 = []
    running_r5 = []
    running_mrr = []
    error_count = 0
    ce_ok_count = 0

    for i, item in enumerate(items):
        query = item["query"]
        expected = item.get("expected_documents", [])

        # Query preprocessing
        search_query = query
        sparse_query = None

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
            latency = time.time() - t0
            row = {
                "type": "query",
                "qid": item["id"],
                "query": query[:80],
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "latency": round(latency, 2),
            }
            writer.write(row)
            error_count += 1
            print(f"  [{i+1}/{len(items)}] ERROR {item['id']} | {str(e)[:60]}", file=sys.stderr)
            continue
        search_ms = round((time.time() - t0) * 1000)

        r1 = check_recall(points, expected, 1, args.fuzzy)
        r3 = check_recall(points, expected, 3, args.fuzzy)
        r5 = check_recall(points, expected, 5, args.fuzzy)
        r10 = check_recall(points, expected, 10, args.fuzzy)
        r20 = check_recall(points, expected, 20, args.fuzzy)
        rr = find_reciprocal_rank(points, expected, args.top_k, args.fuzzy)

        # Базовые метрики
        n_results = len(points)
        expected_rank = find_expected_rank(points, expected, args.fuzzy)
        top5 = points[:5]
        n_channels = len(set(p.get("payload", {}).get("channel", "") for p in top5))
        top5_scores = [round(p.get("score", 0), 4) for p in top5]
        top5_hits = [f"{p['payload']['channel']}:{p['payload']['message_id']}" for p in top5]

        # CE scoring — метрики полезности контекста для LLM
        ce_ok = False
        mean_ce = None
        ce_neg = None
        ce_top = None
        min_ce = None
        if args.reranker_url and top5:
            t_ce = time.time()
            texts = [p.get("payload", {}).get("text", "") for p in top5]
            ce_scores = rerank_texts(query, texts, args.reranker_url)
            ce_ms_val = round((time.time() - t_ce) * 1000)
            if ce_scores is not None:
                ce_ok = True
                mean_ce = round(sum(ce_scores) / len(ce_scores), 2)
                ce_neg = sum(1 for s in ce_scores if s < 0)
                ce_top = round(max(ce_scores), 2)
                min_ce = round(min(ce_scores), 2)

        running_r1.append(r1)
        running_r5.append(r5)
        running_mrr.append(rr)
        if ce_ok:
            ce_ok_count += 1

        row = {
            "type": "query",
            "qid": item["id"],
            "query": query[:80],
            "cat": item.get("category", ""),
            "r1": r1, "r5": r5, "r10": r10, "r20": r20,
            "rr": round(rr, 3),
            "latency": round(search_ms / 1000, 2),
            "search_ms": search_ms,
            "n_results": n_results,
            "expected_rank": expected_rank,
            "n_channels": n_channels,
            "top5_scores": top5_scores,
            "top5_hits": top5_hits,
            "all_docs": [
                {"ch": p.get("payload", {}).get("channel", ""), "mid": p.get("payload", {}).get("message_id", ""), "text": p.get("payload", {}).get("text", "")}
                for p in points[:20]
            ],
            # CE quality metrics (None if reranker not configured)
            "ce_ok": ce_ok,
            "mean_ce": mean_ce,
            "ce_neg": ce_neg,
            "ce_top": ce_top,
            "min_ce": min_ce,
        }
        if ce_ok:
            row["ce_ms"] = ce_ms_val
        writer.write(row)

        # Live stderr
        done = i + 1
        if done % 10 == 0 or r5 == 0:
            avg_r5 = sum(running_r5) / len(running_r5)
            avg_mrr = sum(running_mrr) / len(running_mrr)
            ce_tag = f"CE={ce_ok_count}/{done}" if args.reranker_url else ""
            status = "+" if r5 > 0 else "MISS"
            print(
                f"  [{done}/{len(items)}] {status:4s} r@5={r5:.0f} {search_ms}ms | "
                f"R@5={avg_r5:.3f} MRR={avg_mrr:.3f} {ce_tag} | {query[:40]}",
                file=sys.stderr,
            )

    elapsed = time.time() - t_total

    # ── Summary line ──
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    summary = {
        "type": "summary",
        "timestamp": ts,
        "total_queries": len(items),
        "errors": error_count,
        "recall_at_1": round(avg(running_r1), 4),
        "recall_at_5": round(avg(running_r5), 4),
        "mrr_at_20": round(avg(running_mrr), 4),
        "ce_ok_rate": round(ce_ok_count / len(items), 3) if items else 0,
        "avg_latency": round(elapsed / len(items), 2),
        "total_time": round(elapsed, 1),
    }
    writer.write(summary)
    writer.close()

    # ── Console summary ──
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Queries: {len(items)}, Errors: {error_count}", file=sys.stderr)
    print(f"Total: {elapsed:.1f}s ({elapsed/len(items):.2f}s/query)", file=sys.stderr)
    print(f"R@1={avg(running_r1):.3f}  R@5={avg(running_r5):.3f}  MRR={avg(running_mrr):.3f}", file=sys.stderr)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # ── Backward compat: summary .json ──
    json_path = out_path.rsplit(".", 1)[0] + ".json"
    all_rows = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "query":
                all_rows.append(obj)

    valid = [r for r in all_rows if "error" not in r]
    report = {
        "timestamp": ts,
        **config,
        "total_queries": len(items),
        "recall_at_1": avg([r["r1"] for r in valid]),
        "recall_at_3": avg([r.get("r3", 0) for r in valid]),
        "recall_at_5": avg([r["r5"] for r in valid]),
        "recall_at_10": avg([r["r10"] for r in valid]),
        "recall_at_20": avg([r["r20"] for r in valid]),
        "mrr_at_20": avg([r["rr"] for r in valid]),
        "avg_latency": elapsed / len(items),
        "results": all_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
