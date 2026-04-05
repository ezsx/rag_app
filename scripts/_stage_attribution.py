"""Stage attribution — для каждого permanent miss определить где теряется документ.

Стадии: dense top-100, BM25 top-100, RRF top-60, ColBERT top-20.
Output: JSON + table с hit/miss на каждой стадии.
"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

QDRANT_URL = "http://localhost:16333"
COLLECTION = "news_colbert_v2"
EMBEDDING_URL = "http://localhost:8082"
FUZZY = 5  # msg_id tolerance


# ── Permanent misses из phase 1 best config ──
MISSES = [
    {"id": "ret_030", "query": "Какую конституцию опубликовала Anthropic для своих моделей?", "expected": "theworldisnoteasy:2378", "category": "factual"},
    {"id": "ret_034", "query": "ИИ-модели умеют играть в стратегические игры лучше людей?", "expected": "seeallochnaya:3054", "category": "factual"},
    {"id": "ret_049", "query": "Что нового в генеративных моделях произошло в феврале 2026?", "expected": "ai_newz:4433", "category": "temporal"},
    {"id": "ret_059", "query": "Какие инвестиции привлекались в ИИ-компании в начале 2026?", "expected": "seeallochnaya:3270", "category": "temporal"},
    {"id": "ret_068", "query": "Что в канале llm_under_hood рекомендовали обязательно посмотреть?", "expected": "llm_under_hood:752", "category": "channel_specific"},
    {"id": "ret_075", "query": "Что Постнаука обсуждала с gonzo_ml?", "expected": "gonzo_ml:4666", "category": "channel_specific"},
    {"id": "ret_098", "query": "Сколько OpenAI привлекла в последнем раунде инвестиций?", "expected": "seeallochnaya:3427", "category": "entity"},
    {"id": "ret_109", "query": "че там по трансформерам нового?", "expected": "gonzo_ml:4567", "category": "edge"},
    {"id": "ret_110", "query": "какие нейросетки умеют видосы делать", "expected": "neurohive:1929", "category": "edge"},
    {"id": "ret_111", "query": "Claude Code реально может старые игры запускать?", "expected": "denissexy:11238", "category": "edge"},
    {"id": "ret_112", "query": "нейросети заменят программистов или нет", "expected": "techno_yandex:4978", "category": "edge"},
    {"id": "ret_113", "query": "кто-нить делал ии-музыку через suno?", "expected": "denissexy:10782", "category": "edge"},
]


def embed_query(text: str) -> list[float]:
    """Dense embedding через gpu_server (без prefix, как в production)."""
    body = json.dumps({"inputs": [text], "normalize": True}).encode()
    req = urllib.request.Request(
        f"{EMBEDDING_URL}/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())[0]


def colbert_encode(text: str) -> list[list[float]] | None:
    """ColBERT per-token encoding."""
    try:
        body = json.dumps({"texts": [text], "is_query": True}).encode()
        req = urllib.request.Request(
            f"{EMBEDDING_URL}/colbert-encode",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        result = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return result[0] if result else None
    except Exception:
        return None


def ce_rerank(query: str, texts: list[str]) -> list[float]:
    """Cross-encoder raw scores via gpu_server /rerank."""
    body = json.dumps({
        "query": query,
        "texts": texts,
        "raw_scores": True,
        "truncate": True,
    }).encode()
    req = urllib.request.Request(
        f"{EMBEDDING_URL}/rerank",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    results = json.loads(urllib.request.urlopen(req, timeout=30).read())
    scores = [0.0] * len(texts)
    for item in results:
        scores[item["index"]] = item["score"]
    return scores


def qdrant_query(body: dict) -> list[dict]:
    """Raw Qdrant query API call."""
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    return resp["result"]["points"]


def check_hit(points: list[dict], expected: str, fuzzy: int = FUZZY) -> tuple[bool, int]:
    """Проверить найден ли expected doc. Возвращает (found, rank). rank=0 = not found."""
    parts = expected.split(":", 1)
    if len(parts) != 2:
        return False, 0
    exp_ch, exp_msg = parts[0].lower(), int(parts[1])
    for rank, p in enumerate(points, 1):
        pay = p.get("payload", {})
        if pay.get("channel", "").lower() == exp_ch and abs(pay.get("message_id", 0) - exp_msg) <= fuzzy:
            return True, rank
    return False, 0


def main():
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    results = []
    print(f"Stage attribution for {len(MISSES)} permanent misses")
    print(f"{'='*90}")

    for item in MISSES:
        query = item["query"]
        expected = item["expected"]
        print(f"\n[{item['id']}] ({item['category']}) {query}")
        print(f"  expected: {expected}")

        dense_vec = embed_query(query)
        sparse_result = next(iter(sparse_model.query_embed(query)))
        sparse_q = {
            "indices": sparse_result.indices.tolist(),
            "values": sparse_result.values.tolist(),
        }
        colbert_vecs = colbert_encode(query)

        # Stage 1: Dense top-100
        dense_points = qdrant_query({
            "query": dense_vec,
            "using": "dense_vector",
            "limit": 100,
            "with_payload": ["channel", "message_id"],
        })
        dense_hit, dense_rank = check_hit(dense_points, expected)

        # Stage 2: BM25 top-100
        bm25_points = qdrant_query({
            "query": sparse_q,
            "using": "sparse_vector",
            "limit": 100,
            "with_payload": ["channel", "message_id"],
        })
        bm25_hit, bm25_rank = check_hit(bm25_points, expected)

        # Stage 3: RRF (no ColBERT) top-60
        rrf_points = qdrant_query({
            "prefetch": [
                {"query": dense_vec, "using": "dense_vector", "limit": 40},
                {"query": sparse_q, "using": "sparse_vector", "limit": 100},
            ],
            "query": {"rrf": {"weights": [1.0, 3.0]}},
            "limit": 60,
            "with_payload": ["channel", "message_id"],
        })
        rrf_hit, rrf_rank = check_hit(rrf_points, expected)

        # Stage 4: ColBERT top-20
        colbert_hit, colbert_rank = False, 0
        if colbert_vecs:
            colbert_points = qdrant_query({
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
                "limit": 20,
                "with_payload": ["channel", "message_id", "text"],
            })
            colbert_hit, colbert_rank = check_hit(colbert_points, expected)

            # Stage 5: CE score (if in ColBERT top-20)
            ce_score = None
            if colbert_hit and colbert_points:
                texts = [p["payload"].get("text", "") for p in colbert_points]
                try:
                    scores = ce_rerank(query, texts)
                    # Найти score для expected doc
                    for idx, p in enumerate(colbert_points):
                        pay = p.get("payload", {})
                        if pay.get("channel", "").lower() == expected.split(":")[0].lower():
                            if abs(pay.get("message_id", 0) - int(expected.split(":")[1])) <= FUZZY:
                                ce_score = scores[idx]
                                break
                except Exception as e:
                    print(f"  CE error: {e}")
        else:
            # Без ColBERT — RRF top-20
            colbert_points = rrf_points[:20]
            colbert_hit, colbert_rank = check_hit(colbert_points, expected)
            ce_score = None

        def status(h, r):
            return f"✅ rank={r}" if h else "❌"
        print(f"  dense-100: {status(dense_hit, dense_rank)}")
        print(f"  bm25-100:  {status(bm25_hit, bm25_rank)}")
        print(f"  rrf-60:    {status(rrf_hit, rrf_rank)}")
        print(f"  colbert-20:{status(colbert_hit, colbert_rank)}")
        if ce_score is not None:
            print(f"  CE score:  {ce_score:.3f}")

        # Определяем стадию потери
        if not dense_hit and not bm25_hit:
            lost_at = "not_in_candidates"
        elif not rrf_hit:
            lost_at = "lost_in_rrf"
        elif not colbert_hit:
            lost_at = "lost_in_colbert"
        else:
            lost_at = "found"
        print(f"  → {lost_at}")

        results.append({
            "id": item["id"],
            "query": query,
            "expected": expected,
            "category": item["category"],
            "in_dense_100": dense_hit,
            "dense_rank": dense_rank,
            "in_bm25_100": bm25_hit,
            "bm25_rank": bm25_rank,
            "in_rrf_60": rrf_hit,
            "rrf_rank": rrf_rank,
            "in_colbert_20": colbert_hit,
            "colbert_rank": colbert_rank,
            "ce_score": ce_score,
            "lost_at": lost_at,
        })

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'ID':10s} {'Category':18s} {'Dense':>8s} {'BM25':>8s} {'RRF':>8s} {'ColBERT':>8s} {'Lost at'}")
    print("-" * 90)
    for r in results:
        d = f"r={r['dense_rank']}" if r['in_dense_100'] else "miss"
        b = f"r={r['bm25_rank']}" if r['in_bm25_100'] else "miss"
        f_ = f"r={r['rrf_rank']}" if r['in_rrf_60'] else "miss"
        c = f"r={r['colbert_rank']}" if r['in_colbert_20'] else "miss"
        print(f"{r['id']:10s} {r['category']:18s} {d:>8s} {b:>8s} {f_:>8s} {c:>8s} {r['lost_at']}")

    # Loss distribution
    loss_counts = {}
    for r in results:
        loss_counts[r["lost_at"]] = loss_counts.get(r["lost_at"], 0) + 1
    print("\nLoss distribution:")
    for stage, cnt in sorted(loss_counts.items(), key=lambda x: -x[1]):
        print(f"  {stage}: {cnt}/{len(results)}")

    # Save
    with open("results/ablation/stage_attribution.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved to results/ablation/stage_attribution.json")


if __name__ == "__main__":
    main()
