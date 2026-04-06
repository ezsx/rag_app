"""Сравнение merge стратегий: Round-Robin vs MMR на конкретных запросах.

Для каждого запроса:
1. Query plan + per-subquery search (общий)
2. Round-robin merge (текущий prod)
3. MMR merge (экспериментальный)
4. CE scores для обоих
5. Сравнение top-10
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
os.environ["QDRANT_URL"] = "http://localhost:16333"
os.environ["EMBEDDING_TEI_URL"] = "http://localhost:8082"
os.environ["RERANKER_TEI_URL"] = "http://localhost:8082"
os.environ["QDRANT_COLLECTION"] = "news_colbert_v2"
os.environ["LLM_BASE_URL"] = "http://localhost:8080"

from core.deps import get_hybrid_retriever, get_query_planner, get_reranker

hybrid = get_hybrid_retriever()
planner = get_query_planner()
reranker = get_reranker()

import urllib.request


def embed_text(text: str) -> list[float]:
    """Dense embedding для similarity computation."""
    body = json.dumps({"inputs": [text], "normalize": True}).encode()
    req = urllib.request.Request(
        "http://localhost:8082/embed", data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())[0]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def round_robin_merge(per_query_results: list[list]) -> list:
    """Текущий prod merge."""
    merged = []
    seen = set()
    max_len = max((len(r) for r in per_query_results), default=0)
    for rank in range(max_len):
        for sub in per_query_results:
            if rank < len(sub):
                c = sub[rank]
                if c.id not in seen:
                    merged.append(c)
                    seen.add(c.id)
    return merged


def mmr_merge(per_query_results: list[list], query: str, lam: float = 0.7, top_k: int = 30) -> list:
    """MMR merge: баланс relevance (CE score) и diversity (embedding similarity).

    1. Собрать все unique docs из всех subqueries
    2. Получить CE scores (relevance to original query)
    3. MMR selection: iteratively pick doc that maximizes λ*relevance - (1-λ)*max_sim_to_selected
    """
    # Collect unique docs
    all_docs = []
    seen = set()
    for sub in per_query_results:
        for c in sub:
            if c.id not in seen:
                all_docs.append(c)
                seen.add(c.id)

    if not all_docs:
        return []

    # CE scores as relevance signal
    texts = [c.text[:512] for c in all_docs]
    try:
        indices, scores = reranker.rerank_with_raw_scores(query, texts, top_n=len(texts))
        ce_scores = [0.0] * len(all_docs)
        for idx, score in zip(indices, scores):
            ce_scores[idx] = score
    except Exception:
        # Fallback: use position from first query that found this doc
        ce_scores = [1.0 / (i + 1) for i in range(len(all_docs))]

    # Normalize CE scores to [0, 1]
    if ce_scores:
        min_s, max_s = min(ce_scores), max(ce_scores)
        rng = max_s - min_s if max_s > min_s else 1.0
        ce_norm = [(s - min_s) / rng for s in ce_scores]
    else:
        ce_norm = [0.0] * len(all_docs)

    # Embed all docs for diversity computation
    embeddings = []
    for c in all_docs:
        try:
            emb = embed_text(c.text[:256])
            embeddings.append(emb)
        except Exception:
            embeddings.append([0.0] * 1024)

    # MMR selection
    selected = []
    selected_indices = set()
    remaining = set(range(len(all_docs)))

    for _ in range(min(top_k, len(all_docs))):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = ce_norm[idx]

            # Max similarity to already selected
            if selected_indices:
                max_sim = max(cosine_sim(embeddings[idx], embeddings[s]) for s in selected_indices)
            else:
                max_sim = 0.0

            mmr_score = lam * relevance - (1 - lam) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(all_docs[best_idx])
            selected_indices.add(best_idx)
            remaining.discard(best_idx)

    return selected


def run_comparison(query: str, expected_doc: str | None = None):
    """Прогнать один запрос через оба merge и сравнить."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    if expected_doc:
        print(f"EXPECTED: {expected_doc}")
    print(f"{'='*80}")

    # Query plan
    plan = planner.make_plan(query)
    all_queries = [query] + [q for q in plan.normalized_queries if q.lower() != query.lower()]

    print(f"\nSubqueries ({len(all_queries)} total incl. original):")
    for i, q in enumerate(all_queries):
        label = "ORIGINAL" if i == 0 else f"sq-{i}"
        print(f"  [{label}] {q}")

    # Per-query search
    per_query = []
    for q in all_queries:
        candidates = hybrid.search_with_plan(q, plan)
        per_query.append(candidates)

    total_raw = sum(len(r) for r in per_query)
    print(f"\nPer-query: {' + '.join(str(len(r)) for r in per_query)} = {total_raw} docs (with dupes)")

    # Round-robin merge
    rr_merged = round_robin_merge(per_query)[:30]

    # MMR merge
    mmr_merged = mmr_merge(per_query, query, lam=0.7, top_k=30)

    # CE scores for both
    def get_ce_scores(docs):
        texts = [c.text[:512] for c in docs]
        try:
            indices, scores = reranker.rerank_with_raw_scores(query, texts, top_n=len(texts))
            ce = [0.0] * len(docs)
            for idx, score in zip(indices, scores):
                ce[idx] = score
            return ce
        except Exception:
            return [0.0] * len(docs)

    rr_ce = get_ce_scores(rr_merged)
    mmr_ce = get_ce_scores(mmr_merged)

    # Display comparison
    print(f"\n{'─'*80}")
    print(f"{'ROUND-ROBIN (current prod)':^40s} │ {'MMR (λ=0.7)':^38s}")
    print(f"{'─'*80}")

    max_show = 10
    for i in range(max_show):
        # Round-robin side
        if i < len(rr_merged):
            c = rr_merged[i]
            ch = c.metadata.get('channel', '?')
            mid = c.metadata.get('message_id', 0)
            ce = rr_ce[i]
            rr_str = f"#{i+1} CE={ce:+5.1f} {ch}:{mid}"
            is_expected_rr = ""
            if expected_doc:
                exp_ch, exp_mid = expected_doc.split(":")
                if ch.lower() == exp_ch.lower() and abs(mid - int(exp_mid)) <= 5:
                    is_expected_rr = " ★"
            rr_str += is_expected_rr
        else:
            rr_str = ""

        # MMR side
        if i < len(mmr_merged):
            c = mmr_merged[i]
            ch = c.metadata.get('channel', '?')
            mid = c.metadata.get('message_id', 0)
            ce = mmr_ce[i]
            mmr_str = f"#{i+1} CE={ce:+5.1f} {ch}:{mid}"
            is_expected_mmr = ""
            if expected_doc:
                exp_ch, exp_mid = expected_doc.split(":")
                if ch.lower() == exp_ch.lower() and abs(mid - int(exp_mid)) <= 5:
                    is_expected_mmr = " ★"
            mmr_str += is_expected_mmr
        else:
            mmr_str = ""

        print(f"  {rr_str:38s} │ {mmr_str:38s}")

    # Stats
    rr_mean_ce = sum(rr_ce[:10]) / min(10, len(rr_ce)) if rr_ce else 0
    mmr_mean_ce = sum(mmr_ce[:10]) / min(10, len(mmr_ce)) if mmr_ce else 0

    # Unique channels in top-10
    rr_channels = len(set(c.metadata.get('channel', '') for c in rr_merged[:10]))
    mmr_channels = len(set(c.metadata.get('channel', '') for c in mmr_merged[:10]))

    print(f"{'─'*80}")
    print(f"  Mean CE top-10: {rr_mean_ce:+.1f}             │ Mean CE top-10: {mmr_mean_ce:+.1f}")
    print(f"  Unique channels: {rr_channels}                │ Unique channels: {mmr_channels}")
    print(f"  Total docs: {len(rr_merged)}                    │ Total docs: {len(mmr_merged)}")

    # Check expected doc position
    if expected_doc:
        exp_ch, exp_mid = expected_doc.split(":")
        exp_mid = int(exp_mid)

        def find_rank(docs):
            for i, c in enumerate(docs):
                if (c.metadata.get('channel', '').lower() == exp_ch.lower()
                        and abs(c.metadata.get('message_id', 0) - exp_mid) <= 5):
                    return i + 1
            return None

        rr_rank = find_rank(rr_merged)
        mmr_rank = find_rank(mmr_merged)
        print(f"\n  Expected doc rank: RR={rr_rank if rr_rank else 'not found'} │ MMR={mmr_rank if mmr_rank else 'not found'}")


# ══════════════════════════════════════════════════════════════════════
# 3 запроса для сравнения
# ══════════════════════════════════════════════════════════════════════

queries = [
    ("Что известно про скандал Anthropic с Пентагоном?", "ai_machinelearning_big_data:9601"),
    ("какие нейросетки умеют видосы делать", "neurohive:1929"),
    ("Сколько OpenAI привлекла в последнем раунде инвестиций?", "seeallochnaya:3427"),
]

for query, expected in queries:
    run_comparison(query, expected)

print(f"\n{'='*80}")
print("COMPARISON COMPLETE")
print(f"{'='*80}")
