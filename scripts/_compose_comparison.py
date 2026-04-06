"""Сравнение compose_context стратегий на 3 запросах.

Текущий prod: берёт docs по merge order, budget 4000 tokens, CE threshold=0.0
Вариант B: CE re-sort (по CE score desc) + threshold=1.0

Для каждого запроса показывает какие docs попадают в контекст LLM.
"""
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


def mmr_merge(per_query_results, lam=0.7):
    """MMR merge по dense_score."""
    all_docs = []
    seen = set()
    for sub in per_query_results:
        for c in sub:
            if c.id not in seen:
                all_docs.append(c)
                seen.add(c.id)
    if len(all_docs) <= 1:
        return all_docs

    scores = [float(getattr(c, "dense_score", 0) or 0) for c in all_docs]
    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s if max_s > min_s else 1.0
    norm = [(s - min_s) / rng for s in scores]

    selected = []
    sel_scores = []
    remaining = set(range(len(all_docs)))

    for _ in range(len(all_docs)):
        best_idx, best_mmr = -1, -float("inf")
        for idx in remaining:
            rel = norm[idx]
            max_sim = max((1.0 - abs(norm[idx] - s) for s in sel_scores), default=0.0)
            mmr = lam * rel - (1 - lam) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx
        if best_idx >= 0:
            selected.append(all_docs[best_idx])
            sel_scores.append(norm[best_idx])
            remaining.discard(best_idx)
    return selected


def get_ce_scores(query, docs):
    """CE scores для списка docs."""
    texts = [c.text[:512] for c in docs]
    try:
        indices, scores = reranker.rerank_with_raw_scores(query, texts, top_n=len(texts))
        ce = [0.0] * len(docs)
        for idx, score in zip(indices, scores):
            ce[idx] = score
        return ce
    except Exception:
        return [0.0] * len(docs)


def compose_current(docs, ce_scores, max_chars=16000, ce_threshold=0.0):
    """Текущий prod: merge order, CE filter threshold=0.0."""
    result = []
    used = 0
    for i, c in enumerate(docs):
        if ce_scores[i] < ce_threshold:
            continue
        text_len = len(c.text)
        if used + text_len > max_chars:
            break
        result.append((i, c, ce_scores[i]))
        used += text_len
    return result, used


def compose_resort(docs, ce_scores, max_chars=16000, ce_threshold=1.0):
    """Вариант B: re-sort по CE score, threshold=1.0."""
    # Сортируем по CE score descending
    indexed = [(idx, docs[idx], ce_scores[idx]) for idx in range(len(docs))]
    indexed.sort(key=lambda x: x[2], reverse=True)

    result = []
    used = 0
    for idx, doc, ce in indexed:
        if ce < ce_threshold:
            continue
        text_len = len(doc.text)
        if used + text_len > max_chars:
            break
        result.append((idx, doc, ce))
        used += text_len
    return result, used


def run_query(query, expected=None):
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    # Pipeline: plan → search → MMR merge
    plan = planner.make_plan(query)
    all_queries = [query] + [q for q in plan.normalized_queries if q.lower() != query.lower()]

    per_query = []
    for q in all_queries:
        candidates = hybrid.search_with_plan(q, plan)
        per_query.append(candidates)

    merged = mmr_merge(per_query)[:30]
    ce_scores = get_ce_scores(query, merged)

    print(f"Merged: {len(merged)} docs, CE range: [{min(ce_scores):.1f} .. {max(ce_scores):.1f}]")
    print(f"CE > 0: {sum(1 for s in ce_scores if s > 0)}, CE > 1: {sum(1 for s in ce_scores if s > 1)}, CE > 2: {sum(1 for s in ce_scores if s > 2)}")

    # Вариант A: текущий prod (merge order, CE threshold=0.0)
    a_docs, a_chars = compose_current(merged, ce_scores, ce_threshold=0.0)
    # Вариант B: CE re-sort + threshold=1.0
    b_docs, b_chars = compose_resort(merged, ce_scores, ce_threshold=1.0)

    print(f"\n{'─'*80}")
    print(f"{'A: Current (merge order, CE≥0.0)':^40s}│{'B: CE re-sort, CE≥1.0':^40s}")
    print(f"{'─'*80}")

    max_show = max(len(a_docs), len(b_docs))
    for row in range(min(max_show, 12)):
        # A side
        if row < len(a_docs):
            _i, c, ce = a_docs[row]
            ch = c.metadata.get("channel", "?")[:20]
            mid = c.metadata.get("message_id", 0)
            star = " ★" if expected and _is_match(c, expected) else ""
            a_str = f"[{row+1}] CE={ce:+5.1f} {ch}:{mid}{star}"
        else:
            a_str = ""

        # B side
        if row < len(b_docs):
            _i, c, ce = b_docs[row]
            ch = c.metadata.get("channel", "?")[:20]
            mid = c.metadata.get("message_id", 0)
            star = " ★" if expected and _is_match(c, expected) else ""
            b_str = f"[{row+1}] CE={ce:+5.1f} {ch}:{mid}{star}"
        else:
            b_str = ""

        print(f" {a_str:39s}│ {b_str:39s}")

    a_mean_ce = sum(ce for _, _, ce in a_docs) / len(a_docs) if a_docs else 0
    b_mean_ce = sum(ce for _, _, ce in b_docs) / len(b_docs) if b_docs else 0

    a_channels = len(set(c.metadata.get("channel", "") for _, c, _ in a_docs))
    b_channels = len(set(c.metadata.get("channel", "") for _, c, _ in b_docs))

    print(f"{'─'*80}")
    print(f" Docs in context: {len(a_docs):2d} ({a_chars} chars)     │ Docs in context: {len(b_docs):2d} ({b_chars} chars)")
    print(f" Mean CE: {a_mean_ce:+.1f}                       │ Mean CE: {b_mean_ce:+.1f}")
    print(f" Unique channels: {a_channels}                   │ Unique channels: {b_channels}")
    print(f" CE < 0 included: {sum(1 for _,_,ce in a_docs if ce < 0)}                    │ CE < 0 included: {sum(1 for _,_,ce in b_docs if ce < 0)}")

    if expected:
        a_rank = next((j+1 for j, (_, c, _) in enumerate(a_docs) if _is_match(c, expected)), None)
        b_rank = next((j+1 for j, (_, c, _) in enumerate(b_docs) if _is_match(c, expected)), None)
        print(f"\n Expected doc: A={'not in context' if not a_rank else f'position {a_rank}'} │ B={'not in context' if not b_rank else f'position {b_rank}'}")


def _is_match(c, expected):
    parts = expected.split(":")
    if len(parts) != 2:
        return False
    return (c.metadata.get("channel", "").lower() == parts[0].lower()
            and abs(c.metadata.get("message_id", 0) - int(parts[1])) <= 5)


queries = [
    ("Что известно про скандал Anthropic с Пентагоном?", "ai_machinelearning_big_data:9601"),
    ("какие нейросетки умеют видосы делать", "neurohive:1929"),
    ("Сколько OpenAI привлекла в последнем раунде инвестиций?", "seeallochnaya:3427"),
]

for q, exp in queries:
    run_query(q, exp)

print(f"\n{'='*80}")
print("DONE")
