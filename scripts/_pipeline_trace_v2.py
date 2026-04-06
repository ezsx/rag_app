"""Pipeline trace v2 — с MMR merge, CE re-sort, adaptive filter.

Прогоняет 3 запроса, показывает каждый шаг с числами.
Сравнивает с тем что было (round-robin, merge order, CE threshold=0.0).
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


# ── OLD pipeline functions (для сравнения) ──

def old_round_robin_merge(per_query_results):
    merged, seen = [], set()
    max_len = max((len(r) for r in per_query_results), default=0)
    for rank in range(max_len):
        for sub in per_query_results:
            if rank < len(sub):
                c = sub[rank]
                if c.id not in seen:
                    merged.append(c)
                    seen.add(c.id)
    return merged


def old_ce_filter(docs, ce_scores, threshold=0.0):
    """Старый: merge order сохранён, CE только фильтрует."""
    return [(i, d, ce_scores[i]) for i, d in enumerate(docs) if ce_scores[i] >= threshold]


# ── NEW pipeline functions ──

def new_mmr_merge(per_query_results, lam=0.7):
    all_docs, seen = [], set()
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

    selected, sel_scores, remaining = [], [], set(range(len(all_docs)))
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


def new_ce_resort_adaptive(docs, ce_scores, gap_threshold=2.0, min_docs=5, floor_score=-2.0):
    """Новый: CE re-sort + adaptive filter (gap + top-K + floor)."""
    indexed = sorted(
        [(i, d, ce_scores[i]) for i, d in enumerate(docs)],
        key=lambda x: x[2], reverse=True,
    )
    scores = [ce for _, _, ce in indexed]

    # Gap detection
    gap_cut = len(scores)
    for i in range(len(scores) - 1):
        if scores[i] - scores[i + 1] > gap_threshold:
            gap_cut = i + 1
            break

    # Positive count
    positive_count = sum(1 for s in scores if s > 0)

    # Adaptive cut
    if positive_count >= min_docs:
        cut = max(gap_cut, positive_count)
    elif positive_count > 0:
        cut = max(positive_count, min(min_docs, gap_cut))
    else:
        floor_cut = sum(1 for s in scores if s >= floor_score)
        cut = max(min(min_docs, len(scores)), floor_cut)

    return indexed[:cut]


def get_ce_scores(query, docs):
    texts = [c.text[:512] for c in docs]
    try:
        indices, scores = reranker.rerank_with_raw_scores(query, texts, top_n=len(texts))
        ce = [0.0] * len(docs)
        for idx, score in zip(indices, scores):
            ce[idx] = score
        return ce
    except Exception:
        return [0.0] * len(docs)


def compose_sim(docs_with_info, max_chars=16000):
    """Симуляция compose_context: берёт docs по порядку до бюджета."""
    result = []
    used = 0
    for info in docs_with_info:
        doc = info[1]  # (idx, doc, ce_score)
        text_len = len(doc.text)
        if used + text_len > max_chars:
            break
        result.append(info)
        used += text_len
    return result, used


def run_trace(query, expected=None):
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    if expected:
        print(f"EXPECTED: {expected}")
    print(f"{'='*80}")

    # Step 1: Query Plan
    plan = planner.make_plan(query)
    all_queries = [query] + [q for q in plan.normalized_queries if q.lower() != query.lower()]
    print(f"\nStep 1 — Query Plan: {len(all_queries)} queries, strategy={plan.strategy}")
    for i, q in enumerate(all_queries):
        print(f"  {'ORIG' if i==0 else f'sq-{i}'}: {q}")

    # Step 2: Per-query search
    per_query = []
    for q in all_queries:
        candidates = hybrid.search_with_plan(q, plan)
        per_query.append(candidates)
    total = sum(len(r) for r in per_query)
    print(f"\nStep 2 — Search: {' + '.join(str(len(r)) for r in per_query)} = {total} docs")

    # ── OLD PIPELINE ──
    old_merged = old_round_robin_merge(per_query)[:30]
    old_ce = get_ce_scores(query, old_merged)
    old_filtered = old_ce_filter(old_merged, old_ce, threshold=0.0)
    old_composed, old_chars = compose_sim(old_filtered)

    # ── NEW PIPELINE ──
    new_merged = new_mmr_merge(per_query)[:30]
    new_ce = get_ce_scores(query, new_merged)
    new_filtered = new_ce_resort_adaptive(new_merged, new_ce)
    new_composed, new_chars = compose_sim(new_filtered)

    # ── COMPARISON ──
    print(f"\n{'─'*80}")
    print(f"{'OLD (round-robin, CE≥0, merge order)':^40s}│{'NEW (MMR, CE re-sort, adaptive)':^40s}")
    print(f"{'─'*80}")

    max_show = max(len(old_composed), len(new_composed), 12)
    for row in range(min(max_show, 15)):
        def fmt(composed, row_idx):
            if row_idx >= len(composed):
                return ""
            _idx, doc, ce = composed[row_idx]
            ch = doc.metadata.get("channel", "?")[:22]
            mid = doc.metadata.get("message_id", 0)
            star = ""
            if expected:
                parts = expected.split(":")
                if (doc.metadata.get("channel", "").lower() == parts[0].lower()
                        and abs(doc.metadata.get("message_id", 0) - int(parts[1])) <= 5):
                    star = " ★"
            return f"[{row_idx+1}] CE={ce:+5.1f} {ch}:{mid}{star}"

        old_str = fmt(old_composed, row)
        new_str = fmt(new_composed, row)
        print(f" {old_str:39s}│ {new_str:39s}")

    # Stats
    def stats(composed, chars):
        if not composed:
            return 0, 0.0, 0, 0
        ces = [ce for _, _, ce in composed]
        return len(composed), sum(ces)/len(ces), len(set(d.metadata.get("channel","") for _,d,_ in composed)), sum(1 for c in ces if c < 0)

    on, om, oc, oneg = stats(old_composed, old_chars)
    nn, nm, nc, nneg = stats(new_composed, new_chars)

    print(f"{'─'*80}")
    print(f" Docs to LLM: {on:2d} ({old_chars} chars)         │ Docs to LLM: {nn:2d} ({new_chars} chars)")
    print(f" Mean CE:  {om:+5.1f}                       │ Mean CE:  {nm:+5.1f}")
    print(f" Channels: {oc}                            │ Channels: {nc}")
    print(f" CE < 0:   {oneg}                            │ CE < 0:   {nneg}")

    # Gap info for new
    new_scores = [ce for _, _, ce in new_filtered]
    gaps = [(i, new_scores[i] - new_scores[i+1]) for i in range(len(new_scores)-1)]
    max_gap = max(gaps, key=lambda x: x[1]) if gaps else (0, 0)
    print(f"\n Adaptive filter: {len(new_ce)} docs → {len(new_filtered)} kept")
    print(f" Biggest gap: {max_gap[1]:.1f} between position {max_gap[0]+1} and {max_gap[0]+2}")
    if len(new_filtered) < len(new_ce):
        cut_score = new_scores[len(new_filtered)-1] if new_filtered else 0
        print(f" Cut at score {cut_score:+.1f}")

    # Expected doc
    if expected:
        parts = expected.split(":")
        def find_rank(composed):
            for j, (_, d, _) in enumerate(composed):
                if (d.metadata.get("channel","").lower() == parts[0].lower()
                        and abs(d.metadata.get("message_id",0) - int(parts[1])) <= 5):
                    return j + 1
            return None
        old_rank = find_rank(old_composed)
        new_rank = find_rank(new_composed)
        print(f"\n Expected doc in LLM context: OLD={'no' if not old_rank else f'#{old_rank}'} │ NEW={'no' if not new_rank else f'#{new_rank}'}")


# ══════════════════════════════════════════════════════════════════════

queries = [
    ("Что известно про скандал Anthropic с Пентагоном?", "ai_machinelearning_big_data:9601"),
    ("какие нейросетки умеют видосы делать", "neurohive:1929"),
    ("Сколько OpenAI привлекла в последнем раунде инвестиций?", "seeallochnaya:3427"),
]

for q, exp in queries:
    run_trace(q, exp)

print(f"\n{'='*80}")
print("TRACE COMPLETE")
