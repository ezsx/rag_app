"""
Full-pipeline retrieval evaluation — prod-parity через direct import production модулей.

Тестирует реальный production code path:
  query_plan → multi-query search → round-robin merge → CE filter → channel dedup

Запуск:
    python scripts/evaluate_retrieval_full.py \
        --dataset datasets/eval_retrieval_v3.json \
        --use-query-plan --inject-original-query --use-metadata-filters \
        --ce-filter --ce-threshold 0.0 --channel-dedup 2 \
        --output results/ablation/full_pipeline.json --save-traces
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

# Bootstrap: добавляем src/ в path для import production модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

# Загружаем .env до импорта settings
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Override для standalone запуска (без Docker networking)
os.environ.setdefault("QDRANT_URL", "http://localhost:16333")
os.environ.setdefault("EMBEDDING_TEI_URL", "http://localhost:8082")
os.environ.setdefault("RERANKER_TEI_URL", "http://localhost:8082")
os.environ.setdefault("QDRANT_COLLECTION", "news_colbert_v2")
os.environ.setdefault("HYBRID_ENABLED", "true")
os.environ.setdefault("ENABLE_RERANKER", "true")
os.environ.setdefault("ENABLE_QUERY_PLANNER", "true")
# LLM для query_plan (override Docker hostname)
os.environ.setdefault("LLM_BASE_URL", "http://localhost:8080")


def bootstrap():
    """Инициализация production компонентов без FastAPI DI."""
    from core.deps import get_hybrid_retriever, get_query_planner, get_reranker
    from core.settings import get_settings

    settings = get_settings()
    hybrid = get_hybrid_retriever()
    planner = get_query_planner()
    reranker = get_reranker()

    if not hybrid:
        print("ERROR: HybridRetriever не инициализирован. Проверьте HYBRID_ENABLED и сервисы.")
        sys.exit(1)

    print(f"Bootstrap OK:")
    print(f"  Qdrant: {settings.qdrant_url} / {settings.qdrant_collection}")
    print(f"  Embedding: {settings.embedding_tei_url}")
    print(f"  Reranker: {'ON' if reranker else 'OFF'}")
    print(f"  QueryPlanner: {'ON' if planner else 'OFF'}")

    return settings, hybrid, planner, reranker


def round_robin_merge(per_query_results: list[list]) -> list:
    """Round-robin merge из search.py:20-36. Один code path."""
    merged = []
    seen_ids: set = set()
    max_len = max((len(r) for r in per_query_results), default=0)
    for rank_idx in range(max_len):
        for sub_result in per_query_results:
            if rank_idx < len(sub_result):
                c = sub_result[rank_idx]
                if c.id not in seen_ids:
                    merged.append(c)
                    seen_ids.add(c.id)
    return merged


def channel_dedup(candidates: list, max_per_channel: int) -> list:
    """Channel dedup из hybrid_retriever.py:317-330."""
    if max_per_channel <= 0:
        return candidates
    channel_counts: dict[str, int] = {}
    result = []
    for c in candidates:
        ch = c.metadata.get("channel", "")
        count = channel_counts.get(ch, 0)
        if count < max_per_channel:
            result.append(c)
            channel_counts[ch] = count + 1
    return result


def check_recall(candidates: list, expected_docs: list[str], k: int, fuzzy: int = 5) -> float:
    """Recall@k с fuzzy matching по message_id."""
    if not expected_docs:
        return 0.0
    top_k = candidates[:k]
    matched = 0
    for exp in expected_docs:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for c in top_k:
            meta = c.metadata if hasattr(c, "metadata") else {}
            if (meta.get("channel", "").lower() == exp_ch
                    and abs(meta.get("message_id", 0) - exp_msg) <= fuzzy):
                matched += 1
                break
    return matched / len(expected_docs)


def find_reciprocal_rank(candidates: list, expected_docs: list[str], k: int = 20, fuzzy: int = 5) -> float:
    """1/rank первого найденного expected doc."""
    top_k = candidates[:k]
    for exp in expected_docs:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for rank, c in enumerate(top_k, 1):
            meta = c.metadata if hasattr(c, "metadata") else {}
            if (meta.get("channel", "").lower() == exp_ch
                    and abs(meta.get("message_id", 0) - exp_msg) <= fuzzy):
                return 1.0 / rank
    return 0.0


def llm_call(prompt: str, system: str = "", max_tokens: int = 256) -> str:
    """Один LLM вызов через llama-server /v1/chat/completions."""
    import urllib.request as req_lib
    llm_url = os.environ.get("LLM_BASE_URL", "http://localhost:8080")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    request = req_lib.Request(
        f"{llm_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(req_lib.urlopen(request, timeout=120).read())
    return resp["choices"][0]["message"]["content"].strip()


def llm_rewrite_query(query: str) -> str:
    """R3: LLM single rewrite — перефразировать запрос для поиска."""
    return llm_call(
        prompt=f"Перефразируй запрос для поиска по базе новостей. Ответ строго на языке запроса. Только перефразированный запрос, без пояснений.\n\nЗапрос: {query}",
        system="You are a search query optimizer. Rewrite the query to improve retrieval. Keep the same language as the input. Output only the rewritten query.",
        max_tokens=128,
    )


def llm_hyde(query: str) -> str:
    """D2: HyDE — сгенерировать гипотетический документ-ответ."""
    return llm_call(
        prompt=f"Напиши короткий пост (3-5 предложений) для Telegram-канала о технологиях, который мог бы быть ответом на вопрос: {query}\n\nПиши как реальный пост — с фактами, названиями, датами. Строго на языке запроса.",
        system="You are a tech news writer for a Russian Telegram channel. Generate a realistic short post that answers the user's question. Use the same language as the query. 3-5 sentences max.",
        max_tokens=200,
    )


def run_single_query(
    query: str,
    hybrid,
    planner,
    reranker,
    settings,
    args,
) -> tuple[list, dict]:
    """Прогнать один query через configurable pipeline. Вернуть (candidates, trace)."""
    from schemas.search import SearchPlan, MetadataFilters

    trace: dict[str, Any] = {"original_query": query}

    # Step 0: Single rewrite (R3)
    rewritten_query = None
    if args.single_rewrite:
        try:
            rewritten_query = llm_rewrite_query(query)
            trace["rewritten_query"] = rewritten_query
        except Exception as e:
            trace["rewrite_error"] = str(e)

    # Step 0b: HyDE (D2)
    hyde_text = None
    if args.hyde:
        try:
            hyde_text = llm_hyde(query)
            trace["hyde_text"] = hyde_text
        except Exception as e:
            trace["hyde_error"] = str(e)

    # Step 1: Query plan (LLM)
    subqueries = [query]
    metadata_filters = None
    strategy = "broad"

    # R6: Rule-based filters without LLM planner
    if args.rule_based_filters:
        try:
            from services.query_signals import extract_query_signals
            signals = extract_query_signals(query)
            if signals.channels:
                metadata_filters = MetadataFilters(channel_usernames=signals.channels)
                trace["rule_based_channels"] = signals.channels
            if signals.date_from:
                if metadata_filters:
                    metadata_filters.date_from = signals.date_from
                else:
                    metadata_filters = MetadataFilters(date_from=signals.date_from)
                trace["rule_based_date_from"] = signals.date_from
        except Exception as e:
            trace["rule_filter_error"] = str(e)

    if args.use_query_plan and planner:
        try:
            plan = planner.make_plan(query)
            subqueries = plan.normalized_queries or [query]
            if args.use_metadata_filters and plan.metadata_filters:
                metadata_filters = plan.metadata_filters
            strategy = plan.strategy or "broad"
            trace["subqueries"] = subqueries
            trace["strategy"] = strategy
            if metadata_filters:
                trace["applied_filters"] = metadata_filters.dict(exclude_none=True)
        except Exception as e:
            trace["planner_error"] = str(e)
            subqueries = [query]

    # Step 2: Original query injection
    if args.inject_original_query and query not in subqueries:
        subqueries = [query] + subqueries
        trace["injected_original"] = True

    # Step 3: Per-subquery search + round-robin merge
    k_per_query = settings.search_k_per_query_default or 10
    search_plan = SearchPlan(
        normalized_queries=subqueries,
        metadata_filters=metadata_filters if args.use_metadata_filters else None,
        k_per_query=k_per_query,
        fusion="rrf",
        strategy=strategy,
    )

    per_query_results = []
    for q in subqueries:
        try:
            sub_candidates = hybrid.search_with_plan(q, search_plan)
            per_query_results.append(sub_candidates)
        except Exception as e:
            trace.setdefault("search_errors", []).append(str(e))

    # R3: дополнительный поиск по rewritten query
    if rewritten_query and rewritten_query != query:
        try:
            rewrite_plan = SearchPlan(
                normalized_queries=[rewritten_query],
                k_per_query=k_per_query,
                fusion="rrf",
                strategy="broad",
            )
            rewrite_candidates = hybrid.search_with_plan(rewritten_query, rewrite_plan)
            per_query_results.append(rewrite_candidates)
            trace["rewrite_search_count"] = len(rewrite_candidates)
        except Exception as e:
            trace.setdefault("search_errors", []).append(f"rewrite: {e}")

    # D2: дополнительный поиск по HyDE pseudo-document
    if hyde_text:
        try:
            hyde_plan = SearchPlan(
                normalized_queries=[hyde_text],
                k_per_query=k_per_query,
                fusion="rrf",
                strategy="broad",
            )
            hyde_candidates = hybrid.search_with_plan(hyde_text, hyde_plan)
            per_query_results.append(hyde_candidates)
            trace["hyde_search_count"] = len(hyde_candidates)
        except Exception as e:
            trace.setdefault("search_errors", []).append(f"hyde: {e}")

    if len(per_query_results) > 1:
        candidates = round_robin_merge(per_query_results)
    elif per_query_results:
        candidates = per_query_results[0]
    else:
        candidates = []

    trace["merged_count"] = len(candidates)

    # Step 4: CE filter
    if args.ce_filter and reranker and candidates:
        docs = [c.text for c in candidates]
        try:
            indices, scores = reranker.rerank_with_raw_scores(
                query=query, docs=docs,
                top_n=min(len(docs), settings.reranker_top_n or 80),
            )
            # Если reranker вернул пустые scores (fallback) — не фильтруем
            if scores:
                passed = []
                for idx, score in zip(indices, scores):
                    if score >= args.ce_threshold:
                        passed.append(idx)
                filtered_candidates = [c for i, c in enumerate(candidates) if i in set(passed)]
                trace["ce_filtered_out"] = len(candidates) - len(filtered_candidates)
                trace["ce_scores_sample"] = scores[:5]
                candidates = filtered_candidates
            else:
                trace["ce_error"] = "reranker returned empty scores (fallback)"
                trace["ce_filtered_out"] = 0
        except Exception as e:
            trace["ce_error"] = str(e)

    # Step 5: Channel dedup
    if args.channel_dedup > 0 and candidates:
        before = len(candidates)
        candidates = channel_dedup(candidates, args.channel_dedup)
        trace["dedup_removed"] = before - len(candidates)

    trace["final_count"] = len(candidates)
    return candidates, trace


def main():
    parser = argparse.ArgumentParser(description="Full-pipeline retrieval evaluation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fuzzy", type=int, default=5)
    # Pipeline components
    parser.add_argument("--use-query-plan", action="store_true")
    parser.add_argument("--inject-original-query", action="store_true")
    parser.add_argument("--use-metadata-filters", action="store_true")
    parser.add_argument("--ce-filter", action="store_true")
    parser.add_argument("--ce-threshold", type=float, default=0.0)
    parser.add_argument("--channel-dedup", type=int, default=0, help="0=off, 2=default, 3=wide")
    # Phase 2 new tracks
    parser.add_argument("--single-rewrite", action="store_true", help="R3: LLM single rewrite + raw fusion")
    parser.add_argument("--hyde", action="store_true", help="D2: HyDE auxiliary dense branch + raw")
    parser.add_argument("--rule-based-filters", action="store_true", help="R6: query_signals filters without LLM")
    # Output
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-traces", action="store_true")
    # Subset
    parser.add_argument("--categories", default=None, help='Filter categories, e.g. "edge,temporal,channel_specific"')
    args = parser.parse_args()

    settings, hybrid, planner, reranker = bootstrap()

    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)

    # Фильтрация по категориям
    if args.categories:
        cats = set(args.categories.split(","))
        items = [it for it in items if it.get("category", "") in cats]
        print(f"Filtered to categories {cats}: {len(items)} queries")

    config_str = []
    if args.use_query_plan:
        config_str.append("query_plan")
    if args.inject_original_query:
        config_str.append("inject_original")
    if args.use_metadata_filters:
        config_str.append("metadata_filters")
    if args.ce_filter:
        config_str.append(f"ce_filter(t={args.ce_threshold})")
    if args.channel_dedup > 0:
        config_str.append(f"dedup={args.channel_dedup}")
    print(f"\nDataset: {len(items)} queries")
    print(f"Pipeline: {' → '.join(config_str) or 'raw retriever only'}")
    print()

    results = []
    traces = []
    t_total = time.time()

    for i, item in enumerate(items):
        query = item["query"]
        expected = item.get("expected_documents", [])

        t0 = time.time()
        try:
            candidates, trace = run_single_query(
                query, hybrid, planner, reranker, settings, args,
            )
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            results.append({"id": item["id"], "recall_1": 0, "recall_5": 0, "recall_20": 0, "reciprocal_rank": 0, "error": str(e)})
            continue
        latency = time.time() - t0

        r1 = check_recall(candidates, expected, 1, args.fuzzy)
        r3 = check_recall(candidates, expected, 3, args.fuzzy)
        r5 = check_recall(candidates, expected, 5, args.fuzzy)
        r10 = check_recall(candidates, expected, 10, args.fuzzy)
        r20 = check_recall(candidates, expected, 20, args.fuzzy)
        rr = find_reciprocal_rank(candidates, expected, args.top_k, args.fuzzy)

        results.append({
            "id": item["id"],
            "query": query[:80],
            "category": item.get("category", ""),
            "expected": expected,
            "recall_1": r1, "recall_3": r3, "recall_5": r5, "recall_10": r10, "recall_20": r20,
            "reciprocal_rank": rr,
            "latency": latency,
            "n_candidates": len(candidates),
        })

        if args.save_traces:
            trace["id"] = item["id"]
            trace["latency"] = latency
            traces.append(trace)

        status = "✅" if r5 > 0 else "❌"
        if (i + 1) % 10 == 0 or r5 == 0:
            print(f"  [{i+1}/{len(items)}] {status} r@5={r5:.2f} {latency:.2f}s | {query[:60]}")

    elapsed = time.time() - t_total

    # Агрегация
    valid = [r for r in results if "error" not in r]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    r1_list = [r["recall_1"] for r in valid]
    r3_list = [r["recall_3"] for r in valid]
    r5_list = [r["recall_5"] for r in valid]
    r10_list = [r["recall_10"] for r in valid]
    r20_list = [r["recall_20"] for r in valid]
    mrr = avg([r["reciprocal_rank"] for r in valid])

    print(f"\n{'='*60}")
    print(f"Full Pipeline Evaluation Results")
    print(f"{'='*60}")
    print(f"Queries: {len(items)}, Errors: {sum(1 for r in results if 'error' in r)}")
    print(f"Pipeline: {' → '.join(config_str) or 'raw retriever only'}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(items):.2f}s/query)")
    print()
    print(f"Recall@1:  {avg(r1_list):.3f}  ({sum(1 for r in r1_list if r==1.0)}/{len(r1_list)})")
    print(f"Recall@3:  {avg(r3_list):.3f}  ({sum(1 for r in r3_list if r==1.0)}/{len(r3_list)})")
    print(f"Recall@5:  {avg(r5_list):.3f}  ({sum(1 for r in r5_list if r==1.0)}/{len(r5_list)})")
    print(f"Recall@10: {avg(r10_list):.3f}  ({sum(1 for r in r10_list if r==1.0)}/{len(r10_list)})")
    print(f"Recall@20: {avg(r20_list):.3f}  ({sum(1 for r in r20_list if r==1.0)}/{len(r20_list)})")
    print(f"MRR@20:    {mrr:.3f}")

    # Per-category
    by_category = defaultdict(list)
    for r in valid:
        by_category[r.get("category", "unknown")].append(r)

    if len(by_category) > 1:
        print("\nПо категориям:")
        for cat, cat_results in sorted(by_category.items()):
            cat_r5 = avg([r["recall_5"] for r in cat_results])
            cat_mrr = avg([r["reciprocal_rank"] for r in cat_results])
            print(f"  {cat:20s} n={len(cat_results):3d}  R@5={cat_r5:.2f}  MRR={cat_mrr:.3f}")

    # Сохраняем
    ts = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "timestamp": ts,
        "pipeline": config_str,
        "total_queries": len(items),
        "recall_at_1": avg(r1_list),
        "recall_at_3": avg(r3_list),
        "recall_at_5": avg(r5_list),
        "recall_at_10": avg(r10_list),
        "recall_at_20": avg(r20_list),
        "mrr_at_20": mrr,
        "avg_latency": elapsed / len(items),
        "ce_threshold": args.ce_threshold if args.ce_filter else None,
        "channel_dedup": args.channel_dedup,
        "results": results,
    }

    out_path = args.output or f"results/ablation/full_pipeline_{ts}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

    # Traces
    if args.save_traces and traces:
        trace_path = out_path.replace(".json", "_traces.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        print(f"Traces saved to {trace_path}")


if __name__ == "__main__":
    main()
