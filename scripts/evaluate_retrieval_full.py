"""
Full-pipeline retrieval evaluation — prod-parity через direct import production модулей.

Тестирует реальный production code path:
  query_plan → multi-query search → MMR merge → CE re-sort + adaptive filter → channel dedup

Артефакт: один .jsonl файл с live flush (tail -f для мониторинга).
  Строка 1: header (config, preflight results)
  Строки 2..N: per-query результаты
  Последняя строка: summary (агрегаты)

Запуск:
    python scripts/evaluate_retrieval_full.py \
        --dataset datasets/eval_retrieval_v3.json \
        --use-query-plan --inject-original-query --use-metadata-filters \
        --ce-filter --ce-threshold 0.0 --channel-dedup 2 \
        --output experiments/runs/fp_run_001.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from typing import Any

# Bootstrap: добавляем src/ в path для import production модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

# Загружаем .env до импорта settings
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Force localhost для standalone запуска (не Docker networking).
# setdefault не работает — .env загружает host.docker.internal раньше.
os.environ["QDRANT_URL"] = "http://localhost:16333"
os.environ["EMBEDDING_TEI_URL"] = "http://localhost:8082"
os.environ["RERANKER_TEI_URL"] = "http://localhost:8082"
os.environ["QDRANT_COLLECTION"] = "news_colbert_v2"
os.environ.setdefault("HYBRID_ENABLED", "true")
os.environ.setdefault("ENABLE_RERANKER", "true")
os.environ.setdefault("ENABLE_QUERY_PLANNER", "true")
# LLM для query_plan (override Docker hostname)
os.environ["LLM_BASE_URL"] = "http://localhost:8080"


# ── Live JSONL writer ──────────────────────────────────────────────

class LiveWriter:
    """Пишет jsonl с flush после каждой строки. tail -f для мониторинга."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")
        self._path = path
        self._count = 0

    def write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()
        self._count += 1

    def close(self) -> None:
        self._f.close()

    @property
    def path(self) -> str:
        return self._path


# ── Preflight ──────────────────────────────────────────────────────

def preflight_check() -> dict[str, dict]:
    """Проверить доступность всех сервисов. Возвращает {service: {ok, url, error, ms}}."""
    checks = {
        "qdrant": os.environ["QDRANT_URL"],
        "embedding": os.environ["EMBEDDING_TEI_URL"],
        "reranker": os.environ["RERANKER_TEI_URL"],
        "llm": os.environ.get("LLM_BASE_URL", "http://localhost:8080"),
    }
    # Qdrant → /collections, остальные → /health
    endpoints = {
        "qdrant": "/collections",
        "embedding": "/health",
        "reranker": "/health",
        "llm": "/health",
    }
    results = {}
    for svc, base_url in checks.items():
        url = base_url.rstrip("/") + endpoints[svc]
        t0 = time.time()
        try:
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            ms = (time.time() - t0) * 1000
            results[svc] = {"ok": True, "url": base_url, "status": resp.status, "ms": round(ms)}
        except Exception as e:
            ms = (time.time() - t0) * 1000
            results[svc] = {"ok": False, "url": base_url, "error": str(e)[:120], "ms": round(ms)}
    return results


# ── Bootstrap ──────────────────────────────────────────────────────

def bootstrap():
    """Инициализация production компонентов без FastAPI DI."""
    from core.deps import get_hybrid_retriever, get_query_planner, get_reranker
    from core.settings import get_settings

    settings = get_settings()
    hybrid = get_hybrid_retriever()
    planner = get_query_planner()
    reranker = get_reranker()

    if not hybrid:
        print("ERROR: HybridRetriever не инициализирован.", file=sys.stderr)
        sys.exit(1)

    return settings, hybrid, planner, reranker


# ── Pipeline helpers ───────────────────────────────────────────────

def mmr_merge(per_query_results: list[list]) -> list:
    """MMR merge из production search.py — один code path."""
    from services.tools.search import _mmr_merge
    return _mmr_merge(per_query_results)


def channel_dedup(candidates: list, max_per_channel: int) -> list:
    """Channel dedup: max N docs per channel."""
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


def get_git_sha() -> str:
    """Текущий git commit SHA (short)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def find_expected_rank(candidates: list, expected_docs: list[str], fuzzy: int = 5) -> int | None:
    """Позиция (1-based) первого expected doc в candidates. None если не найден."""
    for exp in expected_docs:
        parts = exp.split(":", 1)
        if len(parts) != 2:
            continue
        exp_ch, exp_msg = parts[0].lower(), int(parts[1])
        for rank, c in enumerate(candidates, 1):
            meta = c.metadata if hasattr(c, "metadata") else {}
            if (meta.get("channel", "").lower() == exp_ch
                    and abs(meta.get("message_id", 0) - exp_msg) <= fuzzy):
                return rank
    return None


def llm_call(prompt: str, system: str = "", max_tokens: int = 256) -> str:
    """Один LLM вызов через llama-server /v1/chat/completions."""
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
    request = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(request, timeout=120).read())
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


# ── Single query pipeline ─────────────────────────────────────────

def run_single_query(
    query: str,
    expected_docs: list[str],
    hybrid,
    planner,
    reranker,
    settings,
    args,
) -> tuple[list, dict]:
    """Прогнать один query через configurable pipeline. Вернуть (candidates, trace)."""
    from schemas.search import MetadataFilters, SearchPlan

    trace: dict[str, Any] = {}
    fuzzy = args.fuzzy

    # Step 0: Single rewrite (R3)
    rewritten_query = None
    if args.single_rewrite:
        try:
            rewritten_query = llm_rewrite_query(query)
            trace["rewritten_query"] = rewritten_query
        except Exception as e:
            trace["rewrite_error"] = str(e)[:100]

    # Step 0b: HyDE (D2)
    hyde_text = None
    if args.hyde:
        try:
            hyde_text = llm_hyde(query)
            trace["hyde_text"] = hyde_text[:200]
        except Exception as e:
            trace["hyde_error"] = str(e)[:100]

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
            trace["rule_filter_error"] = str(e)[:100]

    t_plan = time.time()
    if args.use_query_plan and planner:
        try:
            plan = planner.make_plan(query)
            subqueries = plan.normalized_queries or [query]
            if args.use_metadata_filters and plan.metadata_filters:
                metadata_filters = plan.metadata_filters
            strategy = plan.strategy or "broad"
            trace["n_subqueries"] = len(subqueries)
            trace["strategy"] = strategy
        except Exception as e:
            trace["planner_error"] = str(e)[:100]
            subqueries = [query]
    trace["planner_ms"] = round((time.time() - t_plan) * 1000)

    # Step 2: Original query injection
    if args.inject_original_query and query not in subqueries:
        subqueries = [query, *subqueries]

    # Step 3: Per-subquery search + merge
    k_per_query = settings.search_k_per_query_default or 10
    search_plan = SearchPlan(
        normalized_queries=subqueries,
        metadata_filters=metadata_filters if args.use_metadata_filters else None,
        k_per_query=k_per_query,
        fusion="rrf",
        strategy=strategy,
    )

    t_search = time.time()
    per_query_results = []
    for q in subqueries:
        try:
            sub_candidates = hybrid.search_with_plan(q, search_plan)
            per_query_results.append(sub_candidates)
        except Exception as e:
            trace.setdefault("search_errors", []).append(str(e)[:100])

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
        except Exception as e:
            trace.setdefault("search_errors", []).append(f"rewrite: {str(e)[:80]}")

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
        except Exception as e:
            trace.setdefault("search_errors", []).append(f"hyde: {str(e)[:80]}")

    if len(per_query_results) > 1:
        candidates = mmr_merge(per_query_results)
    elif per_query_results:
        candidates = per_query_results[0]
    else:
        candidates = []
    trace["search_ms"] = round((time.time() - t_search) * 1000)

    trace["merged"] = len(candidates)

    # Stage attribution: отслеживаем expected doc через каждый этап
    rank_after_merge = find_expected_rank(candidates, expected_docs, fuzzy)

    # Step 4: CE re-sort + adaptive filter
    t_ce = time.time()
    ce_ok = False
    ce_top = None
    ce_cut = None
    if args.ce_filter and reranker and candidates:
        docs_text = [c.text for c in candidates]
        try:
            indices, scores = reranker.rerank_with_raw_scores(
                query=query, docs=docs_text,
                top_n=min(len(docs_text), settings.reranker_top_n or 80),
            )
            if scores:
                ce_ok = True
                score_map = {}
                for idx, score in zip(indices, scores):
                    score_map[idx] = score

                scored = [(i, c, score_map.get(i, 0.0)) for i, c in enumerate(candidates)]
                scored.sort(key=lambda x: x[2], reverse=True)

                ce_vals = [s for _, _, s in scored]
                ce_top = round(ce_vals[0], 2) if ce_vals else None

                # Adaptive filter (same logic as state.py _adaptive_ce_filter)
                gap_cut = len(ce_vals)
                for gi in range(len(ce_vals) - 1):
                    if ce_vals[gi] - ce_vals[gi + 1] > 2.0:
                        gap_cut = gi + 1
                        break
                pos_count = sum(1 for s in ce_vals if s > 0)
                if pos_count >= 5:
                    cut = max(gap_cut, pos_count)
                elif pos_count > 0:
                    cut = max(pos_count, min(5, gap_cut))
                else:
                    floor_cut = sum(1 for s in ce_vals if s >= -2.0)
                    cut = max(min(5, len(ce_vals)), floor_cut)
                scored = scored[:cut]
                ce_cut = len(scored)
                trace["mean_ce"] = round(sum(s for _, _, s in scored) / len(scored), 2) if scored else 0
                trace["ce_neg"] = sum(1 for _, _, s in scored if s < 0)
                candidates = [c for _, c, _ in scored]
            else:
                trace["ce_error"] = "empty_scores"
        except Exception as e:
            trace["ce_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    trace["ce_ms"] = round((time.time() - t_ce) * 1000)

    trace["ce_ok"] = ce_ok
    if ce_top is not None:
        trace["ce_top"] = ce_top
    if ce_cut is not None:
        trace["ce_cut"] = ce_cut

    rank_after_ce = find_expected_rank(candidates, expected_docs, fuzzy)

    # Step 5: Channel dedup
    if args.channel_dedup > 0 and candidates:
        before = len(candidates)
        candidates = channel_dedup(candidates, args.channel_dedup)
        trace["dedup_rm"] = before - len(candidates)

    rank_final = find_expected_rank(candidates, expected_docs, fuzzy)

    trace["n_cand"] = len(candidates)
    trace["n_channels"] = len(set(c.metadata.get("channel", "") for c in candidates[:5]))
    trace["expected_rank"] = rank_final

    # Stage attribution: где потерялся expected doc
    if rank_final:
        trace["lost_in"] = None  # found
    elif rank_after_ce:
        trace["lost_in"] = "dedup"
    elif rank_after_merge:
        trace["lost_in"] = "ce_filter"
    else:
        trace["lost_in"] = "not_in_merge"

    return candidates, trace


# ── Main ───────────────────────────────────────────────────────────

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
    # Subset
    parser.add_argument("--categories", default=None, help='Filter categories, e.g. "edge,temporal,channel_specific"')
    args = parser.parse_args()

    # ── Preflight ──
    print("Preflight check...", file=sys.stderr)
    health = preflight_check()
    all_ok = True
    for svc, info in health.items():
        status = "OK" if info["ok"] else "FAIL"
        detail = f"{info['ms']}ms" if info["ok"] else info.get("error", "?")[:60]
        print(f"  {svc:12s} {info['url']:40s} {status} ({detail})", file=sys.stderr)
        if not info["ok"]:
            all_ok = False

    if not all_ok:
        print("\nPreflight FAILED — сервисы недоступны. Прерываю.", file=sys.stderr)
        sys.exit(1)
    print("Preflight OK\n", file=sys.stderr)

    # ── Bootstrap ──
    settings, hybrid, planner, reranker = bootstrap()

    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)

    if args.categories:
        cats = set(args.categories.split(","))
        items = [it for it in items if it.get("category", "") in cats]

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

    # ── Output path ──
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.output or f"experiments/runs/fp_run_{ts}.jsonl"
    if not out_path.endswith(".jsonl"):
        out_path = out_path.rsplit(".", 1)[0] + ".jsonl"

    writer = LiveWriter(out_path)

    # ── Header line ──
    writer.write({
        "type": "header",
        "timestamp": ts,
        "git_sha": get_git_sha(),
        "pipeline": config_str,
        "total_queries": len(items),
        "dataset": args.dataset,
        "preflight": {svc: info["ok"] for svc, info in health.items()},
        "config": {
            "ce_threshold": args.ce_threshold if args.ce_filter else None,
            "channel_dedup": args.channel_dedup,
            "top_k": args.top_k,
            "fuzzy": args.fuzzy,
        },
    })

    print(f"Dataset: {len(items)} queries", file=sys.stderr)
    print(f"Pipeline: {' → '.join(config_str) or 'raw retriever only'}", file=sys.stderr)
    print(f"Output: {out_path}", file=sys.stderr)
    print(file=sys.stderr)

    # ── Eval loop ──
    t_total = time.time()
    running_r1 = []
    running_r5 = []
    running_mrr = []
    ce_ok_count = 0
    error_count = 0

    for i, item in enumerate(items):
        query = item["query"]
        expected = item.get("expected_documents", [])

        t0 = time.time()
        try:
            candidates, trace = run_single_query(
                query, expected, hybrid, planner, reranker, settings, args,
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
        latency = time.time() - t0

        r1 = check_recall(candidates, expected, 1, args.fuzzy)
        r3 = check_recall(candidates, expected, 3, args.fuzzy)
        r5 = check_recall(candidates, expected, 5, args.fuzzy)
        r10 = check_recall(candidates, expected, 10, args.fuzzy)
        r20 = check_recall(candidates, expected, 20, args.fuzzy)
        rr = find_reciprocal_rank(candidates, expected, args.top_k, args.fuzzy)

        running_r1.append(r1)
        running_r5.append(r5)
        running_mrr.append(rr)
        if trace.get("ce_ok"):
            ce_ok_count += 1

        # Per-query jsonl row
        row = {
            "type": "query",
            "qid": item["id"],
            "query": query[:80],
            "cat": item.get("category", ""),
            "r1": r1, "r5": r5, "r10": r10, "r20": r20,
            "rr": round(rr, 3),
            "latency": round(latency, 2),
        }
        row.update(trace)  # ce_ok, ce_top, ce_cut, ce_error, merged, n_cand, dedup_rm, ...
        row["top5_hits"] = [f"{c.metadata.get('channel','')}:{c.metadata.get('message_id','')}" for c in candidates[:5]]
        row["all_docs"] = [
            {"ch": c.metadata.get("channel", ""), "mid": c.metadata.get("message_id", ""), "text": c.text}
            for c in candidates
        ]
        writer.write(row)

        # Live stderr: каждые 10 queries или при R@5=0
        done = i + 1
        if done % 10 == 0 or r5 == 0:
            avg_r5 = sum(running_r5) / len(running_r5)
            avg_mrr = sum(running_mrr) / len(running_mrr)
            ce_tag = f"CE={ce_ok_count}/{done}" if args.ce_filter else ""
            warn = " !!!" if args.ce_filter and ce_ok_count < done * 0.5 else ""
            status = "+" if r5 > 0 else "MISS"
            print(
                f"  [{done}/{len(items)}] {status:4s} r@5={r5:.0f} {latency:.1f}s | "
                f"R@5={avg_r5:.3f} MRR={avg_mrr:.3f} {ce_tag}{warn} | {query[:45]}",
                file=sys.stderr,
            )

    elapsed = time.time() - t_total

    # ── Summary line ──
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    summary = {
        "type": "summary",
        "timestamp": ts,
        "pipeline": config_str,
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
    print(f"Queries: {len(items)}, Errors: {error_count}, CE OK: {ce_ok_count}/{len(items)}", file=sys.stderr)
    print(f"Pipeline: {' → '.join(config_str)}", file=sys.stderr)
    print(f"Total: {elapsed:.1f}s ({elapsed/len(items):.2f}s/query)", file=sys.stderr)
    print(f"R@1={avg(running_r1):.3f}  R@5={avg(running_r5):.3f}  MRR={avg(running_mrr):.3f}", file=sys.stderr)
    print(f"\nSaved: {out_path}", file=sys.stderr)

    # ── Backward compat: summary .json ──
    json_path = out_path.rsplit(".", 1)[0] + ".json"
    # Reread jsonl to build full results for json
    all_rows = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "query":
                all_rows.append(obj)

    valid = [r for r in all_rows if "error" not in r]
    report = {
        "timestamp": ts,
        "pipeline": config_str,
        "total_queries": len(items),
        "recall_at_1": avg([r["r1"] for r in valid]),
        "recall_at_3": avg([r.get("r3", 0) for r in valid]),
        "recall_at_5": avg([r["r5"] for r in valid]),
        "recall_at_10": avg([r["r10"] for r in valid]),
        "recall_at_20": avg([r["r20"] for r in valid]),
        "mrr_at_20": avg([r["rr"] for r in valid]),
        "avg_latency": elapsed / len(items),
        "ce_threshold": args.ce_threshold if args.ce_filter else None,
        "channel_dedup": args.channel_dedup,
        "ce_ok_rate": round(ce_ok_count / len(items), 3) if items else 0,
        "results": all_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
