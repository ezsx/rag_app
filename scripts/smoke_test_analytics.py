"""Smoke test для SPEC-RAG-15: entity_tracker + arxiv_tracker.

Запуск: из WSL2 с активированным venv и PYTHONPATH=src
    cd /mnt/c/llms/rag/rag_app
    source /home/ezsx/infinity-env/bin/activate
    PYTHONPATH=src python3 scripts/smoke_test_analytics.py
"""
import sys

sys.path.insert(0, "src")

from core.deps import get_hybrid_retriever
from services.tools.arxiv_tracker import arxiv_tracker
from services.tools.entity_tracker import _load_alias_map, _normalize_entity, entity_tracker


def main():
    # 0. Entity normalization
    alias_map = _load_alias_map()
    print(f"=== Entity Dictionary: {len(alias_map)} aliases ===")
    tests = [
        ("openai", "OpenAI"),
        ("deepseek v3", "DeepSeek-V3"),
        ("NVIDIA", "NVIDIA"),
        ("unknown_thing", "unknown_thing"),
    ]
    for inp, expected in tests:
        got = _normalize_entity(inp)
        ok = "OK" if got == expected else f"FAIL (expected {expected})"
        print(f"  normalize({inp!r}) = {got!r} [{ok}]")
    print()

    # 1. Get retriever
    retriever = get_hybrid_retriever()
    print(f"=== HybridRetriever ready, collection: {retriever._store.collection} ===")
    print()

    # 2. entity_tracker top
    result = entity_tracker(mode="top", limit=5, hybrid_retriever=retriever)
    print("=== entity_tracker(mode=top, limit=5) ===")
    print(f"  summary: {result.get('summary')}")
    for d in result.get("data", []):
        print(f"    {d['entity']}: {d['count']}")
    print()

    # 3. entity_tracker timeline
    result = entity_tracker(mode="timeline", entity="DeepSeek", hybrid_retriever=retriever)
    print("=== entity_tracker(mode=timeline, entity=DeepSeek) ===")
    print(f"  summary: {result.get('summary')}")
    data = result.get("data", [])
    print(f"  weeks: {len(data)}, total: {result.get('total')}")
    if data:
        top3 = sorted(data, key=lambda x: -x["count"])[:3]
        print(f"  top 3 weeks: {top3}")
    print()

    # 4. entity_tracker co_occurrence
    result = entity_tracker(mode="co_occurrence", entity="NVIDIA", limit=5, hybrid_retriever=retriever)
    print("=== entity_tracker(mode=co_occurrence, entity=NVIDIA, limit=5) ===")
    print(f"  summary: {result.get('summary')}")
    for d in result.get("data", []):
        print(f"    {d['entity']}: {d['count']}")
    print()

    # 5. entity_tracker compare
    result = entity_tracker(mode="compare", entities=["OpenAI", "DeepSeek", "Anthropic"], hybrid_retriever=retriever)
    print("=== entity_tracker(mode=compare, entities=[OpenAI, DeepSeek, Anthropic]) ===")
    print(f"  summary: {result.get('summary')}")
    print()

    # 6. arxiv_tracker top
    result = arxiv_tracker(mode="top", limit=5, hybrid_retriever=retriever)
    print("=== arxiv_tracker(mode=top, limit=5) ===")
    print(f"  summary: {result.get('summary')}")
    for d in result.get("data", []):
        print(f"    arxiv:{d['arxiv_id']}: {d['mentions']} mentions")
    print()

    # 7. arxiv_tracker lookup
    top_arxiv = result.get("data", [{}])[0].get("arxiv_id", "2502.13266")
    result = arxiv_tracker(mode="lookup", arxiv_id=top_arxiv, limit=5, hybrid_retriever=retriever)
    print(f"=== arxiv_tracker(mode=lookup, arxiv_id={top_arxiv}) ===")
    print(f"  summary: {result.get('summary')}")
    for h in result.get("hits", []):
        ch = h["meta"].get("channel", "?")
        dt = h["meta"].get("date", "?")
        snip = h.get("snippet", "")[:80]
        print(f"    [{ch}] {dt}: {snip}...")
    print()

    # 8. Normalization in action — lowercase entity should still return data
    result = entity_tracker(mode="timeline", entity="openai", hybrid_retriever=retriever)
    print("=== entity_tracker(mode=timeline, entity='openai') — normalization test ===")
    print(f"  summary: {result.get('summary')}")
    total = result.get("total", 0)
    ok = "OK — normalized to OpenAI" if total > 0 else "FAIL — no data (normalization broken?)"
    print(f"  result: {ok}")
    print()

    print("=== SMOKE TEST COMPLETE ===")


if __name__ == "__main__":
    main()
