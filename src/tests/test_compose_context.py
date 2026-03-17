from services.tools.compose_context import (
    _compute_coverage,
    _query_term_coverage,
    compose_context,
)


def test_compose_context_basic_limit():
    docs = [
        {"id": "1", "text": "a" * 2000, "metadata": {}},
        {"id": "2", "text": "b" * 2000, "metadata": {}},
    ]
    out = compose_context(docs, max_tokens_ctx=100)  # ~400 chars
    prompt = out["prompt"]
    citations = out["citations"]
    contexts = out["contexts"]
    assert len(prompt) <= 400 + 20  # запас на префиксы [i]
    assert len(citations) == len(contexts)
    assert citations[0]["id"] == "1" and citations[0]["index"] == 1
    assert citations[-1]["index"] == len(citations)


def test_citation_format_includes_metadata():
    docs = [
        {
            "id": "1",
            "text": "DeepSeek выпустит V4 в феврале",
            "metadata": {"channel": "ai_news", "date": "2026-01-10T12:00:00"},
        }
    ]
    out = compose_context(docs, max_tokens_ctx=100)
    assert "[1] (ai_news, 2026-01-10): DeepSeek выпустит V4 в феврале" in out["prompt"]


def test_compute_coverage_empty():
    assert _compute_coverage("query", []) == 0.0


def test_compute_coverage_high_relevance():
    docs = [{"text": "Bitcoin crypto news", "dense_score": 0.82}] * 5
    result = _compute_coverage("bitcoin crypto", docs)
    assert result > 0.65


def test_compute_coverage_low_relevance():
    docs = [{"text": "unrelated text", "dense_score": 0.20}] * 3
    result = _compute_coverage("bitcoin crypto", docs)
    assert result < 0.35


def test_compute_coverage_no_dense_score_fallback():
    docs = [{"text": "bitcoin price crypto market", "dense_score": None}]
    result = _compute_coverage("bitcoin", docs)
    assert 0.0 <= result <= 1.0


def test_query_term_coverage_basic():
    docs = [{"text": "bitcoin price rose sharply"}]
    assert _query_term_coverage("bitcoin price", docs) == 1.0


def test_query_term_coverage_partial():
    docs = [{"text": "bitcoin price rose"}]
    result = _query_term_coverage("bitcoin ethereum", docs)
    assert result == 0.5


def test_compute_coverage_is_capped_at_one():
    docs = [{"text": "test", "dense_score": 0.95}] * 10
    assert _compute_coverage("test", docs) <= 1.0


def test_coverage_higher_than_naive_for_relevant_docs():
    docs = [{"text": f"bitcoin crypto news {i}", "dense_score": 0.75} for i in range(5)]
    naive = 1.0
    composite = _compute_coverage("bitcoin crypto", docs)
    assert 0.5 < composite < naive


def test_coverage_lower_than_naive_for_irrelevant_docs():
    docs = [{"text": "unrelated stuff", "dense_score": 0.10}] * 5
    naive = 1.0
    composite = _compute_coverage("bitcoin crypto", docs)
    assert composite < naive
