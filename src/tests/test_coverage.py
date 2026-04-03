"""Tests for LANCER-style nugget coverage from services/agent/coverage.py."""

from __future__ import annotations

from services.agent.coverage import CoverageResult, compute_nugget_coverage


class TestComputeNuggetCoverage:
    """compute_nugget_coverage: score, uncovered, fallback, threshold."""

    def test_all_nuggets_covered(self) -> None:
        docs = [{"text": "AI transformers architecture attention mechanism"}]
        nuggets = ["AI transformers", "attention mechanism"]
        result = compute_nugget_coverage("AI transformers attention", docs, nuggets)

        assert result.score == 1.0
        assert result.uncovered == []
        # query auto-added + 2 nuggets = 3
        assert result.total_nuggets == 3
        assert result.covered_nuggets == 3

    def test_partial_coverage(self) -> None:
        docs = [{"text": "deep learning neural networks gradient descent"}]
        nuggets = ["deep learning", "quantum computing"]
        result = compute_nugget_coverage("deep learning overview", docs, nuggets)

        assert 0 < result.score < 1.0
        assert "quantum computing" in result.uncovered
        assert result.covered_nuggets < result.total_nuggets

    def test_empty_docs_returns_zero(self) -> None:
        result = compute_nugget_coverage("test query", docs=[])

        assert result.score == 0.0
        assert result.total_nuggets == 0
        assert result.covered_nuggets == 0
        assert result.uncovered == ["test query"]

    def test_empty_docs_with_nuggets(self) -> None:
        result = compute_nugget_coverage("q", docs=[], nuggets=["a", "b"])

        assert result.score == 0.0
        assert result.uncovered == ["a", "b"]

    def test_no_nuggets_fallback_to_query(self) -> None:
        docs = [{"text": "language models generate text tokens"}]
        result = compute_nugget_coverage("language models tokens", docs, nuggets=None)

        assert result.total_nuggets == 1
        assert result.score == 1.0

    def test_query_auto_added_to_nuggets(self) -> None:
        docs = [{"text": "alpha beta gamma delta epsilon"}]
        nuggets = ["alpha beta", "gamma delta"]
        query = "alpha gamma epsilon"
        result = compute_nugget_coverage(query, docs, nuggets)

        # query is not in nuggets list, so effective = [query] + nuggets = 3
        assert result.total_nuggets == len(nuggets) + 1

    def test_query_not_duplicated_when_already_in_nuggets(self) -> None:
        query = "alpha beta"
        docs = [{"text": "alpha beta gamma"}]
        nuggets = ["alpha beta", "gamma"]
        result = compute_nugget_coverage(query, docs, nuggets)

        # query == nuggets[0], no duplication
        assert result.total_nuggets == 2

    def test_threshold_edge_case(self) -> None:
        # Nugget with 2 terms, doc covers only 1 => ratio 0.5
        docs = [{"text": "tensorflow library framework"}]

        # "tensorflow pytorch" -> terms: tensorflow, pytorch (2 terms)
        # doc has "tensorflow" but not "pytorch" -> coverage = 1/2 = 0.5
        # threshold=0.5 -> covered (>=)
        result_low = compute_nugget_coverage(
            "tensorflow pytorch", docs, nugget_threshold=0.5
        )
        assert result_low.score == 1.0

        # threshold=0.6 -> not covered
        result_high = compute_nugget_coverage(
            "tensorflow pytorch", docs, nugget_threshold=0.6
        )
        assert result_high.score == 0.0

    def test_returns_coverage_result_type(self) -> None:
        result = compute_nugget_coverage("x", [{"text": "x y z"}])
        assert isinstance(result, CoverageResult)
        assert isinstance(result.details, dict)
