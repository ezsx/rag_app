"""
Unit-тесты для RerankerService как sync-обёртки над TEIRerankerClient.
"""

from __future__ import annotations

import httpx

from services.reranker_service import RerankerService


class _FakeTEIRerankerClient:
    """Минимальный async-клиент для тестов без сети."""

    def __init__(
        self,
        scores: list[float] | None = None,
        *,
        error: Exception | None = None,
        health: bool = True,
        base_url: str = "http://mock-reranker",
    ) -> None:
        self.base_url = base_url
        self._scores = scores or []
        self._error = error
        self._health = health

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        if self._error is not None:
            raise self._error
        return list(self._scores[: len(passages)])

    async def healthcheck(self) -> bool:
        if self._error is not None:
            raise self._error
        return self._health


def _make_service(
    scores: list[float] | None = None,
    *,
    error: Exception | None = None,
    health: bool = True,
) -> RerankerService:
    """Создаёт сервис с подменённым async TEI-клиентом."""
    return RerankerService(
        _FakeTEIRerankerClient(scores=scores, error=error, health=health)
    )


class TestRerank:
    def test_rerank_returns_sorted_indices(self):
        """rerank() сортирует индексы по убыванию relevance score."""
        svc = _make_service([1.0, 5.0, 3.0])
        try:
            indices = svc.rerank("query", ["doc0", "doc1", "doc2"], top_n=3)
            assert indices == [1, 2, 0]
        finally:
            svc.close()

    def test_rerank_respects_top_n(self):
        """rerank() усекает результат до top_n."""
        svc = _make_service([1.0, 5.0, 3.0])
        try:
            indices = svc.rerank("query", ["doc0", "doc1", "doc2"], top_n=2)
            assert indices == [1, 2]
        finally:
            svc.close()

    def test_rerank_empty_docs_returns_empty(self):
        """rerank() с пустым списком документов возвращает пустой результат."""
        svc = _make_service([])
        try:
            assert svc.rerank("query", [], top_n=5) == []
        finally:
            svc.close()

    def test_rerank_fallback_on_error(self):
        """При ошибке возвращается исходный порядок документов."""
        svc = _make_service(error=httpx.ConnectError("Connection refused"))
        try:
            docs = ["doc0", "doc1", "doc2"]
            assert svc.rerank("query", docs, top_n=3) == [0, 1, 2]
        finally:
            svc.close()

    def test_rerank_ignores_batch_size(self):
        """batch_size сохранён только для обратной совместимости."""
        svc = _make_service([2.0, 1.0])
        try:
            r1 = svc.rerank("query", ["a", "b"], top_n=2, batch_size=1)
            r2 = svc.rerank("query", ["a", "b"], top_n=2, batch_size=64)
            assert r1 == r2 == [0, 1]
        finally:
            svc.close()


class TestRerankWithScores:
    def test_returns_correct_indices_and_scores(self):
        """rerank_with_scores() возвращает индексы и sigmoid scores по убыванию."""
        svc = _make_service([0.0, 2.0, -1.0])
        try:
            indices, scores = svc.rerank_with_scores(
                "query", ["doc0", "doc1", "doc2"], top_n=3
            )
            assert indices == [1, 0, 2]
            assert len(scores) == 3
            assert scores[0] > scores[1] > scores[2]
        finally:
            svc.close()

    def test_sigmoid_values_correct(self):
        """sigmoid(0) должен быть равен 0.5."""
        svc = _make_service([0.0])
        try:
            _, scores = svc.rerank_with_scores("query", ["doc"], top_n=1)
            assert len(scores) == 1
            assert abs(scores[0] - 0.5) < 1e-6
        finally:
            svc.close()

    def test_returns_empty_scores_on_error(self):
        """При ошибке scores очищаются, а индексы возвращаются в исходном порядке."""
        svc = _make_service(error=httpx.ConnectError("Connection refused"))
        try:
            indices, scores = svc.rerank_with_scores("query", ["a", "b", "c"], top_n=3)
            assert indices == [0, 1, 2]
            assert scores == []
        finally:
            svc.close()


class TestHealthcheck:
    def test_healthcheck_true_on_success(self):
        svc = _make_service([], health=True)
        try:
            assert svc.healthcheck() is True
        finally:
            svc.close()

    def test_healthcheck_false_on_error(self):
        svc = _make_service(error=httpx.ConnectError("refused"))
        try:
            assert svc.healthcheck() is False
        finally:
            svc.close()


class TestSigmoid:
    def test_sigmoid_zero(self):
        assert abs(RerankerService._sigmoid(0.0) - 0.5) < 1e-9

    def test_sigmoid_large_positive(self):
        assert RerankerService._sigmoid(10.0) > 0.99

    def test_sigmoid_large_negative(self):
        assert RerankerService._sigmoid(-10.0) < 0.01

    def test_sigmoid_monotone(self):
        values = [-5.0, -1.0, 0.0, 1.0, 5.0]
        results = [RerankerService._sigmoid(v) for v in values]
        assert results == sorted(results)
