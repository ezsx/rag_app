"""Тесты для final_answer tool."""

from services.tools.final_answer import final_answer


def test_sources_extracted_from_text():
    payload = final_answer(answer="DeepSeek выпустит V4 [1]")
    assert payload["sources"] == [1]


def test_empty_sources_adds_warning():
    payload = final_answer(answer="Это ответ без цитат")
    assert payload["sources"] == []
    assert "Источники не указаны" in payload["answer"]


def test_sources_merged():
    payload = final_answer(answer="Факт [2]", sources=[1])
    assert payload["sources"] == [1, 2]
