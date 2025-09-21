"""
Тесты для новых инструментов: verify, math_eval, time_now
"""

import pytest
from unittest.mock import Mock

from services.tools.math_eval import math_eval
from services.tools.time_now import time_now
from services.tools.verify import verify


class TestMathEval:
    """Тесты для math_eval инструмента"""

    def test_basic_arithmetic(self):
        """Тест базовой арифметики"""
        result = math_eval("2 + 3")
        assert result["success"] is True
        assert result["result"] == 5

    def test_multiplication(self):
        """Тест умножения"""
        result = math_eval("4 * 5")
        assert result["success"] is True
        assert result["result"] == 20

    def test_division(self):
        """Тест деления"""
        result = math_eval("10 / 2")
        assert result["success"] is True
        assert result["result"] == 5

    def test_math_functions(self):
        """Тест математических функций"""
        result = math_eval("sqrt(16)")
        assert result["success"] is True
        assert result["result"] == 4

    def test_complex_expression(self):
        """Тест сложного выражения"""
        result = math_eval("2 + 3 * 4")
        assert result["success"] is True
        assert result["result"] == 14

    def test_invalid_expression(self):
        """Тест невалидного выражения"""
        result = math_eval("2 +")
        assert result["success"] is False
        assert "error" in result

    def test_unsafe_expression(self):
        """Тест небезопасного выражения"""
        result = math_eval("__import__('os')")
        assert result["success"] is False
        assert "error" in result

    def test_empty_expression(self):
        """Тест пустого выражения"""
        result = math_eval("")
        assert result["success"] is False
        assert "error" in result


class TestTimeNow:
    """Тесты для time_now инструмента"""

    def test_iso_format(self):
        """Тест ISO формата времени"""
        result = time_now("iso")
        assert "current_time" in result
        assert "timezone" in result
        assert result["timezone"] == "UTC"
        assert result["format_used"] == "iso"

    def test_timestamp_format(self):
        """Тест timestamp формата"""
        result = time_now("timestamp")
        assert "current_time" in result
        assert result["current_time"].isdigit()
        assert result["format_used"] == "timestamp"

    def test_readable_format(self):
        """Тест читаемого формата"""
        result = time_now("readable")
        assert "current_time" in result
        assert "UTC" in result["current_time"]
        assert result["format_used"] == "readable"

    def test_date_format(self):
        """Тест формата только даты"""
        result = time_now("date")
        assert "current_time" in result
        assert len(result["current_time"]) == 10  # YYYY-MM-DD
        assert result["format_used"] == "date"

    def test_time_format(self):
        """Тест формата только времени"""
        result = time_now("time")
        assert "current_time" in result
        assert ":" in result["current_time"]
        assert result["format_used"] == "time"

    def test_additional_fields(self):
        """Тест дополнительных полей"""
        result = time_now()
        assert "year" in result
        assert "month" in result
        assert "day" in result
        assert "hour" in result
        assert "minute" in result
        assert "second" in result
        assert "weekday" in result
        assert "month_name" in result

    def test_unknown_format(self):
        """Тест неизвестного формата"""
        result = time_now("unknown_format")
        assert "current_time" in result
        assert result["format_used"] == "iso (fallback)"
        assert "warning" in result


class TestVerify:
    """Тесты для verify инструмента"""

    def test_verify_with_matching_documents(self):
        """Тест проверки с подходящими документами"""
        # Создаем мок retriever
        mock_retriever = Mock()
        mock_retriever.search.return_value = {
            "documents": ["Это подтверждающий документ", "Еще один документ"],
            "distances": [0.1, 0.2],
        }

        result = verify(
            query="Тестовый запрос",
            claim="Тестовое утверждение",
            retriever=mock_retriever,
        )

        assert "verified" in result
        assert "confidence" in result
        assert "evidence" in result
        assert len(result["evidence"]) == 2
        assert result["documents_found"] == 2

    def test_verify_no_documents_found(self):
        """Тест проверки без найденных документов"""
        mock_retriever = Mock()
        mock_retriever.search.return_value = {"documents": [], "distances": []}

        result = verify(
            query="Тестовый запрос",
            claim="Тестовое утверждение",
            retriever=mock_retriever,
        )

        assert result["verified"] is False
        assert result["confidence"] == 0.0
        assert result["evidence"] == []
        assert "note" in result

    def test_verify_empty_claim(self):
        """Тест проверки пустого утверждения"""
        mock_retriever = Mock()

        result = verify(query="Тестовый запрос", claim="", retriever=mock_retriever)

        assert result["verified"] is False
        assert result["confidence"] == 0.0
        assert "error" in result

    def test_verify_with_search_error(self):
        """Тест проверки с ошибкой поиска"""
        mock_retriever = Mock()
        mock_retriever.search.side_effect = Exception("Search error")

        result = verify(
            query="Тестовый запрос",
            claim="Тестовое утверждение",
            retriever=mock_retriever,
        )

        assert result["verified"] is False
        assert result["confidence"] == 0.0
        assert "error" in result

    def test_verify_confidence_calculation(self):
        """Тест расчета уверенности"""
        mock_retriever = Mock()
        mock_retriever.search.return_value = {
            "documents": ["Очень релевантный документ"],
            "distances": [0.05],  # Очень маленькое расстояние = высокая уверенность
        }

        result = verify(
            query="Тестовый запрос",
            claim="Тестовое утверждение",
            retriever=mock_retriever,
        )

        # С маленьким distance должна быть высокая confidence
        assert result["confidence"] > 0.8
        assert result["verified"] is True  # Должно превысить порог 0.6
