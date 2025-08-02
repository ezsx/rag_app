"""
Тесты для QA Service
"""

import pytest
from unittest.mock import Mock, patch
from services.qa_service import QAService


class TestQAService:

    def test_qa_service_init(self):
        """Тест инициализации QAService"""
        mock_retriever = Mock()
        mock_llm = Mock()

        service = QAService(mock_retriever, mock_llm, top_k=3)

        assert service.retriever == mock_retriever
        assert service.llm == mock_llm
        assert service.top_k == 3

    @patch("services.qa_service.build_prompt")
    def test_answer(self, mock_build_prompt):
        """Тест базового ответа на вопрос"""
        # Настраиваем моки
        mock_retriever = Mock()
        mock_retriever.get_context.return_value = ["doc1", "doc2"]

        mock_llm = Mock()
        mock_llm.return_value = {"choices": [{"text": "Test answer"}]}

        mock_build_prompt.return_value = "test prompt"

        # Создаем сервис и тестируем
        service = QAService(mock_retriever, mock_llm, top_k=2)
        result = service.answer("test query")

        # Проверяем результат
        assert result == "Test answer"
        mock_retriever.get_context.assert_called_once_with("test query", k=2)
        mock_build_prompt.assert_called_once_with("test query", ["doc1", "doc2"])
        mock_llm.assert_called_once()

    @patch("services.qa_service.build_prompt")
    def test_answer_with_context(self, mock_build_prompt):
        """Тест ответа с контекстом"""
        # Настраиваем моки
        mock_retriever = Mock()
        mock_retriever.get_context_with_metadata.return_value = [
            {"document": "doc1", "metadata": {"id": 1}, "distance": 0.1},
            {"document": "doc2", "metadata": {"id": 2}, "distance": 0.2},
        ]

        mock_llm = Mock()
        mock_llm.return_value = {"choices": [{"text": "Detailed answer"}]}

        mock_build_prompt.return_value = "test prompt"

        # Создаем сервис и тестируем
        service = QAService(mock_retriever, mock_llm, top_k=2)
        result = service.answer_with_context("test query")

        # Проверяем результат
        assert result["answer"] == "Detailed answer"
        assert result["query"] == "test query"
        assert result["context_count"] == 2
        assert len(result["context"]) == 2
        assert result["context"][0]["document"] == "doc1"
