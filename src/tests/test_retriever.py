import pytest


def test_dummy():
    assert True


"""
Тесты для Retriever класса
"""

import pytest
from unittest.mock import Mock, patch
from adapters.chroma.retriever import Retriever


class TestRetriever:

    @patch("adapters.chroma.retriever.chromadb")
    @patch("adapters.chroma.retriever.SentenceTransformerEmbeddingFunction")
    def test_retriever_init(self, mock_embedding_fn, mock_chromadb):
        """Тест инициализации Retriever"""
        # Мокаем клиент и коллекцию
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 10
        mock_client.get_collection.return_value = mock_collection

        # Создаем Retriever
        retriever = Retriever(mock_client, "test_collection", "test_model")

        # Проверяем, что все инициализировано правильно
        assert retriever.client == mock_client
        assert retriever.collection_name == "test_collection"
        mock_client.get_collection.assert_called_once()

    @patch("adapters.chroma.retriever.chromadb")
    @patch("adapters.chroma.retriever.SentenceTransformerEmbeddingFunction")
    def test_get_context(self, mock_embedding_fn, mock_chromadb):
        """Тест поиска контекста"""
        # Мокаем клиент и коллекцию
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2", "doc3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[{}, {}, {}]],
        }
        mock_client.get_collection.return_value = mock_collection

        # Создаем Retriever и выполняем поиск
        retriever = Retriever(mock_client)
        context = retriever.get_context("test query", k=3)

        # Проверяем результат
        assert context == ["doc1", "doc2", "doc3"]
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

    @patch("adapters.chroma.retriever.chromadb")
    @patch("adapters.chroma.retriever.SentenceTransformerEmbeddingFunction")
    def test_get_context_with_metadata(self, mock_embedding_fn, mock_chromadb):
        """Тест поиска контекста с метаданными"""
        # Мокаем клиент и коллекцию
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"id": 1}, {"id": 2}]],
        }
        mock_client.get_collection.return_value = mock_collection

        # Создаем Retriever и выполняем поиск
        retriever = Retriever(mock_client)
        context_items = retriever.get_context_with_metadata("test query", k=2)

        # Проверяем результат
        assert len(context_items) == 2
        assert context_items[0]["document"] == "doc1"
        assert context_items[0]["metadata"] == {"id": 1}
        assert context_items[0]["distance"] == 0.1
