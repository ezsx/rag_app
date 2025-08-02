"""
Retriever для поиска релевантных документов в ChromaDB
"""

from typing import List, Dict, Any, Optional
import logging

import numpy as np
import chromadb

try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError:
    # Fallback для более новых версий ChromaDB
    from sentence_transformers import SentenceTransformer
    from chromadb import EmbeddingFunction

    class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
        def __init__(self, model_name: str, device: str = "cpu", **kwargs):
            self._model = SentenceTransformer(model_name, device=device)

        def __call__(self, input):
            return self._model.encode(input)  # np.ndarray → Chroma ok


logger = logging.getLogger(__name__)


class Retriever:
    """Класс для поиска top-k документов в ChromaDB коллекции"""

    def __init__(
        self,
        chroma_client: chromadb.Client,
        collection_name: str = "tg_test1",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.client = chroma_client
        self.collection_name = collection_name
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model, device="cpu"
        )

        # Подключаемся к коллекции или создаем новую
        try:
            self.collection = self.client.get_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            logger.info(
                f"Подключились к коллекции {collection_name}, count={self.collection.count()}"
            )
        except Exception as e:
            logger.warning(f"Коллекция {collection_name} не найдена: {e}")
            # Создаем пустую коллекцию для случая, когда данные еще не загружены
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function,
            )
            logger.info(f"Создана новая коллекция {collection_name}")

    def get_context(self, query: str, k: int = 5) -> List[str]:
        """
        Ищет top-k наиболее релевантных документов для запроса

        Args:
            query: Поисковый запрос
            k: Количество документов для возврата

        Returns:
            Список текстов документов
        """
        try:
            # Выполняем поиск
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            # Извлекаем документы
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            logger.info(
                f"Найдено {len(documents)} документов для запроса: {query[:50]}..."
            )
            for i, (doc, dist) in enumerate(zip(documents[:3], distances[:3])):
                logger.debug(f"  {i+1}. distance={dist:.3f}, doc={doc[:100]}...")

            return documents

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []

    def get_context_with_metadata(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Ищет top-k документов с метаданными

        Returns:
            Список словарей с ключами: document, metadata, distance
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)

            context_items = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                context_items.append(
                    {"document": doc, "metadata": meta, "distance": dist}
                )

            return context_items

        except Exception as e:
            logger.error(f"Ошибка при поиске с метаданными: {e}")
            return []
