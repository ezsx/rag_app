"""
Retriever для поиска релевантных документов в ChromaDB
"""

from typing import List, Dict, Any, Optional, Tuple
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

    def search(
        self, query: str, k: int, filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Выполняет поиск с поддержкой where-фильтров Chroma.
        Перед эмбеддингом добавляет E5 префикс "query: ".
        Возвращает кортеж (documents, distances, metadatas)
        """
        try:
            # Добавляем e5-префикс
            e5_query = f"query: {query}"

            where = None
            if filters and isinstance(filters, dict):
                where = self._build_where(filters)

            results = self.collection.query(
                query_texts=[e5_query],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)

            return documents, distances, metadatas

        except TypeError:
            # Версия Chroma без поддержки where
            logger.warning(
                "Версия Chroma не поддерживает параметр 'where'. Фильтры проигнорированы."
            )
            results = self.collection.query(
                query_texts=[f"query: {query}"],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)
            return documents, distances, metadatas
        except Exception as e:
            logger.error(f"Ошибка при поиске с фильтрами: {e}")
            return [], [], []

    def _build_where(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Строит корректное Chroma where-условие из бизнес-фильтров.
        - Удаляет пустые значения ("", [], None)
        - Объединяет условия через $and при наличии нескольких
        - Использует операторы $eq/$gte/$lte
        Возвращает None, если условий нет.
        """
        conditions: List[Dict[str, Any]] = []

        def _clean_list(values: Optional[List[Any]]) -> List[Any]:
            if not values:
                return []
            return [v for v in values if v not in (None, "", [])]

        # channel_usernames → metadata.channel_username
        usernames = _clean_list(filters.get("channel_usernames"))
        if usernames:
            if len(usernames) == 1:
                conditions.append({"channel_username": {"$eq": usernames[0]}})
            else:
                conditions.append(
                    {"$or": [{"channel_username": {"$eq": u}} for u in usernames]}
                )

        # channel_ids → metadata.channel_id
        channel_ids = _clean_list(filters.get("channel_ids"))
        if channel_ids:
            if len(channel_ids) == 1:
                conditions.append({"channel_id": {"$eq": channel_ids[0]}})
            else:
                conditions.append(
                    {"$or": [{"channel_id": {"$eq": cid}} for cid in channel_ids]}
                )

        # min_views → metadata.views
        min_views = filters.get("min_views")
        if isinstance(min_views, int) and min_views > 0:
            conditions.append({"views": {"$gte": min_views}})

        # reply_to → metadata.reply_to
        reply_to = filters.get("reply_to")
        if isinstance(reply_to, int):
            conditions.append({"reply_to": {"$eq": reply_to}})

        # date_from/date_to → metadata.date
        date_from = filters.get("date_from")
        if isinstance(date_from, str) and date_from.strip():
            conditions.append({"date": {"$gte": date_from.strip()}})
        date_to = filters.get("date_to")
        if isinstance(date_to, str) and date_to.strip():
            conditions.append({"date": {"$lte": date_to.strip()}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
