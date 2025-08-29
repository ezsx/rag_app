import logging
from typing import List, Tuple

from core.settings import get_settings

logger = logging.getLogger(__name__)


class RerankerService:
    """
    CPU CrossEncoder reranker based on sentence-transformers CrossEncoder.
    Default model: BAAI/bge-reranker-v2-m3
    """

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_key
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Загружаем CrossEncoder для ререйка: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device="cpu")
            logger.info("Ререйкер успешно загружен (CPU)")
        except Exception as e:
            logger.error(f"Не удалось загрузить ререйкер: {e}")
            raise

    def rerank(
        self, query: str, docs: List[str], top_n: int, batch_size: int
    ) -> List[int]:
        """
        Возвращает индексы документов, отсортированные по убыванию релевантности к запросу.
        """
        if not docs:
            return []
        try:
            # Формируем пары (query, doc)
            pairs = [(query, d) for d in docs]
            scores = self.model.predict(
                pairs, batch_size=batch_size, show_progress_bar=False
            )
            # Сортируем индексы по score desc
            order = sorted(
                range(len(docs)), key=lambda i: float(scores[i]), reverse=True
            )
            if top_n and top_n > 0:
                order = order[: min(top_n, len(order))]
            return order
        except Exception as e:
            logger.error(f"Ошибка ререйкера: {e}")
            return list(range(min(len(docs), top_n)))
