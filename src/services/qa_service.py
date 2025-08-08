import logging
from typing import List, Dict, Any, AsyncIterator, Optional, Tuple
from utils.prompt import build_prompt
from core.settings import Settings, get_settings
from services.query_planner_service import QueryPlannerService
from utils.ranking import rrf_merge

logger = logging.getLogger(__name__)


class QAService:
    """Сервис для ответов на вопросы с использованием RAG"""

    def __init__(
        self,
        retriever,
        llm,
        top_k: int = 5,
        settings: Optional[Settings] = None,
        planner: Optional[QueryPlannerService] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.settings = settings or get_settings()
        self.planner = planner
        logger.info(f"QAService инициализирован с top_k={top_k}")

    def answer(self, query: str) -> str:
        """
        Отвечает на вопрос пользователя используя RAG подход

        Args:
            query: Вопрос пользователя

        Returns:
            Ответ от LLM
        """
        try:
            # Получаем контекст (с планировщиком если включен)
            logger.info(f"Поиск контекста для запроса: {query[:100]}...")
            context, _ = self._fetch_context(query, return_metadata=False)
            logger.info(f"Найдено {len(context)} документов")

            # Формируем промпт
            prompt = build_prompt(query, context)
            logger.debug(f"Сформирован промпт длиной {len(prompt)} символов")

            # Получаем ответ от LLM
            logger.info("Генерируем ответ с помощью LLM...")
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "\n\nВопрос:", "\n\nUser:", "<s>"],
                echo=False,
            )

            # Извлекаем текст ответа
            answer = response["choices"][0]["text"].strip()
            logger.info(f"Получен ответ длиной {len(answer)} символов")

            return answer

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"

    def answer_with_context(self, query: str) -> Dict[str, Any]:
        """
        Расширенная версия, возвращающая ответ вместе с использованным контекстом

        Returns:
            Словарь с ключами: answer, context, query
        """
        try:
            context, metadatas = self._fetch_context(query, return_metadata=True)
            context_items = [
                {"document": doc, "metadata": meta, "distance": 0.0}
                for doc, meta in zip(context, metadatas)
            ]

            prompt = build_prompt(query, context)

            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "\n\nВопрос:", "\n\nUser:", "<s>"],
                echo=False,
            )

            answer = response["choices"][0]["text"].strip()

            return {
                "answer": answer,
                "context": context_items,
                "query": query,
                "context_count": len(context_items),
            }

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа с контекстом: {e}")
            return {
                "answer": f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}",
                "context": [],
                "query": query,
                "context_count": 0,
            }

    async def stream_answer(
        self, query: str, include_context: bool = False
    ) -> AsyncIterator[str]:
        """
        Генерирует стримящий ответ на вопрос пользователя используя RAG подход

        Args:
            query: Вопрос пользователя
            include_context: Использовать ли контекст из retriever

        Yields:
            str: Токены ответа от LLM
        """
        try:
            logger.info(f"Начинаем стриминг для запроса: {query[:100]}...")

            # Получаем контекст если нужен
            if include_context:
                context, _ = self._fetch_context(query, return_metadata=True)
                logger.info(f"Найдено {len(context)} документов для контекста")
            else:
                context, _ = self._fetch_context(query, return_metadata=False)
                logger.info(f"Найдено {len(context)} документов")

            # Формируем промпт
            prompt = build_prompt(query, context)
            logger.debug(f"Сформирован промпт длиной {len(prompt)} символов")

            # Начинаем стриминг с LLM
            logger.info("Начинаем стриминг ответа от LLM...")
            token_count = 0

            stream = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "\n\nВопрос:", "\n\nUser:", "<s>"],
                echo=False,
                stream=True,
            )

            for chunk in stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    if "text" in choice and choice["text"]:
                        token = choice["text"]
                        token_count += 1
                        yield token

            logger.info(f"Стриминг завершен. Отправлено токенов: {token_count}")

        except Exception as e:
            logger.error(f"Ошибка при стриминге ответа: {e}")
            # Отправляем сообщение об ошибке как финальный токен
            yield f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"

    def _fetch_context(
        self, query: str, return_metadata: bool = False
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self.settings.enable_query_planner and self.planner is not None:
            plan = self.planner.make_plan(query)
            # Проверяем кеш фьюжна
            fusion_key = f"fusion:{hash(query + plan.json())}"
            cached = self.planner.get_cached_fusion(fusion_key)
            if cached is not None:
                merged = cached  # type: ignore
            else:
                results_for_fusion = []
                for q in plan.normalized_queries:
                    docs, dists, metas = self.retriever.search(
                        q,
                        k=plan.k_per_query,
                        filters=(
                            plan.metadata_filters.dict(exclude_none=True)
                            if plan.metadata_filters
                            else None
                        ),
                    )
                    triples = list(zip(docs, dists, metas))
                    results_for_fusion.append(triples)

                merged = rrf_merge(results_for_fusion, k=self.settings.k_fusion)
                self.planner.set_cached_fusion(fusion_key, merged, ttl=300)
            documents = [doc for doc, _dist, _meta in merged][: self.top_k]
            metadatas = [meta for _doc, _dist, meta in merged][: self.top_k]
            return (documents, metadatas) if return_metadata else (documents, [])

        # Старый путь
        if return_metadata:
            items = self.retriever.get_context_with_metadata(query, k=self.top_k)
            documents = [item["document"] for item in items]
            metadatas = [item["metadata"] for item in items]
            return documents, metadatas
        else:
            documents = self.retriever.get_context(query, k=self.top_k)
            return documents, []
