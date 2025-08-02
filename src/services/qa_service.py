import logging
from typing import List, Dict, Any
from utils.prompt import build_prompt

logger = logging.getLogger(__name__)


class QAService:
    """Сервис для ответов на вопросы с использованием RAG"""

    def __init__(self, retriever, llm, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
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
            # Получаем контекст из ChromaDB
            logger.info(f"Поиск контекста для запроса: {query[:100]}...")
            context = self.retriever.get_context(query, k=self.top_k)
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
            context_items = self.retriever.get_context_with_metadata(
                query, k=self.top_k
            )
            context = [item["document"] for item in context_items]

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
