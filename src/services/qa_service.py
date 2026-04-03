from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from core.settings import Settings, get_settings
from schemas.search import SearchPlan
from services.query_planner_service import QueryPlannerService
from utils.prompt import build_prompt

if TYPE_CHECKING:
    from adapters.search.hybrid_retriever import HybridRetriever
    from services.reranker_service import RerankerService

logger = logging.getLogger(__name__)


class QAService:
    """RAG baseline path — прямой pipeline без agent loop.

    plan → search → rerank → LLM answer.
    Используется /v1/qa и /v1/qa/stream.
    НЕ legacy: это дефолтный RAG путь для A/B сравнения с AgentService.
    Рядом потом заведём LlamaIndex baseline — тот же Qdrant, framework wrapper.
    """

    def __init__(
        self,
        retriever,
        llm,
        top_k: int = 5,
        settings: Settings | None = None,
        planner: QueryPlannerService | None = None,
        reranker: RerankerService | None = None,
        hybrid: HybridRetriever | None = None,
    ):
        self.retriever = retriever
        # Ленивая инициализация LLM: поддерживаем как готовый инстанс, так и фабрику (callable)
        self._llm_instance = None
        self._llm_factory = llm if callable(llm) else None
        if not callable(llm):
            self._llm_instance = llm
        self.top_k = top_k
        self.settings = settings or get_settings()
        self.planner = planner
        self.reranker = reranker
        self.hybrid = hybrid
        logger.info("QAService инициализирован с top_k=%s", top_k)

    def _get_llm(self):
        if self._llm_instance is None and self._llm_factory is not None:
            logger.info("Ленивая загрузка LLM по первому запросу…")
            self._llm_instance = self._llm_factory()
            logger.info("LLM загружена и готова к использованию")
        return self._llm_instance

    def answer(self, query: str) -> str:
        """Answer a user question using RAG: plan -> search -> rerank -> LLM."""
        try:
            # Получаем контекст (с планировщиком если включен)
            logger.info("Поиск контекста для запроса: %s...", query[:100])
            context, _ = self._fetch_context(query, return_metadata=False)
            logger.info("Найдено %s документов", len(context))

            # Формируем промпт
            prompt = build_prompt(query, context)
            logger.debug("Сформирован промпт длиной %s символов", len(prompt))

            # Получаем ответ от LLM
            logger.info("Генерируем ответ с помощью LLM...")
            response = self._get_llm()(
                prompt,
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                stop=["</s>", "\n\nВопрос:", "\n\nUser:", "<s>"],
                echo=False,
            )

            # Извлекаем текст ответа
            answer = response["choices"][0]["text"].strip()
            logger.info("Получен ответ длиной %s символов", len(answer))

            return answer

        except Exception as e:  # broad: endpoint safety net
            logger.error("Ошибка при генерации ответа: %s", e)
            return f"Извините, произошла ошибка при обработке вашего запроса: {e!s}"

    def answer_with_context(self, query: str) -> dict[str, Any]:
        """Answer with context: returns {answer, context, query, context_count}."""
        try:
            context, metadatas = self._fetch_context(query, return_metadata=True)
            context_items = [
                {"document": doc, "metadata": meta, "distance": 0.0}
                for doc, meta in zip(context, metadatas)
            ]

            prompt = build_prompt(query, context)

            response = self._get_llm()(
                prompt,
                max_tokens=512,
                temperature=0.3,
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

        except Exception as e:  # broad: endpoint safety net
            logger.error("Ошибка при генерации ответа с контекстом: %s", e)
            return {
                "answer": f"Извините, произошла ошибка при обработке вашего запроса: {e!s}",
                "context": [],
                "query": query,
                "context_count": 0,
            }

    async def stream_answer(
        self, query: str, include_context: bool = False
    ) -> AsyncIterator[str]:
        """Stream RAG answer token-by-token."""
        try:
            logger.info("Начинаем стриминг для запроса: %s...", query[:100])

            # Получаем контекст если нужен
            if include_context:
                context, _ = self._fetch_context(query, return_metadata=True)
                logger.info("Найдено %s документов для контекста", len(context))
            else:
                context, _ = self._fetch_context(query, return_metadata=False)
                logger.info("Найдено %s документов", len(context))

            # Формируем промпт
            prompt = build_prompt(query, context)
            logger.debug("Сформирован промпт длиной %s символов", len(prompt))

            # Начинаем стриминг с LLM
            logger.info("Начинаем стриминг ответа от LLM...")
            token_count = 0

            stream = self._get_llm()(
                prompt,
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                stop=["</s>", "\n\nВопрос:", "\n\nUser:", "<s>"],
                echo=False,
                stream=True,
            )

            for chunk in stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    if choice.get("text"):
                        token = choice["text"]
                        token_count += 1
                        yield token

            logger.info("Стриминг завершен. Отправлено токенов: %s", token_count)

        except Exception as e:  # broad: endpoint safety net
            logger.error("Ошибка при стриминге ответа: %s", e)
            # Отправляем сообщение об ошибке как финальный токен
            yield f"Извините, произошла ошибка при обработке вашего запроса: {e!s}"

    def _fetch_context(
        self, query: str, return_metadata: bool = False
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Fetch context via hybrid retriever: plan -> search -> optional rerank."""
        # Построить план поиска
        if self.settings.enable_query_planner and self.planner is not None:
            plan = self.planner.make_plan(query)
        else:
            plan = SearchPlan(
                normalized_queries=[query],
                k_per_query=self.top_k,
                fusion="rrf",
            )

        # search_with_plan делает BM25+Dense → RRF → ColBERT rerank внутри
        retriever = self.hybrid or self.retriever
        candidates = retriever.search_with_plan(query, plan)
        final_items: list[dict[str, Any]] = [
            {
                "id": c.id,
                "text": c.text,
                "metadata": c.metadata,
                "distance": 0.0,
            }
            for c in candidates
        ]

        # Reranker (опционально)
        if self.settings.enable_reranker and self.reranker and final_items:
            top_n = min(len(final_items), self.settings.reranker_top_n)
            texts = [it["text"] for it in final_items[:top_n]]
            order = self.reranker.rerank(
                query=query,
                docs=texts,
                top_n=top_n,
                batch_size=self.settings.reranker_batch_size,
            )
            reordered = [final_items[i] for i in order]
            if len(final_items) > top_n:
                reordered.extend(final_items[top_n:])
            final_items = reordered

        # Ограничиваем итоговый контекст
        limit = max(1, min(self.top_k, 10))
        final_items = final_items[:limit]

        documents = [it["text"] for it in final_items]
        metadatas = [it.get("metadata", {}) for it in final_items]
        return (documents, metadatas) if return_metadata else (documents, [])

