import logging
from typing import List, Dict, Any, AsyncIterator, Optional, Tuple
import numpy as np
from utils.prompt import build_prompt
from core.settings import Settings, get_settings
from services.query_planner_service import QueryPlannerService
from utils.ranking import rrf_merge, mmr_select, _get_item_id

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
        reranker: Optional[object] = None,
        hybrid: Optional[object] = None,
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
        logger.info(f"QAService инициализирован с top_k={top_k}")

    def _get_llm(self):
        if self._llm_instance is None and self._llm_factory is not None:
            logger.info("Ленивая загрузка LLM по первому запросу…")
            self._llm_instance = self._llm_factory()
            logger.info("LLM загружена и готова к использованию")
        return self._llm_instance

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
            response = self._get_llm()(
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

            response = self._get_llm()(
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

            stream = self._get_llm()(
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
                # Если включен гибрид — используем его единый список кандидатов
                merged_items = []
                if self.settings.hybrid_enabled and self.hybrid is not None:
                    try:
                        candidates = self.hybrid.search_with_plan(query, plan)
                        merged_items = [
                            {
                                "id": c.id,
                                "text": c.text,
                                "metadata": c.metadata,
                                "distance": 0.0,
                            }
                            for c in candidates
                        ]
                    except Exception as e:
                        logger.warning(
                            f"Hybrid retriever failed, fallback to dense-only: {e}"
                        )
                        merged_items = []

                if not merged_items:
                    # Сбор результатов поиска для каждого подзапроса (dense‑ветка)
                    results_for_fusion: List[
                        List[Tuple[str, float, Dict[str, Any]]]
                    ] = []
                    items_by_id: Dict[str, Dict[str, Any]] = {}
                    for q in plan.normalized_queries:
                        items = self.retriever.search(
                            q,
                            k=plan.k_per_query,
                            filters=(
                                plan.metadata_filters.dict(exclude_none=True)
                                if plan.metadata_filters
                                else None
                            ),
                        )
                        triples: List[Tuple[str, float, Dict[str, Any]]] = []
                        for it in items:
                            doc = it.get("text", "")
                            dist = float(it.get("distance", 0.0))
                            meta = it.get("metadata", {})
                            triples.append((doc, dist, meta))
                            item_id = _get_item_id(doc, meta)
                            if item_id not in items_by_id:
                                items_by_id[item_id] = it
                        results_for_fusion.append(triples)

                    merged = rrf_merge(results_for_fusion, k=self.settings.k_fusion)
                    # Кешируем слитый список (как tuples) на 5 минут
                    self.planner.set_cached_fusion(fusion_key, merged, ttl=300)

                    # Преобразуем merged к полным Item через items_by_id
                    merged_items = []
                    for doc, dist, meta in merged:
                        iid = _get_item_id(doc, meta)
                        item = items_by_id.get(iid) or {
                            "id": iid,
                            "text": doc,
                            "metadata": meta,
                            "distance": float(dist),
                        }
                        merged_items.append(item)

            # Если есть кешированный merged без items_by_id, нам нужны эмбеддинги top-N для MMR/ререйка
            # Соберем embeddings для первых N если отсутствуют
            def ensure_embeddings(items: List[Dict[str, Any]], top_n: int) -> None:
                need_indices = [
                    i
                    for i in range(min(len(items), top_n))
                    if "embedding" not in items[i]
                ]
                if need_indices:
                    try:
                        embs = self.retriever.embed_texts(
                            [items[i]["text"] for i in need_indices]
                        )
                        for j, i in enumerate(need_indices):
                            items[i]["embedding"] = np.asarray(embs[j], dtype=float)
                    except Exception:
                        # продолжим без эмбеддингов — далее MMR выбросит понятную ошибку
                        pass

            final_items: List[Dict[str, Any]] = []

            # MMR (опционально)
            if self.settings.enable_mmr and merged_items:
                top_n = min(len(merged_items), self.settings.mmr_top_n)
                ensure_embeddings(merged_items, top_n)
                # Эмбеддинг запроса (E5 query: префикс)
                try:
                    query_emb = self.retriever.embed_texts([f"query: {query}"])[0]
                except Exception:
                    query_emb = None
                if query_emb is None:
                    raise ValueError("Не удалось получить эмбеддинг запроса для MMR")
                docs_embs: List[np.ndarray] = []
                for it in merged_items[:top_n]:
                    emb = it.get("embedding")
                    if emb is None:
                        raise ValueError("Отсутствуют эмбеддинги документов для MMR")
                    docs_embs.append(np.asarray(emb, dtype=float))
                candidates = [
                    {
                        "id": it.get("id"),
                        "text": it.get("text"),
                        "score": float(it.get("distance", 0.0)),
                        "metadata": it.get("metadata", {}),
                    }
                    for it in merged_items[:top_n]
                ]
                selected = mmr_select(
                    candidates=candidates,
                    query_embedding=np.asarray(query_emb, dtype=float),
                    doc_embeddings=np.vstack(docs_embs),
                    lambda_=self.settings.mmr_lambda,
                    out_k=min(self.settings.mmr_output_k, len(candidates)),
                )
                # Восстанавливаем полные items по id, сохраняя порядок MMR
                id_to_item = {it.get("id"): it for it in merged_items}
                selected_ids_in_order = [c.get("id") for c in selected]
                final_items = [
                    id_to_item[i] for i in selected_ids_in_order if i in id_to_item
                ]
            else:
                final_items = merged_items

            # Ререйкер (опционально)
            if self.settings.enable_reranker and self.reranker and final_items:
                top_n = min(len(final_items), self.settings.reranker_top_n)
                texts = [it["text"] for it in final_items[:top_n]]
                order = self.reranker.rerank(
                    query=query,
                    docs=texts,
                    top_n=top_n,
                    batch_size=self.settings.reranker_batch_size,
                )
                # Переупорядочиваем первые top_n, остальные добавляем в конце
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

        # Старый путь
        if return_metadata:
            items = self.retriever.get_context_with_metadata(query, k=self.top_k)
            documents = [item["document"] for item in items]
            metadatas = [item["metadata"] for item in items]
            return documents, metadatas
        else:
            documents = self.retriever.get_context(query, k=self.top_k)
            return documents, []
