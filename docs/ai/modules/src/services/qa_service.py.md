### Модуль: `src/services/qa_service.py`

Назначение: основной сервис RAG‑QA. Получает контекст (dense/hybrid, планировщик), формирует промпт и вызывает LLM; поддерживает синхронный ответ и SSE‑стриминг.

#### Класс `QAService`
- Инициализация: принимает `retriever`, `llm` (инстанс или фабрика), `top_k`, `settings`, опционально `planner`, `reranker`, `hybrid`.
- `_get_llm()` — ленивая загрузка LLM при первом вызове.
- `answer(query)` — контекст → `build_prompt` → вызов LLM (`llama_cpp`) → возвращает строку.
- `answer_with_context(query)` — как `answer`, но возвращает ещё и использованный контекст.
- `stream_answer(query, include_context)` — асинхронная генерация токенов (итератор для SSE).
- `_fetch_context(query, return_metadata)` — если включён планировщик: собирает результаты по `normalized_queries`, сливает RRF, при необходимости MMR (требует эмбеддинги top‑N) и ререйк (CrossEncoder), ограничивает топ‑K. Fallback — старые методы `retriever.get_context*`.

#### Алгоритмические детали
- Ключи кеша для фьюжна формируются из хеша запроса и плана; TTL управляется в `QueryPlannerService`.
- MMR использует `utils.ranking.mmr_select` и np‑вектора (добавляет эмбеддинги для первых N документов по мере необходимости).
- Ререйк выполняется строго после MMR.





