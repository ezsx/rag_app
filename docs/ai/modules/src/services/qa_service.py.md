### Модуль `src/services/qa_service.py`

Назначение: основной RAG‑сервис (retrieval + prompt building + LLM ответ/стрим).

Ключевые элементы:
- Ленивая загрузка LLM через фабрику
- Настройки decoding для ответов: temperature=0.3, top_p=0.9, stop‑последовательности
- Поддержка Planner (SearchPlan) и Hybrid retriever; кеш Fusion via planner

Основные методы:
- `answer(query) -> str` — синхронный ответ
- `answer_with_context(query) -> Dict` — с контекстом и метаданными
- `stream_answer(query, include_context: bool)` — SSE‑подобный стрим

Интеграции: `compose_context`, `rrf_merge`, `mmr_select`, опц. reranker CPU.





