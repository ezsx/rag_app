## Обзор проекта

Проект `rag_app` — это сервис вопросов-ответов (QA) на базе подхода Retrieval-Augmented Generation (RAG) с поддержкой ReAct агентов. Он сочетает полнотекстовый и семантический поиск по коллекции документов (ChromaDB, BM25) и генерацию ответа LLM, предоставляя REST API через FastAPI.

### Назначение
- Предоставление API для поиска и ответа на вопросы с использованием внешнего контекста.
- Поддержка гибридного поиска (BM25 + эмбеддинги) и опционального переранжирования.
- Планирование запросов (Query Planner) для повышения точности за счёт декомпозиции запроса.
- **ReAct агенты** с пошаговым мышлением и использованием инструментов для решения сложных задач.

### Основные сервисы и компоненты
- `src/main.py`: точка входа FastAPI, CORS, маршруты v1, глобальная обработка ошибок.
- `src/api/v1/`: роутер и конечные точки: `qa`, `search`, `models`, `collections`, `ingest`, `system`, **`agent`**.
- `src/core/settings.py`: конфигурация (модели, Chroma, кеши, гибрид/ммр/ререйк, агенты), горячая смена моделей.
- `src/core/deps.py`: фабрики зависимостей (`get_llm`, `get_retriever`, `get_qa_service`, `get_query_planner`, `get_reranker`, **`get_agent_service`** и т.д.).
- `src/services/qa_service.py`: сбор контекста, промптинг и генерация ответа/стриминга.
- `src/services/query_planner_service.py`: построение плана поиска, кеширование планов и результатов слияния.
- `src/services/reranker_service.py`: переранжирование кандидатов (BAAI/bge-reranker-v2-m3).
- **`src/services/agent_service.py`**: ReAct агент с пошаговым мышлением и SSE стримингом.
- **`src/services/tools/`**: инструменты агента (router_select, compose_context, fetch_docs, verify, math_eval, time_now и др.).
- `src/adapters/chroma/retriever.py`: доступ к ChromaDB (HTTP/локально), эмбеддинг‑поиск.
- `src/adapters/search/*`: BM25 индекс/ретривер и гибридный ретривер.
- `src/utils/*`: загрузка/кеширование моделей, сбор промпта, ранжирование (RRF, MMR).
- Данные индекса: `bm25-index/`, векторное хранилище: `chroma-data/`.

### Инварианты и ключевые свойства
- LLM и ретриверы создаются лениво через `lru_cache`; при смене настроек кеши сбрасываются.
- API устойчив к падению зависимостей при старте (warmup опционален, ошибки логируются).
- Планировщик и фьюжн используют встроенный TTL‑кеш (план ~10 мин, фьюжн ~5 мин) при включённом кешировании.
- Гибридный поиск и MMR/ререйк опциональны и настраиваются через переменные окружения.
- Основной контекст хранится в ChromaDB коллекции, имя коллекции — часть конфигурации и может переключаться «на лету».
- **ReAct агенты** используют ToolRunner с таймаутами и JSON-трейсом, поддерживают fallback через QAService.
- **Инструменты агента** регистрируются в deps.py с инъекцией зависимостей (retriever, settings и др.).

### Безопасность и платформа (актуально)
- JWT‑аутентификация и авторизация: `src/core/auth.py`; обязательная проверка в `agent` эндпойнтах.
- Rate limiting middleware с экспоненциальным бэкоффом: `src/core/rate_limit.py`; экспонирование заголовков в CORS.
- Валидация и санитизация ввода (`SecurityManager`): `src/core/security.py`; защита от prompt‑injection, ограничение размера/глубины JSON.
- Санитизация при логировании через `sanitize_for_logging`.

### ReAct агент и инструменты (актуально)
- Decoding‑профиль llama.cpp: temperature=0.2–0.4, top_p=0.9, top_k=40, repeat_penalty=1.2, seed=42.
- Lost‑in‑the‑Middle mitigation и `citation_coverage` в `compose_context`.
- Зарегистрированные инструменты в `deps.py`: router_select, compose_context, fetch_docs, verify, math_eval, time_now, multi_query_rewrite, web_search, temporal_normalize, summarize, extract_entities, translate, fact_check_advanced, semantic_similarity, content_filter, export_to_formats.

### Соответствие research/playbook
- Единые параметры декодирования, ограничение шагов ReAct, таймауты инструментов.
- Кеширование плана/фьюжна в указанные TTL.
- Контроль качества: акцент на покрытие цитат и стабильность форматов.


