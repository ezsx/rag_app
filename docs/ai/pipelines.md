## Пайплайны: ingest, индексация, поиск, rerank

Ниже описаны ключевые сценарии обработки данных и запросов в `rag_app`.

### 1) Ingest (загрузка исходных данных)
- Источник: внешние каналы/файлы (напр., `scripts/ingest_telegram.py`). См. модульную страницу: `docs/ai/modules/scripts/ingest_telegram.py.md`.
- Очистка/нормализация: преобразование в текст + метаданные (канал, дата, id, views и т.п.).
- Эмбеддинг: получение векторных представлений (совместимо с выбранной embedding‑моделью).
- Запись:
  - Векторное хранилище: коллекция в ChromaDB (`CHROMA_COLLECTION`).
  - BM25: офлайновая индексация (`bm25-index/`) через `BM25IndexManager`.

Результат: синхронизированные представления данных — dense (Chroma) и sparse (BM25).

### 2) Индексация (BM25)
- Компоненты: `src/adapters/search/bm25_index.py`, `BM25IndexManager`.
- Хранение: директория `bm25-index/<collection>/...` (файлы индекса).
- Обновления: переиндексация по расписанию/после ingest; быстрые перезагрузки через менеджер индекса (минимальный интервал контролируется настройкой).

### 3) Поиск (Dense/Hybrid)
- Вход: пользовательский запрос.
- Query Planner (опционально): `QueryPlannerService.make_plan` — нормализация запроса в подзапросы, k_per_query, фильтры.
- Извлечение кандидатов:
  - Dense: `adapters/chroma/Retriever.search` по каждому подзапросу.
  - Hybrid (если включён): `HybridRetriever` объединяет BM25 и dense результаты.
- Слияние кандидатов: `utils.ranking.rrf_merge` (RRF).
- Диверсификация (опционально): `utils.ranking.mmr_select` (MMR) по эмбеддингам.
- Переранжирование (опционально): `services.reranker_service.RerankerService`.
- Отбор Top‑K: ограничение по конфигурации/параметрам сервиса.

Итог: список контекстов (тексты + метаданные) для промпта.

### 4) Генерация ответа (LLM)
- Построение промпта: `utils.prompt.build_prompt(query, contexts)`.
- Вызов LLM: фабрика `get_llm()` (через `core.deps`) создаёт `llama_cpp.Llama` с параметрами из `Settings`.
- Режимы:
  - Синхронный ответ: `QAService.answer`.
  - Стриминг ответа: `QAService.stream_answer` (итеративная отдача токенов).

### 5) Кеширование и устойчивость
- DI‑уровень: `@lru_cache` для клиентов (Chroma, LLM, планировщик, сервисы).
- Query Planner: TTL‑кеш планов и результатов слияния (при включённом `ENABLE_CACHE`).
- Redis (опционально): подключается через `get_redis_client` для хранения ответов/планов.
- Fallback’и: Chroma HTTP → локальный Persistent, Planner LLM → основная LLM.

### 6) Конфигурация пайплайнов (основные параметры)
- Коллекция: `CHROMA_COLLECTION` — активная коллекция поиска.
- Модели: `LLM_MODEL_KEY`/`PATH`, `EMBEDDING_MODEL_KEY`, `RERANKER_MODEL_KEY`.
- Гибрид/слияние: `HYBRID_ENABLED`, `FUSION_STRATEGY` (rrf/mmr), `K_FUSION`.
- MMR: `ENABLE_MMR`, `MMR_LAMBDA`, `MMR_TOP_N`, `MMR_OUTPUT_K`.
- Ререйк: `ENABLE_RERANKER`, `RERANKER_TOP_N`, `RERANKER_BATCH_SIZE`.
- Query Planner: `ENABLE_QUERY_PLANNER`, `SEARCH_K_PER_QUERY_DEFAULT`, `MAX_PLAN_SUBQUERIES`.


