## Архитектура и структура модулей

Проект организован как FastAPI‑приложение с чётким разделением на слои: API (роуты), Core (настройки/зависимости), Services (бизнес‑логика), Adapters (интеграции), Utils (вспомогательные функции), а также тесты и данные индексов.

### Обзор директорий (по list_files и анализу)
- `src/main.py` — инициализация приложения, CORS, подключение роутера v1, глобальные обработчики.
- `src/api/v1/router.py` — объединение эндпоинтов; `src/api/v1/endpoints/*.py` — модули: `qa`, `search`, `models`, `collections`, `ingest`, `system`, **`agent`**.
- `src/core/settings.py` — объект `Settings` с параметрами окружения и методами «горячего» переключения LLM/эмбеддинга/коллекции; добавлены настройки агентов.
- `src/core/deps.py` — фабрики: `get_chroma_client`, `get_retriever`, `get_llm`, `get_planner_llm`, `get_query_planner`, `get_bm25_index_manager`, `get_qa_service`, `get_reranker`, **`get_agent_service`**, `release_llm_vram_temporarily`.
- `src/services/qa_service.py` — основной сервис QA: сбор контекста, промптинг, синхронный ответ и стриминг.
- `src/services/query_planner_service.py` — построение поискового плана, TTL‑кеш планов и результатов слияния (RRF/MMR), JSON‑контракты; поддержка строгой генерации через GBNF (`utils/gbnf.py`) и микро‑грамматики для догена подзапросов.
- `src/services/reranker_service.py` — обёртка над BAAI/bge‑reranker‑v2‑m3 для переранжирования.
- **`src/services/agent_service.py`** — ReAct агент с пошаговым мышлением, SSE стримингом и управлением инструментами.
- **`src/services/tools/`** — инструменты агента с единым ToolRunner API: `router_select`, `compose_context`, `fetch_docs`, `dedup_diversify`, `verify`, `math_eval`, `time_now`.
- `src/adapters/chroma/retriever.py` — доступ к ChromaDB (HTTP/Persistent), методы поиска/эмбеддинга.
- `src/adapters/search/bm25_*` — офлайновый BM25 индекс и ретривер; `hybrid_retriever.py` объединяет BM25 и dense‑поиск.
- `src/utils/` — `model_downloader.py` (автоскачивание GGUF/эмбеддингов/ререйкера), `prompt.py` (сбор промпта), `ranking.py` (RRF, MMR).
- Данные: `bm25-index/` (офлайн индекс), `chroma-data/` (векторное хранилище), `models/` (GGUF и HF кэши).

### Связи и зависимости (упрощённо)
- API → Services: эндпоинты вызывают `QAService`, `QueryPlannerService`, `BM25Retriever`, `HybridRetriever`, **`AgentService`** через зависимости из `core.deps`.
- Services → Adapters: `QAService` использует `Retriever` (Chroma) и опционально `HybridRetriever`, `RerankerService`; **`AgentService`** оркестрирует инструменты через `ToolRunner`.
- **Tools → Services/Adapters**: инструменты агента используют `Retriever`, `QAService` и другие компоненты через инъекцию зависимостей.
- Core → External: `deps` создаёт клиентов `chromadb`, `llama_cpp.Llama`, **`ToolRunner`** и управляет кешами через `lru_cache`.
- Utils → External: загрузка моделей через HuggingFace/локальные файлы; математика — `numpy`.

### Диаграмма потоков (высокоуровнево)

#### Классический QA поток:
1. HTTP запрос (FastAPI) → роут `v1/qa`.
2. DI (`core.deps`) создаёт/получает из кеша ретривер, LLM‑фабрику, планировщик, ререйкер.
3. `QAService`:
   - Планировщик (если включён) → нормализует запросы → собирает кандидатов (dense/hybrid).
   - Слияние (RRF), опционально MMR и ререйкер → топ‑K контекст.
   - `utils.prompt.build_prompt` → вызов LLM (`llama_cpp`) → ответ/стриминг.

#### ReAct Agent поток:
1. HTTP запрос (FastAPI) → роут `v1/agent/stream`.
2. DI создаёт `AgentService` с настроенным `ToolRunner` и зарегистрированными инструментами.
3. **ReAct петля** (SSE стриминг):
   - **Thought**: LLM размышляет о следующем действии.
   - **Action**: парсинг и вызов инструмента через `ToolRunner`.
   - **Observation**: результат выполнения инструмента.
   - **Повтор** до получения `FinalAnswer` или достижения `max_steps`.
4. **Fallback**: при превышении лимита шагов — использование обычного `QAService`.

### Конфигурация и инварианты
- Все тяжёлые зависимости обёрнуты `@lru_cache` и создаются лениво.
- Смена модели/эмбеддинга/коллекции сбрасывает соответствующие кеши (включая `AgentService`).
- План/фьюжн кешируются в памяти (TTL), Redis поддерживается опционально.
- Фолбэки: Chroma HTTP → локальный Persistent; Planner LLM → основная LLM при ошибках; **Agent max_steps → QAService**.
- **Инструменты агента** изолированы через `ToolRunner` с индивидуальными таймаутами и JSON-трейсингом.


