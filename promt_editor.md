# Контекст проекта
Мы развиваем RAG‑приложение на FastAPI (Python 3.11). SSE для стриминга уже реализован. Сейчас хотим улучшить качество ретрива за счёт «планирования» семантических запросов перед векторным поиском (Query Planner).

Текущая структура (важное):
- src/api/v1/endpoints/qa.py — REST + SSE для /v1/qa
- src/services/qa_service.py — отвечает на вопросы (retriever + prompt + llama-cpp)
- src/adapters/chroma/retriever.py — работа с Chroma (векторный поиск)
- src/core/deps.py — DI (llm, retriever, qa_service)
- src/utils/prompt.py — сборка промптов
- src/schemas/qa.py — Pydantic-схемы
- Dockerfile.api / docker-compose.yml — уже корректные (НЕ МЕНЯТЬ)

LLM: llama-cpp-python (Vikhr‑7B‑instruct GGUF).  
Embeddings: intfloat/multilingual-e5-large (совместимы с e5‑стилем “query: …”).

# Цель задачи
Добавить слой «Query Planner» перед ретривером, который:
- принимает пользовательский запрос,
- строит нормализованный поисковый план (под‑запросы, must/should фразы, фильтры по метаданным),
- затем ретривер выполняет несколько поисков и «сливает» результаты (RRF, опционально MMR),
- (опционально) reranker пересортировывает top‑N.

# Что нужно сделать (пошагово)

1) Добавить флаги и настройки
- В `src/core/settings.py`:
  - ENABLE_QUERY_PLANNER: bool = true
  - FUSION_STRATEGY: Literal["rrf","mmr"] = "rrf"
  - K_FUSION: int = 60 (параметр rrf)
  - ENABLE_RERANKER: bool = false (плейсхолдер на будущее)
  - SEARCH_K_PER_QUERY_DEFAULT: int = 10
  - MAX_PLAN_SUBQUERIES: int = 5
- Учесть доступ к этим флагам в `deps.py` и сервисах (через Depends(get_settings)).

2) Схемы для поиска и плана
- Создать `src/schemas/search.py`:
  - SearchPlan (вых. план)
    - normalized_queries: List[str]
    - must_phrases: List[str] = []
    - should_phrases: List[str] = []
    - metadata_filters: Optional[dict] (channel_usernames?, channel_ids?, date_from?, date_to?, min_views?, reply_to?)
    - k_per_query: int
    - fusion: Literal["rrf","mmr"]
  - SearchPlanRequest { query: str }
  - SearchRequest { query: str, include_context?: bool = false, plan_debug?: bool = false }
  - SearchResponse { documents: List[str], distances: List[float], metadatas: List[dict], plan?: SearchPlan }
- Валидировать корректность формата дат (ISO), k_per_query > 0, размер normalized_queries ≤ MAX_PLAN_SUBQUERIES.

3) Query Planner Service
- `src/services/query_planner_service.py`:
  - Класс QueryPlannerService(llm, settings).
  - Метод `make_plan(query: str) -> SearchPlan`.
  - LLM‑prompt: строгий JSON‑ответ (без лишнего текста), на русском:
    - Требовать 1–5 нормализованных под‑запросов русским языком.
    - Если есть явные/неявные даты → date_from/date_to (ISO).
    - must/should фразы — ключевые слова.
    - Предлагать разумные метаданные Telegram при наличии намёков (например, «канал РБК» → @rbc).
    - Если неизвестно — поля null/пустые.
  - Жёстко парсить JSON (try/except) → при ошибке: fallback‑план (normalized_queries=[query], k_per_query=SEARCH_K_PER_QUERY_DEFAULT, fusion="rrf").

4) Ретриер: поиск с фильтрами и фьюжн
- В `src/adapters/chroma/retriever.py` добавить метод:
  - `search(query: str, k: int, filters: Optional[dict]) -> Tuple[List[str], List[float], List[dict]]`
    - Передавать фильтры как `where` в Chroma (если поддерживается версией).
    - Перед генерацией эмбеддинга для query — добавлять префикс "query: " (под E5).
- Добавить утилиты для ранжирования `src/utils/ranking.py`:
  - `rrf_merge(list_of_ranked_results, k=60) -> merged_results`
  - (Плейсхолдер) `mmr_merge(results, lambda_, top_n)` — реализовать позже; сейчас оставить NotImplementedError.
- В `retriever` оставить существующие методы (чтобы не ломать обратную совместимость).

5) QAService: интеграция плана
- В `src/services/qa_service.py`:
  - Если `ENABLE_QUERY_PLANNER`:
    1) `plan = planner.make_plan(query)`
    2) Для каждого `q in plan.normalized_queries` вызвать `retriever.search(q, k=plan.k_per_query, filters=plan.metadata_filters)`
    3) Слить результаты через RRF (по rank), вернуть список документов (без дублей, учитывая id если доступны).
  - Если выключен — использовать старый путь (одиночный поиск).
  - Логи: время планирования, количество под‑запросов, время поиска и фьюжна.

6) Эндпоинты API
- `src/api/v1/endpoints/search.py` (новый router):
  - POST `/v1/search/plan` (SearchPlanRequest → SearchPlan) — возвращает только план.
  - POST `/v1/search` (SearchRequest → SearchResponse) — выполняет план (или fallback), возвращает документы и (опционально) `plan` при `plan_debug=true`.
- Подключить router в FastAPI (в `main.py` или индексном модуле роутеров).
- Не ломать текущие `/v1/qa` и SSE‑маршруты.

7) Кэширование (минимум, in‑memory)
- В `QueryPlannerService` (и/или `retriever`) кэшировать:
  - план: ключ `plan:{hash(query)}` на 10 мин
  - результаты фьюжна: `fusion:{hash(query+serialized_plan)}` на 5 мин
- Реализовать простым LRU/TTL словарём (без Redis).
- Флаг `ENABLE_CACHE` в settings (default: true).

8) Тесты
- Добавить `tests/test_query_planner.py`:
  - модель llm замокать → возвращать предсказуемый JSON
  - тест: корректный парсинг, валидация ограничений (≤ MAX_PLAN_SUBQUERIES)
  - тест: некорректный ответ → fallback
- Добавить `tests/test_search_endpoints.py`:
  - `/v1/search/plan` → schema ok
  - `/v1/search` → с plan_debug и без
  - проверка RRF: объединение результатов нескольких под‑запросов
- Время тестов должно быть < 60 сек (мокать llm и chroma, если нужно).

9) Документация
- Обновить README:
  - Кратко описать Query Planner, зачем он.
  - Примеры curl:
    - `POST /v1/search/plan {"query":"..."}`
    - `POST /v1/search {"query":"...", "plan_debug": true}`

# Ограничения и best practices
- НЕ менять Dockerfile.api, Dockerfile.ingest, docker-compose.yml.
- Python 3.11, FastAPI, Pydantic v1 (как сейчас).
- Типы строго, PEP8, без лишних абстракций.
- LLM‑промпт для планировщика: требовать «Только JSON, без комментариев и текста вокруг».
- Если Chroma “where” ограничен версией — фильтры, которые не поддерживаются, игнорировать с предупреждением (лог).
- В RRF учитывать дубликаты (по id метаданных, если доступны; иначе по хэшу текста).
- Производительность: не делать лишних ембеддингов (переприсваивать e5 префикс "query: " только на запросы).
- Backward compatibility: `/v1/qa` должен работать как раньше при `ENABLE_QUERY_PLANNER=false`.

# Acceptance Criteria
- `POST /v1/search/plan` возвращает валидный JSON‑план (даже при плохом ответе LLM — fallback).
- `POST /v1/search` возвращает top‑результаты, объединённые RRF из нескольких под‑запросов.
- Включение/выключение планировщика флагом меняет поведение `/v1/qa`.
- Тесты зелёные: `pytest -q` проходит локально.
- README обновлён (описание, примеры).

# Коммиты (серия)
1) feat(settings): flags for query planner and fusion  
2) feat(schemas): add SearchPlan, SearchRequest/Response  
3) feat(planner): QueryPlannerService with strict JSON parsing + fallback  
4) feat(retriever): filtered search + E5 “query:” prefix + utils.ranking RRF  
5) feat(api): /v1/search/plan and /v1/search endpoints  
6) feat(qa): integrate planner into QAService behind flag  
7) feat(cache): simple in-memory TTL caches for plan and fusion results  
8) test: unit tests for planner and search endpoints  
9) docs: README update (planner and search examples)

Сделай все пункты аккуратно и последовательно, с короткими, содержательными коммитами. Проверяй типы и не меняй существующие Docker‑файлы.