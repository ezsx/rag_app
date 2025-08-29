```prompt
# Контекст проекта
Мы развиваем RAG‑приложение на FastAPI (Python 3.11). Уже есть:
- Dense‑retrieval на Chroma (E5‑семейство эмбеддингов, префикс `query:`).
- Query Planner (LLM → JSON план → под‑запросы/фильтры).
- Слияние результатов через RRF, далее MMR и CPU‑reranker (bge‑reranker).
- SSE‑стриминг ответов /v1/qa/stream.
- Инжест из Telegram → Chroma.

Цель этого этапа — добавить BM25 (pytantivy) и сделать полноценный гибридный поиск BM25 + Dense:
- Инжест Telegram одновременно пишет в Chroma (dense) и в BM25 (pytantivy).
- Появляется BM25Retriever и HybridRetriever.
- /v1/search и /v1/qa должны проходить поток «План → (BM25&Dense) → RRF → (MMR) → (Rerank) → Ответ».
- Добавляем volume для BM25 в docker‑compose (как для chroma).

Важное: старые документы можно не мигрировать — перекачаем при необходимости.

# Ограничения
- Писать чистый Python 3.11, FastAPI, Pydantic v1, без изменения текущих Dockerfile’ов.
- Разрешено обновить docker‑compose.yml, добавив volume для BM25 индекса.
- Производительность: CPU‑реализация BM25 и ререйкера; VRAM/llama‑часть не трогать.
- Backward‑compatible: при флажке HYBRID_ENABLED=false система работает как раньше (только Dense).
- Сохранять префикс `query:` для E5‑запросов.
- Даты фильтровать в Dense ветке пост‑фильтром по ISO (как сейчас); в BM25 — через числовой `date_days` диапазон.
- Единый формат кандидатов (id, text, metadata, score…), дедуп по id (fallback — hash(text)).

# Что нужно сделать (пошагово)

## 1) Настройки (src/core/settings.py)
Добавить:
- BM25_INDEX_ROOT: str = "./bm25-index"
- HYBRID_ENABLED: bool = True
- HYBRID_TOP_BM25: int = 100
- HYBRID_TOP_DENSE: int = 100
- BM25_DEFAULT_TOP_K: int = 100
- BM25_RELOAD_MIN_INTERVAL_SEC: int = 5   # минимум между reload reader() для API
(остальные флаги MMR/Reranker/Planner уже есть — не изменять)

## 2) docker‑compose.yml — volume под BM25
В сервисах api и ingest добавить общий volume:
- ./bm25-index:/app/bm25-index
(либо на уровне compose volumes, но путь внутри контейнера должен соответствовать BM25_INDEX_ROOT)

## 3) Tantivy индекс: менеджер (src/adapters/search/bm25_index.py)
Создать `BM25IndexManager` c API:
- __init__(index_root: str)
- get_or_create(collection: str) -> IndexHandle
  - IndexHandle содержит: schema, index, writer, reader, searcher, пути.
- add_documents(collection: str, docs: List[BM25Doc]) -> None
  - батчевое добавление; коммитить по N (настройка), логировать.
- reload(collection: str) -> None
  - тро́ттлить по времени (BM25_RELOAD_MIN_INTERVAL_SEC)
- search(collection: str, query: BM25Query, top_k: int) -> List[BM25Hit]

BM25Doc поля (минимум):
- doc_id: str  (формат: f"{channel_id}:{msg_id}")
- text: str    (сырое сообщение после лёгкого клинапа)
- channel_id: int
- channel_username: Optional[str]
- date_days: int  (UTC day number, например floor(ts/86400) или date().toordinal())
- date_iso: str (YYYY‑MM‑DD)
- views: Optional[int]
- reply_to: Optional[int]
- msg_id: int

BM25Query:
- must_terms: List[str]     # must (из плана — must_phrases)
- should_terms: List[str]   # should (из плана — should_phrases + normalized_queries при желании)
- filters:
  - channel_usernames?: List[str]
  - channel_ids?: List[int]
  - reply_to?: Optional[int]
  - min_views?: Optional[int]
  - date_from_days?: Optional[int]
  - date_to_days?: Optional[int]

Схема Tantivy:
- doc_id: STRING, STORED (keyword)
- text: TEXT, STORED (анализатор для русского: lowercase + stopwords(ru) + stemmer(ru) — выбрать доступный пайплайн pytantivy; если стеммера нет, оставить lowercase+stopwords)
- channel_id: U64 FAST, STORED
- channel_username: STRING, STORED (keyword)
- date_days: I64 FAST, STORED
- date_iso: STRING, STORED
- views: U64 FAST, STORED
- reply_to: U64 FAST, STORED
- msg_id: U64 STORED

Поиск:
- BooleanQuery:
  - must: TermQueries для must_terms в поле text; фильтры (Term/Range) как must.
  - should: TermQueries для should_terms (boost можно не трогать на MVP).

Результат BM25Hit:
- doc_id, text, metadata (dict со всеми полями), bm25_score (float)

## 4) BM25 Retriever (src/adapters/search/bm25_retriever.py)
Класс `BM25Retriever`:
- __init__(index_manager, settings)
- search(query_text: str, plan: Optional[SearchPlan], k: int) -> List[Candidate]
  - строит BM25Query из плана:
    - must_terms ← plan.must_phrases
    - should_terms ← plan.should_phrases + plan.normalized_queries (на MVP можно просто сложить)
    - filters: usernames / channel_ids / reply_to / min_views / date_from/to → перевод в date_days
  - вызывает index_manager.search(...)
  - нормализует результат к Candidate:
    - id (doc_id), text, metadata, bm25_score, source="bm25"

## 5) Актуализация Dense Retriever (src/adapters/chroma/retriever.py)
- Убедиться, что метод поиска возвращает Candidate:
  - id (doc_id — собирать из metadatas.channel_id + metadatas.msg_id), text, metadata, dense_score=1-distance, source="dense", embedding? (если доступно)
- Префикс `query:` для E5 — уже обязателен.
- Фильтры:
  - where: channel / reply_to / min_views — по мере поддержки версии Chroma; даты — только пост‑фильтр по ISO.
- Добавить метод `embed_texts(texts: List[str]) -> np.ndarray` (CPU) — для MMR, если embeddings не пришли из Chroma.

## 6) Гибридный retriever (src/adapters/search/hybrid_retriever.py)
Класс `HybridRetriever`:
- __init__(bm25_retriever, dense_retriever, ranking_utils, settings)
- search_with_plan(query_text: str, plan: SearchPlan) -> List[Candidate]
  - Вызывает bm25.search(..., k=HYBRID_TOP_BM25) и dense.search(..., k=HYBRID_TOP_DENSE across normalized queries с вашим уже существующим RRF между под‑запросами).
  - Делает RRF поверх объединённых списков BM25 и Dense → дедуп по id.
  - Возвращает список Candidate (ranked), без MMR/Ререйкера — их применит верхний сервис.

## 7) Инжест Telegram (scripts/ingest_telegram.py)
- После успешного добавления батча в Chroma:
  - Сформировать список BM25Doc (см. схему).
  - Лёгкий клинап текста: убрать ведущие/хвостовые пробелы, collapse whitespace; эмодзи/ссылки/boilerplate — пока не вырезаем глубоко (это в будущем).
  - Вызвать `BM25IndexManager.add_documents(collection, docs)`; коммитить пакетно.
- channel_username можно получать из Telethon (если недоступно — пустая строка).
- Итог: Chroma и BM25 пополняются синхронно; ошибки логировать; если Chroma упала — не пишем в BM25.

## 8) Интеграция в сервисы (src/services/qa_service.py и src/api/v1/endpoints/search.py)
- В ранней фазе (до MMR/Ререйкера) выбрать стратегию:
  - На MVP — если HYBRID_ENABLED=true → использовать HybridRetriever.
  - Иначе — старый DenseRetriever.
- Поток (оба эндпоинта):
  1) Планировщик (если ENABLE_QUERY_PLANNER)
  2) Поиск (bm25/dense/hybrid) → список Candidate
  3) RRF (если собирали несколько списков на предыдущем шаге) → дедуп
  4) MMR (если ENABLE_MMR): подготавливаем query_embedding (E5), doc_embeddings (из retriever либо embed_texts() top‑N), вызываем mmr_select(...)
  5) Ререйкер (если ENABLE_RERANKER): берём top_N кандидатов, вызываем reranker.rerank(query, docs, ...) → переупорядочиваем
  6) Обрезаем до размера контекста (6–10), формируем prompt → LLM (в /v1/qa и SSE).
- /v1/search должен по‑прежнему поддерживать plan_debug (вернуть план), и вернуть итоговые {documents, metadatas}.

## 9) Единый формат Candidate
Определить dataclass/TypedDict (например в src/schemas/search.py или utils/types.py):
- id: str        # doc_id
- text: str
- metadata: Dict[str, Any]
- bm25_score: Optional[float] = None
- dense_score: Optional[float] = None
- source: Literal["bm25","dense","hybrid"]

Все retriever’ы должны возвращать такой список.

## 10) Рейтинг/слияние (уточнение)
- RRF: на уровне гибридного retriever (BM25 vs Dense) и/или внутри Dense (между под‑запросами).
- MMR: после объединения и дедупа; вход — query_embedding + doc_embeddings для top‑MMR_TOP_N; output — MMR_OUTPUT_K.
- Ререйкер: забирает первые RERANKER_TOP_N после MMR/или RRF, возвращает переупорядоченные индексы.

## 11) Индексация и reload
- BM25IndexManager должен поддерживать:
  - инициализацию индекса, создание схемы при отсутствии.
  - метод reload(collection) с троттлингом (reader.reload()) — вызывать в API, но не чаще чем раз в BM25_RELOAD_MIN_INTERVAL_SEC.
- Writer commit: по батчам; логика инициализации writer на запись — только в ingest процессе.

## 12) Тесты (скелет)
- Юнит: bm25_index — создать tmp‑индекс, записать N документов, проверить must/should/filters/date_range.
- Юнит: bm25_retriever — мок план → must/should → непустой результат.
- Интеграция: hybrid_retriever — смержить два списка моков через RRF, дедуп.
- Интеграция: /v1/search — с планом (must/should/filters) → гибридный флоу отдаёт адекватный набор; plan_debug возвращает план.
- E2E: /v1/qa — контекст из гибридного флоу, SSE токены приходят; latency в рамках бюджета.

# Acceptance Criteria
- Инжест Telegram пополняет и Chroma, и BM25 (батчами, с логами).
- /v1/search и /v1/qa используют гибридный поиск при HYBRID_ENABLED=true (BM25_TOP=100, DENSE_TOP=100 по умолчанию).
- Фильтры по каналам/реплаю/просмотрам работают в BM25 через BooleanQuery; даты — через date_days‑диапазон. В Dense ветке даты остаются пост‑фильтром ISO.
- Слияние: RRF → дедуп → (MMR) → (Reranker) — единый поток; формат Candidate консистентный.
- В docker‑compose добавлен volume для BM25 индекса.
- При отключенном HYBRID_ENABLED всё ведёт себя как раньше (только Dense).

# Коммиты (серия — короткие, содержательные)
1) feat(settings): BM25/HYBRID flags and paths + docker-compose volume  
2) feat(bm25): BM25IndexManager (pytantivy) with schema, add/search/reload  
3) feat(retrievers): BM25Retriever + Candidate type; unify DenseRetriever output  
4) feat(hybrid): HybridRetriever (BM25 + Dense) with RRF + dedup  
5) feat(ingest): Telegram ingestor writes to BM25 (batched commits)  
6) feat(qa/search): integrate hybrid flow into QAService and /v1/search  
7) test: unit/integration tests for bm25 & hybrid  
8) docs: README update (BM25 index, volume, hybrid usage)

# Примечания
- Очистка текста (клинап) в ingest сейчас минимальная: trim + collapse whitespace (глубокий клинап/дедубликация новостей — в следующих спринтах).
- На уже залитые коллекции можно не обращать внимания; для них перезальём позже.
- Оркестратор (search_orchestrator) пока не вводим; закладываем интерфейсы так, чтобы в будущем было легко добавить Policy Router / ReAct.

Сделай изменения строго по этому плану, без лишних зависимостей. Код — чистый, типизированный, с логами стадий и аккуратной обработкой ошибок.```