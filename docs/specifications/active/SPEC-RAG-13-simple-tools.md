# SPEC-RAG-13: Simple Tools + Dynamic Visibility Update

> **Статус**: Active
> **Создан**: 2026-03-23
> **Research basis**: R16-deep §2, R17-deep §1
> **Зависимости**: SPEC-RAG-12 (payload indexes для facet, channel filter, date filter)
> **Review**: GPT-5.4 review 2026-03-23 — fixes applied

---

## Цель

Добавить 4 новых LLM-visible tools в агент. Обновить dynamic visibility и system prompt.

Итого после этой spec: 7 существующих + 4 новых = **11 tools**, max 5 видимых.

**Решение по review**: `read_post` не добавляем — дублирует системный `fetch_docs`.

## Контекст

**Что уже есть**:
- 7 LLM-visible tools: query_plan, search, temporal_search, channel_search, rerank, compose_context, final_answer
- 2 системных: verify, fetch_docs (получает полные тексты по ID — уже покрывает read_post)
- Dynamic visibility в `_get_step_tools()`: hide по фазе + query signals
- `ToolRunner` с registry и sync timeout (`_run_with_timeout` в ThreadPoolExecutor)
- `HybridRetriever` имеет **dedicated event loop** для sync→async bridge (`_run_sync()`)
- `QdrantStore` использует `AsyncQdrantClient` — **все новые tools должны использовать sync bridge**

**Что добавляем**:

| Tool | Qdrant API | Тип | State machine |
|------|-----------|-----|---------------|
| `list_channels` | `facet("channel")` | navigation | НЕ считается search |
| `related_posts` | `query_points(RecommendQuery)` | exploration | post-search only |
| `cross_channel_compare` | `query_points_groups` с prefetch+RRF | **search** | **Инкрементит search_count** |
| `summarize_channel` | `scroll(filter+order_by)` | **search** | **Инкрементит search_count** |

**Критичные решения из review**:
- `cross_channel_compare` и `summarize_channel` = search tools → **инкрементят `search_count`** → разблокируют compose_context/final_answer
- Все tools — **sync функции** (как существующие). Async Qdrant вызовы через `HybridRetriever._run_sync()` или аналогичный bridge в QdrantStore
- `cross_channel_compare` использует **prefetch + fusion (RRF)**, не dense-only — через `query_points_groups` с prefetch
- Visibility: **phase-based groups**, не priority truncation
- Новые tools возвращают данные в формате совместимом с citation pipeline (`hits[]` с `id`, `text`, `meta`)

---

## Что менять

### 1. Новые файлы в `src/services/tools/`

#### `list_channels.py`

```python
def list_channels(
    sort_by: str = "count",  # "count" | "name"
    channel: str | None = None,  # если указан — вернуть только этот канал с count
    hybrid_retriever=None,   # для sync bridge
) -> dict:
    """Возвращает список каналов с количеством постов (point-level counts).
    Если channel указан — фильтрует до одного канала (для "сколько постов в X?").
    Кэшируемый (TTL 1 час). Qdrant Facet API.
    """
    store = hybrid_retriever._store

    async def _facet():
        return await store._client.facet(
            collection_name=store.collection,
            key="channel",
            limit=50,
            exact=True,
        )

    result = hybrid_retriever._run_sync(_facet())
    channels = [{"channel": h.value, "count": h.count} for h in result.hits]

    # Single-channel mode
    if channel:
        match = [c for c in channels if c["channel"] == channel]
        return {"channels": match, "total": 1 if match else 0}

    if sort_by == "name":
        channels.sort(key=lambda x: x["channel"])
    else:
        channels.sort(key=lambda x: -x["count"])
    return {"channels": channels, "total": len(channels)}
```

**Примечание**: counts = point-level (chunks). Для post-level dedup — client-side по `root_message_id`. Для v1 point counts достаточно.

#### `related_posts.py`

```python
from qdrant_client.models import RecommendQuery

def related_posts(
    post_id: str,
    limit: int = 5,
    hybrid_retriever=None,
) -> dict:
    """Находит похожие посты через Qdrant Recommend API.
    Sync: через HybridRetriever._run_sync() bridge.
    """
    store = hybrid_retriever._store

    async def _recommend():
        return await store._client.query_points(
            collection_name=store.collection,
            query=RecommendQuery(positive=[post_id]),
            using="dense_vector",
            limit=limit,
            with_payload=True,
        )

    results = hybrid_retriever._run_sync(_recommend())
    # Формат совместимый с citation pipeline
    hits = [{
        "id": str(p.id),
        "score": float(p.score),
        "text": p.payload.get("text", ""),
        "snippet": (p.payload.get("text", ""))[:200],
        "meta": {
            "channel": p.payload.get("channel"),
            "date": p.payload.get("date"),
            "url": p.payload.get("url"),
        },
    } for p in results.points]
    return {"hits": hits, "source_id": post_id}
```

#### `cross_channel_compare.py`

```python
def cross_channel_compare(
    topic: str,
    date_from: str | None = None,
    date_to: str | None = None,
    max_channels: int = 10,
    posts_per_channel: int = 2,
    hybrid_retriever=None,
) -> dict:
    """Ищет как разные каналы обсуждают одну тему.
    Использует Qdrant query_points_groups с prefetch + RRF fusion.
    СЧИТАЕТСЯ SEARCH → инкрементит search_count.

    API facts:
    - TEIEmbeddingClient.embed_query() — async
    - QdrantStore.SPARSE_VECTOR = "sparse_vector" (не "bm25")
    - QdrantStore.DENSE_VECTOR = "dense_vector"
    - HybridRetriever._run_sync(coro) — sync bridge
    - HybridRetriever._sparse_encoder.embed() — sync (fastembed)
    """
    retriever = hybrid_retriever

    # Embed topic: async через sync bridge
    dense_vector = retriever._run_sync(
        retriever._embedding_client.embed_query(topic)
    )
    # Sparse: fastembed — sync
    sparse = list(retriever._sparse_encoder.embed([topic]))[0]

    # Построить фильтр по дате
    filter_conditions = []
    if date_from:
        filter_conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(gte=date_from))
        )
    if date_to:
        filter_conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(lte=date_to))
        )
    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    store = retriever._store

    async def _grouped_search():
        return await store._client.query_points_groups(
            collection_name=store.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=store.DENSE_VECTOR,  # "dense_vector"
                    limit=100,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                    using=store.SPARSE_VECTOR,  # "sparse_vector"
                    limit=100,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            group_by="channel",
            limit=max_channels,
            group_size=posts_per_channel,
            query_filter=query_filter,
            with_payload=True,
        )

    results = retriever._run_sync(_grouped_search())

    # Формат совместимый с citation pipeline
    all_hits = []
    groups = []
    for group in results.groups:
        posts = []
        for p in group.hits:
            hit = {
                "id": str(p.id),
                "score": float(p.score) if p.score else 0.0,
                "dense_score": float(p.score) if p.score else 0.0,
                "text": p.payload.get("text", ""),
                "snippet": (p.payload.get("text", ""))[:200],
                "meta": {
                    "channel": p.payload.get("channel"),
                    "date": p.payload.get("date"),
                    "url": p.payload.get("url"),
                },
            }
            posts.append(hit)
            all_hits.append(hit)
        groups.append({"channel": group.id, "posts": posts})

    return {
        "hits": all_hits,  # flat list для rerank/compose_context
        "groups": groups,   # grouped для LLM ответа
        "topic": topic,
        "channels_found": len(groups),
    }
```

#### `summarize_channel.py`

```python
def summarize_channel(
    channel: str,
    time_range: str = "week",  # "day" | "week" | "month"
    limit: int = 20,
    hybrid_retriever=None,
) -> dict:
    """Получает посты канала за период в хронологическом порядке.
    Sync: через HybridRetriever._run_sync() bridge.
    СЧИТАЕТСЯ SEARCH → инкрементит search_count.
    """
    from datetime import datetime, timedelta
    store = hybrid_retriever._store

    now = datetime.utcnow()
    delta = {"day": 1, "week": 7, "month": 30}[time_range]
    date_from = (now - timedelta(days=delta)).isoformat()

    async def _scroll():
        results, _ = await store._client.scroll(
            collection_name=store.collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(
                    key="channel", match=models.MatchValue(value=channel)
                ),
                models.FieldCondition(
                    key="date", range=models.DatetimeRange(gte=date_from)
                ),
            ]),
            order_by=models.OrderBy(key="date", direction="asc"),  # хронологический
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return results

    results = hybrid_retriever._run_sync(_scroll())

    # Формат совместимый с citation pipeline.
    # dense_score=1.0 для всех — digest не имеет relevance scoring,
    # но compose_context использует dense_score для coverage.
    # Фиксированное значение 1.0 означает "контекст полезен" —
    # coverage не будет ложно низким.
    hits = [{
        "id": str(p.id),
        "score": 1.0,
        "dense_score": 1.0,  # фиксированный — digest не scoring-based
        "text": p.payload.get("text", ""),
        "snippet": (p.payload.get("text", ""))[:200],
        "meta": {
            "channel": p.payload.get("channel"),
            "date": p.payload.get("date"),
            "url": p.payload.get("url"),
        },
    } for p in results]

    return {
        "hits": hits,
        "channel": channel,
        "period": time_range,
        "post_count": len(hits),
    }
```

### 2. Регистрация tools в `src/services/tools/__init__.py`

Добавить импорты и экспорты новых tools.

### 3. Обновить AGENT_TOOLS в `agent_service.py`

Добавить 4 новых tool schemas:

```python
# После существующих tools:
{
    "type": "function",
    "function": {
        "name": "list_channels",
        "description": (
            "Показывает доступные Telegram-каналы и количество постов. "
            "Используй когда спрашивают какие каналы есть, сколько постов в канале, "
            "или нужно уточнить название. НЕ используй для поиска по содержимому."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Имя конкретного канала (если нужен count одного канала)",
                },
            },
            "required": [],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "related_posts",
        "description": (
            "Находит посты похожие на указанный. "
            "Используй когда нужно расширить контекст: 'ещё такое же', 'похожие посты'. "
            "НЕ используй для первичного поиска — сначала search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "post_id": {"type": "string", "description": "ID исходного поста"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["post_id"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "cross_channel_compare",
        "description": (
            "Сравнивает как разные каналы обсуждают одну тему. "
            "Используй когда пользователь спрашивает 'сравни', 'как разные каналы', "
            "'мнения экспертов о X', 'X vs Y'. "
            "НЕ используй для поиска в одном канале — для этого channel_search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Тема для сравнения"},
                "date_from": {"type": "string", "description": "Начало периода ISO YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "Конец периода ISO YYYY-MM-DD"},
            },
            "required": ["topic"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "summarize_channel",
        "description": (
            "Получает последние посты канала за период для составления сводки. "
            "Используй когда спрашивают 'что нового в канале X', 'дайджест канала'. "
            "НЕ используй для поиска конкретной темы в канале — для этого channel_search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Имя канала из list_channels"},
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month"],
                    "default": "week",
                },
            },
            "required": ["channel"],
        },
    },
},
```

### 4. Обновить `_get_step_tools()` — phase-based visibility

**Подход**: вместо priority truncation — **phase groups**. Каждая фаза агента имеет фиксированный набор tools. Гарантирует что нужные tools всегда доступны.

```python
def _get_step_tools(self, agent_state) -> List[Dict[str, Any]]:
    """Phase-based visibility — фиксированные наборы по фазе агента.

    Фазы:
    1. PRE-SEARCH: planning + search tools (отфильтрованные по signals)
    2. POST-SEARCH: enrichment + synthesis

    API facts:
    - QuerySignals НЕ содержит original_query — используем self._original_query
    - self._query_signals и self._original_query устанавливаются в stream_agent_response()
    """
    search_done = agent_state.search_count > 0
    signals = getattr(self, "_query_signals", None)
    # original_query хранится в AgentService, не в QuerySignals
    original_query = getattr(self, "_original_query", "") or ""
    query_lower = original_query.lower()

    if search_done:
        # POST-SEARCH: rerank, compose_context, final_answer + enrichment
        visible_names = {
            "rerank", "compose_context", "final_answer",
            "related_posts",  # exploration после search
        }
    else:
        # PRE-SEARCH: query_plan + fallback search (всегда)
        visible_names = {"query_plan", "search"}

        # Signal-based: добавляем релевантный specialized search
        if signals:
            if signals.date_from or signals.strategy_hint == "temporal":
                visible_names.add("temporal_search")
            if signals.channels or signals.strategy_hint == "channel":
                visible_names.add("channel_search")
                visible_names.add("summarize_channel")

        # Keyword-based (из original_query, не signals)
        if any(kw in query_lower for kw in
               ["сравни", "compare", "vs", "мнени", "разн", "каналы обсужд"]):
            visible_names.add("cross_channel_compare")

        if any(kw in query_lower for kw in
               ["какие каналы", "список каналов", "сколько каналов", "сколько постов"]):
            visible_names.add("list_channels")

    # Hard cap: если по совпадению сигналов набралось >5 — убираем fallback search
    # (specialized tools приоритетнее generic search)
    if len(visible_names) > 5 and "search" in visible_names:
        specialized = visible_names - {"search", "query_plan"}
        if len(specialized) >= 2:  # есть минимум 2 specialized → safe to remove generic
            visible_names.discard("search")

    return [t for t in AGENT_TOOLS if t["function"]["name"] in visible_names]
```

**Гарантии**:
- PRE-SEARCH: 2-5 tools (query_plan + search/specialized + 0-2 extras). Hard cap через удаление generic fallback
- POST-SEARCH: всегда 4 tools (rerank + compose_context + final_answer + related_posts)
- `cross_channel_compare` и `summarize_channel` = search tools → `search_count++` → POST-SEARCH
- Navigational (list_channels) добавляется только по keyword match, не забивает slots

### 5. Обновить system prompt

```python
SYSTEM_PROMPT = """Ты — RAG-агент для поиска и анализа AI/ML новостей из 36 Telegram-каналов.

ПОРЯДОК РАБОТЫ:
1. query_plan — декомпозируй запрос на подзапросы
2. ВЫБЕРИ ПОДХОДЯЩИЙ инструмент:
   - temporal_search — даты, периоды ("в январе 2026", "на CES 2026")
   - channel_search — конкретный канал/автор ("gonzo_ml", "Себрант")
   - cross_channel_compare — сравнение мнений ("как разные каналы обсуждают X", "X vs Y")
   - summarize_channel — дайджест канала ("что нового в gonzo_ml за неделю")
   - list_channels — навигация ("какие каналы есть")
   - search — общий поиск, entity-запросы, fallback
3. rerank → compose_context → final_answer

ПОСЛЕ ПОИСКА (если нужно):
   - related_posts — найти похожие посты к уже найденному

ПРАВИЛА:
- При сомнении — используй search
- Отвечай ТОЛЬКО на русском
- Каждое утверждение подкрепляй ссылкой [1], [2]
- Если контекст НЕ содержит информации — честно скажи
"""
```

### 6. State machine integration в `_apply_action_state()`

**Критично**: state update живёт в `_apply_action_state()`, не в `_execute_action()`. Нужно расширить существующий `if action.tool in (...)` блок:

```python
# В _apply_action_state() — расширить существующую проверку:
# БЫЛО:
#   if action.tool in ("search", "temporal_search", "channel_search"):
# СТАЛО:
if action.tool in ("search", "temporal_search", "channel_search",
                    "cross_channel_compare", "summarize_channel"):
    self._last_search_hits = list(action.output.data.get("hits", []) or [])
    self._last_search_route = action.output.data.get("route_used")
    self._agent_state.search_count += 1
    self._agent_state.strategy = action.output.data.get("strategy", "broad")
    self._agent_state.routing_source = action.output.data.get("routing_source", "default")
    # ... existing logging ...
    return
```

Новые tools возвращают `{"hits": [...]}` в том же формате что и `search` — `rerank` и `compose_context` работают без изменений.

`list_channels` и `related_posts` **НЕ** попадают в этот блок — не search tools.

### 7. Регистрация в ToolRunner

В `AgentService.__init__()` или `_execute_action()` — добавить обработку новых tool names, пробросить зависимости (qdrant_store, hybrid_retriever).

---

## Acceptance Criteria

1. **11 tools зарегистрированы** в AGENT_TOOLS (7 старых + 4 новых)
2. **list_channels** возвращает 36 каналов с point-level counts через facet API
3. **related_posts** возвращает 5 семантически похожих постов в формате `{"hits": [...]}`
4. **cross_channel_compare** возвращает grouped results по каналам с prefetch+RRF, формат `{"hits": [...], "groups": [...]}`
5. **summarize_channel** возвращает хронологические посты канала за period в формате `{"hits": [...]}`
6. **State machine**: `cross_channel_compare` и `summarize_channel` инкрементят `search_count` → разблокируют compose_context/final_answer
7. **Dynamic visibility**: phase-based groups, 2-4 tools в pre-search, 4 tools в post-search
8. **Citation pipeline**: hits от новых search-tools проходят через rerank → compose_context → final_answer без ошибок
9. **Async/sync**: все tools — sync функции, Qdrant вызовы через `_run_sync()` bridge
10. **System prompt** обновлён
11. **Smoke test**: агент корректно выбирает новые tools (3 запроса: navigational, comparative, digest)
12. **Regression**: 5 старых eval вопросов — recall не упал

---

## Новые eval вопросы (добавить в dataset)

```json
[
  {"id": "v4_nav01", "query": "Какие каналы у тебя есть?", "category": "navigational"},
  {"id": "v4_nav02", "query": "Сколько постов в канале gonzo_ml?", "category": "navigational"},
  {"id": "v4_cmp01", "query": "Как разные каналы освещали выход GPT-5?", "category": "comparative"},
  {"id": "v4_cmp02", "query": "Сравни мнения экспертов о DeepSeek-V3", "category": "comparative"},
  {"id": "v4_dig01", "query": "Что нового в канале techsparks за последнюю неделю?", "category": "digest"},
  {"id": "v4_dig02", "query": "Дайджест llm_under_hood за месяц", "category": "digest"},
  {"id": "v4_exp01", "query": "Покажи похожие посты на тему MoE архитектур", "category": "exploratory"},
  {"id": "v4_exp02", "query": "Найди похожие посты на тему Claude Opus 4.6", "category": "exploratory"}
]
```

---

## Чеклист реализации

### Код — tools (все sync, через _run_sync bridge)
- [ ] `src/services/tools/list_channels.py` — facet API
- [ ] `src/services/tools/related_posts.py` — RecommendQuery
- [ ] `src/services/tools/cross_channel_compare.py` — query_points_groups с prefetch+RRF
- [ ] `src/services/tools/summarize_channel.py` — scroll с filter+order_by
- [ ] Обновить `src/services/tools/__init__.py`

### Код — интеграция
- [ ] Обновить AGENT_TOOLS в `agent_service.py` (4 новых schemas)
- [ ] Обновить `_get_step_tools()` — phase-based visibility
- [ ] Обновить `_apply_action_state()` — добавить cross_channel_compare, summarize_channel в search tools set
- [ ] Обновить SYSTEM_PROMPT
- [ ] Добавить `self._original_query` lifecycle: set в `stream_agent_response()`, clear в `finally`
- [ ] Регистрация в ToolRunner — проброс `hybrid_retriever`
- [ ] Проверить что hits от новых tools совместимы с rerank/compose_context

### Тестирование
- [ ] Каждый tool — unit test (sync вызов, валидный результат)
- [ ] Phase visibility: navigational/comparative/digest/factual запрос → правильные tools
- [ ] State machine: cross_channel_compare → search_count++ → compose_context доступен
- [ ] Citation pipeline: cross_channel_compare → rerank → compose_context → final_answer
- [ ] Agent e2e: 3 запроса через SSE endpoint
- [ ] Regression: 5 старых eval вопросов — recall не упал

### Eval dataset
- [ ] Добавить 6+ новых вопросов (navigational, comparative, digest)
- [ ] Обновить `datasets/eval_dataset_v4.json`

### Документация
- [ ] Обновить `always_on.md` — новые tools
- [ ] Обновить `agent_context/modules/agent.md`
- [ ] Decision log — DEC-XXXX
