# Deep Research: Tool Router + Adaptive Retrieval Architecture

## Резюме рекомендаций

**Рекомендуемый подход**: расширение существующего `query_plan` tool — zero extra LLM calls, минимальные изменения в ReAct loop. Не отдельный router, а обогащённый JSON-output query_plan с полем `strategy` и параметрами фильтрации.

**Набор tools**: 4-5 (не больше). `broad_search` (текущий), `temporal_search`, `channel_search`, `entity_search`. `comparative_search` и `fact_check` — это composites из существующих, не отдельные tools.

**Ожидаемый эффект**: +5-15% recall@5 на v2 категориях (temporal, entity, recency), -20-30% latency на простых запросах за счёт более точной фильтрации и меньшего candidate pool.

---

## 1. Архитектура Tool Router

### 1.1 Анализ вариантов

**LLM-as-router (отдельный call)** — отклонён. Qwen3-30B-A3B с 3B active params — это фактически модель уровня 3B для inference. Отдельный routing call добавит 10-15с latency и создаст точку отказа. При этом accuracy routing'а для 3B модели на сложных запросах будет ~70-80% (по данным Adaptive-RAG paper, даже T5-Large с fine-tuning достигает ~85% accuracy на 3-class classification).

**Lightweight classifier (fine-tuned BERT)** — отклонён для текущего этапа. 20 eval вопросов — недостаточно для обучения. Нужно минимум 200-500 labeled examples. Это хороший next step после накопления данных, но не сейчас.

**Rule-based** — частично применим. Для детекции дат, имён каналов, сравнительных конструкций regex работает надёжно. Но не покрывает семантические случаи ("что нового в NVIDIA" → temporal+entity).

**Рекомендация: Hybrid через обогащённый query_plan.**

### 1.2 Конкретная реализация

Текущий `query_plan` уже вызывается как первый tool и генерирует JSON с subqueries + metadata filters. Расширить его output schema:

```json
{
  "subqueries": ["Meta acquired Manus", "Manus AI acquisition price"],
  "strategy": "entity_search",
  "filters": {
    "date_from": null,
    "date_to": null,
    "channels": [],
    "entities": ["Meta", "Manus"]
  },
  "k": 10,
  "reasoning": "Query asks about specific entities and a factual event"
}
```

Почему это работает лучше отдельного router:

1. **Zero extra LLM calls** — routing интегрирован в существующий query_plan call
2. **LLM видит контекст** — при генерации subqueries модель уже "понимает" запрос, добавление strategy/filters — минимальная дополнительная нагрузка
3. **Graceful degradation** — если модель не заполнит filters корректно, fallback на broad_search (текущее поведение)
4. **Testable** — можно eval'ить routing accuracy отдельно от retrieval quality

### 1.3 Prompt engineering для query_plan

Ключевые изменения в system prompt для query_plan tool:

```
You must output a JSON object with these fields:
- subqueries: list of 2-5 search queries
- strategy: one of "broad", "temporal", "channel", "entity"
- filters: object with date_from (ISO), date_to (ISO), channels (list), entities (list)
- k: number of results (5-20)

Strategy selection rules:
- "temporal": query mentions specific dates, months, periods, "recently", "latest"
- "channel": query mentions a specific channel name or author
- "entity": query asks about specific product, company, person, or technology
- "broad": default for general/comparative/multi-topic queries

ALWAYS extract dates if mentioned. ALWAYS extract channel names if mentioned.
If unsure about strategy, use "broad" with extracted filters.
```

Принцип: rules-in-prompt. Модель получает чёткие if-then правила, а не абстрактные инструкции. Для 3B active params модели это надёжнее, чем просить "decide the best strategy".

### 1.4 Rule-based pre-processing (до LLM call)

Дополнительный deterministic слой перед query_plan:

```python
import re
from datetime import datetime

def extract_hints(query: str) -> dict:
    hints = {"strategy_hint": None, "filters": {}}
    
    # Date patterns (русский + английский)
    date_patterns = [
        r'(?:в |in )?(январ[еёя]|феврал[еёя]|март[еа]?|апрел[еёя]|ма[еёя]|июн[еёя]|'
        r'июл[еёя]|август[еа]?|сентябр[еёя]|октябр[еёя]|ноябр[еёя]|декабр[еёя])\s*(\d{4})',
        r'(january|february|march|april|may|june|july|august|september|'
        r'october|november|december)\s*(\d{4})',
        r'(\d{4})[-./](\d{1,2})',
    ]
    
    # Channel names (из payload)
    known_channels = ['gonzo_ml', 'llm_under_hood', 'ai_newz', 'seeallochnaya', 
                      'boris_again', 'techsparks', 'data_secrets', 'rybolos', ...]
    
    query_lower = query.lower()
    
    for channel in known_channels:
        if channel in query_lower or channel.replace('_', ' ') in query_lower:
            hints["filters"]["channels"] = [channel]
            hints["strategy_hint"] = "channel"
    
    # Temporal hints
    if any(w in query_lower for w in ['последн', 'недавн', 'latest', 'recent', 
                                       'на прошлой неделе', 'last week', 'вчера',
                                       'за последний месяц', 'в 2026', 'в 2025']):
        hints["strategy_hint"] = "temporal"
    
    # Inject hints into query_plan prompt
    return hints
```

Эти hints передаются в prompt query_plan как дополнительный контекст:
```
System detected hints: strategy_hint=temporal, filters.date_from=2026-01-01
Use these hints to guide your strategy selection. You may override if incorrect.
```

Модель может override hints — это не жёсткое ограничение, а подсказка.

---

## 2. Набор специализированных Tools

### 2.1 Оптимальный набор — 4 tools

Исследование "Less is More" (2024) прямо показывает: уменьшение числа доступных tools повышает success rate для малых моделей. Для Qwen2-7b с 4-bit quantization tool accuracy выросла с ~35% (все tools) до 87% (subset). Для 3B active params модели это ещё критичнее.

**Рекомендуемые tools:**

| Tool | Что делает | Когда вызывается |
|------|-----------|-----------------|
| `search` | Текущий broad hybrid search (BM25+Dense→RRF→ColBERT→Rerank) | Всегда — default fallback |
| `temporal_search` | `search` + Qdrant date filter (`must: [{range: {date: {gte, lte}}}]`) | strategy=temporal |
| `channel_search` | `search` + Qdrant channel filter (`must: [{match: {channel: X}}]`) | strategy=channel |
| `entity_search` | `search` с entity-boosted queries + optional channel/date filters | strategy=entity |

**Отклонённые tools:**

- `comparative_search` → это два вызова `search` с разными queries. Реализуется как логика в `search`, а не отдельный tool. Модель итак генерирует multiple subqueries в query_plan.
- `fact_check` → семантически идентичен `search` с другим промптом для final_answer. Не нужен отдельный tool.
- `trending_search` → частный случай `temporal_search` с sort by date. Добавить `sort_by` параметр в temporal_search.

### 2.2 Tool schemas для function calling

```json
{
  "name": "search",
  "description": "Search the knowledge base for AI/ML news and articles. Use when no specific date range or channel filter is needed, or as fallback.",
  "parameters": {
    "type": "object",
    "properties": {
      "queries": {
        "type": "array",
        "items": {"type": "string"},
        "description": "2-5 search queries, diverse phrasings of the user's question"
      },
      "k": {
        "type": "integer",
        "description": "Number of results to return (5-20)",
        "default": 10
      }
    },
    "required": ["queries"]
  }
}
```

```json
{
  "name": "temporal_search",
  "description": "Search within a specific date range. Use for questions about 'what happened in [month/period]', 'latest news about X', 'recent developments'.",
  "parameters": {
    "type": "object",
    "properties": {
      "queries": {
        "type": "array",
        "items": {"type": "string"}
      },
      "date_from": {
        "type": "string",
        "description": "Start date ISO format YYYY-MM-DD"
      },
      "date_to": {
        "type": "string",
        "description": "End date ISO format YYYY-MM-DD"
      },
      "k": {"type": "integer", "default": 15}
    },
    "required": ["queries", "date_from", "date_to"]
  }
}
```

```json
{
  "name": "channel_search",
  "description": "Search within a specific Telegram channel. Use when user mentions a channel name or author (e.g., 'gonzo_ml', 'llm_under_hood', 'techsparks').",
  "parameters": {
    "type": "object",
    "properties": {
      "queries": {
        "type": "array",
        "items": {"type": "string"}
      },
      "channel": {
        "type": "string",
        "description": "Channel name exactly as stored: gonzo_ml, llm_under_hood, ai_newz, etc."
      },
      "k": {"type": "integer", "default": 10}
    },
    "required": ["queries", "channel"]
  }
}
```

```json
{
  "name": "entity_search",
  "description": "Search for information about a specific entity (company, product, person, technology). Combines keyword and semantic search with entity-focused queries.",
  "parameters": {
    "type": "object",
    "properties": {
      "entity": {
        "type": "string",
        "description": "Primary entity name (e.g., 'NVIDIA', 'GPT-5', 'Vera Rubin')"
      },
      "queries": {
        "type": "array",
        "items": {"type": "string"},
        "description": "2-5 queries combining entity with context"
      },
      "date_from": {"type": "string", "description": "Optional start date ISO"},
      "date_to": {"type": "string", "description": "Optional end date ISO"},
      "k": {"type": "integer", "default": 15}
    },
    "required": ["entity", "queries"]
  }
}
```

### 2.3 Реализация на стороне Qdrant

Все tools — это один и тот же `HybridRetriever` с разными `qdrant_filter`:

```python
class AdaptiveRetriever:
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever
    
    async def search(self, queries: list[str], k: int = 10, 
                     filters: dict | None = None) -> list[Document]:
        """Unified search with optional Qdrant filters."""
        qdrant_filter = self._build_filter(filters) if filters else None
        return await self.retriever.search(
            queries=queries, k=k, qdrant_filter=qdrant_filter
        )
    
    async def temporal_search(self, queries: list[str], 
                               date_from: str, date_to: str, k: int = 15):
        filters = {"date_range": {"gte": date_from, "lte": date_to}}
        # Увеличиваем k для BM25 т.к. date filter сужает pool
        return await self.search(queries, k=k, filters=filters)
    
    async def channel_search(self, queries: list[str], 
                              channel: str, k: int = 10):
        filters = {"channel": channel}
        return await self.search(queries, k=k, filters=filters)
    
    async def entity_search(self, entity: str, queries: list[str],
                             date_from: str = None, date_to: str = None, 
                             k: int = 15):
        # Entity name всегда добавляется в queries для BM25 keyword match
        enriched_queries = [entity] + queries
        filters = {}
        if date_from and date_to:
            filters["date_range"] = {"gte": date_from, "lte": date_to}
        return await self.search(enriched_queries, k=k, 
                                  filters=filters if filters else None)
    
    def _build_filter(self, filters: dict) -> models.Filter:
        conditions = []
        if "channel" in filters:
            conditions.append(
                models.FieldCondition(
                    key="channel", 
                    match=models.MatchValue(value=filters["channel"])
                )
            )
        if "date_range" in filters:
            conditions.append(
                models.FieldCondition(
                    key="date",
                    range=models.Range(
                        gte=filters["date_range"].get("gte"),
                        lte=filters["date_range"].get("lte")
                    )
                )
            )
        return models.Filter(must=conditions) if conditions else None
```

### 2.4 Dynamic tool visibility

Текущая система уже скрывает `final_answer` до выполнения search. Расширить:

- **Фаза 1** (до search): показать `search`, `temporal_search`, `channel_search`, `entity_search`, `query_plan`
- **Фаза 2** (после search): скрыть search tools, показать `compose_context`, `final_answer`

Не показывать все 4 search tools одновременно, если rule-based hints однозначны. Если hint говорит strategy=temporal → показать только `temporal_search` + `search` (fallback). Это снижает когнитивную нагрузку на модель.

---

## 3. Интеграция с текущим ReAct Loop

### 3.1 Где вставить router

**Внутри query_plan — не отдельным step.** Текущий flow:

```
query_plan → search → rerank → compose_context → final_answer
```

Новый flow:

```
query_plan (+ strategy + filters) → adaptive_search → rerank → compose_context → final_answer
```

`query_plan` возвращает strategy и filters. `ToolRunner` интерпретирует strategy и вызывает соответствующий метод `AdaptiveRetriever`. С точки зрения ReAct loop ничего не меняется — это тот же один tool call.

### 3.2 Изменения в AgentState

```python
@dataclass
class AgentState:
    search_count: int = 0
    compose_count: int = 0
    coverage: float = 0.0
    # Новые поля
    strategy: str = "broad"  # broad|temporal|channel|entity
    applied_filters: dict = field(default_factory=dict)
    routing_source: str = "llm"  # llm|rules|fallback
```

Зачем `routing_source`: для eval. Позволяет понять, кто принял решение о стратегии — LLM, rule-based hints, или fallback. Это важно для debugging и улучшения routing accuracy.

### 3.3 Fallback механизм

```python
async def execute_search(self, plan: QueryPlan, state: AgentState):
    strategy = plan.strategy
    results = []
    
    # Попытка специализированного поиска
    if strategy == "temporal" and plan.filters.date_from:
        results = await self.retriever.temporal_search(
            queries=plan.subqueries,
            date_from=plan.filters.date_from,
            date_to=plan.filters.date_to
        )
    elif strategy == "channel" and plan.filters.channels:
        results = await self.retriever.channel_search(
            queries=plan.subqueries,
            channel=plan.filters.channels[0]
        )
    elif strategy == "entity" and plan.filters.entities:
        results = await self.retriever.entity_search(
            entity=plan.filters.entities[0],
            queries=plan.subqueries,
            date_from=plan.filters.date_from,
            date_to=plan.filters.date_to
        )
    else:
        results = await self.retriever.search(plan.subqueries)
    
    # Fallback: если мало результатов, broad search
    if len(results) < 3:
        state.routing_source = "fallback"
        broad_results = await self.retriever.search(plan.subqueries)
        results = self._merge_deduplicate(results, broad_results)
    
    return results
```

### 3.4 Multi-strategy запросы

"Что писал gonzo_ml в январе 2026" — это `channel_search` + `temporal_search`. Решение: **не делать мультивызов**. Один tool call с комбинированным фильтром:

```python
# channel_search с date filter
filters = {
    "channel": "gonzo_ml",
    "date_range": {"gte": "2026-01-01", "lte": "2026-01-31"}
}
```

Это реализуется через `entity_search` tool, который уже принимает optional date_from/date_to. Альтернативно — добавить optional date параметры в `channel_search`.

Лучше: сделать все tools с опциональными date_from/date_to и channel фильтрами. Тогда любой tool может комбинировать фильтры.

---

## 4. Влияние на Latency и Reliability

### 4.1 Latency budget

Текущий breakdown:
```
query_plan:     ~12с (LLM inference)
search:         ~5с  (Qdrant + BM25 + Dense + RRF)
ColBERT:        ~2.5с (MaxSim rerank)
cross-encoder:  ~1.5с (bge-reranker)
compose:        ~12с (LLM inference)
final_answer:   ~12с (LLM inference)
Total:          ~45с
```

С adaptive retrieval:
```
query_plan (enriched): ~13с (+1с за дополнительные поля в output)
adaptive_search:       ~3-5с (уменьшен candidate pool за счёт фильтров)
ColBERT:               ~1.5-2.5с (меньше candidates → быстрее)
cross-encoder:         ~1-1.5с (аналогично)
compose:               ~12с (без изменений)
final_answer:          ~12с (без изменений)
Total:                 ~42-46с
```

**Net effect на latency: ±0-3с.** Ожидание "минус 35% latency" из R11 report — это для production систем с быстрым LLM inference (<1с). При 12с/call на V100 экономия на retrieval (2-3с) незначительна относительно суммарной latency.

Реальный выигрыш latency возможен только через:
- `--parallel 2` на llama-server для параллельных subquery searches
- KV cache preloading для system prompt
- Batched reranking (уже реализовано)

### 4.2 Параллельное выполнение

llama-server с `--parallel 2` позволяет 2 concurrent requests. Но tools в ReAct loop вызываются последовательно по дизайну (LLM генерирует один tool call → ждёт результат → генерирует следующий).

Параллелизм можно использовать внутри tool execution:

```python
async def search(self, queries: list[str], ...):
    # Параллельно: BM25 и Dense search по всем queries
    bm25_tasks = [self.bm25_search(q) for q in queries]
    dense_tasks = [self.dense_search(q) for q in queries]
    
    bm25_results, dense_results = await asyncio.gather(
        asyncio.gather(*bm25_tasks),
        asyncio.gather(*dense_tasks)
    )
    # RRF merge...
```

Это уже на уровне Qdrant client, а не LLM. Qdrant обрабатывает запросы быстро (~10-50ms каждый), bottleneck — LLM inference.

### 4.3 Recovery при неправильном routing

Три уровня recovery:

1. **Fallback в execute_search** (описан выше): если специализированный поиск вернул <3 результата → автоматический broad_search.

2. **Coverage-driven refinement** (существующий механизм): если coverage < 0.65 после первого поиска → LLM может вызвать search повторно с другими parameters.

3. **Forced search** (существующий механизм): если LLM не вызывает tools (finish_reason=stop) → принудительный broad_search с оригинальным запросом.

Не добавлять retry с другой стратегией — это +12с latency за каждый retry. Лучше: при fallback использовать broad search с max k=20 и полагаться на reranker для precision.

---

## 5. Production примеры и Evaluation

### 5.1 Релевантные papers с измеримыми результатами

**Adaptive-RAG (Jeong et al., NAACL 2024)**
- Classifier (T5-Large fine-tuned) роутит между no-retrieval, single-step, multi-step
- Accuracy 85-92% для routing classifier
- Улучшение на multi-hop datasets, снижение compute на простых запросах
- Code: github.com/starsuzi/Adaptive-RAG

**RAP-RAG (2025, MDPI)**
- Adaptive planner выбирает между Vector, Local, Topology retrieval
- +2-4% accuracy vs fixed strategy, latency overhead 0.2-0.4с
- Работает с SLMs — заявляется совместимость с малыми моделями

**Production benchmarks (ASCII.co.uk, January 2026)**
- Query-adaptive routing: 35% P50 latency reduction, 8% accuracy improvement
- Complexity classification: 85-92% accuracy с Phi-2, Llama-2-7B at <1ms/query
- Single-hop: 80-120ms, 92-96% precision; Multi-hop: 400-800ms, 85-92% accuracy

**"Less is More" (2024)**
- Reducing available tools improves small LLM function calling dramatically
- Qwen2-7b: tool accuracy 35% (all tools) → 87% (subset of k=5)
- Execution time reduction 21-70% depending on model

**A-RAG (Feb 2026, arxiv 2602.03442)**
- Hierarchical retrieval interfaces, autonomous strategy selection
- Показывает что даже Naive Agentic RAG (простой выбор стратегии) превосходит fixed pipelines

### 5.2 Применимость к данному кейсу

Ключевое отличие: все перечисленные papers работают с open-domain QA (Natural Questions, TriviaQA, HotpotQA). В данном кейсе — closed-domain (13K Telegram posts, AI/ML), что проще для routing:

- Vocabulary ограничен доменом
- Channel names — closed set (36 каналов)
- Date range — 9 месяцев (July 2025 — March 2026)
- Entity types — ограничены AI/ML (companies, models, papers, researchers)

Это значит rule-based routing будет надёжнее, чем в open-domain. Детекция "gonzo_ml" или "январь 2026" — тривиальна. Сложные случаи ("Vera Rubin NVIDIA") требуют entity extraction, но entity_search с BM25 boost по entity name решает это без NER.

### 5.3 Evaluation strategy

**Routing accuracy eval** — отдельный от recall:

```python
# routing_eval.py
routing_test_cases = [
    {"query": "Что писал gonzo_ml про трансформеры", 
     "expected_strategy": "channel", 
     "expected_filters": {"channel": "gonzo_ml"}},
    {"query": "Новости за декабрь 2025", 
     "expected_strategy": "temporal",
     "expected_filters": {"date_from": "2025-12-01", "date_to": "2025-12-31"}},
    {"query": "Сравни GPT-5 и Claude", 
     "expected_strategy": "broad"},
    {"query": "NVIDIA Vera Rubin анонс 2026", 
     "expected_strategy": "entity",
     "expected_filters": {"entities": ["NVIDIA", "Vera Rubin"]}},
]

def eval_routing(query_plan_output, expected):
    strategy_match = query_plan_output.strategy == expected["expected_strategy"]
    filter_match = check_filters(query_plan_output.filters, expected.get("expected_filters", {}))
    return {"strategy_correct": strategy_match, "filters_correct": filter_match}
```

**End-to-end eval**: расширить v2 dataset (20→50+ вопросов) с пометкой ожидаемой стратегии. Измерять recall@5 per-strategy. Это покажет, где routing помогает, а где нет.

**Regression testing**: сохранить текущие v1+v2 результаты как baseline. После внедрения adaptive retrieval прогнать те же 20 вопросов — recall@5 не должен упасть на v1 (где broad search работает хорошо).

---

## 6. Roadmap реализации (3-5 дней)

### День 1: Query plan enrichment
- Расширить JSON schema query_plan (strategy, filters)
- Rule-based hint extraction (dates, channels, entities)
- Обновить system prompt для query_plan
- Тест: 20 запросов через обновлённый query_plan, измерить routing accuracy

### День 2: Adaptive retriever
- `AdaptiveRetriever` class с методами temporal/channel/entity search
- Qdrant filter builder
- Fallback logic (<3 results → broad search)
- Unit tests с mock Qdrant

### День 3: Integration
- Обновить `ToolRunner` для dispatch по strategy
- Обновить `AgentState` (strategy, applied_filters, routing_source)
- Dynamic tool visibility (сузить tool set по strategy hint)
- E2E test: 5-10 запросов через полный pipeline

### День 4: Evaluation
- Расширить eval dataset (v2 → v3, 30+ вопросов с strategy labels)
- Routing accuracy eval
- Recall@5 per-strategy breakdown
- A/B comparison: adaptive vs broad-only

### День 5: Buffer
- Отладка edge cases
- Tune BM25 weight / k parameters для filtered searches
- Документация, commit

---

## 7. Решение конкретных провалов v2

| Провал | Корневая причина | Как adaptive retrieval помогает |
|--------|-----------------|-------------------------------|
| Q1: entity (Карпаты) | data_secrets:8021 не в candidate pool | `entity_search("Карпаты")` + BM25 keyword boost по entity name |
| Q3: fact_check (лицензия) | rybolos:1562 не найден | `entity_search` с entity = ключевое слово лицензии |
| Q6: recency (NVIDIA 2026) | LLM не знает "Vera Rubin" | Rule-based: "2026" → `temporal_search` + "NVIDIA" → entity. Даже без знания "Vera Rubin", temporal+entity filter сузит pool |
| Q7: numeric (Deep Think цена) | seeallochnaya:2711 не в pool | `entity_search("Deep Think")` — entity name как BM25 keyword |
| Q8: long_tail (Kandinsky) | Правильный канал, fuzzy ±5 strict | Не решается routing — это проблема eval metric tolerance |

Q6 — показательный кейс. Текущий pipeline: LLM не знает "Vera Rubin" → генерирует неправильные subqueries → recall=0. С adaptive retrieval: rule-based детектирует "NVIDIA" + "2026" → temporal_search(date_from="2026-01-01") + entity queries с "NVIDIA" → значительно более узкий candidate pool → Vera Rubin posts попадают в top-K через BM25 match по "NVIDIA".

---

## 8. Что НЕ делать

1. **Не добавлять >5 tools.** Research показывает деградацию для малых моделей при >5-7 tools. 4 search-related tools + query_plan + compose_context + final_answer = 7 total.

2. **Не делать отдельный LLM router call.** +10-15с latency не окупается при текущем железе. Routing встраивается в query_plan.

3. **Не тренировать classifier на 20 примерах.** Нужно 200+. Пока — rules + LLM-in-prompt.

4. **Не делать NER pipeline.** Для 13K коротких постов entity extraction через BM25 keyword match + cross-encoder reranking достаточен. NER добавит complexity без значительного gain.

5. **Не ожидать "-35% latency".** Это цифра для систем с быстрым LLM (<1с inference). При 12с/call на V100 bottleneck — LLM, не retrieval. Retrieval-side оптимизация сэкономит 2-3с из 45.
