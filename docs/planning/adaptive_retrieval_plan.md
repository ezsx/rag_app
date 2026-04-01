# Plan: Adaptive Retrieval + Tool Router

> Подробный план внедрения Phase 3.2. Основан на 4 исследованиях (R13-quick, R13-deep, R14-quick, R14-deep).
> Создан: 2026-03-20. Статус: **COMPLETED** — реализовано в SPEC-RAG-11/13/15/16/17.
> 15 tools, dynamic visibility, data-driven routing, analytics short-circuit.
> Этот файл — исторический артефакт планирования.

---

## Контекст и мотивация

### Корневая проблема

Pipeline **линейный** — каждый запрос идёт одним путём:
```
query_plan → search (hybrid BM25+Dense → RRF → ColBERT) → rerank → compose_context → final_answer
```

Это значит:
- "Что было в январе 2026" → **нет date filter**, идёт generic search
- "Что писал gonzo_ml" → **нет channel filter**, идёт generic search
- "NVIDIA Vera Rubin" → LLM не знает entity, не генерирует правильный subquery
- "Сколько стоит Deep Think" → нужен exact fact lookup, а идёт broad search

**Это то, что фреймворки (LlamaIndex/LangChain) тоже НЕ делают из коробки.** RouterQueryEngine в LlamaIndex роутит между index'ами, а не генерирует dynamic Qdrant filters.

### Доказательная база (из ресерчей)

| Paper | Результат | Применимость |
|-------|-----------|--------------|
| Adaptive-RAG (NAACL 2024) | +5-31pp на mixed-complexity | Routing по complexity level → у нас routing по query type |
| CRAG (Yan et al., 2024) | +7% PopQA, +37% PubHealth | ColBERT scores → quality gate без доп. модели |
| Self-RAG (ICLR 2024) | +31.4pp long-tail queries | Демонстрирует magnitude improvement; требует fine-tuning — не берём |
| RouteRAG (Bai et al., 2025) | 97.6-100% EM, 10x latency reduction | Rule-driven features из запроса — прямая аналогия |
| A-RAG (Du et al., Feb 2026) | 94.5% HotpotQA | Hierarchical tools (BM25/dense/ColBERT) — модель решает |
| "Less is More" (2024) | 35%→87% tool accuracy при subset | Max 5 tools для Qwen3-30B (3B active params) |
| FAIR-RAG (2025) | F1 0.453 HotpotQA (+8.3pp) | Iterative refinement с adaptive query generation |
| ChronoQA (Nature 2025) | 72% ошибок RAG от temporal | Temporal retrieval — must have для news aggregator |

---

## Архитектура

### Общая схема

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│ Rule-based Pre-Validator (<1ms) │  ← regex: dates, @channels, entities
│ → QuerySignals: strategy_hint,  │
│   extracted_filters, confidence  │
└─────────────┬───────────────────┘
              │ hints injected into prompt
              ▼
┌─────────────────────────────────┐
│ query_plan (LLM, ~12s)         │  ← enriched JSON output:
│ → strategy: enum               │     broad | temporal | channel | entity
│ → filters: {date_from, date_to,│
│    channels, entities}          │
│ → subqueries: [...]            │
└─────────────┬───────────────────┘
              │ rule override if confidence > 0.8
              ▼
┌─────────────────────────────────┐
│ Dynamic Tool Visibility         │  ← hide irrelevant tools
│ → No dates detected → hide      │     temporal_search
│ → No @channel → hide            │     channel_search
│ → Show 2-4 tools instead of 4   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ Tool Dispatch (by strategy)     │
│ ├─ broad    → base_search()     │  ← текущий hybrid search
│ ├─ temporal → base_search() +   │     Qdrant DatetimeRange filter
│ ├─ channel  → base_search() +   │     Qdrant MatchValue filter
│ └─ entity   → [entity]+queries  │     + BM25 keyword boost
│              + optional filters  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ Quality Gate (ColBERT scores)   │  ← CRAG-lite
│ ├─ score > 0.6  → Correct      │     use results
│ ├─ 0.3 ≤ score ≤ 0.6 → Ambig  │     expand search (relax filters)
│ └─ score < 0.3  → Incorrect    │     fallback to broad_search
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 3-Tier Fallback Chain           │
│ 1. Specialized search           │
│ 2. Broadened (relax filters)    │
│ 3. broad_search (full corpus)   │
└─────────────────────────────────┘
```

### Ключевые принципы (из ресерчей)

1. **Zero extra LLM calls** — routing внутри существующего query_plan. Отдельный LLM-router = +10-15с latency = неприемлемо на V100.
   - Ref: R13-quick §1.1, R13-deep §1

2. **Max 5 tools visible** — "Less is More" показывает деградацию accuracy при >5-7 tools для малых моделей.
   - Ref: R13-quick §2.1, R13-deep §2

3. **Thin wrappers** — все tools это один base_search() с разными Qdrant filters, не отдельные pipelines.
   - Ref: R13-quick §2.3, R13-deep §2

4. **Filter composition** — multi-strategy запросы ("gonzo_ml в январе 2026") = combined filter в одном запросе, не sequential tool calls.
   - Ref: R13-quick §3.4, R13-deep §3

5. **Graceful degradation** — неправильный routing → broad search (текущее поведение), не failure.
   - Ref: R13-quick §1.2, R13-deep §3

---

## Компоненты

### 1. Rule-based Pre-Validator

**Что делает**: Deterministic regex extraction перед LLM call. <1ms.

**Зачем**: Safety net для LLM routing. Если regex однозначно детектирует дату или канал — override LLM если он ошибся.

**Паттерны для extraction**:
```
Temporal:
  - Русские месяцы: "в январе 2026", "декабрь 2025"
  - Английские: "january 2026", "last week"
  - Relative: "последняя неделя", "вчера", "недавно", "за последний месяц"
  - ISO: "2026-01", "2025-12"

Channel:
  - Exact match против known_channels list (36 каналов)
  - @mentions: "@gonzo_ml", "канал gonzo_ml"
  - Author names: "Сапунов" → gonzo_ml, "Себрант" → techsparks

Entity:
  - Product patterns: "GPT-5", "Claude 4", "Gemini", "Llama", "Vera Rubin"
  - Company: "NVIDIA", "OpenAI", "Google", "Meta", "Anthropic"
  - Comparison markers: "vs", "сравни", "отличия", "compare"
```

**Output**: `QuerySignals(strategy_hint, confidence, extracted_filters)`

**Ref**: R13-quick §1.4, R13-deep §1 (rule-based pre-validator)

### 2. Enriched query_plan

**Что меняется**: Расширяется JSON schema output query_plan tool.

**Текущий output**:
```json
{
  "subqueries": ["transformers architecture", "attention mechanism"]
}
```

**Новый output**:
```json
{
  "subqueries": ["transformers architecture", "attention mechanism"],
  "strategy": "temporal",
  "filters": {
    "date_from": "2026-01-01",
    "date_to": "2026-01-31",
    "channels": [],
    "entities": ["NVIDIA"]
  }
}
```

**Strategy enum**: `broad` | `temporal` | `channel` | `entity`

**Prompt engineering**: Rules-in-prompt approach (explicit if-then для 3B active params модели):
```
Strategy selection rules:
- "temporal": query mentions dates, months, periods, "recently", "latest"
- "channel": query mentions a specific channel name or author
- "entity": query asks about specific product, company, person, technology
- "broad": default for general/comparative/multi-topic queries
ALWAYS extract dates if mentioned. ALWAYS extract channel names if mentioned.
If unsure, use "broad" with extracted filters.
```

**Grammar enforcement**: llama-server `--jinja` + JSON schema constraint → model constrained to valid enum values.

**Ref**: R13-quick §1.2-1.3, R13-deep §1

### 3. Specialized Tools (4 штуки)

Все tools — thin wrappers вокруг единого `base_search()` с разными Qdrant filters.

#### 3.1 `broad_search(queries, k)` — default/fallback
- Текущий hybrid search без изменений
- BM25 top-100 + Dense top-20 → RRF 3:1 → ColBERT rerank → cross-encoder rerank

#### 3.2 `temporal_search(queries, date_from, date_to, k)`
- base_search() + Qdrant `FieldCondition(key="date", range=Range(gte=date_from, lte=date_to))`
- Увеличенный k (×1.5) т.к. date filter сужает candidate pool
- Optional: recency decay `score *= exp(-λ × days_old)` (R14-deep: ChronoQA)

#### 3.3 `channel_search(queries, channel, k)`
- base_search() + Qdrant `FieldCondition(key="channel", match=MatchValue(value=channel))`
- Optional: date_from/date_to как дополнительные параметры

#### 3.4 `entity_search(entity, queries, date_from?, date_to?, k)`
- Entity name ВСЕГДА добавляется в queries → BM25 keyword boost
- Optional date/channel filters
- Увеличенный k (×1.5) для BM25 catch

**Filter builder** — единый метод:
```python
def _build_filter(self, filters: dict) -> models.Filter:
    conditions = []
    if "channel" in filters:
        conditions.append(FieldCondition(key="channel", match=MatchValue(value=...)))
    if "date_range" in filters:
        conditions.append(FieldCondition(key="date", range=Range(gte=..., lte=...)))
    return Filter(must=conditions) if conditions else None
```

**Ref**: R13-quick §2.2-2.3, R13-deep §2

### 4. Dynamic Tool Visibility

**Текущее**: `final_answer` скрыт до search.

**Новое**: search tools тоже динамически фильтруются по query signals.
- Нет temporal hints → скрыть `temporal_search`
- Нет channel hints → скрыть `channel_search`
- Однозначный hint (confidence > 0.8) → показать только matching tool + `broad_search` (fallback)

Результат: вместо 4 search tools LLM видит 2-3, что повышает accuracy выбора.

**Ref**: R13-quick §2.4, R13-deep §2

### 5. Quality Gate (CRAG-lite)

**Что делает**: После retrieval, проверяет ColBERT rerank scores.

| Score range | Action | Описание |
|-------------|--------|----------|
| > 0.6 | **Correct** | Документы релевантны, использовать |
| 0.3 — 0.6 | **Ambiguous** | Расширить поиск: увеличить k, ослабить filters |
| < 0.3 | **Incorrect** | Fallback на broad_search |

**Почему ColBERT**: уже есть в pipeline, zero overhead. CRAG paper использовал T5-Large evaluator — у нас ColBERT scores выполняют ту же функцию.

**Ref**: R13-deep §3, R14-quick §1.2 (CRAG), R14-deep (CRAG с ColBERT)

### 6. Fallback Chain

```
Tier 1: Specialized search (temporal/channel/entity)
  │ if len(results) < 3 or max(colbert_scores) < 0.3
  ▼
Tier 2: Broadened search (relax filters: expand date ×2, remove channel)
  │ if still < 3 results
  ▼
Tier 3: broad_search (full corpus, no filters)
```

Максимум 1 extra retrieval call (+5с), не extra LLM call.

**Ref**: R13-quick §3.3, R13-deep §3

### 7. AgentState расширение

```python
@dataclass
class AgentState:
    # Existing
    search_count: int = 0
    compose_count: int = 0
    coverage: float = 0.0
    # New — adaptive retrieval
    strategy: str = "broad"              # broad|temporal|channel|entity
    applied_filters: dict = field(default_factory=dict)
    routing_source: str = "llm"          # llm|rules|fallback
    strategies_attempted: list = field(default_factory=list)
    result_quality_score: float = 0.0    # max ColBERT score
```

`routing_source` — для eval: кто принял решение (LLM, rules, fallback). Позволяет отлаживать routing accuracy.

**Ref**: R13-quick §3.2, R13-deep §3

---

## Какие v2 провалы решает

| Провал | Текущая проблема | Решение через adaptive retrieval |
|--------|-----------------|----------------------------------|
| **Q6** (NVIDIA 2026, recall=0.0) | LLM не знает "Vera Rubin" → неправильные subqueries | Rule-based: "2026" + "NVIDIA" → `temporal_search(date_from="2026-01-01") + entity_search("NVIDIA")`. Qdrant date filter сужает pool → NVIDIA posts всплывают через BM25 |
| **Q1** (Карпаты, recall=0.50) | data_secrets:8021 не в candidate pool | `entity_search("Карпаты")` — BM25 keyword boost по entity name расширяет pool |
| **Q3** (лицензия, recall=0.50) | rybolos:1562 не в candidate pool | `entity_search("OpenAI лицензия")` + BM25 boost |
| **Q7** (Deep Think цена, recall=0.50) | seeallochnaya:2711 не попадает | `entity_search("Deep Think")` — exact keyword match через BM25 |
| **Q8** (Kandinsky, recall=0.0) | Правильный канал, fuzzy ±5 strict | **НЕ решается routing** — проблема eval metric, не retrieval |

---

## Evaluation стратегия

### Routing accuracy eval (новый, отдельный от recall)

```python
routing_test_cases = [
    {"query": "Что писал gonzo_ml про трансформеры",
     "expected_strategy": "channel",
     "expected_filters": {"channels": ["gonzo_ml"]}},
    {"query": "Новости за декабрь 2025",
     "expected_strategy": "temporal",
     "expected_filters": {"date_from": "2025-12-01", "date_to": "2025-12-31"}},
    {"query": "Vera Rubin NVIDIA анонс 2026",
     "expected_strategy": "entity",
     "expected_filters": {"entities": ["NVIDIA", "Vera Rubin"]}},
    {"query": "Сравни GPT-5 и Claude",
     "expected_strategy": "broad"},
]
```

**Метрики**: strategy accuracy, filter extraction accuracy, routing_source distribution.

### End-to-end eval

- Прогнать v1+v2 (20 Qs) → recall не должен упасть (regression)
- Прогнать retrieval eval (100 Qs) → recall не должен упасть
- Добавить v3 dataset с strategy labels → per-type recall breakdown

**Ref**: R13-quick §5.3, R13-deep §6 (Day 4)

---

## Реализация: порядок работы

### День 1: Schema + Rule Engine + Tool Wrappers

**Утро**: Спецификация и подготовка
- [ ] Прочитать текущий код: query_plan tool, search tool, HybridRetriever, AgentState
- [ ] Написать техническую спецификацию (что менять в каких файлах)
- [ ] Создать Pydantic schemas: QueryPlan (enriched), QuerySignals, SearchFilters

**День**: Реализация core
- [ ] Rule-based pre-validator (regex patterns для dates, channels, entities)
- [ ] Расширить query_plan tool: новый JSON schema output + prompt
- [ ] 4 tool wrappers: broad_search, temporal_search, channel_search, entity_search
- [ ] Filter builder: _build_qdrant_filter(filters) → models.Filter
- [ ] Unit тесты rule-based validator

**Вечер**: Проверка
- [ ] Single query verification: temporal query ("новости за январь 2026")
- [ ] Single query verification: channel query ("что писал gonzo_ml")
- [ ] Убедиться что broad queries не сломались

### День 2: Integration + Dynamic Visibility + Fallback

**Утро**: Integration в ReAct loop
- [ ] AgentState расширение (strategy, applied_filters, routing_source)
- [ ] ToolRunner: dispatch по strategy → соответствующий tool
- [ ] Rule override logic: if hint.confidence > 0.8 → override LLM strategy

**День**: Dynamic visibility + fallback
- [ ] Dynamic tool visibility: скрытие irrelevant tools по query signals
- [ ] Quality gate: ColBERT score thresholds → Correct/Ambiguous/Incorrect
- [ ] 3-tier fallback chain
- [ ] Logging: strategy decision, routing_source, applied_filters

**Вечер**: E2E проверка
- [ ] 5-10 запросов через полный pipeline
- [ ] Temporal + channel + entity + broad → все работают
- [ ] Fallback: temporal query без результатов → broadened → broad

### День 3: Eval + Tuning + Commit

- [ ] Full eval: v1 (10 Qs) → recall не ниже 0.76
- [ ] Full eval: v2 (10 Qs) → recall выше 0.61
- [ ] Routing accuracy: сколько запросов правильно классифицированы
- [ ] Tune ColBERT score thresholds если нужно
- [ ] Tune rule-based patterns если нужно
- [ ] Commit с подробным описанием
- [ ] Обновить playbook с новыми результатами

### День 4-5 (если время есть): Eval Expansion + Ablation

- [ ] Расширить dataset до 30-50 вопросов с strategy labels
- [ ] Routing accuracy eval (отдельный скрипт)
- [ ] Per-category recall breakdown
- [ ] Начать ablation study: adaptive vs linear на разных query types

---

## Что НЕ делать (anti-patterns из ресерчей)

1. **Не добавлять >5 tools** — "Less is More" paper: деградация для 3B active params
2. **Не делать отдельный LLM router call** — +10-15с latency неприемлемо
3. **Не тренировать classifier** — 20 вопросов мало, нужно 200+
4. **Не делать NER pipeline** — BM25 keyword match + cross-encoder достаточно для 13K docs
5. **Не ожидать "-35% latency"** — bottleneck это LLM (12с/call), не retrieval
6. **Не делать comparative_search как отдельный tool** — это два broad_search + merge
7. **Не использовать --parallel 2** на llama-server — KV cache splitting tight на V100

**Ref**: R13-quick §8, R13-deep §4

---

## Ссылки на ресерчи

| Ресерч | Файл | Ключевое для этого плана |
|--------|------|--------------------------|
| R13-quick | `../research/reports/R13-quick-tool-router-architecture.md` | 4 tools, query_plan schema, Qdrant filter builder, tool schemas, fallback logic |
| R13-deep | `../research/reports/R13-deep-tool-router-architecture.md` | Grammar enforcement, 3-tier fallback, parallel Qdrant, ColBERT quality gate, published results |
| R14-quick | `../research/reports/R14-quick-beyond-frameworks-techniques.md` | A-RAG pattern, CRAG evaluator, interview strategy, ablation table template |
| R14-deep | `../research/reports/R14-deep-beyond-frameworks-techniques.md` | Temporal reasoning (ChronoQA), NLI verification, Speculative RAG, multi-source synthesis, 5-day plan |

### Papers (для deeper dive)

- Adaptive-RAG: `github.com/starsuzi/Adaptive-RAG` — routing classifier, eval suite
- A-RAG: `github.com/Ayanami0730/arag` — hierarchical retrieval, Qwen3-Embedding-0.6B (как у нас!)
- CRAG reproduction: `arXiv:2603.16169` — open-source CRAG + SHAP анализ
- "Less is More": `arXiv:2411.15399` — tool count vs accuracy для SLMs
- RouteRAG: rule-driven features, 97.6-100% EM accuracy
- ChronoQA: 72% ошибок RAG от temporal retrieval failures
- RAGAS: `github.com/explodinggradients/ragas` — evaluation framework
- NirDiamant/RAG_Techniques: 24.7K stars, self-contained notebooks
