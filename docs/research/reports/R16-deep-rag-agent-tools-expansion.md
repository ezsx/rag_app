# Расширение инструментов RAG-агента для новостного корпуса

**Анализ 9 кандидатов на новые tools, архитектуры dynamic visibility и стратегии маршрутизации для ReAct-агента на Qwen3-30B-A3B показывает: оптимальным решением является добавление 5 инструментов (из 9 рассмотренных), управляемых гибридной системой видимости (regex + embedding similarity) с латентностью ~5 мс.** Ключевой вывод из литературы — «less is more»: сокращение видимых tools до 4–5 повышает точность function calling на **25+ п.п.** (Anthropic, 2025) и снижает латентность на **70%** (Paramanayakam et al., 2024). A-RAG (Du et al., 2026) подтверждает, что **3 хорошо спроектированных инструмента дают прирост 5–13 п.п.** в QA-accuracy по сравнению с flat RAG. Для Qwen3-30B-A3B с 3B активных параметров это особенно критично: модель демонстрирует отличные результаты в single-turn function calling, но деградирует при параллельных вызовах (**37.5%** accuracy на live_parallel в BFCL).

---

## 1. Что говорят исследования: papers, бенчмарки, индустрия

### Академические работы (2024–2026)

**A-RAG** (arXiv:2602.03442, февраль 2026) — ключевая работа для данного проекта. Авторы из USTC предложили иерархические интерфейсы retrieval: всего три инструмента (keyword search, semantic search, chunk read) на разных уровнях гранулярности. Агент **спонтанно обобщает разнообразные workflow** под типы запросов. Прирост **5–13 п.п.** над flat RAG. Главный инсайт: инструменты должны отражать разные уровни доступа к данным, а не дублировать один уровень.

**«Less is More» для function calling** (arXiv:2411.15399, ноябрь 2024, DATE 2025) — Paramanayakam et al. показали, что динамическое сокращение видимых tools без дообучения модели даёт: снижение latency на **70%**, энергопотребления на **40%**, и рост accuracy. Это fine-tuning-free подход, напрямую валидирующий constraint max 5 tools.

**Tool-to-Agent Retrieval** (arXiv:2511.01854, ноябрь 2025) — PwC предложили эмбеддить tools и агентов в единое векторное пространство. Результат: **+19.4% Recall@5** и **+17.7% nDCG@5** на LiveMCPBench (527 tools). Подход architecture-agnostic — улучшения воспроизводятся на 8 разных embedding-моделях.

**AutoTool** (arXiv:2511.14650, ноябрь 2025) использует граф исторических вызовов tools: ноды — инструменты, рёбра — вероятности перехода. Результат — **снижение inference-cost на 30%** при сохранении task completion rate. Ценно для оптимизации последовательностей вызовов в multi-turn.

**ToolACE** (ICLR 2025, arXiv:2409.00920) демонстрирует, что качество описаний tools критичнее количества — модель ToolACE-8B конкурирует с GPT-4 на BFCL и API-Bank исключительно за счёт высококачественных синтетических данных.

### Бенчмарки: количество tools vs accuracy

**BFCL v4** (Berkeley, ICML 2025) — наиболее релевантный бенчмарк. Qwen3-30B-A3B занимает позицию, уступая только QwQ-32B и GPT-4o среди open-source моделей. Qwen3.5-35B-A3B набирает **67.3 на BFCL-V4** (GPT-5 mini — 55.5). Критический паттерн для малых моделей: **«not calling a tool beats calling the wrong one»** — 5 из 8 малых моделей вызывали get_weather при любом упоминании «weather», независимо от контекста.

**ToolBench** (ICLR 2024) с 16K+ API показал, что baseline semantic search часто извлекает «похожие, но нерабочие» API. GRETEL (execution-based filtering) улучшил **Pass@10 на +13.6 п.п.** Для tool selection это значит: similarity ≠ relevance, нужны дополнительные сигналы.

**T-Eval** (arXiv:2312.14033) декомпозирует tool-use на 6 sub-процессов: instruction following, planning, reasoning, retrieval, understanding, review. Позволяет находить конкретные bottleneck-и — например, модель может хорошо планировать, но плохо извлекать правильный tool.

### Индустрия и open-source

**Anthropic Tool Search Tool** (ноябрь 2025) — наиболее прямой прецедент. При 58 tools из 5 MCP-серверов потребление контекста составило ~55K токенов. Tool Search снизил это до ~8.7K (**85% reduction**). Точность Opus 4 выросла с **49% до 74%**. Рекомендация Anthropic: **keep 3–5 most-used tools always loaded, defer the rest**.

Anthropic также документирует, что добавление `input_examples` к сложным tools улучшает accuracy с **72% до 90%**. Описание tool — «by far the most important factor». Минимум 3–4 предложения на инструмент с указанием когда использовать, когда *не* использовать, и форматов параметров.

**LangGraph Dynamic Tool Calling** (август 2025) реализует изменение набора доступных tools на разных этапах выполнения: «give your agent the right toolbox at the right time». **Perplexity Focus Modes** — контекстные фильтры (Web, Academic, Social, Finance), маршрутизирующие к разным backend-ам и source-типам. Прямо применимо к новостному RAG как паттерн pre-defined modes.

**vLLM Semantic Router** (arXiv:2510.08731) — **тестировался именно на Qwen3-30B-A3B** с vLLM v0.10.1. Результат: **+10.2% accuracy на MMLU-Pro** при **снижении latency на 48.5%**. Написан на Rust (HuggingFace Candle), использует ModernBERT для intent detection. Это валидация того, что semantic routing + Qwen3-30B-A3B — рабочая production-комбинация.

---

## 2. Рекомендуемый каталог инструментов

Из 9 кандидатов **рекомендуются 5**, **2 рекомендуются условно**, **2 отклоняются**.

### Рекомендуемые (приоритет 1)

| Tool | Зачем | Сложность | Qdrant API |
|------|-------|-----------|------------|
| `compare_search` | Сравнение сущностей — частый паттерн в AI/ML | Средняя | `query_batch_points` |
| `read_post` | Чтение полного текста — необходим для глубокого анализа | Низкая | `retrieve` по ID |
| `list_channels` | Навигация по корпусу — быстрый, почти zero-cost | Низкая | `facet` API |
| `related_posts` | Exploration — «показать похожее» | Низкая | `RecommendQuery` |
| `summarize_channel` | Дайджест по каналу — высокая пользовательская ценность | Средняя | `scroll` + filter |

### Рекомендуемые условно (приоритет 2)

| Tool | Зачем | Условие |
|------|-------|---------|
| `fact_check` | Кросс-проверка по каналам | Реализовать как вариант `compare_search` с другой семантикой |
| `entity_search` | Поиск по сущности | Полноценно работает только с NER в payload |

### Отклонённые

| Tool | Причина отклонения |
|------|-------------------|
| `trending_search` / `digest` | Qdrant не имеет нативных time-series aggregations. Требует scroll + client-side aggregation — слишком сложно для описания в ≤50 слов, latency непредсказуемая. Лучше реализовать как offline pipeline (cron), а не как tool |
| `author_search` | Дублирует `channel_search` с другим фильтром. Поле `author` в Telegram-каналах часто совпадает с `channel` (один автор = один канал). Можно добавить параметр `author` в существующий `channel_search` |

### Детальные описания рекомендуемых tools

#### compare_search

```json
{
  "name": "compare_search",
  "description": "Сравнивает две или более сущности (модели, компании, технологии) по упоминаниям в корпусе. Используй когда пользователь спрашивает 'X vs Y', 'сравни X и Y', 'чем X отличается от Y'. НЕ используй для поиска одной сущности.",
  "parameters": {
    "entities": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4, "description": "Список сущностей для сравнения"},
    "time_range": {"type": "string", "enum": ["week", "month", "quarter", "all"], "default": "month"},
    "top_k": {"type": "integer", "default": 5, "description": "Кол-во результатов на сущность"}
  }
}
```

**Под капотом:** Для каждой entity генерирует embedding запроса «{entity} в контексте AI/ML», отправляет `query_batch_points` с N параллельными запросами (один на entity, каждый с date-range filter). Результаты мёржатся, дедуплицируются и группируются по entity. Добавляет count per entity для количественного сравнения coverage.

**Qdrant-запрос:**
```python
client.query_batch_points("news", requests=[
    QueryRequest(query=embed(entity), filter=date_filter, limit=top_k)
    for entity in entities
])
```

**Latency:** ~20–50 мс (batch из 2–4 запросов, 13K документов). Укладывается в лимит 1с.

**Пример запроса пользователя:** «Сравни Llama 4 и Qwen3 по обсуждениям за последний месяц»

#### read_post

```json
{
  "name": "read_post",
  "description": "Читает полный текст поста по его ID. Используй после search/rerank когда нужен полный контекст найденного поста, а не только сниппет. НЕ используй без предварительного поиска.",
  "parameters": {
    "post_id": {"type": "string", "description": "ID поста из результатов поиска"},
    "include_replies": {"type": "boolean", "default": false}
  }
}
```

**Под капотом:** `client.retrieve(collection_name="news", ids=[post_id], with_payload=True)`. Если `include_replies=true`, дополнительно scroll с фильтром `reply_to == post_id`. Возвращает полный текст, метаданные (канал, дата, ссылки, медиа).

**Latency:** <1 мс для retrieve, +2–5 мс для replies.

**Пример:** После поиска агент находит релевантный пост, но сниппет недостаточен — вызывает `read_post` для полного текста.

#### list_channels

```json
{
  "name": "list_channels",
  "description": "Показывает список доступных Telegram-каналов с количеством постов в каждом. Используй когда пользователь спрашивает какие каналы есть, или нужно уточнить название канала для channel_search.",
  "parameters": {
    "sort_by": {"type": "string", "enum": ["name", "count"], "default": "count"}
  }
}
```

**Под капотом:** `client.facet("news", key="channel", limit=50, exact=True)`. Возвращает `[{channel_name, post_count}, ...]`, сортировка client-side.

**Latency:** ~2–5 мс. Результат кэшируемый (TTL 1 час).

**Пример:** «Какие каналы у тебя есть?» / «Сколько постов в канале X?»

#### related_posts

```json
{
  "name": "related_posts",
  "description": "Находит посты, похожие на указанный пост. Используй когда пользователь хочет 'ещё такое же', 'похожие посты', или когда нужно расширить контекст уже найденного релевантного поста.",
  "parameters": {
    "post_id": {"type": "string", "description": "ID исходного поста"},
    "limit": {"type": "integer", "default": 5},
    "exclude_channel": {"type": "boolean", "default": false, "description": "Исключить посты из того же канала"}
  }
}
```

**Под капотом:** `query_points` с `RecommendQuery(positive=[post_id])`. Если `exclude_channel=true`, добавляется filter `must_not` по каналу исходного поста. Стратегия `best_score` (default в Qdrant ≥1.6).

**Latency:** ~5–15 мс.

**Пример:** «Покажи похожие посты на этот» / агент самостоятельно расширяет контекст для ответа.

#### summarize_channel

```json
{
  "name": "summarize_channel",
  "description": "Получает последние посты канала за период для составления сводки. Используй когда пользователь спрашивает 'что нового в канале X', 'дайджест канала за неделю'. НЕ используй для поиска конкретной темы в канале — для этого есть channel_search.",
  "parameters": {
    "channel": {"type": "string", "description": "Точное название канала из list_channels"},
    "time_range": {"type": "string", "enum": ["day", "week", "month"], "default": "week"},
    "limit": {"type": "integer", "default": 20}
  }
}
```

**Под капотом:** `client.scroll("news", scroll_filter=Filter(must=[channel_match, date_range]), order_by=OrderBy(key="date", direction="desc"), limit=limit, with_payload=True)`. Возвращает тексты постов в хронологическом порядке. LLM суммаризует на этапе `final_answer`.

**Latency:** ~5–15 мс для scroll. Основная latency — в генерации суммаризации LLM.

#### fact_check (условный — как расширение compare_search)

Реализуется аналогично `compare_search`, но с другой семантикой: один факт, множество каналов. Под капотом — `query_batch_points` с одним query-вектором и N фильтрами по каналам. Рекомендация: не делать отдельным tool, а добавить параметр `mode: "compare" | "verify"` в `compare_search`. Это экономит слот в tool schema.

#### entity_search (условный)

При наличии NER-полей в payload — мощный инструмент. Без NER — де-факто дублирует `search` с лучшим промптом. Рекомендация: добавлять NER entities при ingestion (spaCy/Natasha для русского), затем реализовать как `query_points` с MatchAny-фильтром по entity-полю + vector rerank.

---

## 3. Четыре подхода к dynamic visibility и почему выбран гибрид

### Сравнительная таблица

| Подход | Accuracy | Latency | Сложность | Cold start | Подходит для 10–15 tools |
|--------|----------|---------|-----------|------------|--------------------------|
| Rule-based (regex) | Средняя | <0.1 мс | Низкая | Нет | Да, но хрупкий |
| Embedding-based | Высокая | 3–10 мс | Низкая–средняя | Нужны эмбеддинги | Да, оптимален |
| Classifier-based | Высокая | <1 мс | Средняя | Нужны 50–200 примеров | Да, но overkill |
| RATS (2-agent) | Наивысшая | 200–500 мс | Средняя–высокая | Нет | Overkill, +latency |

**Rule-based** — текущая реализация проекта. Работает для явных паттернов (даты → `temporal_search`, «сравни» → `compare_search`), но ломается на парафразах: «как эволюционировал GPT» не матчит regex для trending. Для русского языка — дополнительная сложность из-за морфологии. Подходит как первый слой, но не как единственный.

**Embedding-based** — «RAG для RAG». Offline: эмбеддим описания 12–15 tools (один вызов Qwen3-Embedding-0.6B, уже есть в pipeline). Online: эмбеддим запрос, cosine similarity с 15 векторами в in-memory FAISS. При 15 точках — это фактически 15 dot-product-ов, **<0.1 мс** на поиск. Embedding generation — **1–3 мс** (модель уже загружена). Метод Tool2Vec (arXiv:2409.02141) предлагает эмбеддить не описания tools, а **примеры запросов** — прирост **+30.5% recall**. Это закрывает семантический разрыв между «developer description» и «user query».

**Classifier-based** — для 10–15 tools избыточен. Требует сбора training data (50–200 примеров на tool). Zero-shot вариант (BART-MNLI) добавляет 5–20 мс и модель в 400M параметров. При наличии embedding-подхода не даёт существенных преимуществ, но создаёт maintenance burden при добавлении новых tools.

**RATS** (Retrieval-Augmented Tool Selection) — двух-агентный паттерн: лёгкий router-агент выбирает tools, тяжёлый execution-агент работает с ними. Идеален для 50–1000+ tools, но для 10–15 tools overhead не оправдан. Добавляет 200–500 мс latency на LLM-inference router-а, что критично при лимите first-token <3с.

### Рекомендуемая архитектура: гибрид rules + embedding

```
User Query
    │
    ├─► [Layer 1: Rule-Based] (~0 мс)
    │   • always_visible: [search, final_answer]  — базовый набор
    │   • regex «сравни|vs|versus» → +compare_search
    │   • regex дата-паттерны → +temporal_search
    │   • regex «канал|channel» → +channel_search, +list_channels
    │
    ├─► [Layer 2: Embedding Similarity] (~3–5 мс)
    │   • Qwen3-Embedding-0.6B (уже в pipeline)
    │   • In-memory: 15 tool vectors (Tool2Vec: по 5–10 примеров запросов на tool)
    │   • Top-5 по cosine similarity, threshold 0.35
    │
    ├─► [Layer 3: Category Expansion] (~0 мс)
    │   • search → +rerank, +compose_context
    │   • compare_search → +rerank
    │   • summarize_channel → +list_channels
    │
    └─► Merge + Deduplicate → Cap at 5 → Qwen3-30B-A3B
```

**Суммарная latency:** ~5 мс. vLLM Semantic Router (arXiv:2510.08731) валидирует этот подход именно на Qwen3-30B-A3B: **+10.2% accuracy, −48.5% latency**.

Ключевое архитектурное решение — **Tool2Vec вместо embedding описаний**. Для каждого tool создаём 5–10 примеров запросов на русском, эмбеддим, усредняем в один вектор. Это даёт гораздо лучший match между пользовательскими запросами и tools, потому что пользователь формулирует запрос — не описание инструмента.

---

## 4. Группировка и маршрутизация: schema для 12 tools

### Финальный набор tools (7 существующих + 5 новых)

Существующие: `search`, `temporal_search`, `channel_search`, `query_plan`, `rerank`, `compose_context`, `final_answer`.

Новые: `compare_search`, `read_post`, `list_channels`, `related_posts`, `summarize_channel`.

Итого: **12 tools**, max 5 видимых. Группировка — **по фазе + типу запроса** (гибрид):

### Группы по фазе

| Фаза | Tools | Когда активна |
|------|-------|---------------|
| **Always-on** | `final_answer` | Виден всегда (1 слот) |
| **Planning** | `query_plan`, `list_channels` | Первый шаг сложного запроса |
| **Retrieval** | `search`, `temporal_search`, `channel_search`, `compare_search`, `summarize_channel` | Основной поиск |
| **Enrichment** | `read_post`, `related_posts`, `rerank` | После первичного поиска |
| **Synthesis** | `compose_context`, `final_answer` | Формирование ответа |

### Маршрутизация по типу запроса

| Тип запроса | Пример | Visible tools (5 слотов) |
|-------------|--------|--------------------------|
| **Factual** | «Что такое LoRA?» | `search`, `rerank`, `compose_context`, `read_post`, `final_answer` |
| **Temporal** | «Что нового за неделю?» | `temporal_search`, `summarize_channel`, `rerank`, `compose_context`, `final_answer` |
| **Comparative** | «Qwen3 vs Llama 4» | `compare_search`, `search`, `rerank`, `compose_context`, `final_answer` |
| **Exploratory** | «Что обсуждают про RAG?» | `search`, `related_posts`, `list_channels`, `compose_context`, `final_answer` |
| **Navigational** | «Покажи каналы» / «Прочитай пост» | `list_channels`, `channel_search`, `read_post`, `summarize_channel`, `final_answer` |

Динамическая перекомпозиция: после первого tool call агент может запросить tools из следующей фазы. Реализуется через category expansion — если агент вызвал `search`, на следующем шаге автоматически добавляются `read_post`, `related_posts`, `rerank`.

---

## 5. Конкретный implementation plan

### Фаза 1: Quick wins (1–2 недели)

**Приоритет: `list_channels`, `read_post`, `related_posts`**

Все три реализуются одним Qdrant API-вызовом каждый, не требуют дополнительной инфраструктуры:

- `list_channels` → `client.facet("news", key="channel", limit=50)`. Кэшировать результат (TTL 1 час). ~20 строк кода
- `read_post` → `client.retrieve(ids=[post_id])`. ~15 строк кода
- `related_posts` → `client.query_points(query=RecommendQuery(positive=[id]))`. ~25 строк кода

Обновить regex pre-scan: добавить паттерны для навигационных запросов. Написать tool descriptions по Anthropic guidelines (3–4 предложения, когда использовать / не использовать).

### Фаза 2: Core tools (2–3 недели)

**Приоритет: `compare_search`, `summarize_channel`**

- `compare_search` → batch queries + merge logic. ~80 строк кода. Тестирование на парах entity-запросов
- `summarize_channel` → scroll + date filter + order_by. ~50 строк кода. Проверить наличие datetime-индекса на поле `date`

Добавить keyword-индекс на поле `author` (если нет) и datetime-индекс на `date`.

### Фаза 3: Embedding-based visibility (2–3 недели)

Заменить regex-only visibility на гибрид:

- Сгенерировать по 5–10 примеров запросов для каждого из 12 tools (всего ~100 примеров)
- Эмбеддить через Qwen3-Embedding-0.6B, усреднить в tool vectors
- Реализовать in-memory cosine search (numpy, без FAISS — 12 точек)
- Добавить category expansion mappings
- A/B тест: regex-only vs hybrid на реальных запросах
- Замерить accuracy tool selection и end-to-end quality

### Фаза 4: Оптимизация (1–2 недели)

- Собрать логи tool calls, проанализировать паттерны использования
- Настроить similarity threshold (начать с 0.35, тюнить по precision/recall)
- Реализовать кэширование `list_channels` и популярных facet-запросов
- Оптимизировать tool descriptions на основе ошибок агента (Anthropic: итеративное prompt-engineering описаний)

### Фаза 5 (опционально): NER pipeline

- Добавить NER при ingestion (Natasha для русского, spaCy для английского)
- Сохранять entities как keyword-массив в payload Qdrant
- Реализовать `entity_search` как filter + vector rerank

---

## 6. Anti-patterns: что НЕ делать

**Не добавлять `trending_search` / `digest` как real-time tool.** Qdrant не имеет нативных time-series aggregations. Реализация через scroll + client-side aggregation требует обработки сотен документов, непредсказуемой latency (100–500 мс), и сложного описания для LLM. Правильное решение — offline cron job, формирующий дайджест в отдельную коллекцию, и простой tool для чтения готового дайджеста.

**Не создавать `author_search` как отдельный tool.** В Telegram-каналах author ≈ channel в большинстве случаев. Добавление параметра `author` в существующий `channel_search` экономит слот и снижает когнитивную нагрузку на модель. Бенчмарки BFCL показывают: **keyword matching — главный failure mode** малых моделей. Два похожих tool-а (`channel_search` и `author_search`) провоцируют именно эту ошибку.

**Не описывать tool в >50 слов.** Anthropic рекомендует 3–4 предложения, но constraint в 2000 токенов на все видимые descriptions (5 tools × ~400 токенов max = 2000) требует баланса. Формула: **name (5 слов) + description (30–40 слов) + parameters (JSON schema)**. Тестировать каждое описание: может ли человек, прочитав только description, правильно выбрать tool для запроса?

**Не реализовывать multi-step tools.** Tool, который внутри делает search → rerank → summarize — это «скрытый агент». Qwen3-30B-A3B должен видеть и контролировать каждый шаг. Multi-step tools нарушают ReAct-цикл и делают debugging невозможным. Каждый tool — одна атомарная операция.

**Не дублировать функциональность.** `fact_check` как отдельный tool — это `compare_search` с одним запросом по нескольким каналам. Реализовать как параметр `mode` в `compare_search`, а не как отдельный инструмент. Каждый добавленный tool — это +150–400 токенов в schema, +1 вариант для ошибки выбора.

**Не полагаться только на regex для visibility.** Текущий подход хрупок к парафразам и морфологии русского языка. «Эволюция GPT» не матчит regex для trending, «различия между моделями» — не матчит regex для compare. Embedding-based слой критичен для robustness.

**Не реализовывать tools без данных в корпусе.** Entity search без NER в payload — это просто semantic search с другим именем. Code search, image search, structured data query — бессмысленны для Telegram-текстов. Каждый tool должен иметь backing data.

---

## 7. Ожидаемый эффект на метрики

### Прогноз на основе литературы

**Tool selection accuracy.** Переход от regex-only к hybrid (regex + embedding) visibility должен дать **+15–25 п.п.** на tool selection accuracy. Базис: Anthropic Tool Search даёт +25 п.п. (49→74%) на 58 tools; при 12 tools эффект меньше, но модель менее мощная (3B active vs 200B+).

**End-to-end answer quality.** Добавление `read_post` и `related_posts` позволит агенту строить более полный контекст. A-RAG показывает **+5–13 п.п.** QA accuracy от multi-granularity tools. Ожидаемый прирост для данного проекта: **+5–8 п.п.** на пользовательских запросах, требующих глубокого контекста.

**Latency.** Новые tools добавляют **<50 мс** к pipeline (Qdrant-запросы на 13K документов). Hybrid visibility добавляет ~5 мс. Суммарный бюджет — **<100 мс**, что укладывается в constraint «tools не должны добавлять >1s». «Less is More» paper показывает, что сокращение видимых tools с 10+ до 5 может **снизить** общую latency за счёт более быстрого принятия решений моделью.

**Token efficiency.** Текущий constraint — <2000 токенов на tool descriptions. С 5 видимыми tools это ~400 токенов/tool. Для Qwen3-30B-A3B с 3B active params каждый сэкономленный токен в schema — это больше бюджета на reasoning. Hybrid visibility позволяет показывать только релевантные tools, экономя **40–60%** токенов schema по сравнению со static набором.

**Охват типов запросов.** Текущие 7 tools покрывают factual и temporal запросы. Добавление 5 tools расширяет покрытие на comparative (`compare_search`), navigational (`list_channels`, `read_post`), и exploratory (`related_posts`, `summarize_channel`) типы. Ожидаемый прирост покрытия: с ~60% до **~85–90%** типов пользовательских запросов.

### Метрики для отслеживания

При внедрении необходимо отслеживать: долю запросов, где агент вызвал tool, который не был в top-5 visibility (→ нужно тюнить threshold); долю «пустых» tool calls (tool вернул 0 результатов → неправильный выбор); среднее количество tool calls на запрос (рост >4 — тревожный сигнал); latency breakdown по фазам (visibility selection, tool execution, LLM generation); и user satisfaction на A/B тесте с/без новых tools.