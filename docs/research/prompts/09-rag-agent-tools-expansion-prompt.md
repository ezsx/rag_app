# Deep Research: Расширение инструментов RAG-агента — какие тулы добавить и как управлять видимостью

## О проекте

**rag_app** — portfolio-grade RAG система для собеседований на позицию **Applied LLM Engineer**. Поисковик/агрегатор новостей из 36 AI/ML Telegram-каналов (~13K документов). ReAct агент с native function calling на Qwen3-30B-A3B (3B active params, V100).

**Ключевая проблема**: у агента всего 3 search-инструмента (search, temporal_search, channel_search). Роутер по инструментам — наша архитектурная фишка и главное отличие от LlamaIndex/LangChain, но при 3 tools он не раскрывается. Яндекс на конференции YaC AI Meetup (март 2026) показал системы с сотнями/тысячами тулов и динамической подкладкой.

---

## Железо и стек

- **LLM**: Qwen3-30B-A3B GGUF (MoE, 3B active) на V100 SXM2 32GB, llama-server.exe, Windows Host
- **GPU-сервер** (RTX 5060 Ti 16GB, WSL2 native, gpu_server.py):
  - Qwen3-Embedding-0.6B (dense 1024-dim)
  - bge-reranker-v2-m3 (cross-encoder, 568M)
  - jina-colbert-v2 (ColBERT per-token, 560M, 128-dim)
- **Qdrant v1.17**: dense + sparse(BM25) + ColBERT(multi-vector MaxSim), коллекция `news_colbert`
- **Docker**: API + Qdrant (CPU only)
- **Оркестрация**: native function calling через `/v1/chat/completions`

---

## Домен и корпус

- 13124 документа из 36 Telegram-каналов
- AI/ML тематика, русский + английский
- 36 каналов 8 категорий:
  - **LLM новости**: ai_newz, neurohive, denissexy
  - **Research papers**: gonzo_ml (CTO Intento), seeallochnaya, dendi_math_ai, complete_ai (AIRI)
  - **Production ML**: llm_under_hood, varim_ml, boris_again, cryptovalerii (ex-Yandex)
  - **Open-source**: scientific_opensource, ruadaptnaya
  - **AI индустрия**: techsparks (Себрант, Яндекс), addmeto (Bobuk), aioftheday, singularityfm (Т-Банк), oulenspiegel_channel
  - **NLP**: rybolos_channel, stuffynlp
  - **MLOps**: MLunderhood
  - **CV/Data/Other**: deep_school, ai_machinelearning_big_data, data_secrets, techno_yandex и др.
- Период: 2025-07-01 → 2026-03-18 (9 месяцев)
- Payload каждого документа: `channel`, `date`, `message_id`, `author`, `links`, `media_types`, `is_forward`, `reply_to`, `lang`, `hash`

---

## Текущие инструменты агента

### LLM-visible tools — полная OpenAI-совместимая schema

```python
AGENT_TOOLS = [
    {
        "name": "query_plan",
        "description": "Декомпозирует сложный запрос на 3-5 подзапросов с фильтрами. Вызывай первым для планирования поиска.",
        "parameters": {"query": "string (required)"}
    },
    {
        "name": "search",
        "description": "Широкий поиск по всей базе AI/ML новостей из Telegram-каналов. Используй когда НЕ нужен фильтр по дате или каналу, или для общих/сравнительных запросов. Это fallback если другие инструменты не подходят.",
        "parameters": {"queries": "array<string> (required)", "k": "int (default 10)"}
    },
    {
        "name": "temporal_search",
        "description": "Поиск новостей за конкретный период времени. Используй когда в запросе есть даты, месяцы, периоды. Примеры: 'Что произошло в январе 2026?', 'Новинки на CES 2026'. НЕ используй для вопросов без привязки ко времени.",
        "parameters": {"queries": "array<string> (required)", "date_from": "string YYYY-MM-DD (required)", "date_to": "string YYYY-MM-DD (required)", "k": "int (default 15)"}
    },
    {
        "name": "channel_search",
        "description": "Поиск в конкретном Telegram-канале. Используй когда упоминается канал или автор. Примеры: 'Что писал gonzo_ml про трансформеры?'. НЕ используй для общих вопросов без указания канала.",
        "parameters": {"queries": "array<string> (required)", "channel": "string (required)", "k": "int (default 10)"}
    },
    {
        "name": "rerank",
        "description": "Переранжирует найденные документы по семантической близости к запросу. Вызывай после search. Документы подставляются автоматически.",
        "parameters": {"query": "string (required)", "top_n": "int (default 5)"}
    },
    {
        "name": "compose_context",
        "description": "Собирает контекст из ВСЕХ найденных документов с цитатами [1], [2] и считает coverage. Вызывай после rerank. Не передавай параметры.",
        "parameters": {}
    },
    {
        "name": "final_answer",
        "description": "Формирует финальный ответ пользователю на русском языке, опираясь только на собранный контекст.",
        "parameters": {"answer": "string (required)", "sources": "array<int> (required)"}
    }
]
```

### Системные tools (вызываются AgentService автоматически, НЕ в LLM schema):
- `verify` — проверка покрытия ответа (coverage threshold 0.65)
- `fetch_docs` — получение полных текстов по ID

### System prompt агента (полный текст):
```
Ты — RAG-агент для поиска информации в базе новостей из Telegram-каналов.

ПОРЯДОК РАБОТЫ:
1. query_plan — декомпозируй запрос на подзапросы
2. ВЫБЕРИ ПОДХОДЯЩИЙ инструмент поиска:
   - temporal_search — если в запросе есть даты, месяцы, периоды
   - channel_search — если упоминается конкретный канал или автор
   - search — для общих, сравнительных, entity-запросов без привязки ко времени или каналу
3. rerank — переранжируй документы по исходному запросу
4. compose_context — собери контекст из лучших документов
5. final_answer — дай итоговый ответ на основе контекста

ПРАВИЛА ВЫБОРА ИНСТРУМЕНТА:
- Если есть дата/период → temporal_search
- Если есть имя канала/автора → channel_search
- Если нет ни дат, ни каналов → search
- При сомнении — используй search

ПРАВИЛА ОТВЕТА:
- Отвечай ТОЛЬКО на русском языке
- Каждое утверждение подкрепляй ссылкой [1], [2]
- Если контекст НЕ содержит информации — честно скажи
- В final_answer ОБЯЗАТЕЛЬНО заполни поле sources
```

### Dynamic visibility — реализация в коде:
```python
def _get_step_tools(self, agent_state):
    """2 уровня visibility:
    1. final_answer/compose_context скрыты до search (архитектурная гарантия)
    2. temporal_search/channel_search скрыты по query signals (regex pre-scan)
    """
    if agent_state.search_count > 0:
        # После search: post-search tools (скрываем search-tools)
        return [t for t in AGENT_TOOLS if t["function"]["name"] not in
                ("temporal_search", "channel_search")]

    # До search: скрываем post-search tools + фильтруем search tools по signals
    hidden = {"final_answer", "compose_context"}
    signals = getattr(self, "_query_signals", None)
    if signals:
        if not signals.date_from and signals.strategy_hint != "temporal":
            hidden.add("temporal_search")
        if not signals.channels and signals.strategy_hint != "channel":
            hidden.add("channel_search")
    return [t for t in AGENT_TOOLS if t["function"]["name"] not in hidden]
```

### Constraint:
- **Max 5 tools** видимых для Qwen3-30B (3B active) — "Less is More" (R13 research, подтверждено Яндексом). При >5-7 tools accuracy деградирует.
- Tool descriptions должны быть в пределах **2000 токенов** суммарно (Яндекс R15: 10K+ токенов описаний → падение accuracy всех моделей, даже сильных).
- Total pool может быть 10-15, но dynamic visibility ограничивает до 5.
- Все search tools под капотом вызывают один `search()` с разными фильтрами — thin wrappers.

---

## Retrieval pipeline (под капотом search tools)

Все search tools вызывают один и тот же pipeline:
1. **Query decomposition** (query_plan) → 3-5 subqueries
2. **Per-subquery search**:
   - BM25 sparse: top-100 (Qdrant native sparse vector, без внешнего индекса)
   - Dense: top-20 (Qwen3-Embedding-0.6B, 1024-dim)
   - **Weighted RRF** fusion: `score = 1/(k+rank)`, BM25 weight = 3, Dense weight = 1
   - **ColBERT rerank**: jina-colbert-v2, MaxSim per-token scoring, top-10
3. **Round-robin merge** subquery results (preserving per-subquery ranking)
4. **Cross-encoder rerank** (bge-reranker-v2-m3) → top_n
5. **Channel dedup**: max 2 docs per channel
6. **Coverage check** → refinement if coverage < 0.65

Доступные metadata фильтры в Qdrant (payload fields):
- `channel` (string) — username канала (@gonzo_ml, @techsparks, ...)
- `date` (datetime) — дата публикации (range filter)
- `channel_id` (int), `message_id` (int) — уникальный ID
- `author` (string), `is_forward` (bool), `reply_to` (string|null)
- `links` (array<string>), `media_types` (array<string>)
- `lang` (string: "ru"|"en"), `hash` (string)

---

## Паттерны запросов пользователей (из eval dataset v3, 30 вопросов)

| Категория | Кол-во | Примеры | Текущий tool |
|-----------|--------|---------|--------------|
| **temporal** | 7 | "Что произошло в декабре 2025?", "Что было на GTC 2026?" | temporal_search ✅ |
| **channel** | 7 | "О чём писал gonzo_ml?", "Что Борис Цейтлин думает о Claude?" | channel_search ✅ |
| **entity** | 7 | "Что нового у DeepSeek?", "Что известно про FLUX.2?" | search ⚠️ (нет entity-специфичного) |
| **broad** | 5 | "Какие open-source модели для генерации?", "AI в медицине?" | search ✅ |
| **negative** | 2 | "Существует ли GPT-7?", "Выходила ли Bard 3?" | search ✅ (должен отказать) |
| **comparative** | 0 | — НЕТ в eval | — НЕТ tool |
| **trending/digest** | 0 | — НЕТ в eval | — НЕТ tool |
| **multi-hop** | 0 | — НЕТ в eval | — НЕТ tool |

**Проблема**: 23% запросов (entity) не имеют оптимального инструмента. Ещё 3 категории (comparative, trending, multi-hop) вообще не покрыты — и не тестируются.

---

## Ключевые findings из предыдущих ресерчей

### R13-deep: Tool Router Architecture
- **"Less is More"** (arXiv:2411.15399): Llama3.1-8B failed на 13+ tools, но **95%+ accuracy на 5 tools**. Qwen2-7B аналогично.
- **Thin wrappers**: все search tools = обёртки над одной base search function + разные фильтры. Это правильная архитектура.
- **Dynamic visibility**: regex pre-scan по query signals → скрытие ненужных tools. Уже реализовано.
- **A-RAG paper** (Feb 2026): 3 tools (keyword_search, semantic_search, chunk_read) → outperformed GraphRAG, RAPTOR, LightRAG.

### R14-deep: Beyond Frameworks
- **Ablation study** = "секретное оружие" для портфолио. Показать что каждый компонент вносит вклад.
- **Query complexity routing**: простые запросы → single-shot, сложные → multi-step agent.
- **Tool diversity** vs **tool count**: лучше 5 разных типов tools чем 10 вариаций одного.

### R15: Яндекс YaC AI Meetup
- **Яндекс нейроюрист**: 1000+ тулов, динамическая подкладка через "RAG для RAG" (embedding описаний → retrieve по запросу).
- **MarketEye (Вихров)**: tool design от Anthropic — консолидация, semantic IDs, explicit "когда НЕ использовать".
- **FC синтетика (Цымбой)**: сложность FC = f(суммарный размер tool descriptions), НЕ количество tools. 45K tools при обучении, 4 уровня сложности.
- **Irrelevance subset**: модель должна уметь НЕ вызывать tools. Наш forced search — противоположность.

---

## Что показал Яндекс (R15 analysis, YaC AI Meetup 2026-03-21)

1. **Соколов (RAG для Алисы/нейроюриста)**: "Не все 1000 тулов в контекст, а отбор релевантных" — динамическая подкладка через "RAG для RAG" (embedding описаний тулов → retrieve по запросу).
2. **Вихров (MarketEye)**: Принципы Anthropic — консолидация (4+ API → 1 tool), явное описание "когда использовать / когда НЕ", разделение ответственности (hardcoded params в сервисе, информативные — генерирует агент).
3. **Цымбой (Function Calling)**: 45K инструментов при обучении. Сложность от simple до complex. Irrelevance subset — нужный инструмент исключается, модель учится давать текстовый ответ.

---

## Что я хочу получить от ресерча

### 1. Каталог инструментов для новостного RAG-агента

Мне нужен **конкретный список инструментов** которые имеет смысл добавить в RAG-агент работающий над новостным корпусом (Telegram AI/ML каналы). Для каждого инструмента:
- **Имя и описание** (как в OpenAI function calling schema)
- **Параметры** (что генерирует LLM)
- **Что делает под капотом** (какие Qdrant запросы, агрегации, вычисления)
- **Зачем нужен** — какой тип запросов решает, чего нельзя добиться существующими tools
- **Пример запроса** от пользователя, который этот tool решает лучше чем broad search
- **Оценка сложности** реализации (low/medium/high)

Кандидаты которые мы уже рассматриваем (нужно подтвердить или отклонить с аргументами):
- `compare_search` — сравнение двух+ сущностей/тем
- `trending_search` / `digest` — что обсуждали чаще всего за период
- `author_search` — поиск по конкретному автору (не каналу)
- `fact_check` — кросс-проверка факта по нескольким каналам
- `summarize_channel` — сводка по каналу за период
- `entity_search` — поиск по конкретной сущности (модель, компания, технология)
- `read_post` — прочитать полный текст поста по ID (для iterative refinement)
- `list_channels` — показать доступные каналы
- `related_posts` — найти похожие посты к уже найденному

### 2. Архитектура dynamic visibility

Исследовать подходы к управлению видимостью инструментов:
- **Rule-based** (наш текущий подход): regex + heuristics по query signals
- **Embedding-based** (Яндекс "RAG для RAG"): embed query + embed tool descriptions → top-k tools
- **Classifier-based**: обученный классификатор query → tools subset
- **RATS pattern** (Retrieval-Augmented Tool Selection): retrieve tools из реестра по запросу

Для каждого подхода: плюсы, минусы, сложность реализации, requirements.

### 3. Группировка и маршрутизация

Как организовать 10-15 tools так чтобы LLM видел не больше 5:
- **По типу запроса**: factual, analytical, temporal, comparative, exploratory
- **По фазе**: pre-search tools, search tools, post-search tools
- **По специализации**: data retrieval, data analysis, navigation

### 4. Papers и evidence

Нужны конкретные ссылки на:
- Papers про tool design в agentic RAG (2024-2026)
- Papers про dynamic tool selection / tool routing
- Industry case studies (Perplexity Focus Modes, You.com agents, Bing Chat tools)
- Open-source реализации (LlamaIndex tools, LangChain agent tools, CrewAI)
- Benchmarks: как количество и тип tools влияет на accuracy (ToolBench, BFCL, etc.)

### 5. Anti-patterns

Что **НЕ стоит** добавлять в RAG-агент:
- Tools которые дублируют друг друга
- Tools которые слишком сложны для описания в 50 слов
- Tools которые требуют multi-step внутри одного вызова
- Tools для которых нет данных в нашем корпусе

---

## Ограничения

1. **Закрытая база** — нет web search, нет API к внешним сервисам. Все данные = Qdrant коллекция с Telegram постами.
2. **Single-agent** — не мультиагентная система. Один LLM с набором tools.
3. **Qwen3-30B-A3B** — MoE модель с 3B active params. Хорошо справляется с function calling (Яндекс: сопоставим с GPT-4o при правильном SFT), но чувствителен к количеству tools в schema.
4. **Latency** — целевой first-token <3s. Tools не должны добавлять >1s к pipeline.
5. **Qdrant-only** — все поисковые tools работают через Qdrant. Нет внешних баз, графов знаний, SQL.
6. **Русский язык** — основной язык запросов и контента.

---

## Контекст для портфолио

Этот проект — основа портфолио для позиции Applied LLM / MLOps Engineer (Middle+). Расширение тулов должно показать:
1. **Глубину архитектуры** — не "LangChain + 3 промпта", а продуманная система с routing, visibility, fallback
2. **Понимание trade-offs** — почему именно эти tools, почему не больше, как выбирали
3. **Ablation story** — каждый tool вносит измеримый вклад (eval before/after)
4. **Alignment с industry** — ссылки на papers, подходы Яндекса/Anthropic/Perplexity
5. **Practical value** — tools решают реальные query patterns, не теоретические

---

## Формат ответа

Структурированный отчёт на русском с:
1. Обзор литературы (papers, industry, open-source)
2. Рекомендуемый каталог tools (таблица + подробные описания каждого)
3. Архитектура dynamic visibility (сравнение подходов)
4. Группировка и маршрутизация (schema)
5. Конкретный implementation plan (что добавить первым, что вторым)
6. Anti-patterns и что НЕ делать
7. Ожидаемый эффект на метрики

Все рекомендации с ссылками на papers/sources. Без воды — конкретные решения для нашего домена.
