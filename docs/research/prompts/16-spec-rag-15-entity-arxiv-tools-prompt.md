# Промпт для анализа и написания SPEC-RAG-15: Entity Analytics Tools

## Контекст задачи

Ты — Claude Opus, разрабатываешь RAG-агент для поиска по AI/ML Telegram-каналам.
После compact тебе нужно:
1. Прочитать ключевые файлы (ниже список)
2. Проанализировать как именно реализовать entity_tracker и arxiv_tracker
3. Обсудить с пользователем ключевые решения
4. Написать SPEC-RAG-15

## Что уже сделано

### Инфраструктура (SPEC-RAG-12)
- Коллекция `news_colbert_v2`: 13088 points, 36 каналов, 38 недель (2025-W27 → 2026-W12)
- **16 keyword indexes** включая: `entities`, `entity_orgs`, `entity_models`, `arxiv_ids`, `year_week`, `year_month`, `channel`
- Entity dictionary: 95 AI/ML entities в 6 категориях (org/model/framework/technique/conference/tool), case_sensitive flag
- Regex NER при ingest → entities[] массив в payload каждого point

### Реальные данные в Qdrant
**Top entities**: OpenAI=1597, Google=1127, Claude=794, Gemini=713, Anthropic=701, GPT-5=680, NVIDIA=525, Qwen=427, DeepSeek=344
**Arxiv papers**: max 4 mentions per paper, ~50-70 уникальных paper IDs. Примеры: 2502.13266 (4), 1706.03762=Attention Is All You Need (2)
**Timeline**: 38 year_week значений, ~300-450 posts/week

### Facet API verification
Протестированные запросы:
- `facet("entities", limit=20)` → top entities с counts ✓
- `facet("arxiv_ids", limit=15)` → top papers ✓
- `facet("year_week", filter=entities contains "DeepSeek")` → timeline DeepSeek (видны пики W48-W49) ✓
- `facet("entities", filter=entities contains "NVIDIA")` → co-occurrence entities ✓

### Текущие 11 tools
query_plan, search, temporal_search, channel_search, cross_channel_compare, summarize_channel, list_channels, rerank, compose_context, final_answer, related_posts

### Паттерн существующих Facet-based tools
`list_channels` использует `store._client.facet()` через `hybrid_retriever._run_sync()` (sync bridge). Это референсная реализация.

## Что нужно реализовать

### entity_tracker
Аналитический tool: timeline, comparison, co-occurrence сущностей.

Возможные operations (mode parameter):
1. **top** — топ-N сущностей (overall или за period) → `facet("entities", filter=year_week/year_month)`
2. **timeline** — как entity менялось по неделям → `facet("year_week", filter=entities contains X)`
3. **compare** — сравнение двух entities по timeline → два facet-запроса
4. **co-occurrence** — что упоминается вместе с entity X → `facet("entities", filter=entities contains X)`

Фильтры: year_week range, channel, category (org/model/...)

### arxiv_tracker
Аналитический tool: popular papers, lookup, timeline.

Возможные operations:
1. **top** — самые обсуждаемые papers → `facet("arxiv_ids", filter=optional year_week)`
2. **lookup** — кто обсуждал paper X → `scroll(filter=arxiv_ids contains X)` → посты + каналы
3. **timeline** — когда обсуждался paper X → `facet("year_week", filter=arxiv_ids contains X)`

## Ключевые вопросы для анализа

### 1. Output format
Entity tools возвращают **counts/aggregations**, не documents. Как это интегрируется с agent flow?
- Вариант A: tool возвращает текстовый summary → LLM использует в final_answer напрямую
- Вариант B: tool возвращает counts + top posts → можно провести через compose_context

### 2. Tool schema для LLM
Один tool с `mode` parameter? Или несколько tools (entity_timeline, entity_compare, ...)?
R15 (Яндекс): "Less is More" — меньше tools = лучший FC accuracy. Но mode с enum — это тоже decision point.

### 3. Visibility rules
В какой фазе видны? PRE-SEARCH или отдельная ANALYTICS фаза? Keyword signals?

### 4. State machine
Increment search_count? Или это отдельная ветка (аналитика, не retrieval)?

### 5. Arxiv data sparsity
Max 4 mentions per paper. Достаточно ли для полезного tool? Или arxiv_tracker = nice-to-have?

### 6. Golden dataset
3 future_baseline вопроса уже в eval_golden_v1.json — обновить key_tools и expected для них.

## Файлы для чтения (по tool policy)

### Код (через repo-semantic-search → serena)
1. `src/services/tools/list_channels.py` — reference Facet API tool (паттерн)
2. `src/services/tools/cross_channel_compare.py` — reference для complex tool
3. `src/services/agent_service.py` → AGENT_TOOLS, _get_step_tools(), SYSTEM_PROMPT
4. `src/core/deps.py` → tool registration pattern
5. `scripts/payload_enrichment.py` → extract_from_text(), entity dictionary loading
6. `scripts/migrate_collection.py` → payload indexes list
7. `datasets/entity_dictionary.json` — 95 entities с categories

### Research
8. `docs/research/reports/R17-deep-domain-specific-tools.md` — entity_tracker, arxiv_tracker design
9. `docs/research/reports/R16-deep-rag-agent-tools-expansion.md` — generic tools, visibility

### Specs (completed, как reference)
10. `docs/specifications/completed/SPEC-RAG-13-simple-tools.md` — формат spec, tool schemas

## Deliverable

SPEC-RAG-15 в формате `docs/specifications/active/SPEC-RAG-15-entity-analytics-tools.md` с:
- Цель и контекст (ссылки на R17)
- Tool schemas (JSON для AGENT_TOOLS)
- Implementation details (Facet API queries, sync bridge)
- Visibility rules
- State machine integration
- Golden dataset updates
- Acceptance criteria
- Checklist
