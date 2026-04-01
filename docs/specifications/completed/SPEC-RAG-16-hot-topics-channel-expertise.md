# SPEC-RAG-16: hot_topics + channel_expertise — Pre-computed Analytics

> **Status**: Draft → Review
> **Research base**: R17-deep-domain-specific-tools.md (T2 + T5)
> **Depends on**: SPEC-RAG-12 (payload enrichment — done), SPEC-RAG-15 (analytics tools — done)
> **Scope**: два новых LLM tool + cron pipeline + две auxiliary Qdrant коллекции

---

## Цель

Добавить `hot_topics` и `channel_expertise` — два tool'а, работающих на **pre-computed данных** (weekly/monthly cron), а не на real-time retrieval. Это завершает аналитическую триаду: entity_tracker (real-time facets) + hot_topics (pre-computed trends) + channel_expertise (pre-computed profiles).

**Interview value**: показывает production-grade cron pipeline, BERTopic clustering, pre-computed aggregations — то, что ни один RAG фреймворк не даёт из коробки.

---

## Архитектура

### Общая схема

```
[Weekly cron script]
  ├── Qdrant scroll (ALL posts 13K+, with pre-computed embeddings)
  ├── BERTopic fit_transform on full corpus (stable topic space)
  ├── Filter to this week → hot_score computation
  ├── LLM digest generation (Qwen3-30B)
  └── Upsert → weekly_digests collection

[Monthly cron script]
  ├── Qdrant scroll (all posts, 13K+)
  ├── Per-channel aggregation (entity counts, topic distribution, posting patterns)
  ├── Authority score computation
  └── Upsert → channel_profiles collection

[Agent runtime]
  ├── hot_topics tool → scroll weekly_digests → instant response (<10ms)
  └── channel_expertise tool → scroll/search channel_profiles → instant response (<10ms)
```

### Auxiliary коллекции

**`weekly_digests`** (~38 points, одна на неделю, 9 месяцев данных):

```json
{
  "id": "2026-W12",
  "vector": [/* embedded summary, 1024-dim */],
  "payload": {
    "period": "2026-W12",
    "date_from": "2026-03-16",
    "date_to": "2026-03-22",
    "post_count": 347,
    "summary": "На этой неделе основные темы...",
    "topics": [
      {
        "label": "DeepSeek V3 API и pricing",
        "hot_score": 0.87,
        "post_count": 42,
        "channels": ["ai_newz", "seeallochnaya", "cryptovalerii"],
        "representative_post_ids": ["uuid1", "uuid2"],
        "keywords": ["deepseek", "v3", "api", "pricing", "rate limits"]
      }
    ],
    "top_entities": [
      {"entity": "DeepSeek", "count": 89},
      {"entity": "OpenAI", "count": 72}
    ],
    "burst_events": [
      {"topic": "DeepSeek V3 release", "channels": 8, "first_seen": "2026-03-18T10:00:00Z"}
    ]
  }
}
```

**`channel_profiles`** (36 points, одна на канал):

```json
{
  "id": "gonzo_ml",
  "vector": [/* embedded profile summary, 1024-dim */],
  "payload": {
    "channel": "gonzo_ml",
    "display_name": "Григорий Сапунов",
    "total_posts": 287,
    "post_frequency": {"avg_per_week": 7.2, "active_weeks": 38},
    "top_entities": [
      {"entity": "Transformer", "count": 45},
      {"entity": "OpenAI", "count": 38}
    ],
    "top_topics": ["architecture", "papers", "training"],
    "authority_score": 0.82,
    "speed_score": 0.65,
    "breadth_score": 0.71,
    "profile_summary": "Эксперт по архитектурам нейросетей...",
    "updated_at": "2026-03-28T00:00:00Z"
  }
}
```

---

## Tool 1: `hot_topics`

### API

```python
def hot_topics(
    period: str = "this_week",  # "this_week" | "last_week" | "YYYY-WNN" | "this_month"
    top_n: int = 5
) -> dict
```

### Response

```json
{
  "period": "2026-W12",
  "post_count": 347,
  "summary": "...",
  "topics": [
    {
      "label": "DeepSeek V3 API и pricing",
      "hot_score": 0.87,
      "post_count": 42,
      "channels": ["ai_newz", "seeallochnaya"],
      "keywords": ["deepseek", "v3", "api"]
    }
  ],
  "top_entities": [...],
  "burst_events": [...]
}
```

### Hot score formula (R17)

```
hot_score = 0.3 × volume_norm + 0.3 × channel_diversity + 0.3 × recency + 0.1 × velocity

volume_norm = post_count / max_post_count_across_topics
channel_diversity = unique_channels / 36
recency = exponential_decay(median_post_date, half_life=3_days)
velocity = (this_week_count - last_week_count) / max(last_week_count, 1)
```

### Реализация

- **Файл**: `src/services/tools/hot_topics.py`
- **Qdrant access**: tool создаёт свой `QdrantClient` для коллекции `weekly_digests` (НЕ через DI singleton `QdrantStore` который привязан к `news_colbert_v2`). URL берётся из `settings.qdrant_url`. Аналогично entity_tracker/arxiv_tracker которые работают через `hybrid_retriever._store.client`.
- **Period resolution**: "this_week" → текущий ISO week, "this_month" → агрегация последних 4 weekly_digests
- **Без citations**: hot_topics — analytics tool, verify bypass (аналогично entity_tracker)

---

## Tool 2: `channel_expertise`

### API

```python
def channel_expertise(
    channel: str | None = None,     # Конкретный канал или None для ranking
    topic: str | None = None,       # Найти каналы по теме
    metric: str = "authority"       # "authority" | "speed" | "volume" | "breadth"
) -> dict
```

### Modes

1. **channel=X**: вернуть профиль конкретного канала (scroll by id)
2. **topic=Y**: semantic search по profile summaries → каналы-эксперты по теме (vector search)
3. **channel=None, topic=None**: ranking всех каналов по metric (scroll all + sort)

### Authority score (решение по дыре из R17)

R17 не определяет формулу authority. Определяем:

```
authority_score = 0.4 × entity_coverage + 0.3 × consistency + 0.2 × uniqueness + 0.1 × volume_norm

entity_coverage = unique_entities_mentioned / total_known_entities (из entity_dictionary)
consistency = active_weeks / total_weeks (регулярность постинга)
uniqueness = fraction_of_posts_that_are_original (не forwards)
volume_norm = total_posts / max_posts_across_channels
```

Дополнительные метрики:
- **speed** = для каждой entity в канале: days_after_first_global_mention (дата первого поста в канале - дата первого поста об entity в любом канале). `speed_score = 1 - median(delays) / max_delay`. Считается из payload полей `date` + `entities[]`, без event detection. Низкий delay = высокий speed. **V1 scope**: считать только для top-20 entities по частоте, остальные — default 0.5.
- **breadth** = unique_topics / total_topics (topic diversity, из BERTopic topic assignments)

### Реализация

- **Файл**: `src/services/tools/channel_expertise.py`
- **Qdrant access**: свой `QdrantClient` для коллекции `channel_profiles` (аналогично hot_topics — НЕ через DI QdrantStore). Vector search для topic mode, scroll для channel/ranking mode.
- **Без citations**: analytics tool, verify bypass

---

## Cron pipeline

### Weekly cron: `scripts/compute_weekly_digest.py`

```
1. Scroll ALL posts из news_colbert_v2 (13K+, with_vectors=True)
   — брать только `point.vector["dense_vector"]` (1024-dim Qwen3 embedding),
     игнорировать sparse/ColBERT vectors.
   — полный корпус нужен для consistent BERTopic topic space (R17 §3)
2. BERTopic fit_transform на ВСЁМ корпусе:
   - UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
   - HDBSCAN(min_cluster_size=20, min_samples=5, prediction_data=True)
   - CountVectorizer(min_df=3, max_df=0.9, ngram_range=(1,2))
   - Tokenizer: default CountVectorizer (sufficient для mixed RU/EN AI/ML text,
     т.к. n-grams (1,2) ловят ключевые термины без морфологии)
   - ~60s на V100 с pre-computed embeddings
3. Filter topics to THIS WEEK's posts (last 7 days) → compute hot_score
4. Cross-channel burst detection: topic in 5+ channels within 48h → burst
5. Generate summary via LLM (Qwen3-30B, top 5 topics + entities → 300 words)
6. Embed summary (Qwen3-Embedding-0.6B)
7. Upsert point в weekly_digests
```

**Важно**: BERTopic fit на всём корпусе, hot_score фильтрует по неделе. Это даёт stable topic IDs для channel_profiles (monthly cron использует тот же topic space).

**Решение по русскому тексту в BERTopic**: CountVectorizer по умолчанию (без pymorphy2). Обоснование:
- AI/ML текст ~50% английских терминов (model names, frameworks, concepts)
- CountVectorizer с n-grams (1,2) ловит "open source", "language model", "нейронная сеть"
- BERTopic использует embeddings (Qwen3) для clustering, CountVectorizer только для c-TF-IDF labels
- pymorphy2 добавляет dependency + ~2s overhead, marginal gain для topic labels
- Если labels неинформативны → LLM-based labeling (уже в pipeline, step 6)

**Время выполнения**: ~2 минуты на V100 (BERTopic ~60s + LLM summary ~30s + embedding ~5s).

### Monthly cron: `scripts/compute_channel_profiles.py`

```
1. Load latest BERTopic model (сохраняется weekly cron → pickle/safetensors)
   — использует тот же topic space что и weekly_digests
2. Scroll all posts (13K+, grouped by channel)
3. Per channel:
   a. Count entities (из payload), topics (из BERTopic assignments), post frequency
   b. Compute authority/speed/breadth/volume scores
   c. Generate profile summary via LLM (100 words)
   d. Embed summary
4. Upsert 36 points в channel_profiles
```

**BERTopic model persistence**: weekly cron сохраняет fitted model в `datasets/bertopic_model/`. Monthly cron загружает его для topic assignments. Это гарантирует consistent topic space между weekly_digests и channel_profiles.

**Время**: ~10 минут на V100.

### Запуск

Cron скрипты запускаются вручную или через системный cron/task scheduler:

```bash
# Weekly (можно включить в Docker cron или запускать вручную)
docker compose -f deploy/compose/compose.dev.yml run --rm api \
  python scripts/compute_weekly_digest.py

# Monthly
docker compose -f deploy/compose/compose.dev.yml run --rm api \
  python scripts/compute_channel_profiles.py
```

---

## Agent integration

### Dynamic visibility

- `hot_topics`: keyword routing — "горячие", "тренды", "trending", "hot", "обсуждали на этой неделе", "дайджест недели"
- `channel_expertise`: keyword routing — "эксперт", "авторитет", "какой канал лучше", "кто разбирается", "профиль канала"

Добавить в `datasets/tool_keywords.json` → `tool_keywords` section.

### Agent state

- `hot_topics` → `analytics_done = True` (как entity_tracker)
- `channel_expertise` → `analytics_done = True`
- Verify bypass для обоих (analytics-only ответы без citations)

### System prompt

Добавить в SYSTEM_PROMPT описание двух новых tools. Описание должно описывать **выбор**, не хардкодить конкретный tool (feedback: system prompt must list tool CHOICES).

### AGENT_TOOLS schema

+2 tool definitions → 15 LLM-visible tools total. Dynamic visibility max 5 сохраняется.

---

## Acceptance criteria

1. `hot_topics("this_week")` возвращает топ-5 тем с hot_score, channels, keywords за <100ms
2. `hot_topics("2026-W10")` возвращает historical digest
3. `channel_expertise(channel="gonzo_ml")` возвращает профиль с authority/speed/breadth
4. `channel_expertise(topic="трансформеры")` возвращает топ каналов по теме
5. Weekly cron выполняется за <5 минут
6. Monthly cron выполняется за <15 минут
7. LLM корректно выбирает hot_topics для "что обсуждали на этой неделе?"
8. LLM корректно выбирает channel_expertise для "кто лучше пишет про NLP?"
9. Eval: добавить 3-5 вопросов в golden dataset для новых tools

---

## Не входит в scope

- GLiNER tier 2 NER (entities уже из regex dictionary — достаточно для 13K docs)
- Daily cron (incremental ingest) — отдельный scope, пока manual ingest
- Topic lifecycle analysis (emerging/mature/declining) — future, после baseline
- pymorphy2 integration — fallback если labels будут плохие (checkpoint после first run)

---

## Порядок реализации

```
1. Создать auxiliary коллекции в Qdrant (weekly_digests, channel_profiles)
2. Weekly cron script (BERTopic + hot_score + LLM summary + upsert)
3. hot_topics tool (scroll weekly_digests, format response)
4. Monthly cron script (channel aggregation + authority scores + upsert)
5. channel_expertise tool (scroll/search channel_profiles)
6. Agent integration (tool_keywords.json, AGENT_TOOLS, system prompt, state machine)
7. Eval: add 3-5 golden questions
8. Smoke test full pipeline
```

Estimated effort: **3-5 дней**.

---

## Dependencies

- BERTopic: `pip install bertopic` (+ hdbscan, umap-learn, scikit-learn)
- Embeddings: Qwen3-Embedding-0.6B через gpu_server.py (уже запущен)
- LLM: Qwen3-30B-A3B через llama-server.exe (уже запущен)
- Qdrant: docker compose (уже запущен)
