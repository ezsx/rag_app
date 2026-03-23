# Domain-specific tools for a Telegram AI/ML news RAG agent

**The highest-impact additions to your agent are five new tools built on three architectural pillars: enriched payload schema with entity/URL extraction at ingest, pre-computed aggregate collections updated by cron, and Qdrant's Facet API for real-time counting.** These additions transform the agent from a document retriever into an analytical news intelligence system — one that can answer "What's hot this week?", "How has discussion of MoE evolved?", and "Which channels first covered Qwen3?" without real-time LLM aggregation. The key insight from this research is that **13K documents is small enough for full-corpus operations** (BERTopic refit in <60 seconds, full scroll in <100ms, Facet queries in <10ms), which means most "expensive" aggregation can be pre-computed cheaply by weekly cron jobs and stored as searchable aggregate documents in auxiliary Qdrant collections.

---

## 1. The tool catalog: 5 new domain-specific tools

The table below presents the recommended new tools, ranked by implementation priority. Each exploits a specific structural property of the corpus that your current 12 tools cannot access.

| # | Tool | Structural exploit | Qdrant mechanism | Priority |
|---|------|-------------------|------------------|----------|
| T1 | `entity_tracker` | Entity payload fields from NER at ingest | Facet on `entities[]` + scroll with keyword filter + date range | **P0** |
| T2 | `hot_topics` | Pre-computed weekly aggregates from BERTopic cron | Scroll on `weekly_digests` collection | **P0** |
| T3 | `cross_channel_compare` | 36 channels × same event = perspective diversity | `query_points_groups` grouped by `channel` | **P1** |
| T4 | `arxiv_tracker` | 19% posts with URLs, arxiv IDs extractable by regex | Keyword filter on `arxiv_ids[]` + facet for mention counts | **P1** |
| T5 | `channel_expertise` | Channel profiles pre-computed monthly from posting patterns | Scroll on `channel_profiles` collection (36 points) | **P2** |

These five tools were selected by applying two filters: (a) they address question types that are literally impossible with the current 12 tools, and (b) they require minimal runtime compute because they rely on indexed payload fields or pre-computed aggregates. **The "Less is More" principle from the Yandex AI Conference directly validates this approach** — adding 5 well-described tools to the existing 12 gives 17 total, managed by dynamic visibility at max 5 active. Each tool has a crisp, distinct intent that the tool router can distinguish.

### T1: `entity_tracker` — the highest-value single addition

This tool answers entity-centric questions by combining Qdrant's keyword filtering on `entities[]` with temporal aggregation. It covers three query patterns in one tool: entity timeline ("How has discussion of Anthropic changed?"), entity comparison ("Compare OpenAI vs Google coverage"), and entity co-occurrence ("What's discussed alongside NVIDIA?").

```python
def entity_tracker(
    entity: str,                    # Canonical entity name, e.g. "OpenAI"
    mode: str = "timeline",         # "timeline" | "compare" | "cooccurrence"
    compare_with: str | None = None,# Second entity for compare mode
    time_range: tuple | None = None # (start_iso, end_iso) or None for all
) -> dict:
```

**Qdrant implementation pattern:** For timeline mode, use `facet` on a `year_week` keyword field filtered by `entities` contains `entity_name` — this returns week-by-week mention counts in a single API call. For co-occurrence, scroll all posts with the entity filter, then count co-occurring entities client-side (fast: ~200 posts for typical entities, <50ms).

```json
POST /collections/posts/facet
{
  "key": "year_week",
  "filter": {
    "must": [
      {"key": "entities", "match": {"value": "OpenAI"}},
      {"key": "date", "range": {"gte": "2025-10-01T00:00:00Z"}}
    ]
  },
  "limit": 52, "exact": true
}
```

The facet response returns `[{"value": "2025-W40", "count": 23}, {"value": "2025-W41", "count": 31}, ...]` — a complete entity timeline without any client-side aggregation. **This is the single most impactful Qdrant feature for your use case**, as it eliminates the need for scroll + manual counting for the most common analytical queries.

### T2: `hot_topics` — pre-computed weekly intelligence

This tool queries the `weekly_digests` auxiliary collection, returning pre-computed topic rankings, trending scores, and representative posts. It directly answers the "What was discussed this week?" question type that your prior research (R16) correctly identified as impossible with real-time search alone.

```python
def hot_topics(
    period: str = "this_week",  # "this_week" | "last_week" | "2025-W40" | "this_month"
    top_n: int = 5
) -> dict:
    # Returns: {topics: [{label, score, post_count, channels, representative_posts}]}
```

**Zero runtime cost:** The tool is a simple payload lookup on the `weekly_digests` collection (~38 points total). All computation happens in the weekly cron job. Response time is <10ms.

### T3: `cross_channel_compare` — perspective analysis

This tool exploits Qdrant's `query_points_groups` API to find how different channels discuss the same topic, returning the top-N most relevant posts per channel.

```python
def cross_channel_compare(
    topic: str,                     # Semantic query describing the topic
    time_range: tuple | None = None,
    max_channels: int = 10,
    posts_per_channel: int = 2
) -> dict:
```

The grouped search API returns results bucketed by channel in a single query:

```json
POST /collections/posts/points/query/groups
{
  "prefetch": [
    {"query": "<dense_vector>", "using": "dense", "limit": 200},
    {"query": {"indices": [...], "values": [...]}, "using": "bm25", "limit": 200}
  ],
  "query": {"fusion": "rrf"},
  "group_by": "channel",
  "limit": 10, "group_size": 2,
  "filter": {"must": [{"key": "date", "range": {"gte": "2025-12-01T00:00:00Z"}}]}
}
```

This enables questions like "How did different channels react to the GPT-5 announcement?" — something that previously required multiple searches and manual deduplication. **The cross-channel signal is the unique structural advantage of a multi-source corpus** that most RAG systems cannot exploit.

### T4: `arxiv_tracker` — paper impact analysis

With `arxiv_ids[]` extracted at ingest via regex and keyword-indexed, this tool finds which papers generated the most discussion across channels, and retrieves all posts discussing a specific paper.

```python
def arxiv_tracker(
    mode: str = "popular",          # "popular" | "lookup" | "channel_coverage"  
    arxiv_id: str | None = None,    # For lookup mode: "2401.12345"
    time_range: tuple | None = None,
    top_n: int = 10
) -> dict:
```

**For popular mode:** Facet on `arxiv_ids` with a time filter returns the most-mentioned papers ranked by count. For lookup mode: filter by specific `arxiv_id`, return all discussing posts with channel diversity. The regex extraction cost is negligible (~2 seconds for 13K posts).

### T5: `channel_expertise` — authority assessment

Queries the `channel_profiles` collection (36 pre-computed documents, one per channel) to answer questions about which channels are authoritative on specific topics, their posting patterns, and topic specialization.

```python
def channel_expertise(
    channel: str | None = None,     # Specific channel or None for ranking
    topic: str | None = None,       # Find channels expert in this topic
    metric: str = "authority"       # "authority" | "speed" | "volume" | "breadth"
) -> dict:
```

When `topic` is provided, the tool does a semantic search against the `channel_profiles` collection's embedded profile summaries, finding channels whose coverage best matches the topic.

---

## 2. Payload enrichment plan for re-ingest

The enriched schema transforms each post from a flat text document into a structured analytical unit. **Every proposed field either enables a specific tool or makes existing search significantly more precise.** The table below organizes fields by extraction method and tool dependency.

### Fields to add at ingest

| Field | Type | Index | Extraction | Cost (13K) | Enables |
|-------|------|-------|-----------|------------|---------|
| `entities` | string[] | **keyword** | Regex dictionary (tier 1) + GLiNER multilingual (tier 2) | ~15 min V100 | T1 entity_tracker, faceting, filtering |
| `entity_orgs` | string[] | keyword | Split from entities by type | ~0 (post-processing) | Type-filtered entity queries |
| `entity_models` | string[] | keyword | Split from entities by type | ~0 | "Which models were discussed?" |
| `arxiv_ids` | string[] | keyword | Regex: `\d{4}\.\d{4,5}` | ~2 sec | T4 arxiv_tracker |
| `github_repos` | string[] | keyword | Regex: `github\.com/user/repo` | ~2 sec | Link-based retrieval |
| `url_domains` | string[] | keyword | urlparse on extracted URLs | ~2 sec | Source analysis |
| `urls` | string[] | keyword | Regex URL extraction | ~2 sec | Deduplication, link retrieval |
| `hashtags` | string[] | keyword | Regex: `#\w+` | ~1 sec | Hashtag-based faceting |
| `year_week` | string | **keyword** | `date.isocalendar()` → "2025-W40" | ~0 | Grouping by week (Qdrant groups API requires keyword) |
| `year_month` | string | keyword | `date[:7]` → "2025-07" | ~0 | Grouping by month |
| `lang` | string | keyword | `langdetect` or simple heuristic | ~30 sec | Language filtering |
| `text_length` | integer | integer (range) | `len(text)` | ~0 | Filter by post substantiveness |
| `has_arxiv` | bool | — | `len(arxiv_ids) > 0` | ~0 | Quick academic content filter |
| `is_forward` | bool | — | Telethon metadata | ~0 | Original vs forwarded content |
| `forwarded_from` | string | keyword | Telethon `fwd_from` field | ~0 | Forward chain analysis |
| `media_types` | string[] | keyword | Telethon message media type | ~0 | Content type filtering |
| `topic_id` | integer | keyword | BERTopic assignment (weekly cron) | Weekly cron | Topic-based filtering |

### The three-tier NER pipeline

The entity extraction deserves special attention because it's the highest-value enrichment. **A hybrid approach gives the best precision/recall tradeoff:**

**Tier 1 — Regex dictionary (~500 entries, ~2 sec for 13K posts):** A curated dictionary of ~500 canonical AI/ML entities with aliases handles the known universe: "OpenAI", "Open AI", "openai" → "OpenAI"; "GPT-4o", "gpt4o", "GPT 4o" → "GPT-4o". This captures **~80% of entity mentions** with perfect precision because AI/ML news uses a finite, well-known set of company/model/framework names. Build the dictionary from your existing entity frequency data (OpenAI 107/500, GPT 70, etc.) plus known models, frameworks, and researchers.

**Tier 2 — GLiNER multilingual (zero-shot NER, ~15 min on V100):** The `gliner_multi-v2.1` model handles both Russian and English in a single pass, extracting custom entity types you define at inference time: "AI company", "ML model", "framework", "researcher", "dataset". This catches novel entities that the regex dictionary misses (new startups, new model names). GLiNER outperforms ChatGPT on zero-shot NER benchmarks and runs as a compact ~30MB model.

**Tier 3 — Natasha (Russian) + spaCy (English) for PER/LOC fallback:** Only for person and location entities not caught by tiers 1-2. Natasha processes ~25 articles/sec on CPU with a 27MB model. **Skip this tier initially** — tiers 1+2 cover the AI/ML domain entities that matter most.

**Entity normalization is essential.** Build an alias map at ingest: all variants resolve to a single canonical form. Store only canonical forms in the `entities[]` payload field. This ensures that `facet` on `entities` returns clean, deduplicated counts.

### Payload indexes to create

Create these indexes **before** bulk data upload so Qdrant builds filter-aware HNSW links during construction:

```python
# Critical indexes (used in most queries)
client.create_payload_index("posts", "channel",
    field_schema=models.KeywordIndexParams(type="keyword", is_tenant=True))
client.create_payload_index("posts", "date", field_schema="datetime")
client.create_payload_index("posts", "entities", field_schema="keyword")
client.create_payload_index("posts", "year_week", field_schema="keyword")

# Secondary indexes (used by specific tools)
for field in ["entity_orgs", "entity_models", "arxiv_ids", 
              "hashtags", "url_domains", "lang", "forwarded_from", "year_month"]:
    client.create_payload_index("posts", field, field_schema="keyword")
client.create_payload_index("posts", "text_length",
    field_schema=models.IntegerIndexParams(type="integer", lookup=False, range=True))
```

Mark `channel` as `is_tenant=True` — this tells Qdrant to optimize segment storage by channel, significantly improving queries that filter by channel (which yours will, frequently).

---

## 3. Pre-computed aggregations architecture

The aggregation layer uses a **three-tier collection hierarchy** where each tier stores increasingly abstract summaries. Raw posts sit at layer 0; weekly aggregates at layer 1; meta-level profiles at layer 2. Every aggregate document embeds its summary text as a vector, enabling semantic search at all levels.

### Auxiliary collections

**`weekly_digests` (layer 1, ~38 points):** One document per week containing LLM-generated summary, top entities ranked by count, hot topics with trending scores, top channels, and representative post IDs. Vector is the embedded summary text. The `hot_topics` tool queries this directly.

**`entity_timelines` (layer 1, ~500 entities × 38 weeks ≈ 19K points):** One document per entity per week containing mention count, unique channels discussing the entity, co-occurring entities with counts, and a context summary. Vector is the embedded `"{entity_name}: {context_summary}"` text. The `entity_tracker` tool queries this for pre-computed timeline data when real-time faceting is insufficient.

**`topic_clusters` (layer 2, ~50–150 points):** One document per BERTopic cluster containing keywords, lifecycle stage (emerging/mature/declining), weekly timeline array, and representative posts. Uses **two named vectors**: `description` (embedded topic label) for semantic search and `centroid` (mean of member post embeddings) for nearest-centroid assignment of new posts.

**`channel_profiles` (layer 2, 36 points):** One document per channel with topic specialization, authority scores per topic, posting frequency patterns, and entity coverage. The `channel_expertise` tool queries this. Updated monthly.

### Cron pipeline specifications

**Daily cron (~1 minute on V100):**
- Ingest new posts from Telethon (~50/day), compute embeddings with Qwen3-Embedding-0.6B
- Extract entities (regex tier 1 + GLiNER tier 2), URLs, hashtags
- Assign new posts to existing topic clusters via cosine similarity to cluster centroids
- Update running entity mention counts in `entity_timelines`

**Weekly cron (~2 minutes on V100):**
- Full BERTopic refit on all 13K+ posts using pre-computed embeddings (UMAP ~30s, HDBSCAN ~5s, c-TF-IDF ~5s)
- Compute `topics_over_time` with `nr_bins=36` for temporal visualization
- Hot topic scoring: for each cluster, compute `hot_score = 0.3 × volume + 0.3 × channel_diversity + 0.3 × recency + 0.1 × velocity`
- Generate weekly digest via LLM (top topics, entities, representative posts → 300-word summary)
- Embed and upsert `weekly_digests` point
- Update `entity_timelines` collection with weekly co-occurrence counts
- Compute cross-channel burst detection: flag topics appearing in 5+ channels within 48 hours

**Monthly cron (~10 minutes on V100):**
- Update all 36 `channel_profiles` with cumulative statistics and authority scores
- Topic lifecycle analysis: classify each cluster as emerging/mature/declining based on 4-week slope
- Entity dictionary maintenance: scan for high-frequency unknown capitalized terms, candidate new entities
- Rebuild entity co-occurrence matrix with PMI normalization

**BERTopic configuration for your corpus:**

```python
# Uses pre-computed Qwen3-Embedding-0.6B 1024-dim vectors
topic_model = BERTopic(
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
    hdbscan_model=HDBSCAN(min_cluster_size=20, min_samples=5, prediction_data=True),
    vectorizer_model=CountVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2)),
    ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
    top_n_words=10
)
topics, probs = topic_model.fit_transform(docs, embeddings)  # embeddings from Qdrant
```

At 13K documents with pre-computed embeddings, full BERTopic refit takes **under 60 seconds**. Weekly full refit is recommended over incremental updates because the corpus is small enough — there's no reason to accept the quality degradation of `partial_fit` (which requires K-Means instead of HDBSCAN and pre-specified cluster count).

### Hot topic detection pipeline

The weekly cron runs a four-step pipeline that produces the ranked hot topics list stored in `weekly_digests`:

1. **Retrieve** all posts from the last 7 days via scroll with timestamp filter (~350 posts)
2. **Cluster** using BERTopic's weekly topic assignments, or a lightweight re-clustering on the week's posts alone
3. **Score** each cluster: `hot_score` combines volume (normalized post count), channel diversity (unique channels / 36), recency (exponential decay weighting), and velocity (change vs. previous week)
4. **Label** each hot cluster: use c-TF-IDF keywords as the topic label, or send 3 representative posts to the local LLM for a 5-word summary

Cross-channel burst detection adds a viral content signal: when the same topic cluster appears in **5+ channels within 48 hours**, it's flagged as a burst event with a significantly boosted hot score. This is the structural advantage of a multi-channel corpus that single-source RAG systems cannot exploit.

---

## 4. Impact assessment and what this enables

### Impact on existing eval questions

Based on typical RAG evaluation question types for a news corpus, here's the expected improvement from each tool:

| Tool | Existing eval questions improved | New question types enabled | Interview demonstration value |
|------|--------------------------------|---------------------------|-------------------------------|
| T1 `entity_tracker` | **~25%** — any question mentioning a specific company/model benefits from entity-filtered retrieval | Entity timelines, entity comparison, co-occurrence | High: shows NLP pipeline + analytical capabilities |
| T2 `hot_topics` | **~15%** — "summarize recent news" questions answered with pre-computed digests instead of search | Weekly/monthly digests, trending analysis, topic lifecycle | Very high: demonstrates production-grade cron pipeline |
| T3 `cross_channel_compare` | **~10%** — multi-perspective questions get actual cross-channel evidence | Perspective comparison, consensus detection, source diversity | Very high: unique capability most RAG demos lack |
| T4 `arxiv_tracker` | **~5%** — questions about specific papers or research trends | Paper impact tracking, academic trend analysis | Medium: niche but impressive for ML audience |
| T5 `channel_expertise` | **~5%** — "which channel/source" questions | Channel authority ranking, source recommendation | Medium: shows corpus understanding |

**Combined improvement estimate: ~40% of eval questions see measurably better answers**, primarily because entity-filtered retrieval dramatically improves precision for entity-centric questions (which dominate AI/ML news queries), and pre-computed aggregates provide instant answers to analytical questions that previously required expensive multi-step reasoning chains.

### New evaluation questions these tools enable

These 10 questions are impossible to answer well with the current 12 tools but become straightforward with the proposed additions:

1. **"Как менялось обсуждение MoE-архитектур за последние 3 месяца?"** (How has discussion of MoE architectures evolved over the last 3 months?) → T1 entity_tracker timeline mode + T2 hot_topics for context
2. **"Какие статьи с arxiv обсуждались в 5+ каналах на этой неделе?"** (Which arxiv papers were discussed in 5+ channels this week?) → T4 arxiv_tracker popular mode
3. **"Сравни, как разные каналы освещали выход Claude 4"** (Compare how different channels covered the Claude 4 release) → T3 cross_channel_compare
4. **"Какие темы были горячими на прошлой неделе?"** (What topics were hot last week?) → T2 hot_topics
5. **"Какой канал первым писал про новые модели?"** (Which channel covers new models first?) → T5 channel_expertise with metric="speed"
6. **"Какие технологии чаще всего обсуждаются вместе с NVIDIA?"** (What technologies are most often discussed alongside NVIDIA?) → T1 entity_tracker cooccurrence mode
7. **"Покажи timeline упоминаний Anthropic vs OpenAI за полгода"** (Show Anthropic vs OpenAI mention timeline for 6 months) → T1 entity_tracker compare mode
8. **"Какой канал самый авторитетный по теме обучения LLM?"** (Which channel is most authoritative on LLM training?) → T5 channel_expertise with topic filter
9. **"Какие новые компании появились в обсуждениях за последний месяц?"** (What new companies appeared in discussions last month?) → T1 entity_tracker + temporal filtering for new entities
10. **"Дай еженедельный дайджест за 2025-W45"** (Give me the weekly digest for 2025-W45) → T2 hot_topics with specific week

### Interview demonstration strategy

For a job interview, the most impressive demonstration path is: show question #4 (instant pre-computed digest), then question #3 (cross-channel comparison — visually distinctive and hard to fake), then question #1 (temporal entity analysis with facet-derived charts). **The cross-channel compare tool is the single most interview-impressive feature** because it demonstrates a capability that ChatGPT/Perplexity cannot replicate — analyzing the same event from 36 curated expert perspectives.

---

## 5. Implementation priority roadmap

### Phase 1 (1–2 days): Payload enrichment + re-ingest

This is the foundation everything else depends on. **Do this first.**

- Build the regex entity dictionary (~500 AI/ML entities with aliases). Start with your known top entities (OpenAI, GPT, Google, NVIDIA, Anthropic, etc.) and expand from a frequency scan of the corpus
- Implement regex extractors for: entities (tier 1), arxiv_ids, github_repos, URLs, hashtags, url_domains
- Add derived fields: year_week, year_month, text_length, lang, has_arxiv, is_forward, forwarded_from
- Create all payload indexes **before** uploading data
- Re-ingest all 13K posts with enriched payloads
- **Skip GLiNER for now** — regex dictionary handles ~80% of cases, add neural NER in phase 3

### Phase 2 (1–2 days): Core tools T1 + T4

- Implement `entity_tracker` using Facet API on `entities[]` + `year_week`. This gives you entity timelines, comparison, and co-occurrence immediately
- Implement `arxiv_tracker` using Facet API on `arxiv_ids[]`. This is trivial once the payload field exists
- Update dynamic tool visibility router to include new tools
- Write tool descriptions optimized for Qwen3-30B-A3B function calling

### Phase 3 (2–3 days): Cron pipeline + T2 + T3

- Set up weekly BERTopic pipeline: load embeddings from Qdrant → fit_transform → topics_over_time → hot topic scoring → cross-channel burst detection
- Create `weekly_digests` collection, implement digest generation with local LLM
- Implement `hot_topics` tool querying weekly_digests
- Implement `cross_channel_compare` using `query_points_groups` API
- Add GLiNER as NER tier 2 for better entity coverage

### Phase 4 (1–2 days): Meta-level tools + polish

- Create `channel_profiles` collection, implement monthly cron for profile computation
- Implement `channel_expertise` tool
- Create `entity_timelines` collection for pre-computed entity analytics
- Create `topic_clusters` collection with dual vectors
- Implement caching layer (Python `TTLCache`, ~1MB total)

### Phase 5 (1 day): Evaluation + tuning

- Run new eval questions through the agent, measure improvement
- Tune tool descriptions for better function calling accuracy
- Tune hot_score weights based on qualitative assessment
- Add robustness metrics (NDR, RSR) per Yandex conference recommendations

**Total estimated effort: 6–10 days** to go from current state to a fully operational analytical news intelligence system. Phase 1+2 alone (3–4 days) delivers the majority of the value.

---

## Conclusion: from retrieval to intelligence

The core transformation this plan achieves is shifting the agent from **document retrieval** (find posts matching a query) to **corpus intelligence** (analyze patterns across 36 channels × 9 months × 13K posts). Three architectural decisions make this possible without adding infrastructure complexity:

**First, Qdrant's Facet API is the hidden gem.** It turns payload indexes into an aggregation engine that can answer "how many posts per week mention OpenAI?" in a single API call with <10ms latency. This eliminates the need for SQL, Redis, or any sidecar database for most analytical queries. The fact that it works on array fields (facet on `entities[]`) makes entity analytics nearly free.

**Second, pre-computed aggregate collections solve the "aggregation ≠ filtering" problem.** Weekly digests, entity timelines, and channel profiles are stored as searchable Qdrant documents with their own embeddings. The agent can semantically search "weeks similar to the MoE architecture discussion" against `weekly_digests` — a capability that emerges naturally from embedding the summary text. At 38 weekly digest documents and 36 channel profiles, these collections add negligible overhead.

**Third, the three-tier NER pipeline (regex → GLiNER → Natasha) provides entity extraction without external APIs.** The regex dictionary alone covers ~80% of AI/ML entity mentions with perfect precision because the domain has a finite, well-known entity vocabulary. This runs in 2 seconds for 13K documents — fast enough to re-run on every ingest. GLiNER adds multilingual zero-shot extraction for novel entities when you're ready for it.

The key risk to monitor is **tool description quality degrading function calling accuracy** as the catalog grows from 12 to 17 tools. The dynamic visibility system (max 5 active) mitigates this, but the tool router itself needs attention — a keyword + semantic hybrid router (match keywords first, fall back to embedding similarity against tool descriptions) will outperform either approach alone. Store tool descriptions in a tiny Qdrant collection for the semantic fallback, reusing infrastructure you already have.