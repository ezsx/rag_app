# Prompt 41: Custom RAG vs Framework RAG — benchmark design

## Цель

Спроектировать объективный бенчмарк: наш custom RAG pipeline vs лучший RAG-фреймворк (LlamaIndex / LangChain / другой) на одних данных и вопросах. Результат — артефакт для портфолио, демонстрирующий инженерную зрелость и понимание tradeoffs.

## Полное описание нашего проекта

### Что это

Production RAG-система для агрегации и поиска по русскоязычным Telegram-каналам (новости AI/ML/LLM). Полностью custom, zero frameworks. FastAPI backend, SSE streaming, web UI.

**Масштаб данных:** ~50 Telegram-каналов, ~200K+ постов, enriched payloads (entities, arxiv_ids, URLs, language, year_week).

**Codebase:** 14.5K LOC Python, 22 test files (105 passing), mypy strict 0 errors, CI (ruff + pytest + mypy). Code quality 9/10 по независимому аудиту.

### Retrieval pipeline — пошагово

```
User query
  → QueryPlannerService (LLM generates SearchPlan via /v1/chat/completions, JSON mode)
    → SearchPlan: normalized_queries[], must_phrases[], should_phrases[], metadata_filters, k_per_query, fusion, strategy
  → HybridRetriever.search_with_plan()
    → For each subquery (round-robin merge across all LLM subqueries):
      1. BM25 sparse search (top-100) via fastembed Qdrant/bm25
      2. Dense embedding search (top-20) via pplx-embed-v1-0.6B (bf16, mean pooling, no instruction prefix)
      3. Weighted RRF fusion (BM25 weight=3.0, dense weight=1.0) — BM25-heavy for Russian keyword matching
      4. ColBERT MaxSim rerank via jina-colbert-v2 (560M, 128-dim per-token)
         → Qdrant native multivector query (query_points with prefetch + RRF + ColBERT rescore)
    → Channel dedup (max 2 results per channel)
  → Cross-encoder rerank via Qwen3-Reranker-0.6B (seq-cls, chat template, padding_side=left, logit scoring)
  → Top-K candidates with scores
```

**Ключевые решения:**
- BM25 weight 3:1 vs dense — русский текст с аббревиатурами (LLM, SSM, MoE) плохо ложится на dense-only
- ColBERT как third-stage reranker, не first-stage — дорогой, но точный per-token matching
- Query planning: LLM генерирует 2-5 subqueries, round-robin merge — multi-faceted coverage
- Original query always injected в subqueries для BM25 keyword match

### Qdrant schema

Collection: `news_colbert_v2`

**Named vectors:**
- `dense_vector`: 1024-dim (pplx-embed-v1, cosine)
- `sparse_vector`: BM25 via fastembed (IDF modifier, in-memory index)
- `colbert_vector`: multivector, 128-dim per token (jina-colbert-v2, cosine)

**Payload (enriched):**
```
text, channel, channel_id, message_id, date, author, views,
root_message_id, url, forwarded_from_id,
entities_persons[], entity_orgs[], entity_models[],
arxiv_ids[], hashtags[], url_domains[],
lang, year_week, year_month, text_length
```

**16 payload indexes** для Facet API (analytics tools):
- KEYWORD: channel, channel_id, date, year_week, entities_persons, entity_orgs, entity_models, arxiv_ids, hashtags, url_domains, lang, forwarded_from_id, year_month, root_message_id, author
- INTEGER (range): message_id, text_length

### ReAct Agent — полная архитектура

**Оркестрация:** native function calling через llama-server `/v1/chat/completions` (не regex parsing, не LangChain agent).

**LLM:** Qwen3.5-35B-A3B (MoE, Q4_K_M GGUF) на V100 SXM2 32GB через llama-server.exe, 32K context.

**15 LLM tools:**

| Tool | Назначение | Особенности |
|------|-----------|-------------|
| `query_plan` | Генерация SearchPlan | LLM → JSON, multi-query decomposition |
| `search` | Основной hybrid search | BM25+Dense+RRF+ColBERT, round-robin merge |
| `temporal_search` | Поиск по дате | Virtual tool → search с date filters |
| `channel_search` | Поиск по каналу | Virtual tool → search с channel filter |
| `rerank` | Cross-encoder reranker | Qwen3-Reranker, batch scoring |
| `related_posts` | Похожие посты | Qdrant Recommend API |
| `compose_context` | Формирование контекста | Fetch full docs + format citations |
| `final_answer` | Финальный ответ | Markdown с citations |
| `list_channels` | Список каналов | Navigation short-circuit |
| `cross_channel_compare` | Сравнение каналов | Multi-channel synthesis |
| `summarize_channel` | Профиль канала | Channel-level analytics |
| `entity_tracker` | NER аналитика | top/timeline/compare/co_occurrence modes |
| `arxiv_tracker` | ArXiv аналитика | top/lookup modes |
| `hot_topics` | Горячие темы | BERTopic weekly digests из Qdrant |
| `channel_expertise` | Экспертиза канала | Per-channel profiles |

**Dynamic tool visibility:**
- Phase-based: PRE-SEARCH → POST-SEARCH → NAV-COMPLETE → ANALYTICS-COMPLETE
- Max 5 tools visible per step (LLM token budget)
- Data-driven keyword routing из `datasets/tool_keywords.json`
- Signal + keyword matching для tool selection

**Coverage & refinement:**
- LANCER-inspired nugget coverage: subqueries from query_plan = nuggets
- Threshold 0.75 (3/4 nuggets covered by retrieved docs)
- Max 1 targeted refinement (re-search uncovered nuggets only)

**Short-circuits:**
- Navigation: `list_channels` → skip forced search, only `final_answer` visible
- Analytics: `entity_tracker`/`arxiv_tracker`/`hot_topics`/`channel_expertise` → skip forced search, skip verify
- Refusal: negative intent detection (data-driven from tool_keywords.json) + deterministic refusal trim

**Forced search:** если LLM не вызывает tools на первом шаге, принудительный `search` с оригинальным запросом. Bypass только для negative intent + refusal markers.

**SSE streaming:** `/v1/agent/stream` — events: step_started, thought, tool_invoked, observation, citations, final.

### Evaluation pipeline

**Golden dataset v2:** 36 questions:
- 18 retrieval (entity-centric, cross-channel, temporal, product-specific, fact-check, multi-hop)
- 13 analytics (entity_tracker, arxiv_tracker, hot_topics, channel_expertise)
- 2 navigation (list_channels, channel info)
- 3 refusal (out-of-scope, harmful, ambiguous)

**Metrics (per question):**
- `factual_correctness` (0-1): LLM judge, consensus Claude+Codex
- `usefulness` (0-2): LLM judge
- `key_tool_accuracy` (KTA): did agent call the right tools?
- `tool_call_f1`: precision/recall of tool invocations
- `strict_anchor_recall`: did citations point to expected documents?
- `retrieval_sufficiency` (0-3): judge rating
- IR metrics: Precision@5, MRR, nDCG@5
- BERTScore: semantic answer quality
- Latency: time_to_first_token, total_time

**Current baseline (golden v2):**
- Factual accuracy: ~0.875
- Usefulness: ~1.53/2
- KTA: 1.000
- Faithfulness (NLI): 0.91, 0 hallucinations

**Evaluation script:** `scripts/evaluate_agent.py` (~1200 LOC) — full pipeline, LLM judge, offline Codex consensus judge, failure attribution.

### Ingest pipeline

```
Telegram API (Telethon) → messages
  → smart_chunk (paragraph-aware, target 1200 chars)
  → TEI embed (pplx-embed-v1, batch 32)
  → fastembed BM25 sparse
  → jina-colbert-v2 per-token vectors
  → NER enrichment (entities, arxiv_ids, URLs)
  → PointDocument → QdrantStore.upsert()
```

Docker-based: `docker compose run --rm ingest --channel @name --since YYYY-MM-DD --until YYYY-MM-DD`

### Infrastructure

- **LLM:** llama-server.exe (V100 SXM2 32GB, Windows host, port 8080)
- **Embedding + Reranker + ColBERT:** gpu_server.py (RTX 5060 Ti, WSL2 native, port 8082)
- **Vector store:** Qdrant (Docker, CPU, port 16333)
- **API:** FastAPI (Docker, CPU, port 8001)
- **Observability:** Langfuse v3 self-hosted (7 instrumentation points)

### Структура кодовой базы

```
src/
  api/v1/endpoints/     — FastAPI (agent SSE, search, auth, ingest, system, models)
  core/                 — Settings (Pydantic), deps (DI), auth (JWT), security, protocols
  services/             — agent_service (551 LOC orchestrator), qa_service, query_planner, reranker
  services/agent/       — 11 modules: state, guards, visibility, routing, executor,
                          formatting, finalization, refinement, llm_step, coverage, prompts
  services/tools/       — 15 tools + registry.py (partial bindings)
  adapters/qdrant/      — QdrantStore (named vectors, 16 payload indexes)
  adapters/search/      — HybridRetriever (BM25+Dense+RRF+ColBERT pipeline)
  adapters/tei/         — TEIEmbeddingClient, TEIRerankerClient
  adapters/llm/         — LlamaServerClient (/v1/completions + /v1/chat/completions)
  schemas/              — Pydantic models
  tests/                — 22 files, 105 passing tests
scripts/
  gpu_server.py         — Embedding + Reranker + ColBERT HTTP (PyTorch, RTX 5060 Ti)
  evaluate_agent.py     — Agent eval (1200 LOC, LLM judge, failure attribution)
  evaluate_retrieval.py — Retrieval eval (direct Qdrant queries)
  ingest_telegram.py    — Telegram → Qdrant (BM25 + dense + ColBERT + NER enrichment)
datasets/
  eval_dataset_quick_v2.json — Golden v2 (36 Qs, 4 categories)
  tool_keywords.json         — Dynamic routing + policies
  planner_stopwords.json     — Query sanitization
```

## Что нужно от Deep Research

### 1. Выбор фреймворка

- **LlamaIndex vs LangChain vs Haystack vs другие** — какой наиболее fair для сравнения с нашим pipeline?
- Критерии: RAG-first (не generic agent), поддержка Qdrant, hybrid search (BM25+dense), reranking, multi-query, agent with tools
- LlamaIndex кажется closest match (RAG-centric). Подтверждается ли?
- Есть ли фреймворки специализированные на production RAG (Ragflow, Cognita, Kotaemon, RAGAS)?
- **Важно:** нам нужен фреймворк который hiring manager узнает. Экзотика не подходит.

### 2. Feature parity mapping

Для выбранного фреймворка — **точное соответствие** компонентов:

| Наш компонент | Что делает | Эквивалент в фреймворке? |
|---------------|-----------|--------------------------|
| HybridRetriever | BM25(100) + Dense(20) → weighted RRF 3:1 → ColBERT rerank | ? |
| QueryPlannerService | LLM → SearchPlan (subqueries, filters, phrases) | ? |
| Round-robin merge | All subqueries merged, not just first | ? |
| ColBERT MaxSim | Third-stage rerank via Qdrant multivector | ? |
| Cross-encoder rerank | Qwen3-Reranker-0.6B seq-cls | ? |
| AgentService (ReAct) | 15 tools, native function calling | ? |
| Dynamic tool visibility | Phase-based, max 5 visible, keyword routing | ? |
| LANCER nugget coverage | Subquery-based retrieval sufficiency | ? |
| Forced search | Deterministic fallback if LLM skips tools | ? |
| Navigation/analytics short-circuits | Skip unnecessary steps | ? |
| Channel dedup | Max 2 results per source | ? |
| SSE streaming | Real-time events | ? |

Для каждого: есть из коробки / можно настроить / нужно писать custom / невозможно.

### 3. Дизайн бенчмарка

**Controlled variables (одинаковые для обоих):**
- Одна и та же LLM (Qwen3.5-35B-A3B через llama-server)
- Одни и те же embeddings (pplx-embed-v1 через gpu_server.py)
- Один и тот же Qdrant collection (news_colbert_v2)
- Одни и те же данные (~200K постов)
- Одни и те же вопросы (golden v2 + расширенный набор)

**Independent variable:** оркестрация (custom vs framework)

**Metrics:**
- Retrieval: Recall@K, MRR, nDCG@5, Precision@5
- E2E: factual_correctness, usefulness, KTA, BERTScore
- Latency: TTFT, total time, per-step breakdown
- Cost: LOC, abstraction count, time-to-implement
- Flexibility: effort to add new tool, change model, modify pipeline

**Вопросы:**
- Хватит ли 36 вопросов для статистически значимого сравнения или нужно 100+?
- Нужен ли третий вариант (naive baseline: vector search + LLM, без agent) для triangulation?
- Как измерять "debugging ease" и "flexibility" объективно?
- Какие evaluation frameworks подходят (RAGAS, ARES, другие)?

### 4. Настройка фреймворка — best-effort, не strawman

- Как настроить фреймворк **оптимально** для нашего use case?
- Какие конфиги, промты, параметры нужно тюнить?
- Сколько LOC потребуется для эквивалентного pipeline в фреймворке?
- Есть ли known limitations фреймворка которые honest comparison должен упомянуть?

### 5. Presentation для портфолио

- Как представить результаты чтобы hiring manager увидел engineering maturity?
- **Не** "framework bashing" — а "informed architectural decision with data"
- Формат: blog post? GitHub README section? Jupyter notebook? Interactive comparison?
- Существующие примеры таких сравнений в production settings (не academic)?

### 6. Существующие бенчмарки и papers

- Published comparisons: custom RAG vs LlamaIndex/LangChain на одних данных?
- RAGAS, ARES, RECALL — что подходит для multilingual, Telegram news, agent with tools?
- Benchmarks: BEIR, MTEB, RAGBench — применимы ли к нашему domain?
- Как позиционируют framework vs custom в industry (не academia)?

### 7. Implementation plan

- Порядок: что реализовать первым для быстрого MVP сравнения?
- Можно ли использовать наш evaluate_agent.py для обоих pipeline?
- Как организовать код: отдельный dir? отдельный repo? branch?
- Estimation: сколько часов на MVP сравнение (retrieval only → agent → full)?
