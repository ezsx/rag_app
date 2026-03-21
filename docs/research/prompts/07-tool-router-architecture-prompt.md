# Deep Research: Tool Router + Adaptive Retrieval Architecture

## О проекте

**rag_app** — portfolio-grade RAG система для собеседований на позицию **Applied LLM Engineer** ($2-3k/мес). Не "ещё один RAG на LlamaIndex", а система которая **принципиально лучше** фреймворков на сложных запросах.

**Суть**: поисковик/агрегатор новостей из 36 AI/ML Telegram-каналов. Telegram → ingest → Qdrant → Hybrid Retrieval → ReAct Agent → SSE ответ. Закрытая база знаний (не web search) — аналог enterprise RAG над корпоративными источниками.

**Автор**: backend-разработчик (Python), опыт ML (обнаружение аневризм мозга MRI, внедрён в больницах), VPN-сервис с нуля. Понимание Transformer, CUDA на уровне концепций. Цель — показать на собесе умение строить production RAG от ingestion до evaluation.

---

## Железо

- **LLM**: Qwen3-30B-A3B GGUF (MoE, 3B active params) на V100 SXM2 32GB. Windows Host, llama-server.exe, порт 8080. `--jinja --reasoning-budget 0 --cache-type-k q8_0 --cache-type-v q8_0 -c 16384 --parallel 2`
- **GPU-сервер**: RTX 5060 Ti 16GB в WSL2 native (Docker GPU не работает — V100 TCC отравляет NVML). gpu_server.py (PyTorch cu128) на порту 8082, обслуживает 3 модели:
  - Qwen3-Embedding-0.6B (dense embedding, 1024-dim)
  - bge-reranker-v2-m3 (cross-encoder reranker, 568M params)
  - jina-colbert-v2 (ColBERT per-token reranker, 560M params, 128-dim)
- **Vector Store**: Qdrant v1.17 (Docker, CPU). Коллекция `news_colbert`: dense_vector(1024, cosine) + sparse_vector(BM25) + colbert_vector(128, multi-vector MaxSim)
- **API**: FastAPI в Docker (CPU only), SSE streaming

---

## Корпус данных

- 13124 документа из 36 Telegram-каналов (AI/ML тематика, русский + английский)
- Период: 2025-07-01 → 2026-03-18 (9 месяцев)
- Каналы: gonzo_ml (CTO Intento, papers), llm_under_hood (production LLM), ai_newz (PhD ex-Meta), seeallochnaya (бенчмарки), boris_again (ex-eBay), techsparks (директор Яндекса) и 30 других
- Payload в Qdrant: channel, channel_id, date, message_id, is_forward, links, media_types, text
- Chunking: posts <1500 chars целиком, >1500 recursive split. UUID5 deterministic IDs.

---

## Текущая архитектура

### ReAct Agent

Оркестрация: native function calling через `/v1/chat/completions` (llama-server), без regex-парсинга. AgentService управляет loop, ToolRunner исполняет tools.

### Текущий pipeline — ЛИНЕЙНЫЙ

```
query_plan (LLM генерирует 3-5 subqueries + metadata filters)
  → search (все subqueries через HybridRetriever, round-robin merge)
    → Qdrant: BM25 top-100 + Dense top-20 → weighted RRF (BM25 weight=3, dense weight=1)
      → ColBERT MaxSim rerank (top-50 → top-k×2)
    → channel dedup (max 2 docs/channel)
  → rerank (bge-reranker-v2-m3, cross-encoder, top-5)
  → compose_context (собирает контекст, считает coverage)
  → final_answer (LLM генерирует ответ с цитатами)
```

### Механизмы качества

- **Dynamic tools**: final_answer скрыт до выполнения search (LLM не может пропустить поиск)
- **Forced search**: если LLM не вызывает tools (finish_reason=stop), принудительный search с оригинальным запросом
- **Original query injection**: оригинальный запрос всегда в subqueries для BM25 keyword match
- **Multi-query search**: все LLM subqueries выполняются (раньше использовался только первый — critical bug, recall +33% после fix)
- **Coverage threshold**: 0.65, max 2 refinements
- **Grounding**: citation-forced generation, source validation

---

## Текущие метрики (2026-03-20)

### Agent eval (full pipeline через LLM, ~40с/запрос)

| Dataset | Кол-во | Recall@5 | Coverage | Answer rate |
|---------|--------|----------|----------|-------------|
| v1 (factual, temporal, channel, comparative, multi_hop) | 10 | **0.76** | 0.86 | 10/10 |
| v2 (entity, product, fact_check, cross_channel, recency, numeric, long_tail) | 10 | **0.61** | 0.80 | 10/10 |

### Retrieval eval (прямые Qdrant queries, без LLM, ~5с/запрос)

| Pipeline | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|----------|----------|----------|-----------|-----------|
| BM25+Dense → RRF | 0.36 | 0.55 | 0.64 | 0.70 |
| **BM25+Dense → RRF → ColBERT** | **0.71** | **0.73** | **0.73** | **0.74** |

### История recall@5

0.15 (baseline) → 0.33 (pure RRF) → 0.59 (+ orig query) → 0.70 (+ weighted RRF + forced search) → 0.76 (+ per-category matching) → **0.61 на сложных v2 запросах**

### Конкретные провалы v2

| Q | Тип | Recall | Проблема |
|---|-----|--------|----------|
| Q1 | entity (Карпаты) | 0.50 | data_secrets:8021 не в candidate pool |
| Q3 | fact_check (лицензия) | 0.50 | rybolos:1562 не в candidate pool |
| Q6 | recency (NVIDIA 2026) | 0.00 | LLM не знает "Vera Rubin", не генерирует правильный subquery |
| Q7 | numeric (Deep Think цена) | 0.50 | seeallochnaya:2711 не в candidate pool |
| Q8 | long_tail (Kandinsky) | 0.00 | Нашёл правильный канал, fuzzy ±5 strict |

**Корневая проблема**: pipeline линейный, одна стратегия для всех запросов. Temporal query "что было в январе 2026" идёт тем же путём что и factual "за сколько Meta купила Manus". Date filter не применяется, channel filter не применяется, entity extraction не делается.

---

## Что я хочу реализовать: Adaptive Retrieval

### Идея

Вместо одного generic `search(queries, k)` — набор **специализированных инструментов**. Агент выбирает стратегию в зависимости от запроса:

```
"Meta купила Manus"           → entity_search("Meta", "Manus") + fact_lookup
"Что было в декабре 2025"     → temporal_search(date_from="2025-12-01", date_to="2025-12-31")
"Что писал gonzo_ml про X"   → channel_search("gonzo_ml", "X")
"Сравни GPT-5 и Claude"      → comparative_search("GPT-5", "Claude")
"Vera Rubin NVIDIA"           → broad_search() + temporal_search(date_from="2026-01-01")
```

Это то, что **фреймворки не делают из коробки**. LlamaIndex/LangChain используют один retriever для всех запросов.

---

## Конкретные вопросы для исследования

### 1. Архитектура Tool Router

Как определить тип запроса и выбрать стратегию?

Варианты:
- **LLM-as-router**: отдельный LLM call. Но Qwen3-30B (3B active) ненадёжен как classifier, и +10-15с latency.
- **Lightweight classifier**: fine-tuned BERT/distilbert. Нужны labeled data (у нас 20 вопросов — мало).
- **Rule-based**: regex + keyword matching (даты, имена каналов). Быстро, хрупко.
- **Hybrid**: rules для очевидного, LLM для сложного.
- **Интеграция в query_plan**: LLM уже генерирует subqueries — можно добавить поле strategy/tool_selection в JSON output query_plan.

Какой подход рекомендуется? Конкретные примеры реализации для ReAct agent с function calling.

### 2. Набор специализированных Tools

Какие tools реально полезны для нашего кейса (13K коротких Telegram постов, AI/ML)?

Кандидаты:
- `temporal_search(query, date_from, date_to)` — Qdrant filter по дате
- `channel_search(query, channel_name)` — filter по каналу
- `entity_search(entity_name, entity_type)` — NER + payload filter
- `broad_search(query, k)` — текущий hybrid search (fallback)
- `comparative_search(topic_a, topic_b)` — два параллельных поиска + merge/contrast
- `trending_search(period_days)` — последние N дней, sort by date
- `fact_check(claim)` — найти подтверждение или опровержение утверждения

Вопросы:
- Оптимальное количество tools для Qwen3-30B? (>7-8 → путается)
- Как описать tool schema для native function calling чтобы LLM правильно выбирал?
- Каждый tool = отдельная стратегия Qdrant search, или tools комбинируются?
- Можно ли динамически показывать/скрывать tools? (У нас уже есть dynamic tools — final_answer скрыт до search.)

### 3. Интеграция с текущим ReAct Loop

Текущий код: `AgentService._run_agent_loop()` → LLM вызывает tools через function calling → `ToolRunner` исполняет.

- Где вставить router — до loop или как первый tool?
- Текущий `query_plan` уже делает query expansion. Совместить с routing? Или отдельный step?
- `AgentState` уже трекает `search_count`, `compose_count`, `coverage`. Добавить `strategy` field?
- Fallback: если специализированный tool не нашёл → автоматически broad_search?
- Как обрабатывать multi-strategy запросы ("Что писал gonzo_ml в январе 2026")?

### 4. Влияние на latency и reliability

- Каждый дополнительный tool call = +10-15с (LLM inference на V100). Как минимизировать?
- Router как первый call vs router внутри query_plan (zero extra calls)?
- Что если LLM выбирает неправильный tool? Механизм recovery?
- Parallel tool execution (llama-server --parallel 2) — можно ли вызывать tools параллельно?

### 5. Production примеры + evaluation

- Конкретные open-source реализации adaptive retrieval для RAG agents
- Papers с измеримым improvement от tool routing (цифры, не теория)
- Как тестировать routing accuracy — отдельный eval или через общий recall?
- R11 report говорит "+8% accuracy, -35% latency" — при каких условиях?

---

## Ограничения

- Qwen3-30B (3B active MoE) — маленькая модель, ненадёжно следует сложным инструкциям
- Latency: ~30-40с total (V100 inference ~10-15с, retrieval ~5с, ColBERT ~2.5с)
- RTX 5060 Ti 16GB — 3 модели загружены (~4-5 GB VRAM), остаётся ~11GB. Можно добавить ещё модели на GPU если нужно.
- VPN блокирует сеть в WSL2 — модели скачиваются на Windows, копируются
- 20 eval вопросов (v1+v2) — стат. незначимо, нужно 50+ для regression testing
- Время на реализацию: 3-5 дней
