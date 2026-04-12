# Experiment History

> Полная история экспериментов с per-question таблицами и подробными описаниями техник.
> Полная хронология. Сводка метрик — в [project_scope.md](project_scope.md).
> Последнее обновление: 2026-04-10
> 57+ eval прогонов, 8 milestone phases, 8 formal runs (RUN-001–008), 39+ retrieval experiments, NDR/RSR/ROR robustness

---

## Experiment Tables


#### Agent Eval (full pipeline через LLM, ~40с/запрос)

Dataset v1 (10 Qs): factual ×3, temporal ×2, channel ×2, comparative ×1, multi_hop ×1, negative ×1.

| # | Дата | Изменение | Recall@5 | Δ | Файл | Коммит |
|---|------|-----------|----------|---|------|--------|
| 1-3 | 2026-03-19 | Baseline (broken SSE, dense re-score) | 0.00 | — | eval_results_20260319-12* | — |
| 4 | 2026-03-19 | Убрали dense re-score → чистый RRF | 0.15 | — | eval_results_20260319-123202 | — |
| 5 | 2026-03-19 | + Orig query в subqueries | 0.33 | +0.18 | eval_results_20260319-125400 | 72efb31 |
| 6-7 | 2026-03-19 | + MMR post-process (0.7/0.9) | 0.11 | -0.22 | eval_results_20260319-13* | reverted |
| 8 | 2026-03-19 | Pure RRF + orig query | **0.59** | +0.26 | eval_results_20260319-133619 | e0bd871 |
| **11** | **2026-03-19** | **+ Weighted RRF 3:1 + forced search** | **0.70** | **+0.11** | **eval_results_20260319-175853** | **036e54f** |
| 12 | 2026-03-19 | + Whitening 512-dim | 0.56 | -0.14 | eval_results_20260319-* | reverted |
| 13 | 2026-03-19 | + bge-reranker-v2-m3 | 0.70 | 0 | eval_results_20260319-* | 4d43183 |
| **15** | **2026-03-20** | **+ Per-category matching (±50 temporal)** | **0.76** | **+0.06** | **eval_results_20260320-*** | **b676687** |
| 16 | 2026-03-20 | + ColBERT rerank (news_colbert) | 0.76 | 0 | eval_results_20260320-* | 6961cab |

Dataset v2 (10 Qs): entity ×1, product ×3, fact_check ×1, cross_channel ×1, recency ×1, numeric ×1, long_tail ×1, negative ×1.

| # | Дата | Изменение | Recall@5 | Файл | Примечание |
|---|------|-----------|----------|------|------------|
| **19** | **2026-03-20** | **ColBERT + RRF 3:1 (v2 dataset)** | **0.46** | **eval_results_20260320-*** | Сложнее v1: entity/product/cross-channel |
| 20 | 2026-03-20 | + Channel dedup max 2/channel (v2 Q1+Q3 точечно) | 0.50 | точечный тест | Diversity ↑, recall без изменений |
| **21** | **2026-03-20** | **CRITICAL: multi-query search (v1)** | **0.76** | **eval_results_20260320-*** | v1 стабильно |
| **22** | **2026-03-20** | **CRITICAL: multi-query search (v2)** | **0.61** | **eval_results_20260320-*** | **v2: 0.46→0.61 (+33%)** |
| 23 | 2026-03-21 | + Adaptive retrieval (internal routing only) | 0.574 | eval_results_20260321-151103 | Rule-based strategy, no LLM tool selection |
| **24** | **2026-03-21** | **+ LLM tool selection (temporal/channel/broad)** | **0.685** | **eval_results_20260321-*** | **v2: 0.61→0.685 (+12.3%). LLM выбирает tool.** |

#### Retrieval Eval (прямые Qdrant queries, без LLM)

**v1 dataset** (100 auto-generated queries, 2026-03-20):

| # | Pipeline | Fusion | Recall@1 | Recall@5 | Recall@20 | Файл |
|---|----------|--------|----------|----------|-----------|------|
| 17b | BM25+Dense → RRF | RRF (3:1) | 0.36 | 0.55 | 0.70 | retrieval_eval_20260320-155419 |
| **17** | **BM25+Dense → RRF → ColBERT** | **RRF (3:1)** | **0.71** | **0.73** | **0.74** | **retrieval_eval_20260320-155000** |
| 18 | BM25+Dense → DBSF → ColBERT | DBSF | 0.70 | 0.72 | 0.73 | retrieval_eval_20260320-180240 |

**v2 dataset** (100 hand-crafted queries, свежие из текущей коллекции, 2026-03-31):
Dataset: `datasets/eval_retrieval_calibration.json`. Script: `scripts/calibrate_coverage.py`.

| Pipeline | r@1 | r@3 | r@5 | r@10 | r@20 | Monotonic |
|----------|-----|-----|-----|------|------|-----------|
| **BM25+Dense → RRF 3:1 → ColBERT** | **0.80** | **0.97** | **0.97** | **0.98** | **0.98** | **OK** |
| + CE reranking after ColBERT | 0.81 | 0.94 | 0.96 | 0.98 | 0.98 | **BROKEN** (r@3 drops) |
| Pipeline v2 (RRF→CE→ColBERT) | 0.79 | — | — | — | — | +0.02 r@2, marginal |

**Ключевые результаты**:
- Recall@3 = **0.97** — практически потолок, только 3 queries из 100 не находят документ в top-3
- Recall@20 = **0.98** — 2 queries не находят документ вообще (специфичные формулировки)
- CE reranking **портит** r@3 (0.97→0.94) → заменён на confidence filter (DEC-0045)
- Pipeline v2 marginal (+0.02 r@2) — не стоит усложнения
- Monotonicity OK — recall не падает при увеличении k

### Framework comparison benchmark (SPEC-RAG-29, 2026-04-03)

4 pipeline: naive (dense-only), LI-stock (LlamaIndex default hybrid), LI-maxed (LlamaIndex + weighted RRF + CE reranker), custom (BM25+Dense → RRF 3:1 → ColBERT).

**Auto-generated dataset** (eval_retrieval_100.json, 100 Qs):

| Pipeline | R@1 | R@5 | R@20 | MRR | Latency p50 |
|----------|-----|-----|------|-----|-------------|
| naive | 0.820 | 0.920 | 0.940 | 0.861 | 0.1s |
| LI-stock | 0.820 | 0.920 | 0.940 | 0.861 | 0.1s |
| LI-maxed | 0.880 | 0.940 | 0.940 | 0.907 | 1.4s |
| **custom** | **0.939** | **0.949** | **0.949** | **0.944** | **0.2s** |

**Calibration dataset** (eval_retrieval_calibration.json, 100 Qs, hand-crafted):

| Pipeline | R@1 | R@5 | R@20 | MRR | Latency p50 |
|----------|-----|-----|------|-----|-------------|
| naive | 0.730 | 0.940 | 0.990 | 0.825 | 0.1s |
| LI-stock | 0.730 | 0.940 | 0.990 | 0.825 | 0.1s |
| LI-maxed | 0.780 | 0.980 | 0.980 | 0.865 | 1.4s |
| **custom** | **0.780** | **0.970** | **0.980** | **0.866** | **0.2s** |

**Ключевые результаты:**
- LI-stock = naive: LlamaIndex default hybrid (relative_score_fusion) = zero gain
- Weighted RRF = main differentiator: +5-12% R@1, +4-10% MRR
- ColBERT ≈ cross-encoder на calibration dataset (R@1 0.78 vs 0.78, MRR 0.866 vs 0.865)
- ColBERT > cross-encoder на auto-generated (R@1 0.94 vs 0.88) — exact token matching на цитатах
- Custom pipeline 7x быстрее LI-maxed (0.2s vs 1.4s) — framework abstraction overhead
- Retrieval-only benchmark не покрывает query planning, LANCER, multi-query — agent E2E нужен

### Agent E2E benchmark (SPEC-RAG-29 Phase 2, 2026-04-03)

4 pipeline × 17 retrieval questions (golden v2). Judge: Claude Opus 4.6.
Только retrieval-evidence вопросы (17 из 36): оставшиеся 19 (analytics, navigation, refusal) исключены — LlamaIndex pipeline не имеет аналогов `entity_tracker`, `hot_topics`, `channel_expertise`, `list_channels`, и интеграция этих tools в LI потребовала бы полной переписки агентского слоя.

| Pipeline | Factual (avg) | Usefulness (avg) | Grounding (avg) | Avg Latency | Avg Tools |
|----------|:---:|:---:|:---:|:---:|:---:|
| naive | 0.55 | 1.04 | 0.28 | ~4s | 0 |
| LI-stock | 0.51 | 1.13 | 0.46 | ~9s | 1 |
| LI-maxed | 0.54 | 1.21 | 0.48 | ~11s | 1 |
| **custom** | **0.84** | **1.77** | **0.88** | **~30s** | **4.6** |

**Delta custom vs best-of-three: factual +0.30, usefulness +0.56, grounding +0.40.**

**Ключевые результаты:**
- Custom доминирует: 1.5x factual, 1.5x usefulness, 3x grounding vs naive
- li_maxed ≈ li_stock: weighted RRF + CE reranker дали +0.03 factual (marginal)
- Killer questions: q11 (channel-specific entity search), q15/q16 (channel digests) — custom единственный кто ответил
- Retrieval failure: li_stock/li_maxed на q03 (hallucinated refusal "Meta не покупала Manus AI")
- Grounding = главный дифференциатор custom (0.88 vs 0.48) — inline [1][2][3] через compose_context
- Multi-query planning + LANCER coverage + specialized tools = основной source of gain (не reranker)

**TODO — ablation experiments:**
- [ ] DBSF fusion vs RRF
- [ ] BM25 weight sweep (2:1, 3:1, 4:1, 5:1)
- [ ] Multi-query retrieval (query planning subqueries)
- [ ] Instruction prefix impact на разных типах запросов
- [ ] ColBERT query formatting (is_query vs без)

CE score distribution (2000 docs):

| | Relevant (n=143) | Irrelevant (n=1857) |
|---|---|---|
| median | **8.35** | **-1.11** |
| При filter_threshold=0.0 | keep 92% relevant | remove 55% irrelevant |

### Подробные результаты agent eval v1 (dataset v1, #21, recall@5=0.76)

| Q | Тип | Вопрос | Recall | Cov | Статус | Citations (top-4) |
|---|-----|--------|--------|-----|--------|-------------------|
| Q1 | factual | Financial Times ЧГ 2025 | **1.0** | 0.88 | ✅ | ai_ml_big_data:9245 |
| Q2 | factual | GPT OSS параметры | **1.0** | 0.91 | ✅ | rybolos:1563 (±1 от :1562) |
| Q3 | factual | Meta + Manus AI | **1.0** | 0.83 | ✅ | ai_newz:4355 |
| Q4 | temporal | Декабрь 2025 Google/NVIDIA | 0.0 | 0.85 | ✅ ответ | Нашёл декабрьские, но не msg 9245/9226 |
| Q5 | temporal | Январь 2026 AI-каналы | 0.33 | 0.80 | ✅ partial | boris_again:3703 (±2 от :3701) |
| Q6 | channel | llm_under_hood reasoning GPT-5 | **1.0** | 0.92 | ✅ | llm_under_hood:648 |
| Q7 | channel | boris_again Gemini 3 Flash | **1.0** | 0.91 | ✅ | boris_again:3701 |
| Q8 | comparative | Deep Think vs o3-pro | **1.0** | 0.84 | ✅ | seeallochnaya:2711 |
| Q9 | multi_hop | LLM production 2 канала | 0.50 | 0.84 | ✅ partial | llm_under_hood:641/723 |
| Q10 | negative | GPT-6 | N/A | 0.86 | ✅ отказ | — |

### Подробные результаты agent eval v2 (dataset v2, #22, recall@5=0.61)

**До multi-query fix (#19)**: recall@5 = 0.46. **После (#22)**: recall@5 = **0.61** (+33%).

| Q | Тип | Вопрос | #19 | #22 | Δ | Анализ |
|---|-----|--------|-----|-----|---|--------|
| Q1 | entity | Карпаты + LLM | 0.50 | 0.50 | = | aioftheday:3655 ✅, data_secrets:8021 не в candidate pool |
| Q2 | product | Mamba 3 | **1.0** | **1.0** | = | gonzo_ml:4242 ✅ |
| Q3 | fact_check | OpenAI лицензия | 0.50 | 0.50 | = | complete_ai:750 ✅, rybolos:1562 не в candidate pool |
| Q4 | cross_channel | Meta+Manus каналы | 0.67 | **1.0** | **+0.33** | ai_newz:4355 + aioftheday:3988 + data_secrets:8582 — multi-query нашёл все 3 |
| Q5 | product | Granite 4.0 | **1.0** | **1.0** | = | ai_ml_big_data:8680 ✅ |
| Q6 | recency | NVIDIA начало 2026 | 0.0 | 0.0 | = | boris_again:3693 не найден — LLM не генерирует "Vera Rubin" |
| Q7 | numeric | Deep Think цена | 0.50 | 0.50 | = | aioftheday:3885 ✅, seeallochnaya:2711 не попадает |
| Q8 | long_tail | Kandinsky 5.0 | 0.0 | 0.0 | = | dendi_math_ai:83 вместо :100 (±17, fuzzy ±5 strict) |
| Q9 | product | Sora 2 | 0.0 | **1.0** | **+1.0** | aioftheday:3577 — multi-query subquery "sora 2" нашёл через ColBERT |
| Q10 | negative | Claude 4 | N/A | N/A | = | ✅ корректный отказ |

**Что улучшил multi-query fix:**
- Q4 (cross_channel): 3 разных subquery нашли 3 разных канала → full recall
- Q9 (Sora 2): subquery "sora 2 описание" нашёл через BM25 → ColBERT поднял на top-5

**Оставшиеся провалы:**
- Q6 (recency): LLM не знает "Vera Rubin" → не генерирует правильный subquery

### Подробные результаты agent eval v2 (#24, adaptive retrieval + LLM tool selection, recall@5=0.685)

**Ключевое изменение**: LLM видит specialized tools (temporal_search, channel_search) и **сам выбирает** какой инструмент вызвать. Dynamic tool visibility скрывает irrelevant tools.

| Q | Тип | #22 (baseline) | #24 (adaptive) | Δ | Анализ |
|---|-----|---------------|----------------|---|--------|
| Q1 | entity | 0.50 | 0.50 | = | aioftheday:3655 ✅, data_secrets:8021 не в pool |
| Q2 | product | **1.0** | **1.0** | = | gonzo_ml:4242 ✅ |
| Q3 | fact_check | 0.50 | 0.50 | = | complete_ai:750 ✅, rybolos:1562 не в pool |
| Q4 | cross_channel | **1.0** | 0.67 | -0.33 | data_secrets:8582 не попал (LLM variance) |
| Q5 | product | **1.0** | **1.0** | = | ai_ml_big_data:8680 ✅ |
| Q6 | recency | 0.0 | 0.0 | = | boris_again:3693 не в top-5; NVIDIA docs найдены из других каналов |
| Q7 | numeric | 0.50 | 0.50 | = | aioftheday:3885 ✅ |
| Q8 | long_tail | 0.0 | **1.0** | **+1.0** | dendi_math_ai:83/:94/:103 — fuzzy ±50 матчит :100/:101 |
| Q9 | product | **1.0** | **1.0** | = | aioftheday:3577 ✅ |
| Q10 | negative | N/A | N/A | = | ✅ корректный отказ |

**Итого**: recall@5 = 0.61 → **0.685** (+12.3%). Q8 из 0→1.0 — главное улучшение.

**Что LLM выбирает** (по логам):
- Temporal запросы → `temporal_search` с date_from/date_to
- Channel запросы → `channel_search` с channel=name
- Остальные → `search` (broad)

**Q6 — почему всё ещё 0.0**: temporal_search с date filter находит NVIDIA документы из января-марта 2026 (Vera Rubin, CES, GTC), но expected source `boris_again:3693` не в top-5 — другие каналы релевантнее. Проблема eval dataset, не retrieval.
- Q8 (long_tail): fuzzy ±5 слишком strict (dendi_math_ai:83 vs :100)
- Q1, Q3, Q7 (partial): один expected doc не в candidate pool даже при multi-query

### Eval v3 quick (6 Qs, #25, system prompt fix + hints injection)

**Два фикса** (2026-03-21):
1. **System prompt** — раньше хардкодил "2. search", LLM 30/30 раз игнорировал temporal_search/channel_search. Заменили на выбор из 3 tools с if-then правилами (R13-quick §1.3, R13-deep §2).
2. **Hints injection** — query signals (strategy_hint, date_from, channels) теперь передаются в system message как подсказка LLM.

**Strict recall@5 = 0.167** (1.0/6) — **некорректная метрика** для open-ended temporal вопросов.
**LLM judge score = 0.71** (4.25/6) — реальное качество ответов.

| Q | Category | Strict Recall | LLM Judge | Анализ |
|---|----------|---------------|-----------|--------|
| q01 | temporal | 0.50 | 0.90 | NVIDIA Rubin, Razer AVA, Google Gemini — ответ полный. Нашёл 1/2 expected docs + 4 других релевантных |
| q02 | temporal | 0.00 | 0.40 | **Реальный miss**. Нашёл Alibaba Qwen за декабрь, но пропустил Лекун+Лебрун и DeepSeek-V3 |
| q03 | temporal | 0.00 | 0.70 | Agibot роботы (совпадает с expected), $5трлн инвестиции. Не нашёл Opus 4.6. Near-miss: techsparks:5439 vs 5444 (off by 5) |
| q04 | temporal | 0.00 | 0.95 | GPT-5.3/5.4 — **точно совпадает с expected answer**. Strict recall=0 потому что нашёл из других каналов |
| q05 | temporal | 0.00 | 0.50 | GPT-5, WAIC 2025 — реальные события лета. Пропустил Авито AI лабу. Near-miss: gonzo_ml:3839 vs 3830 (off by 9) |
| q06 | channel | 0.50 | 0.80 | Channel search сработал, все 6 citations из gonzo_ml. HRM, TRM — релевантная тема |

**Ключевой вывод**: strict document matching неадекватна для temporal/broad запросов. Нужен LLM judge как основная метрика. Strict recall оставить как вспомогательную для entity/product/channel запросов где expected doc однозначен.

**Проблема eval methodology**:
- Для "Что произошло в марте 2026?" корректных ответов — десятки. Захардкодить 2 документа и судить по ним — ложная метрика.
- LLM judge (пока — Claude при ручном review) оценивает: фактическая корректность, покрытие ожидаемого ответа, цитирование.
- Автоматический LLM judge — следующий шаг (SPEC для отдельного eval pipeline).

### Корневые проблемы

| # | Проблема | Статус | Решение |
|---|----------|--------|---------|
| 1 | Embedding anisotropy (cosine 0.78-0.83) | ✅ Решено | ColBERT per-token MaxSim обходит single-vector collapse |
| 2 | Attractor documents | ✅ Решено | Weighted RRF 3:1 + ColBERT rerank + channel dedup |
| 3 | Реранкер suboptimal (bge-m3) | ✅ Решено | bge-reranker-v2-m3 (logit gap 8→18) |
| 4 | Cosine-based MMR | ✅ Решено | Отключён, заменён на channel dedup |
| 5 | Single-query search (critical bug) | ✅ Решено | Multi-query + round-robin merge (v2 recall +33%) |
| 6 | LLM не знает entity names | Открыто | Q6: LLM не генерирует "Vera Rubin" → пост не находится |
| 7 | Fuzzy ±5 слишком strict для long-tail | Открыто | Q8: dendi_math_ai:83 vs :100 (±17) |
| 8 | Partial recall на multi-doc queries | Открыто | Q1, Q3, Q7: один из expected docs вне candidate pool |
| 9 | **System prompt хардкодил tool name** | ✅ Решено | Prompt говорил "2. search" → LLM 30/30 игнорировал temporal_search/channel_search. Fix: if-then rules + hints injection |
| 10 | **Strict doc matching — ложная метрика** | Открыто | Temporal/broad вопросы имеют десятки валидных source docs. Strict recall@5 = 0.167, LLM judge = 0.71 на тех же ответах. Нужен LLM judge |


---

## Phase 3.4–3.6: Analytics, Pipeline Cleanup, NLI (2026-03-25 → 2026-04-01)

### SPEC-RAG-15: Entity Analytics (2026-03-25)

entity_tracker (top/timeline/compare/co_occurrence) + arxiv_tracker. Facet API, point-level counts. Golden dataset расширен до 30 Qs.

| Метрика | Значение |
|---------|----------|
| KTA | 0.926 (35/36) |
| Consensus factual (Claude + Codex) | 1.79/2 |
| Consensus useful | 1.72/2 |
| Eval file | eval_results_20260325-192924.json |

### SPEC-RAG-16: Hot Topics + Channel Expertise (2026-03-28)

BERTopic weekly digests → `weekly_digests` collection. Channel profiles → `channel_profiles` collection. 15 tools total.

### SPEC-RAG-18: Golden v2 + Offline Judge (2026-03-30)

Golden v2 dataset: 36 Qs (18 retrieval_evidence, 13 analytics, 2 navigation, 3 refusal). Consensus judge: Claude + Codex.

| Метрика | Значение |
|---------|----------|
| Consensus factual | ~0.80 |
| Consensus useful | ~1.53/2 |
| KTA | 1.000 |
| Eval file | eval_results_20260330-035118.json |

### Observability + Qwen3.5 swap (2026-03-30)

Langfuse v3 integration (SPEC-RAG-19). Qwen3 → Qwen3.5-35B-A3B swap (DEC-0039).

| Метрика | Значение |
|---------|----------|
| Consensus factual | 0.833 |
| Consensus useful | 1.611 |
| KTA | 0.970 |

### SPEC-RAG-20d: Pipeline Cleanup + Coverage Redesign (2026-03-31 → 2026-04-01)

32 code changes. LANCER nugget coverage (DEC-0044), CE confidence filter (DEC-0045), observability audit (15 findings), retrieval calibration (100 queries).

| Метрика | До | После |
|---------|-----|-------|
| Factual (Claude judge) | ~0.80 | **0.847** |
| Useful | ~1.53 | **1.861** |
| KTA | 1.000 | **1.000** |
| Strict recall | 0.461 | **0.637** |
| Latency | 26.4s | **23.6s** |
| Eval file | | eval_results_20260401-014638.json |

Retrieval calibration (100 hand-crafted queries, `datasets/eval_retrieval_calibration.json`):
- r@1=0.80, r@3=0.97, r@20=0.98, monotonic
- CE reranking degrades r@3 (0.97→0.94) → replaced with filter
- Pipeline v2 (RRF→CE→ColBERT): +0.02 r@2, not worth complexity
- CE score: relevant median=8.35, irrelevant median=-1.11

Smoke tests latency: 55-76s → 14-49s (−40-65%).

### q27 SecurityManager Fix (2026-04-01)

SecurityManager.check_sql_injection() false positive: `text.count(";") > 2` блокировал `final_answer` с русской пунктуацией. Стоило 3% factual.

| Метрика | До | После |
|---------|-----|-------|
| Factual | 0.847 | **0.875** |
| Useful | 1.861 | **1.917** |

Fix: добавлены `final_answer`, `verify`, `fetch_docs` в `_skip_security`.

### SPEC-RAG-21: NLI Citation Faithfulness (2026-04-01)

Первый baseline faithfulness через independent NLI verification. Pipeline: Claude decomposition → ruBERT NLI → per-claim verification.

| Метрика | Значение |
|---------|----------|
| Factual (Claude judge, 0.1 scale) | **0.842** |
| Useful | **1.778/2** |
| KTA | **1.000** |
| Faithfulness (raw, lenient) | 0.792 |
| Faithfulness (corrected) | **~0.91** |
| Real contradictions | **0** (19 raw → all false positives) |
| Retrieval Qs | 17/36 |
| Claims verified | 171 |
| Eval file | eval_results_20260401-091242.json |
| NLI file | nli_scores_20260401_full.json |
| Full analysis | [nli_faithfulness_analysis_20260401.md](../../experiments/legacy/reports/nli_faithfulness_analysis_20260401.md) |

Model: rubert-base-cased-nli-threeway (180M, 0.36 GB). Выбрана после A/B с xlm-roberta-large-xnli (560M) — ruBERT дала ent=0.948 vs XLM-R ent=0.006 на тех же парах.

19 contradictions проверены вручную: 12 false positives (paraphrase failures), 5 wrong-doc matches, 2 borderline. 0 реальных hallucinations.

### SPEC-RAG-23: NDR/RSR/ROR Robustness Baseline (2026-04-02)

Bypass pipeline (прямые Qdrant + llama-server, без agent). BERTScore F1 scoring (ruBert-large, layer 18).
36 golden Qs, 151 LLM calls, ~40 мин compute.

| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| Метрика | BERTScore (proxy) | **Claude Judge (final)** |
|---------|-------------------|--------------------------|
| **NDR** | 0.818 | **0.963** (26/27) |
| **RSR** | 0.706 | **0.941** (16/17) |
| **ROR** | 0.974 | **0.959** |
| **Composite** | 0.826 | **0.954** |

**Ключевая находка**: pipeline работает отлично. BERTScore как proxy **провалился** — занижал NDR, показывал ложные RSR violations. Claude judge обязателен.

**RSR монотонность подтверждена**: k=3 (0.52) < k=5 (0.59) < k=10 (0.60) < k=20 (0.63). Единственный violation: q11 (boris confusion, k=10→k=20).

**Retrieval критически важен**: avg factual без docs = 0.10, с docs = 0.63 (delta +0.53).

**Known issues**: q11 boris confusion, q12/q15/q06 missing docs (data quality, не pipeline).

Raw: `experiments/legacy/robustness/ndr_rsr_ror_raw_20260402-082135.json`
Report: `experiments/legacy/robustness/ndr_rsr_ror_report_20260402-082135.md`

#### Методология: наша vs Cao et al. (2025, arXiv:2505.21870)

Три метрики рекомендованы Яндексом (R15, Андрей Соколов):

| Параметр | Cao et al. (full) | Наш (simplified) | Ratio |
|----------|-------------------|-------------------|-------|
| Questions | 1500 | 36 (17 for RSR/ROR) | 42× меньше |
| k values | 6: {5,10,25,50,75,100} | 4: {3,5,10,20} | 1.5× меньше |
| Orderings per k | 3 | 1 (NDR/RSR), 3 (ROR) | 1-3× меньше |
| Total LLM calls | ~55,000 | **~160** | 340× меньше |
| Scoring | Llama-3.3-70B judge | BERTScore proxy + Claude subset | Different |

Simplified подход: NDR = k=20 vs k=0 (одно сравнение per query), RSR = chain monotonicity k=3→5→10→20, ROR = 3 orderings при k=20. Теряем k×ordering взаимодействие, но за 67 мин вместо сотен часов.

#### Per-question NDR анализ (Claude judge, 27 scored)

| Question | RAG score | noRAG score | Δ | Причина |
|----------|----------|------------|---|---------|
| **q26** | 0.5 | **0.8** | **-0.30** | Единственный failure: parametric > retrieval (AI-стартапы инвестиции) |
| q01 | **1.0** | 0.0 | **+1.00** | RAG critical: FT человек года — Qwen не знает |
| q09 | **1.0** | 0.0 | **+1.00** | RAG critical: llm_under_hood про GPT-5 reasoning |

Средний factual: k=0 = **0.10**, k=20 = **0.63**. Retrieval даёт **+0.53** absolute improvement.

#### RSR: 1 violation (17 retrieval Qs)

| k | Avg factual | Δ vs prev |
|---|------------|-----------|
| k=3 | 0.518 | — |
| k=5 | 0.588 | +0.070 |
| k=10 | 0.600 | +0.012 |
| k=20 | **0.629** | +0.029 |

Единственный violation: **q11** (boris_again confusion) — k=10: 1.0, k=20: 0.1. При k=20 docs про другого Бориса путают модель.

#### ROR: 12/17 = σ=0, max σ=0.115 (q04). Qwen3.5 полностью устойчив к порядку docs.

---

## Phase 3.9–4.0: Ablation Protocol + Post-Protocol Validation (2026-04-08 → 2026-04-10)

### Experiment protocol formalization (2026-04-08)

После двух production bugs (CE URL drift, embedding prefix drift) eval workflow формализован:
- `experiments/PROTOCOL.md` — spec-before-run, parity check, preflight, structured artifacts
- `experiments/baseline.yaml` — frozen production config
- `experiments/log.md` — summary всех formal runs
- `scripts/parity_check.py` — drift detection
- `scripts/compute_confidence.py` — canonical bootstrap CI / paired bootstrap script

### RUN-004–008 summary

| Run | Change | Result | Decision |
|-----|--------|--------|----------|
| RUN-004 | `compose_context` 1800→4000 | factual raw ~0.83, no material gain | rejected |
| RUN-005 | channel dedup 2→3 | вернулся 3rd doc/channel на q08 | **adopted** |
| RUN-006 | dual scoring (`norm_linear`, `rrf_ranks`) | ломает CE gap detection, хуже citations | rejected |
| RUN-007 | cosine recall guard | CE precision + bi-encoder recall, q08 repaired | **adopted** |
| RUN-008 | full baseline с adopted changes | factual **0.858 corrected**, useful **1.708**, refusal **3/3** | **baseline** |

### Dataset audit (2026-04-09)

Выявлена методологическая проблема: **7/36 open-ended вопросов** в golden v2 имели слишком узкий `expected_answer`, из-за чего judge penalized корректные, но более широкие ответы.

Fix:
- создан `datasets/eval_golden_v2_fixed.json`
- исправлены `q04`, `q06`, `q08`, `q10`, `q14`, `q15`, `q16`
- для open-ended queries expected rewritten как acceptance criteria, а не список узких формулировок

Impact:
- factual **0.803 → 0.858** after correction
- useful unchanged materially
- refusal remains **3/3**

### Statistical confidence (2026-04-10)

Bootstrap CI по corrected baseline (`RUN-008`, 10K resamples, `scripts/compute_confidence.py`):

| Metric | Mean | 95% CI | n |
|--------|------|--------|---|
| Factual (all) | **0.858** | **[0.792, 0.917]** | 36 |
| Factual (retrieval) | **0.888** | **[0.782, 0.965]** | 17 |
| Factual (analytics) | **0.793** | **[0.679, 0.893]** | 14 |
| Useful (all) | **1.708** | **[1.606, 1.803]** | 36 |

Вывод: baseline уже пригоден для portfolio/outreach, но интервалы широкие. Для statistically tighter claims нужен golden v3 на 100-120 вопросов.

### Golden v3 dataset expansion (2026-04-10)

Создан финальный `datasets/golden_v3/eval_golden_v3.json` на 120 reviewed questions без смены frozen pipeline baseline.

Методология:
- `datasets/golden_v3/eval_golden_v3_draft.json`: 125 candidates, 5 rejected kept for audit
- `datasets/golden_v3/golden_v3_plan.md`: review protocol and status
- `datasets/golden_v3/golden_v3_review_packet_001.md` ... `008.md`: semi-manual review packets with source snippets
- `scripts/build_golden_v3_draft.py`: candidate builder
- `scripts/export_golden_v3_review_packet.py`: review packet exporter

Final mix:

| Eval mode | Count |
|-----------|------:|
| retrieval_evidence | 65 |
| analytics | 32 |
| navigation | 8 |
| refusal | 15 |

Validation:
- `scripts.evaluate_agent.load_dataset(...)`: 120 final / 120 reviewed / 125 draft
- strict-anchor source lookup: 60/60 source ids found in `news_colbert_v2`
- `ruff`: pass for changed scripts/security files
- `pytest src/tests/test_security.py`: 30 passed

Следующий шаг: сделать full eval + judge на `eval_golden_v3.json`, затем повторить bootstrap CI и обновить README только после v3 judge.

### Корневые проблемы (обновлено 2026-04-10)

| # | Проблема | Статус |
|---|----------|--------|
| 6 | LLM не знает entity names (Vera Rubin) | ✅ Не актуально — golden_v2 q07 (GTC 2026) работает, temporal_search находит |
| 10 | Strict doc matching — ложная метрика | ✅ Решено — primary metric = judge factual + faithfulness |
| 11 | **SecurityManager false positive на ";"** | ✅ Решено — _skip_security для final_answer. МИНА: нужен полный рефакторинг |
| 12 | **q15 routing: summarize_channel без citations** | Открыто — нужен search + compose_context вместо summarize_channel |
| 13 | **NLI false positives на cross-lingual/paraphrase** | Открыто — ruBERT 180M accuracy drops на mixed RU/EN text |

---

## Technique Details

## Archived Technique Details (подробности ранних экспериментов)

> Результаты суммированы в таблицах выше. Здесь — детали для reference.

### Embedding Whitening (PCA 1024→512)
- **Результат #1** (2026-03-19, 1024→512): recall 0.70→0.56. Слишком агрессивный cutoff.
- **Результат #2** (2026-03-20, 1024→1024): паритет. Dense не bottleneck при BM25 3:1.
- **Статус**: отклонено. Whitening params в `datasets/whitening_params.npz`.
  - Коллекция `news_whitened` (1024-dim) сохранена. Whitening params: `datasets/_whitening_mean.npy`, `datasets/_whitening_transform.npy`.
  - **Ключевой инсайт**: при weighted RRF 3:1 BM25 доминирует, dense вносит малый вклад. Whitening улучшает dense, но dense уже не bottleneck. Bottleneck — reranking quality и query expansion.
- **Статус**: [x] whitening 512 — recall 0.56, откат. [x] whitening 1024 — паритет, коллекция сохранена. **Dense не является bottleneck при текущей архитектуре (BM25 3:1).**
- **Ссылки**: Su et al. 2021 "BERT-whitening", WhitenRec 2024, WhiteningBERT (Huang et al. EMNLP 2021)

### Weighted RRF, DBSF, Channel Dedup, Reranker
- **RRF 3:1** (2026-03-19): recall 0.33→0.59. BM25 limit=100, dense limit=20.
- **DBSF** (2026-03-20): recall 0.72 vs RRF 0.73. Отклонено.
- **Channel dedup** (2026-03-20): +diversity, не +recall. Оставлен как default.
- **Reranker upgrade** (2026-03-19): bge-m3 → bge-reranker-v2-m3. Logit gap 8→18. Позже → Qwen3-Reranker-0.6B-seq-cls (DEC-0043).

---

## Technique Status (обновлено 2026-04-01)

### Реализовано

| Техника | Результат | Spec/Decision |
|---------|-----------|---------------|
| Weighted RRF 3:1 (BM25 dominate) | recall 0.33→0.59 (+79%) | — |
| ColBERT reranking (jina-colbert-v2) | recall@1 +97% (0.36→0.71) | — |
| Multi-query search (round-robin merge) | v2 recall +33% (0.46→0.61) | — |
| Channel dedup (max 2/channel) | +diversity, not +recall | — |
| bge-reranker-v2-m3 → Qwen3-Reranker | logit gap 8→18 | DEC-0043 |
| Query classifier + strategy router | 15 tools, dynamic visibility, data-driven routing | SPEC-RAG-11/13 |
| Entity extraction | 95 entities, 16 payload indexes, Facet API | SPEC-RAG-12 |
| LANCER nugget coverage | -45% latency, 0 лишних refinements | DEC-0044 |
| CE confidence filter (CRAG-style) | keep 92% relevant, remove 55% irrelevant | DEC-0045 |
| NLI citation faithfulness | faithfulness 0.91, 0 hallucinations | SPEC-RAG-21 |

### Протестировано и отклонено (с evidence)

| Техника | Результат | Почему отклонено |
|---------|-----------|-----------------|
| Cosine MMR | recall 0.70→0.11 | Re-promotes attractor documents |
| Dense re-score после RRF | recall 0.33→0.15 | Стирает BM25 вклад |
| PCA whitening 1024→512 | recall 0.70→0.56 | Слишком агрессивный cutoff |
| Whitening 1024→1024 | паритет | Dense не bottleneck при BM25 3:1 |
| DBSF fusion | recall 0.72 vs RRF 0.73 | RRF чуть лучше |
| CE reranking after ColBERT | r@3 degrades 0.97→0.94 | Заменён на filter (DEC-0045) |
| Pipeline v2 (RRF→CE→ColBERT) | +0.02 r@2 | Не стоит усложнения |
| Lost-in-middle reorder | — | Docs уже reranked ColBERT, reorder hurts |
| XLM-RoBERTa-large-xnli для NLI | ent=0.006 на очевидных парах | ruBERT в 150x точнее на русском |

### Не реализовано (backlog)

| Техника | Expected impact | Blocker |
|---------|----------------|---------|
| Fine-tune CE reranker на domain data | -3% degradation | 500 query-doc pairs needed |
| Contextual Retrieval (Anthropic) | +10-20% recall | 8-15 часов one-time LLM compute |
| Reranker-as-Fusion (CE вместо RRF) | Потенциально лучше RRF | Pipeline v2 A/B показал +0.02 — marginal |

---

## Archived: Technique Details

> Ранние эксперименты Tier 3 (ColBERT, Contextual Retrieval, Fine-tune, CRAG) и Tier 4 (HyDE, Multi-Collection, Link Expansion и др.) — описания перенесены в research reports. ColBERT реализован (R@1 +97%). CRAG частично покрыт DEC-0044 (LANCER) + DEC-0045 (CE filter). Contextual Retrieval и Fine-tune отложены.
>
> Adaptive Retrieval + Tool Router (R13/R14) — **полностью реализован** в SPEC-RAG-11/13/15/16/17. 15 tools, dynamic visibility, data-driven routing.

---

## Research Track: Тематическая кластеризация (R12)

> **Статус**: исследование завершено. **Вердикт: отложено** — effort > impact при 13K docs.
> Кластеризация оправдана при 50K+ документов. Подробности: [R12](../research/reports/R12-cluster-based-retrieval.md).

Кластеризация (20-40ч) даёт **меньше recall** чем ColBERT reranking (4-8ч). Cross-encoder "bypasses the cosine floor entirely".

Ключевые findings: per-cluster whitening математически некорректен при <600 docs/cluster (rank-deficient covariance). Embedding collapse на topically narrow corpus — expected behavior (Ethayarajh 2019, Zhou et al. ACL 2025). BERTopic с отдельной embedding моделью — правильный инструмент (не TF-IDF).

---

## Backlog: Entity Dictionary Enrichment (Future)

> **Статус**: отложено. Текущий словарь (95 entities, regex tier-1) покрывает ~80% упоминаний.
> Расширять когда: tier-2 GLiNER добавлен, или словарь начал пропускать частые entities.

**Готовых AI/ML словарей с русскими aliases не существует** (исследовано 2026-03-24).

Пайплайн для масштабирования до 500-1000 entities:
1. **Wikidata SPARQL** — компании, модели, конференции + русские labels/aliases. Класс `Q21198342` (LLM), `Q2385804` (conference), filter по AI industry
2. **HuggingFace API** — top-1000 моделей по downloads → извлечь org names + model families. `GET /api/models?sort=downloads&limit=1000`
3. **Papers with Code API** — ~1000 методов (Transformer, LoRA, MoE...). `pip install paperswithcode-client`
4. **LLM-генерация русских aliases** — транслитерация (DeepSeek→дипсик), сленг (GPT→жпт)
5. **Валидация на корпусе** — frequency scan, фильтрация шума

Источники: исследование в Claude web chat (2026-03-24), R17 §2 NER pipeline.

---

## Что НЕ стоит пробовать (дополнительно к "Протестировано и отклонено")

| Техника | Почему |
|---------|--------|
| SPLADE для русского | Нет production-ready мультиязычных моделей. WSDM Cup 2026: "struggled to remain competitive" |
| HyDE как primary fix | +1-3с latency, не решает collapse в document space. Only complementary |
| Scaling embedding alone | Length-Induced Collapse + domain-specific collapse = фундаментальные ограничения single-vector |
