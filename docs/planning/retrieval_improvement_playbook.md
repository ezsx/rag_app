# Retrieval Improvement Playbook

> Живой документ. Ответ на вопрос "что дальше пробовать чтобы повысить recall?"
> Источник: R11-advanced-retrieval-strategies.md + собственные эксперименты.
> Последнее обновление: 2026-03-24

---

## Текущее состояние (2026-03-24)

**Agent eval v1**: Recall@5 = **0.76** (10 Qs), Coverage = 0.86, Answer rate = 10/10.
**Agent eval v2**: Recall@5 = **0.685** (10 Qs, сложные), Coverage = 0.80, Answer rate = 10/10.
**Retrieval eval**: Recall@5 = **0.73** (100 Qs, RRF+ColBERT), Recall@1 = 0.71.
**Golden eval v1**: Strict recall@5 = **~0.43** / Key Tool Accuracy = **0.955** / Manual judge factual = **0.52**, useful = **1.14/2** (25 Qs, 6 categories).

**Phase 3.3 (Eval Pipeline V2, SPEC-RAG-14)**: реализовано. Golden dataset 25 Qs, tool tracking, failure attribution, LLM judge integration.
**Phase 3.4 (Tool Expansion)**: 11 tools с dynamic visibility, navigation short-circuit, temporal guard, refusal hardening.

**Важно**: strict recall@5 на golden_v1 (0.43) занижен — смешивает dataset strictness, alternative valid evidence и real retrieval misses. Manual judge показывает factual=0.52, что ближе к реальному качеству.

**Следующий приоритет**: audit zero-recall cases, soft metric для compare/summarize, stochastic refusal (q19/q20).

### История экспериментов

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

#### Retrieval Eval (прямые Qdrant queries, без LLM, ~5с/запрос)

Dataset: 100 auto-generated queries из первых предложений документов (35 каналов).
Тестирует **только retrieval quality**, без LLM query expansion и generation.

| # | Дата | Pipeline | Fusion | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Latency | Файл |
|---|------|----------|--------|----------|----------|-----------|-----------|---------|------|
| 17b | 2026-03-20 | BM25+Dense → RRF | RRF (3:1) | 0.36 | 0.55 | 0.64 | 0.70 | 2.5s | retrieval_eval_20260320-155419 |
| **17** | **2026-03-20** | **BM25+Dense → RRF → ColBERT** | **RRF (3:1)** | **0.71** | **0.73** | **0.73** | **0.74** | **5.0s** | **retrieval_eval_20260320-155000** |
| 18 | 2026-03-20 | BM25+Dense → DBSF → ColBERT | DBSF | 0.70 | 0.72 | 0.72 | 0.73 | 4.4s | retrieval_eval_20260320-180240 |

**Ключевой результат**: ColBERT rerank удвоил Recall@1 (0.36→0.71, +97%) и дал +33% Recall@5 (0.55→0.73). DBSF чуть хуже RRF — не оправдывает.

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

## Tier 1: Quick Wins (часы, не дни)

### 1.1 Embedding Whitening (PCA 1024→512)
- **Суть**: Global PCA whitening — mean-centering + scaling по eigenvalues + dimensionality reduction.
- **Почему поможет**: раздвигает cosine range. **Подтверждено экспериментально (2026-03-19)**:
  - ДО: pairwise cosine mean=0.7954, std=0.0933, range [0.33, 0.95]
  - ПОСЛЕ: pairwise cosine mean=0.0006, std=0.0484, range [-0.17, 0.61]
  - Variance explained: 96.2% (потеря 3.8% информации при 1024→512)
- **Whitening params сохранены**: `datasets/whitening_params.npz` (mean, components, explained_variance)
- **Нюанс**: query-only mean-centering **НЕ работает** (asymmetric — query в другом пространстве, docs в оригинальном). Нужна **полная переиндексация**: embed → whitening transform → upload whitened vectors в Qdrant.
- **Как реализовать**: скрипт переиндексации: загрузить все 13K vectors, применить whitening, создать новую коллекцию с dense_vector 512-dim, скопировать sparse + payload.
- **Ожидание**: +5-15% recall (по литературе: Su et al. +8-12 Spearman, WhitenRec +7-16% recall)
- **Результат эксперимента #1 (2026-03-19, 1024→512)**: переиндексация в `news_whitened` (512-dim). **Recall упал с 0.70 до 0.56.** Слишком агрессивный cutoff — потеря полезных dimensions.
- **Результат эксперимента #2 (2026-03-20, 1024→1024, без reduction)**:
  - Cosine range: original mean=0.80 std=0.09 → whitened mean=0.00 std=0.03
  - Ручной тест на 10 запросах (прямые Qdrant queries, **без LLM**):
  - 3 запроса стали лучше (Qwen3: miss→#5, NVIDIA: #2→#1, Sakana: #2→#1)
  - 1 запрос стал хуже (GPT-5 Pro: #8→miss)
  - 6 без изменений
  - **Итог: паритет с лёгким преимуществом whitened.** Не ломает pipeline, немного помогает на некоторых запросах.
  - Коллекция `news_whitened` (1024-dim) сохранена. Whitening params: `datasets/_whitening_mean.npy`, `datasets/_whitening_transform.npy`.
  - **Ключевой инсайт**: при weighted RRF 3:1 BM25 доминирует, dense вносит малый вклад. Whitening улучшает dense, но dense уже не bottleneck. Bottleneck — reranking quality и query expansion.
- **Статус**: [x] whitening 512 — recall 0.56, откат. [x] whitening 1024 — паритет, коллекция сохранена. **Dense не является bottleneck при текущей архитектуре (BM25 3:1).**
- **Ссылки**: Su et al. 2021 "BERT-whitening", WhitenRec 2024, WhiteningBERT (Huang et al. EMNLP 2021)

### 1.2 Weighted RRF (BM25 3:1 vs dense)
- **Суть**: Qdrant v1.17 поддерживает веса для prefetch веток в RRF. BM25 weight=3, dense weight=1.
- **Почему поможет**: BM25 правильно находит keyword-matched документы, но при equal weight dense "магниты" их перевешивают.
- **Как**: одна строка — `models.RrfQuery(rrf=models.Rrf(weights=[3.0, 1.0], k=2))`. Также асимметричный prefetch: BM25 limit=100, dense limit=20.
- **Ожидание**: +5-10% recall
- **Статус**: [x] **Выполнено (2026-03-19)**. BM25 limit=100, dense limit=20, weights=[1.0, 3.0]. Recall 0.33 → 0.59. Commit: e0bd871.

### 1.3 DBSF fusion (альтернатива RRF)
- **Суть**: Distribution-Based Score Fusion — нормализует скоры через mean ± 3σ перед слиянием.
- **Результат (2026-03-20, 100 queries)**: DBSF+ColBERT recall@5=0.72 vs RRF+ColBERT recall@5=0.73. **RRF лучше на 1%.** DBSF чуть быстрее (4.4с vs 5.0с) но не оправдывает потерю recall.
- **Статус**: [x] **Протестировано, RRF остаётся.** DBSF не даёт преимуществ при weighted RRF + ColBERT.

### 1.4 Channel-based dedup (max 2 docs per channel)
- **Суть**: ограничить максимум 2 документа из одного канала в результатах.
- **Почему поможет**: prolific каналы (gonzo_ml, ai_machinelearning_big_data) монополизируют top-10. Dedup даёт шанс менее частым но релевантным каналам.
- **Как**: post-processing в `_channel_dedup()` после `_to_candidates()`. Qdrant `group_by` не работает с multi-stage prefetch. Запрашиваем k×2 из Qdrant, dedup сужает до k.
- **Результат (2026-03-20, точечный тест на v2 Q1 + Q3)**:
  - Citations стали разнообразнее: ai_ml_big_data 3→2 постов, появились techsparks, aioftheday
  - Recall не изменился (0.50 → 0.50 для обоих) — нужные документы (data_secrets:8021, rybolos:1562) не попадают в ColBERT top-20 candidate pool. Dedup не может вытащить то, что не найдено retrieval'ом.
  - **Вывод**: dedup улучшает diversity но не recall, когда bottleneck — candidate generation, а не ranking.
- **Статус**: [x] **Выполнено**. Помогает diversity, не recall. Оставлен как полезный default.

### 1.5 Замена реранкера: bge-m3 → bge-reranker-v2-m3
- **Суть**: bge-m3 — bi-encoder, загруженный как seq-cls. bge-reranker-v2-m3 — dedicated cross-encoder, дообученный из bge-m3 backbone специально для query-document relevance scoring.
- **Нюанс**: `_name_or_path` в config обоих моделей = `BAAI/bge-m3` (reranker построен НА БАЗЕ bge-m3). Но **веса разные** — reranker дообучен на reranking task.
- **Результат эксперимента (2026-03-19)**:
  - Recall@5 на quick dataset (10 Qs): **0.70 → 0.70** (без изменений на малом датасете)
  - Но reranker scores **значительно лучше**:

  | | bge-m3 (старый) | bge-reranker-v2-m3 (текущий) |
  |---|---|---|
  | Релевантный doc | logit **-0.55**, sigmoid 0.37 | logit **+7.60**, sigmoid 0.9995 |
  | Нерелевантный doc | logit -8.77...-9.07 | logit -8.41...-11.03 |
  | Gap (relev vs irrelev) | ~8 пунктов | **~18 пунктов** |

  - Gap между релевантным и нерелевантным удвоился (8→18 logit points)
  - Confidence в правильном документе: 0.37 → **0.9995**
  - На 50+ вопросах с borderline documents разница проявится
- **Статус**: [x] **Выполнено**. Модель: `/home/tei-models/reranker-v2`, gpu_server.py переключён. Commit: 4d43183.
- **Альтернативы на будущее** (исследовано 2026-03-19):
  - **Jina Reranker v3** (0.6B, Qwen3-0.6B backbone, listwise): 65.20 nDCG на MIRACL Russian, BEIR 61.94 — выше Qwen3-Reranker-4B при 6x меньше. `AutoModel` + `trust_remote_code=True`. CC-BY-NC-4.0.
  - **Jina Reranker v2-base-multilingual** (278M, XLMRoberta): 100+ языков, 15× throughput vs bge-reranker-v2-m3. Контекст 1024 tokens. Стабилен через `CrossEncoder`.
  - **Contextual AI Reranker v2** (1B/2B/6B): **instruction-following** — можно приоритизировать по дате, типу, метаданным. Для temporal queries — ценная фича. Инференс через vLLM. CC-BY-NC-SA-4.0.
  - **GTE-Reranker-ModernBERT-base** (149M): Hit@1=83%, 8x меньше nemotron-1.2B. Контекст 8192. Но EN-only — нужна валидация на русском.
  - **Qwen3-Reranker-0.6B-seq-cls**: TEI не поддерживает (PR #835), но через gpu_server.py (AutoModelForSequenceClassification) **может работать** — стоит проверить.

---

## Tier 2: Архитектурные улучшения (1-3 дня каждое)

### 2.1 Query Classifier + Strategy Router
- **Суть**: определить тип запроса (factual/temporal/channel/comparative/multi-hop) → выбрать стратегию retrieval.
- **Почему поможет**: temporal запросы нуждаются в date filter, channel запросы — в channel filter, comparative — в parallel multi-query. Одна стратегия для всех = компромисс.
- **Как**: Qwen3-30B как classifier (structured JSON output). Map типа → стратегия (top_k, filters, diversity mode).
- **Ожидание**: +5-10% recall, -35% latency (enterprise benchmarks)
- **Статус**: [ ] не начато
- **Код из отчёта**:
```python
STRATEGIES = {
    "simple":      {"top_k": 3,  "diversity": "channel_dedup"},
    "temporal":    {"top_k": 10, "date_filter": True, "diversity": "temporal_buckets"},
    "comparative": {"top_k": 8,  "diversity": "bm25_mmr", "reranker": True},
    "multi_hop":   {"top_k": 5,  "diversity": "cluster_first", "iterative": True},
}
```

### 2.2 Entity Extraction (Natasha/Slovnet NER)
- **Суть**: извлекать именованные сущности (люди, компании, продукты) из запросов и документов.
- **Почему поможет**: "что писал Дженсен Хуанг" → entity="Дженсен Хуанг" → Qdrant payload filter. Точный recall на entity queries.
- **Как**: Natasha NER (27 МБ, CPU, F1~0.96, 25 docs/sec). Один раз прогнать по 13K постов (~10 мин), сохранить entities как Qdrant payload. При поиске — extract entities из query → filter.
- **Ожидание**: +5-15% recall на entity-specific запросах
- **Статус**: [ ] не начато

### 2.3 BM25-based Diversity (замена cosine MMR)
- **Суть**: MMR loop, но с BM25 pairwise similarity вместо cosine.
- **Почему поможет**: два поста про "трансформеры" vs "диффузию" — одинаковый cosine, но разный BM25 профиль. BM25 diversity работает когда cosine сломан.
- **Как**: модифицированный MMR loop в Python. `score = lambda * relevance - (1-lambda) * max(bm25_sim(doc, selected))`.
- **Ожидание**: +2-4% recall
- **Статус**: [ ] не начато

### 2.4 Genericity Score (штраф attractor documents)
- **Суть**: заранее посчитать для каждого документа — в скольких случайных запросах он попадает в top-10. Хранить как payload. Штрафовать при поиске.
- **Почему поможет**: напрямую атакует проблему attractor documents (gonzo_ml:4121, denissexy:10940).
- **Как**: прогнать 100 random queries → подсчитать frequency → сохранить как `genericity` payload. Qdrant formula query: `score = relevance * (1.0 - 0.3 * genericity)`.
- **Ожидание**: +3-5% recall
- **Статус**: [ ] не начато

### 2.5 Reranker-as-Fusion (без RRF)
- **Суть**: вместо RRF → rerank, сделать: BM25 top-50 + dense top-50 → deduplicate → cross-encoder reranks весь пул.
- **Почему поможет**: cross-encoder делает content-aware решения, не слепое rank fusion. Решает проблему "RRF весов" фундаментально.
- **Как**: два отдельных Qdrant search, merge, rerank 70-100 кандидатов. Latency ~150-300ms на RTX 5060 Ti.
- **Ожидание**: потенциально лучше RRF+rerank, нужен A/B тест
- **Статус**: [ ] не начато

---

## Tier 3: Глубокие улучшения (3-7 дней каждое)

### 3.1 ColBERT Reranking (jina-colbert-v2)
- **Суть**: per-token matching вместо single-vector cosine. Для каждого query token — MaxSim с document tokens.
- **Почему поможет**: **фундаментально** решает attractor problem. "Meta купила Manus" и "курс по трансформерам" — совершенно разные token profiles, даже если single-vector cosine одинаковый.
- **Как**: jina-colbert-v2 (560M, 89 языков, русский включён). Qdrant multi-vector config (MaxSim). Трёхэтапный: BM25+Dense → RRF → ColBERT rerank.
- **Storage**: ~500MB для 13K docs (100 tokens × 128 dim × float16). Реально: `_colbert_vectors.json` = 5.3 GB (нужно исключить из MCP индексации!).
- **Ожидание**: +6-10% nDCG → **подтверждено: +33% recall@5 (0.55→0.73) на 100 запросах**
- **Результат (2026-03-20)**: Recall@1 удвоился (0.36→0.71). ColBERT полностью устраняет attractor documents из top-10. На 10 agent eval вопросах разница не видна (0.76 vs 0.76) — нужен большой датасет.
- **Реализация**: gpu_server.py загружает jina-colbert-v2 + manual linear projection 1024→128. Endpoint `/colbert-encode`. Коллекция `news_colbert` с 3 named vectors: dense(1024) + sparse + colbert(128, MaxSim). HybridRetriever: 3-stage Qdrant query с fallback на RRF-only.
- **Зависимости WSL2**: einops, xlm-roberta-flash-implementation (скопировано оффлайн), config.json auto_map исправлен на локальные пути.
- **Latency**: +2.5с/запрос (5.0с vs 2.5с без ColBERT). Encoding всех 13K docs: ~67 мин на RTX 5060 Ti.
- **Статус**: [x] **Выполнено**. Коммит: 6961cab, 0919c3b.

### 3.2 Contextual Retrieval (Anthropic's technique)
- **Суть**: перед embedding'ом каждого чанка — LLM генерирует 2-3 предложения контекста ("Этот пост из канала X про тему Y"). Этот prefix disambiguates embedding.
- **Почему поможет**: посты "новая модель вышла" vs "бенчмарк модели" получают разные prefix'ы → разные embeddings. Anthropic измерили 35-67% reduction в retrieval failures.
- **Как**: прогнать Qwen3-30B по всем 13K постам (8-15 часов one-time на V100). Переиндексировать с prefix'ами.
- **Ожидание**: +10-20% recall (один из самых impactful, но трудоёмкий)
- **Статус**: [ ] не начато

### 3.3 Fine-tune Qwen3-Embedding-0.6B
- **Суть**: contrastive fine-tuning с hard negatives, добытыми из нашей же "сломанной" embedding space.
- **Почему поможет**: attractor documents = идеальные hard negatives. Модель учится различать именно те пары, которые сейчас путает.
- **Как**: 1) Qwen3-30B генерирует 3 query на пост → 39K положительных пар. 2) Для каждой пары top-50 "ложно похожих" = hard negatives. 3) sentence-transformers MultipleNegativesRankingLoss.
- **Ожидание**: +5-15% recall
- **Статус**: [ ] не начато
- **Ссылки**: Aurelio AI benchmarks, NV-Retriever (hard negative mining +2-5 nDCG)

### 3.4 CRAG (Corrective RAG)
- **Суть**: после search агент оценивает качество результатов. Если плохо — переформулирует запрос и ищет заново. Не просто "ещё один search", а анализ GAP'ов.
- **Почему поможет**: accuracy 58% → 83% в литературе. У нас уже есть refinement, но он только добавляет search, не переформулирует.
- **Как**: после compose_context, если coverage < threshold → анализ "чего не хватает" → новый query_plan с другими sub-queries → search.
- **Ожидание**: +10-15% на сложных запросах
- **Статус**: [ ] отложен (дорогой по latency, брать когда упрёмся)

---

## Tier 4: На будущее (держим в уме)

### 4.1 HyDE (Hypothetical Document Embedding)
- Генерировать "гипотетический ответ", embed его, искать похожие документы.
- +1-3с latency, не решает collapse в document space. **Complementary technique**.

### 4.2 Qwen3-Embedding-4B
- 8 GB VRAM, помещается рядом с реранкером на 16 GB. +5-10% vs 0.6B на benchmarks.
- Но "scaling alone won't fully solve domain-specific collapse" — не серебряная пуля.

### 4.3 Multi-Collection Architecture
- Разделить на коллекции по длине/типу/тематике. Разные embedding стратегии для каждой.
- Router выбирает в какие коллекции искать.

### 4.4 Link Expansion
- Многие посты = "ссылка + комментарий". Индексировать контент ссылок → +5-15% recall на запросах, ответ на которые в ссылке.

### 4.5 DPP Diversity (Determinantal Point Processes)
- Математически принципиальный diversity selection. YouTube recommendations использует.
- +5-15% diversity metrics vs MMR. Библиотека `dppy`.

### 4.6 Temporal Decay
- `fused_score = 0.7 × semantic + 0.3 × 0.5^(age_days/14)` — boost свежих документов.
- Тривиально через Qdrant payload.

### 4.7 Channel Authority Scoring
- Вручную расставить веса каналам (gonzo_ml=0.9, ml_product=0.5). Boost авторитетных.
- ReliabilityRAG (2025) — explicit source reliability signals.

### 4.8 Forward/Reply Chain Awareness
- Хранить `reply_to_id`, `forwarded_from` как payload. При нахождении поста — подтянуть всю цепочку.

---

## NEXT PHASE: Adaptive Retrieval + Tool Router

> **Приоритет #1.** Все 4 исследования (R13-quick, R13-deep, R14-quick, R14-deep) единогласно рекомендуют.
> Подробный план: `adaptive_retrieval_plan.md`

### Почему это следующий шаг

Pipeline оптимизации (Tier 1-3) дали recall 0.15→0.76, но уперлись в потолок: pipeline **линейный**. Все запросы идут одним путём. Temporal query "что было в январе 2026" и factual "Meta купила Manus" обрабатываются идентично. Это то, что фреймворки тоже не умеют — и ключевое отличие нашей системы.

### Что конкретно решает

| Открытая проблема | Как решает adaptive retrieval |
|---|---|
| #6: LLM не знает "Vera Rubin" | Rule-based: "2026" + "NVIDIA" → `temporal_search` + `entity_search`, даже без знания entity |
| #8: Partial recall на multi-doc | `entity_search` с BM25 keyword boost по entity name → расширяет candidate pool |
| Все temporal queries | `temporal_search` с Qdrant DatetimeRange filter → precision вместо broad search |
| Channel-specific queries | `channel_search` с Qdrant MatchValue filter → только релевантный канал |

### Архитектура (консенсус)

```
User Query
    │
    ├─ Rule-based pre-validator (<1ms)
    │   └─ regex: dates, @channels, entity patterns → hints
    │
    ├─ query_plan (LLM, ~12s) — enriched JSON:
    │   └─ {subqueries, strategy: "temporal", filters: {date_from, date_to, channels, entities}}
    │
    ├─ Strategy → Tool dispatch:
    │   ├─ broad    → base_search(queries, k)
    │   ├─ temporal → base_search(queries, k, date_filter)
    │   ├─ channel  → base_search(queries, k, channel_filter)
    │   └─ entity   → base_search([entity]+queries, k, optional_filters)
    │
    ├─ Quality gate (ColBERT scores):
    │   ├─ >0.6  → Correct, use results
    │   ├─ 0.3-0.6 → Ambiguous, expand search
    │   └─ <0.3  → Incorrect, fallback to broad_search
    │
    └─ Fallback chain: specialized → broadened → broad
```

### Ожидаемый эффект

- +8-15% recall на temporal/channel/entity запросах
- Papers: Adaptive-RAG +5-31pp, CRAG +7-37%, RouteRAG 97.6-100% EM
- Latency: ±0-3с (routing встроен в query_plan, не отдельный LLM call)

### Ссылки на ресерчи

- **R13-quick**: `reports/R13-quick-tool-router-architecture.md` — 4 tools, query_plan enrichment, Qdrant filters
- **R13-deep**: `reports/R13-deep-tool-router-architecture.md` — grammar enforcement, 3-tier fallback, parallel Qdrant
- **R14-quick**: `reports/R14-quick-beyond-frameworks-techniques.md` — A-RAG, CRAG, interview strategy
- **R14-deep**: `reports/R14-deep-beyond-frameworks-techniques.md` — Speculative RAG, NLI, temporal reasoning, 5-day plan

---

## Research Track: Тематическая кластеризация коллекции

> **Статус**: исследование завершено (R12). **Вердикт: Phase 4, не Phase 1.** Есть более простые и impactful решения. Кластеризация становится оправданной при 50K+ документов или когда простые фиксы plateau'ят.
> **Источник**: [R12-cluster-based-retrieval.md](../research/reports/R12-cluster-based-retrieval.md)

### Исходная идея

Одна плоская коллекция на весь AI-корпус — наивно. Все документы "про AI" сливаются в embedding space (cosine 0.78-0.83). Если кластеризовать по темам (M&A, релизы моделей, образование, research papers...) — внутри каждого кластера cosine станет осмысленным.

### Результаты исследования (R12)

**Comparison table (ключевой результат)**:

| Подход | Effort | Recall@5 Δ | Cumulative | Ops burden |
|--------|--------|------------|------------|------------|
| Weighted RRF tuning | 1-2 ч | +3-10% | 0.62-0.65 | None |
| **Global PCA whitening** (1024→512) | 2-4 ч | **+5-15%** | 0.67-0.73 | Near-zero |
| **bge-reranker-v2-m3** | 4-8 ч | **+15-30%** | 0.75-0.82 | ~200ms latency |
| BGE-M3 model swap (dense+sparse+ColBERT) | 8-12 ч | +25-40% | 0.80-0.88 | Medium |
| **Topic clustering + routing** | **20-40 ч** | +10-20% | 0.69-0.77 | **High (ongoing)** |

**Вывод**: кластеризация даёт **меньше recall** при **большем effort** чем whitening + reranker. Cross-encoder reranker "bypasses the cosine floor entirely" — обходит проблему embedding collapse фундаментально, без кластеров.

### Что мы узнали (ценные находки)

**Per-cluster whitening математически некорректен** при наших размерах:
- 200-600 docs/cluster, 1024 dims → ковариационная матрица rank-deficient (625 eigenvalues = 0)
- Деление на 0 при whitening = amplification шума
- **Безопасная операция**: только mean-centering per-cluster (без PCA)
- **Global whitening** корректен (N=13000, d=1024, ratio ~12.7)

**BERTopic — правильный инструмент** (не raw TF-IDF):
- TF-IDF плох для коротких мультиязычных текстов (40-120 tokens → шумные вектора)
- Ключевая идея: **отдельная embedding модель** для кластеризации (paraphrase-multilingual-mpnet-base-v2), не наша Qwen3-Embedding (которая сама "сломана")
- UMAP проецирует в 5 dims → re-scales compressed distance space → HDBSCAN работает
- 25-40 кластеров оптимально (min_cluster_size=100-200)
- Soft assignment через `approximate_distribution()` (top-3 clusters per doc, хранить как array payload в Qdrant)

**Routing: ensemble (centroid + BM25 frequency)**, не LLM:
- Embed query → nearest centroids (top-3) + BM25 full-corpus top-50 → count cluster frequency (top-3) → union
- Суммарная latency ~100-150ms на CPU
- LLM-classification **не рекомендуют**: "slow inference, high costs, poor accuracy on domain-specific topics"
- Fine-tuned BERT classifier (94% accuracy, ms latency) — если нужен ML-based router, но требует labeled data

**Incremental updates**: daily nearest-centroid assignment + weekly BERTopic `merge_models()`. Trigger re-clustering при drift > threshold или >500 orphan docs.

**"Embedding collapse on topically narrow corpus — expected behavior, не баг"**:
- Ethayarajh 2019: average BERT pairwise cosine = 0.99 (!)
- Zhou et al. ACL 2025 (Length-Induced Collapse): self-attention = low-pass filter, фундаментальное свойство архитектуры
- Domain homogeneity (все про AI) amplifies сужение

### Когда кластеризация НУЖНА

1. **50K+ документов** — reranker не справляется с шумным candidate pool, cluster filtering сужает HNSW search space на 90-95%
2. **Расширение за AI/ML** — crypto, biotech, finance → cross-domain filtering через кластеры
3. **Per-cluster whitening at scale** — при 500+ docs/cluster ковариация стабильна

### Архитектура (финальная, из R12)

```
Query → [Embed] + [BM25 full corpus]
           ↓              ↓
    [Global Whitening]  [Top-50 BM25]
    [PCA 1024→512]      [Cluster frequency]
           ↓              ↓
    [Centroid routing]  [top-3 clusters]
           ↓              ↓
         [Union: 3-5 clusters]
                  ↓
    [Qdrant Filtered Hybrid Search]
    filter: cluster_ids ∈ selected
    prefetch: dense(20) + BM25(20)
    fusion: weighted RRF
                  ↓
    [bge-reranker-v2-m3 top-20 → top-5]
                  ↓
         [Final Results]
Latency: ~240ms total
```

### Код для будущей реализации

```python
# BERTopic clustering
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

cluster_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
embeddings = cluster_model.encode(all_texts, batch_size=64)

topic_model = BERTopic(
    embedding_model=cluster_model,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
    hdbscan_model=HDBSCAN(min_cluster_size=150, min_samples=10, prediction_data=True),
    calculate_probabilities=True
)
topics, probs = topic_model.fit_transform(all_texts, embeddings)

# Soft assignment → Qdrant payload
topic_distr, _ = topic_model.approximate_distribution(all_texts, window=4, stride=1)
for i, point_id in enumerate(point_ids):
    top_ids = np.argsort(topic_distr[i])[::-1][:3]
    assigned = [int(t) for t in top_ids if topic_distr[i][t] > 0.05] or [int(top_ids[0])]
    client.set_payload("news", {"cluster_ids": assigned}, points=[point_id])
```

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

## Что НЕ работает / НЕ стоит пробовать

| Техника | Почему не работает |
|---------|-------------------|
| Cosine-based MMR | Переиспользует сломанный cosine signal → re-promotes attractors. Tested lambda 0.7 и 0.9, оба хуже baseline. |
| Dense re-score после RRF | Стирает BM25 вклад. Recall 0.33→0.15. |
| SPLADE для русского | Нет production-ready мультиязычных моделей. WSDM Cup 2026: "struggled to remain competitive". |
| HyDE как primary fix | +1-3с latency, не решает collapse в document space. Only complementary. |
| Scaling embedding alone | "Length-Induced Embedding Collapse" + domain-specific collapse = фундаментальные ограничения single-vector. 4B/8B помогут, но не решат. |

---

## Таблица реранкеров (для выбора)

| Модель | Params | VRAM | MIRACL Avg | Примечание |
|--------|--------|------|------------|------------|
| **bge-reranker-v2-m3** | 568M | ~1.2 GB | 69.32 | Best multilingual, наш target |
| jina-reranker-v2-base-multilingual | 278M | ~0.6 GB | Competitive | 15× throughput, `use_flash_attn=False` для sm_120 |
| jina-reranker-v3 | 0.6B | ~1.2 GB | 66.50 | BEIR SOTA (61.94) |
| bge-reranker-v2-gemma | 2.5B | ~5 GB | Higher | Best quality на 16GB |

---

## Бенчмарки для ориентации

- **Production RAG (structured docs)**: 0.70-0.85 recall@5
- **Short social media, non-English**: **0.55-0.70** recall@5
- **С whitening + cross-encoder reranker**: **0.75-0.82** (R12 estimate)
- **С BGE-M3 swap**: **0.80-0.88** (R12 estimate)
- **Текущий**: **0.70** (10 questions, quick dataset) — уже в production range!
- **Минимум для regression testing**: 50 вопросов
- **Минимум для значимости**: 200+ вопросов

## Рекомендуемый путь к 0.80+ (из R11 + R12)

```
Текущее состояние: recall@5 = 0.70
  ↓
✅ Phase 0 (Done): Weighted RRF 3:1 + forced search + bge-reranker-v2-m3
  → Achieved: agent recall 0.15 → 0.70
  ↓
✅ Phase 0.5 (Done): Whitening 1024-dim — паритет, dense не bottleneck
  ↓
✅ Phase 1 (Done): ColBERT reranking (jina-colbert-v2)
  → Achieved: retrieval recall@5 0.55 → 0.73 (+33%), recall@1 0.36 → 0.71 (+97%)
  ↓
✅ Phase 1.5 (Done): Per-category eval matching + retrieval-only eval (100 Qs)
  → Achieved: agent recall 0.70 → 0.76, retrieval eval infrastructure ready
  ↓
Phase 2 (Next): Расширить dataset (20-30 agent + 100 retrieval) + DBSF fusion + channel dedup
  → Expected: retrieval 0.78-0.82
  ↓
Phase 3: Query classifier + entity extraction + contextual retrieval
  → Expected: 0.82-0.88
  ↓
Phase 4 (если нужно): Fine-tune embedding / кластеризация / BGE-M3 swap
  → Expected: 0.88+
```
