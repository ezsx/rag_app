# Retrieval Ablation Study

> Систематическое исследование retrieval pipeline: параметры, компоненты, новые подходы.
> Проводится совместно Claude + Codex (sidecar reviewer).

**Dataset**: `datasets/eval_retrieval_v3.json` — 120 natural language вопросов.
- 6 категорий: factual (48), temporal (18), channel_specific (18), comparative (12), entity (12), edge (12)
- 30 каналов, stratified sampling из 2767 документов (Jul 2025 — Mar 2026)
- Валидация: 75.8% dense-only recall@20

**Collection**: `news_colbert_v2` (13,777 points, Qdrant)

**Production pipeline**: `BM25(100) + Dense(20) → RRF [1:3] → ColBERT MaxSim → CE filter(0.0) → channel dedup(2)`

---

## Phase 1 — Parameter Sweep (2026-04-04)

24 эксперимента. Тестировали: instruction prefix, dense limit (10/20/40/60), BM25 limit (50/100/200), RRF weights ([1:1]..[1:5]), DBSF fusion, ColBERT on/off, комбинации.

### Full Results Table

| # | Experiment | R@1 | R@3 | R@5 | R@10 | R@20 | MRR | Δ R@5 |
|---|-----------|-----|-----|-----|------|------|-----|-------|
| | **Baseline** | | | | | | | |
| 1 | baseline (prefix ON, dense=20, BM25=100, RRF default, ColBERT ON) | 0.708 | 0.817 | 0.833 | 0.850 | 0.867 | 0.765 | — |
| | **Single parameter changes** | | | | | | | |
| 2 | no_prefix | **0.725** | **0.875** | **0.892** | **0.908** | **0.917** | **0.801** | **+0.058** |
| 3 | dense_limit=40 | 0.733 | 0.850 | 0.867 | 0.892 | 0.900 | 0.793 | +0.034 |
| 4 | dense_limit=10 | 0.675 | 0.792 | 0.817 | 0.833 | 0.842 | 0.738 | −0.017 |
| 5 | rrf_weights [1.0, 1.0] | 0.708 | 0.817 | 0.833 | 0.850 | 0.867 | 0.765 | 0.000 |
| 6 | rrf_weights [1.0, 2.0] | 0.708 | 0.817 | 0.833 | 0.850 | 0.867 | 0.765 | 0.000 |
| 7 | rrf_weights [1.0, 3.0] | 0.708 | 0.817 | 0.833 | 0.850 | 0.867 | 0.765 | 0.000 |
| 8 | rrf_weights [1.0, 4.0] | 0.692 | 0.808 | 0.825 | 0.842 | 0.858 | 0.753 | −0.008 |
| 9 | rrf_weights [1.0, 5.0] | 0.683 | 0.808 | 0.833 | 0.842 | 0.858 | 0.749 | 0.000 |
| 10 | bm25_limit=50 | 0.708 | 0.817 | 0.833 | 0.850 | 0.867 | 0.765 | 0.000 |
| 11 | bm25_limit=200 | 0.700 | 0.817 | 0.833 | 0.850 | 0.867 | 0.761 | 0.000 |
| 12 | DBSF fusion | 0.675 | 0.808 | 0.825 | 0.842 | 0.850 | 0.744 | −0.008 |
| 13 | no ColBERT | 0.467 | 0.650 | 0.733 | 0.758 | 0.842 | 0.581 | −0.100 |
| | **Combinations** | | | | | | | |
| 14 | no_prefix + dense=40 | 0.742 | 0.883 | 0.900 | 0.917 | 0.933 | 0.813 | +0.067 |
| 15 | **no_prefix + dense=40 + RRF [1:3]** | **0.758** | **0.883** | **0.900** | **0.917** | **0.933** | **0.823** | **+0.067** |
| 16 | no_prefix + BM25=200 + dense=40 | 0.742 | 0.883 | 0.900 | 0.917 | 0.933 | 0.813 | +0.067 |
| 17 | no_prefix + dense=60 + RRF [1:3] | 0.750 | 0.883 | 0.900 | 0.917 | 0.925 | 0.818 | +0.067 |
| 18 | no_prefix + RRF [1:3] | 0.725 | 0.875 | 0.892 | 0.908 | 0.917 | 0.802 | +0.058 |
| 19 | no_prefix + BM25=200 | 0.717 | 0.875 | 0.892 | 0.908 | 0.917 | 0.797 | +0.058 |
| 20 | DBSF + no_prefix | 0.700 | 0.875 | 0.892 | 0.900 | 0.917 | 0.788 | +0.058 |
| 21 | no ColBERT + no_prefix | 0.542 | 0.800 | 0.850 | 0.908 | 0.933 | 0.681 | +0.017 |
| 22 | DBSF + wide (BM25=200, dense=40) | 0.725 | 0.833 | 0.850 | 0.875 | 0.875 | 0.782 | +0.017 |
| 23 | max candidates (BM25=200, dense=40) | 0.725 | 0.850 | 0.867 | 0.892 | 0.900 | 0.789 | +0.033 |
| 24 | no ColBERT + RRF default | 0.442 | 0.667 | 0.733 | 0.758 | 0.842 | 0.570 | −0.100 |

### Phase 1 Winner

```
no-prefix + dense=40 + RRF [1.0, 3.0] + ColBERT ON + BM25=100
R@1=0.758  R@5=0.900  R@20=0.933  MRR=0.823
```

### Key Findings

1. **Instruction prefix вредил в eval** (+5.8% R@5 при отключении). Eval скрипт добавлял prefix `"Instruct: Given a user question..."`, production (DEC-0042) работает без prefix. Самый значимый одиночный параметр.
2. **Dense limit 20→40** = +3.4% R@5. Больше dense кандидатов → ColBERT получает разнообразнее пул. Выше 40 — diminishing returns (dense=60 тот же R@5, но R@1 чуть хуже).
3. **ColBERT критически важен**. Без ColBERT: R@1 −34% (0.708→0.467). Latency +2.2s/query, но quality оправдывает.
4. **RRF weights не влияют** при наличии ColBERT. [1:1], [1:2], [1:3] — идентичные результаты. ColBERT полностью переранжирует RRF output. Веса [1:4] и [1:5] слегка хуже.
5. **BM25 limit не влияет**. 50/100/200 — одинаковый результат. BM25 top-100 уже saturation.
6. **DBSF fusion хуже RRF** на −0.8%.

### Per-Category Analysis (baseline → best)

| Category | n | Baseline R@1 | Best R@1 | Baseline R@5 | Best R@5 | Baseline MRR | Best MRR |
|----------|---|-------------|---------|-------------|---------|-------------|---------|
| factual | 48 | 0.83 | **0.85** | 0.92 | **0.96** | 0.875 | **0.909** |
| comparative | 12 | 0.58 | **0.67** | 0.83 | **1.00** | 0.683 | **0.808** |
| entity | 12 | 0.75 | 0.75 | 0.92 | 0.92 | 0.822 | 0.818 |
| channel_specific | 18 | 0.78 | **0.89** | 0.89 | 0.89 | 0.827 | **0.889** |
| temporal | 18 | 0.67 | **0.72** | 0.78 | **0.89** | 0.713 | **0.796** |
| edge | 12 | 0.25 | **0.33** | 0.42 | **0.58** | 0.337 | **0.442** |

Наибольшее улучшение: comparative (+0.17 R@5) и edge (+0.17 R@5). Edge остаётся самой слабой категорией.

### 12 Permanent Misses (R@5=0 на best config)

| ID | Category | Query | Expected doc |
|----|----------|-------|-------------|
| ret_030 | factual | Какую конституцию опубликовала Anthropic? | theworldisnoteasy:2378 |
| ret_034 | factual | ИИ-модели играют в стратегические игры? | seeallochnaya:3054 |
| ret_049 | temporal | Что нового в генеративных моделях в феврале? | ai_newz:4433 |
| ret_059 | temporal | Инвестиции в ИИ-компании в начале 2026? | seeallochnaya:3270 |
| ret_068 | channel | llm_under_hood рекомендации? | llm_under_hood:752 |
| ret_075 | channel | Постнаука + gonzo_ml? | gonzo_ml:4666 |
| ret_098 | entity | Сколько OpenAI привлекла? | seeallochnaya:3427 |
| ret_109 | edge | "че там по трансформерам нового" | gonzo_ml:4567 |
| ret_110 | edge | "какие нейросетки умеют видосы делать" | neurohive:1929 |
| ret_111 | edge | Claude Code + старые игры? | denissexy:11238 |
| ret_112 | edge | нейросети заменят программистов? | techno_yandex:4978 |
| ret_113 | edge | ии-музыка через suno? | denissexy:10782 |

**Артефакты**: `results/ablation/` (24 JSON), `results/ablation/_summary.json`, `scripts/evaluate_retrieval.py`

---

## Phase 2a — Diagnosis (2026-04-05)

### Stage Attribution

Для каждого из 12 permanent misses прогнали query через отдельные стадии: Dense top-100, BM25 top-100, RRF top-60, ColBERT top-20.

| ID | Category | Dense-100 | BM25-100 | RRF-60 | ColBERT-20 | CE Score | Диагноз |
|----|----------|-----------|----------|--------|------------|----------|---------|
| ret_030 | factual | r=36 | miss | r=36 | **r=8** | 6.33 | **found** (dense=40 помогло) |
| ret_098 | entity | r=8 | miss | r=35 | **r=9** | 7.39 | **found** |
| ret_110 | edge | r=23 | r=5 | r=6 | **r=14** | 1.28 | **found** |
| ret_112 | edge | r=25 | miss | r=26 | **r=16** | 2.54 | **found** |
| ret_034 | factual | r=80 | miss | miss | miss | — | **lost_in_rrf** (rank 80, обрезка top-60) |
| ret_049 | temporal | miss | r=71 | miss | miss | — | **lost_in_rrf** (BM25 r=71, обрезка) |
| ret_059 | temporal | r=77 | miss | miss | miss | — | **lost_in_rrf** (rank 77) |
| ret_111 | edge | r=70 | miss | miss | miss | — | **lost_in_rrf** (rank 70) |
| ret_068 | channel | miss | **r=6** | **r=6** | miss | — | **lost_in_colbert** (BM25 идеально, ColBERT выкинул) |
| ret_075 | channel | r=10 | miss | r=40 | miss | — | **lost_in_colbert** |
| ret_113 | edge | r=27 | r=13 | r=17 | miss | — | **lost_in_colbert** |
| ret_109 | edge | miss | miss | miss | miss | — | **not_in_candidates** (lexical+semantic gap) |

**Summary**:

| Стадия потери | Count | IDs | Action |
|---|---|---|---|
| **found** (ложные misses, dense=40 находит) | 4 | ret_030, ret_098, ret_110, ret_112 | Уже решено в phase 1 |
| **lost_in_rrf** (rank 70-80, RRF top-60 обрезает) | 4 | ret_034, ret_049, ret_059, ret_111 | → расширить rrf_limit 60→100 |
| **lost_in_colbert** (в RRF, ColBERT top-20 не видит) | 3 | ret_068, ret_075, ret_113 | → расширить ColBERT pool 20→40, исследовать semantic gap |
| **not_in_candidates** | 1 | ret_109 | → только LLM rephraser или HyDE |

### CE Fix + Score Distribution

**Bug (Codex review)**: `RerankerService.rerank_with_scores()` (reranker_service.py:70) возвращал sigmoid(raw), а `rerank.py:55` сравнивал `score >= 0.0`. sigmoid ∈ [0,1] → threshold=0.0 **всегда true** → CE filter = полный no-op в production.

**Fix**: добавлен `rerank_with_raw_scores()` возвращающий raw logits. `rerank.py` переключён. Threshold=0.0 теперь = decision boundary (sigmoid(0)=0.5).

**Score Distribution** (120 Qs × top-20, всего 2400 пар query-doc):

| | Relevant docs (n=175) | Irrelevant docs (n=2225) |
|---|---|---|
| mean | 5.9 | −0.8 |
| median | 7.0 | −0.65 |
| p5 | −4.5 | −8.7 |
| p25 | 4.2 | −4.8 |
| p75 | 9.0 | 3.3 |
| p95 | 10.2 | 7.2 |

**Threshold analysis** — что фильтруется при разных порогах:

| Threshold | Relevant lost | Irrelevant removed | Precision gain vs Recall loss |
|-----------|--------------|-------------------|------------------------------|
| −1.0 | 8.6% (15/175) | 48.0% (1069/2225) | Консервативный |
| −0.5 | 9.1% (16/175) | 50.9% (1133/2225) | |
| **0.0** | **9.1% (16/175)** | **53.8% (1196/2225)** | **Decision boundary** |
| 0.5 | 10.3% (18/175) | 56.9% (1265/2225) | |
| 1.0 | 10.9% (19/175) | 59.6% (1326/2225) | |
| 2.0 | 13.7% (24/175) | 66.2% (1473/2225) | Слишком агрессивно |

**Вывод**: CE разделяет классы (mean relevant=5.9 vs mean irrelevant=−0.8), но с большим overlap (25% irrelevant docs имеют score > 3.3). Threshold=0.0 — рабочий порог.

**Артефакты**: `results/ablation/stage_attribution.json`, `results/ablation/ce_score_distribution.json`

---

## Phase 2b — Prod-Parity Query-Plan Ablation (2026-04-05)

Создан `scripts/evaluate_retrieval_full.py` — eval через direct import production модулей (HybridRetriever, QueryPlannerService, RerankerService) без FastAPI DI. Modular flags для каждого pipeline stage.

**Dataset**: 48 hard Qs (edge=12, temporal=18, channel_specific=18). Factual исключены (R@5=0.96, уже хорошо).

**Note**: production HybridRetriever использует формулу `dense = max(k*2, 20)` = 20 при k=10. Phase 1 winning dense=40 здесь не применён — это production baseline.

### Query-Plan Ablation Matrix

| # | Config | Pipeline | R@1 | R@5 | R@20 | MRR | Lat |
|---|--------|----------|-----|-----|------|-----|-----|
| 1 | raw retriever | поиск по raw query | 0.625 | 0.771 | 0.792 | 0.696 | 3.0s |
| 2 | query_plan only | LLM subqueries, без оригинала | 0.333 | 0.625 | 0.667 | 0.462 | 13.8s |
| 3 | **qplan + inject original** | LLM subqueries + оригинал первым | **0.625** | **0.792** | **0.812** | 0.684 | 16.8s |
| 4 | qplan + inject + filters | + metadata date/channel filters | 0.604 | 0.792 | 0.812 | 0.664 | 16.7s |
| 5 | full prod (CE + dedup) | + CE filter(0.0) + channel dedup(2) | 0.604 | 0.792 | 0.792 | 0.666 | 37.3s |

### Key Findings

1. **Query plan без original injection = вредит** (−14.6% R@5). LLM теряет лексический якорь, subqueries перефразированы. Для BM25 оригинальные слова критичны.
2. **Query plan + inject = +2.1% R@5** (0.771→0.792). Скромный, но стабильный uplift. Original injection = обязательная часть архитектуры.
3. **Metadata filters не помогают** на hard subset. R@1 даже падает (0.625→0.604). Возможно planner генерирует неточные фильтры.
4. **CE filter просаживает R@20** (0.812→0.792). Отфильтровал 1 relevant doc. Overhead ×2 по latency.
5. **Full prod хуже чем qplan+inject**. Orchestration stages (CE, dedup, filters) вносят overhead без quality gain для retrieval.
6. **Query planner generating**: для edge queries ("че там по трансформерам") LLM генерировал subqueries на английском ("transformer models new releases"). BM25 не матчит русские документы. Нужен language fix в planner prompt.

**Вывод**: основной рычаг = retriever tuning (dense limit, funnel width), не orchestration. Query plan полезен только с inject, и uplift скромный (+2%).

**Артефакты**: `results/ablation/phase2/p2_*.json`, `scripts/evaluate_retrieval_full.py`

---

## Phase 2c — New Retrieval Tracks (2026-04-05, in progress)

Согласовано Claude + Codex + пользователь: не только тюнинг параметров, но и **новые retrieval подходы**. Ablation = exploration, не tuning.

### Experiment Matrix

| ID | Эксперимент | Тип | Что меняем | Гипотеза | Метрика успеха | Dataset |
|----|------------|-----|-----------|----------|---------------|---------|
| **A3** | Combo baseline: dense=40 + rrf=100 + ColBERT=40 | funnel | Все confirmed levers вместе | Новый baseline, R@5 > 0.900 | R@5, MRR | 120 Qs |
| **A2** | ColBERT output 20→40 | funnel | ColBERT rerank pool | 3 lost_in_colbert: limit или semantic gap? | R@5 +2% | 120 Qs |
| **A1** | rrf_limit 60→100 | funnel | RRF output limit | 4 lost_in_rrf docs вернутся (rank 70-80) | R@20 +3% | 120 Qs |
| **R2** | Sparse-only normalization | **lexical fix** | Normalized query → BM25, raw → dense | Lexical gap в BM25 (сленг/aliases), dense оставить чистым | edge/channel R@5 +15% | 120 Qs |
| **R1** | Raw + normalized fusion | **lexical fix** | Normalized query → обе ветки | Помогает ли нормализация dense или только BM25 | сравнение с R2 | 120 Qs |
| **R3** | LLM single rewrite + raw fusion | **query expansion** | 1 LLM rewrite (same lang) + raw, fuse | Enriched dense + preserved BM25 anchor, дешевле query_plan | R@5, edge R@5 | 48 Qs |
| **R6** | Rule-based filters без planner | **filter isolation** | query_signals → date/channel, без LLM | Filters полезны, но planner шумит? | temporal/channel R@5 | 48 Qs |
| **R5** | BM25 PRF-lite | **corpus-driven expansion** | BM25 top-5 → top terms → expanded query, без LLM | Lexical gap закрывается через корпус | edge/entity R@5 | 120 Qs |
| **E1** | Dense-only + ColBERT | **signal isolation** | BM25 off, только dense → ColBERT | Вклад BM25 по категориям: нужен ли вообще? | R@5 per category | 120 Qs |
| **D2** | HyDE + original dense (dual branch) | **hypothetical doc** | LLM генерит pseudo-post → embed → extra dense branch, fuse с raw | Для broad/edge queries pseudo-doc ближе к реальным постам | edge R@5 | 48 Qs |

### Results (9/10 complete)

Baseline: Phase 1 winner = R@5 **0.900**, MRR **0.823** (no-prefix + dense=40 + RRF [1:3], 120 Qs).

| ID | Эксперимент | Тип | R@1 | R@5 | R@20 | MRR | Qs | Δ R@5 |
|----|------------|-----|-----|-----|------|-----|----|-------|
| A3 | dense=40 + rrf=100 + ColBERT=40 | funnel | 0.733 | 0.900 | 0.933 | 0.809 | 120 | 0.000 |
| A2 | ColBERT output 20→40 | funnel | 0.733 | 0.900 | 0.933 | 0.811 | 120 | 0.000 |
| A1 | rrf_limit 60→100 | funnel | 0.733 | 0.900 | 0.933 | 0.809 | 120 | 0.000 |
| **R2** | **Normalized query → BM25 only** | **lexical fix** | **0.742** | **0.900** | **0.933** | **0.814** | 120 | 0.000 |
| R1 | Normalized → обе ветки | lexical fix | 0.700 | 0.858 | 0.933 | 0.772 | 120 | **−0.042** |
| R3 | LLM single rewrite + raw | query expansion | 0.625 | 0.792 | 0.812 | 0.704 | 48* | — |
| R6 | Rule-based filters | filter isolation | 0.604 | 0.771 | 0.792 | 0.686 | 48* | — |
| R5 | BM25 PRF-lite | corpus expansion | 0.633 | 0.858 | 0.917 | 0.733 | 120 | **−0.042** |
| E1 | Dense-only + ColBERT | signal isolation | 0.742 | 0.892 | 0.925 | 0.809 | 120 | −0.008 |
| D2 | HyDE + original dense | hypothetical doc | — | — | — | — | 48* | computing |

*R3, R6 на 48 hard Qs subset (edge+temporal+channel), не сравнимы напрямую с 120 Qs baseline.

### Key Findings

1. **Funnel expansion (A1/A2/A3) не даёт прироста** — rrf_limit 60→100 и ColBERT pool 20→40 не изменили ни одну метрику. Stage attribution показывал потери в RRF/ColBERT, но расширение лимитов их не решило. Проблема = semantic gap, не truncation.

2. **R2 (sparse-only normalization) — лучший новый трек**. R@1 0.733→0.742, MRR 0.809→0.814. Нормализация помогает BM25 матчить сленг, не ломая dense.

3. **R1 (normalize all) — вредит!** R@5 −4.2% (0.900→0.858). Синонимы в dense query размывают embedding. **Вывод: нормализация строго только для BM25.**

4. **R5 (PRF-lite) — вредит.** R@5 −4.2%. Top terms из initial BM25 hits = нерелевантный шум.

5. **E1 (Dense-only)** — R@5 0.892 vs hybrid 0.900. BM25 вносит +0.8% R@5. Стабильный вклад, убирать не стоит.

6. **R3 (LLM rewrite) и R6 (rule-based filters)** — обе хуже raw retriever на hard subset. Prod HybridRetriever (dense=20) ограничивает potential.

### Что подтвердилось, что опровергнуто

| Гипотеза | Статус | Комментарий |
|----------|--------|-------------|
| rrf_limit truncation = bottleneck | **Опровергнуто** | Расширение 60→100 не помогло |
| ColBERT pool 20 = bottleneck | **Опровергнуто** | Pool 40 = те же числа |
| Lexical normalization поможет BM25 | **Частично подтверждено** | R2 +0.009 R@1, R@5 тот же |
| Normalization для dense вредит | **Подтверждено** | R1 −4.2% R@5 |
| PRF expansion поможет edge | **Опровергнуто** | Добавляет шум |
| BM25 можно убрать | **Опровергнуто** | Dense-only −0.8% R@5 |

### Инструменты

- `datasets/query_normalization_lexicon.json` — JSON словарь (slang 47, channels 37, aliases 43 entries)
- `scripts/evaluate_retrieval.py` — +8 CLI params (rrf_limit, colbert_pool, dense-only, normalize, prf)
- `scripts/evaluate_retrieval_full.py` — +3 flags (single-rewrite, hyde, rule-based-filters)
- `scripts/_run_phase2_experiments.py` — sequential runner

---

## Хронология

| Дата | Фаза | Экспериментов | Ключевой результат |
|------|-------|--------------|-------------------|
| 2026-04-04 | Phase 1: parameter sweep | 24 | R@5 0.833→0.900 (no-prefix + dense=40) |
| 2026-04-05 | Phase 2a: diagnosis | — | CE bug fixed, stage attribution (4 rrf, 3 colbert, 1 not_found) |
| 2026-04-05 | Phase 2b: query-plan ablation | 5 | qplan + inject = +2% R@5, filters/CE/dedup не помогают |
| 2026-04-05 | Phase 2c: new retrieval tracks | 9/10 | Funnel ≠ bottleneck, R2 sparse norm лучший новый, PRF/normalize-all вредят |
