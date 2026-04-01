# Retrieval Improvement Playbook

> Операционный документ. "Что попробовали, где мы сейчас, что дальше."
> Последнее обновление: 2026-04-01

---

## Текущее состояние

**Основная метрика** — manual judge (strict recall@5 ненадёжен для temporal/analytics запросов).

| Eval | Factual | Useful | KTA | Faithfulness | Questions | Report |
|------|---------|--------|-----|-------------|-----------|--------|
| **SPEC-RAG-21 NLI + granular judge (2026-04-01)** | **0.842** | **1.778** | **1.000** | **0.91** (corrected) | 36 | [nli_faithfulness_analysis](../../results/reports/nli_faithfulness_analysis_20260401.md) |
| SPEC-RAG-20d + q27 fix (Claude judge, 2026-04-01) | 0.875 | 1.917 | 1.000 | — | 36 | [claude_judge_20260401.json](../../results/raw/claude_judge_20260401.json) |
| Qwen3.5 + observability (consensus, 2026-03-30) | 0.833 | 1.611 | 0.970 | 36 | [claude_judge_verdicts.md](../../results/eval_qwen35_langfuse/claude_judge_verdicts.md) |
| Golden v2 baseline (Qwen3, SPEC-RAG-18) | ~0.80 | ~1.53 | 1.000 | 36 | [R26-golden-v2-eval-baseline.md](../research/reports/R26-golden-v2-eval-baseline.md) |
| Golden v1 + SPEC-RAG-15 | 1.79/2 | 1.72/2 | 0.926 | 30 | [eval_judge_20260325_spec15.md](../../results/reports/eval_judge_20260325_spec15.md) |
| Agent v1 (legacy) | — | — | — | 10, recall@5=0.76 | [details](experiment_history.md#agent-eval-v1) |
| Agent v2 (legacy) | — | — | — | 10, recall@5=0.685 | [details](experiment_history.md#agent-eval-v2) |
| Retrieval-only | — | — | — | 100, recall@5=0.73 | [details](experiment_history.md#retrieval-eval) |

**Текущий pipeline**: 15 tools, dynamic visibility (max 5), data-driven routing (`datasets/tool_keywords.json`).
**Analytics / precomputed tools**: entity_tracker, arxiv_tracker, hot_topics, channel_expertise. Analytics short-circuit включён, cron-backed tools работают через auxiliary collections.
**Eval pipeline**: SPEC-RAG-18 golden_v2 (36 Qs, 4 eval_modes, offline judge workflow, batch review). Consensus judge: Claude Opus 4.6 + Codex GPT-5.4.

### Что реализовано

| Phase | Что | Spec | Результат |
|-------|-----|------|-----------|
| 3.1 | Payload enrichment (NER, arxiv, temporal) | SPEC-RAG-12 | 16 payload indexes, 91 entities |
| 3.2 | Simple tools (list_channels, related, compare, summarize) | SPEC-RAG-13 | 11 tools, dynamic visibility |
| 3.3 | Eval pipeline v2 | SPEC-RAG-14 | Golden dataset, tool tracking, failure attribution |
| 3.4 | Entity analytics tools | SPEC-RAG-15 | 13 tools, analytics state machine, data-driven routing |
| 3.4 | Precomputed analytics tools | SPEC-RAG-16 | 15 tools, weekly digests, channel profiles, BERTopic cron pipeline |
| 3.4 | Production hardening | SPEC-RAG-17 | Request isolation, auth hardening, cooperative deadline, rate limiter fix |
| 3.5 | Pipeline cleanup + observability | SPEC-RAG-20d | Serialize fix, trim atomic blocks, Langfuse enrichment, 15 obs findings fixed |
| 3.5 | LANCER nugget coverage | SPEC-RAG-20d/DEC-0044 | query_plan subqueries = nuggets, targeted refinement, 45% latency reduction |
| 3.5 | CE confidence filter (CRAG-style) | SPEC-RAG-20d/DEC-0045 | Cross-encoder фильтрует мусор, ColBERT порядок сохраняется |
| 3.5 | Retrieval calibration | — | 100-query dataset, recall@1-20, CE score distribution, pipeline v2 A/B test |
| 3.6 | NLI citation faithfulness | SPEC-RAG-21 | ruBERT NLI, faithfulness 0.91 corrected, 0 real hallucinations. [Analysis](../../results/reports/nli_faithfulness_analysis_20260401.md) |

### Remaining issues

| # | Проблема | Impact | Fix direction |
|---|----------|--------|---------------|
| 1 | **q33**: monthly hot_topics path даёт fallback на одну неделю вместо month aggregation | Quality | Debug monthly digest end-to-end |
| 2 | **q36**: channel_expertise routing miss — LLM выбирает list_channels вместо channel_expertise | Routing | Keyword/description fix + убрать list_channels из acceptable_alternatives |
| 3 | **q21**: out-of-timerange refusal не срабатывает — агент генерирует нерелевантный ответ | Refusal | Deterministic temporal guard |
| 4 | **q01**: Qwen3 false refusal → ungrounded direct answer (coverage=0, нет citations) | LLM limitation | Known issue, не чиним сейчас |
| 5 | Stale tests | Proof layer gap | Удалить dead tests, покрыть analytics/state machine |
| 6 | Данные устарели / weekly digests частичные | Demo reliability | Re-ingest + full weekly digests + profile re-compute |
| 7 | required_claims не decomposed (q01-q25) | Eval quality | Декомпозировать при expansion до 100 Qs |

---

## Что пробовали (summary)

Полная история с per-question таблицами: [experiment_history.md](experiment_history.md)

### Pipeline evolution (24 experiments)

```
0.00 → 0.15 → 0.33 → 0.59 → 0.70 → 0.76 (v1) / 0.685 (v2)
```

| Milestone | Change | Recall impact | Date |
|-----------|--------|---------------|------|
| Weighted RRF 3:1 | BM25 доминирует, dense добавляет diversity | 0.33→0.59 (+79%) | 2026-03-19 |
| Forced search | LLM не вызывал tools → принудительный search | 0.59→0.70 (+19%) | 2026-03-19 |
| ColBERT rerank | Per-token MaxSim, устраняет attractor docs | Recall@1 0.36→0.71 (+97%) | 2026-03-20 |
| Multi-query search | Round-robin merge sub-queries | v2: 0.46→0.61 (+33%) | 2026-03-20 |
| LLM tool selection | Dynamic visibility, temporal/channel routing | v2: 0.61→0.685 (+12%) | 2026-03-21 |
| SPEC-RAG-15 analytics | entity_tracker + arxiv_tracker | Factual: 0.52→1.79 | 2026-03-25 |

### Что НЕ сработало

| Technique | Result | Why |
|-----------|--------|-----|
| Cosine MMR | recall 0.70→0.11 | Re-promotes attractor documents |
| Dense re-score after RRF | recall 0.33→0.15 | Erases BM25 contribution |
| PCA whitening 1024→512 | recall 0.70→0.56 | Too aggressive dimensionality cut |
| Whitening 1024→1024 | parity | Dense isn't bottleneck at BM25 3:1 |
| DBSF fusion | 0.72 vs RRF 0.73 | RRF slightly better |

### Эксперименты 2026-03-31 — 2026-04-01 (SPEC-RAG-20d сессия)

**Retrieval calibration** (100 hand-crafted queries, news_colbert_v2, 13777 docs):

| Метрика | Результат |
|---------|-----------|
| Recall@1 | 0.800 (80 full, 20 zero) |
| Recall@3 | 0.970 (97 full, 3 zero) |
| Recall@5 | 0.970 |
| Recall@20 | 0.980 (2 queries not found at all) |
| Monotonicity | OK (recall never drops) |
| Dense cosine (pplx-embed) | median 0.47, range 0.07-0.86 |
| ColBERT MaxSim score | median 10.0, range 4.7-17.3 |

**Cross-encoder (Qwen3-Reranker-0.6B) как reranker поверх ColBERT:**

| Метрика | Retrieval | + CE rerank | Delta |
|---------|-----------|-------------|-------|
| r@1 | 0.800 | 0.810 | +0.01 |
| r@3 | 0.970 | 0.940 | **-0.03** |
| r@5 | 0.970 | 0.960 | -0.01 |
| Детали | — | 9 improved, 3 degraded, 88 unchanged | net +6 |

**Вывод:** CE reranking marginal и портит r@3 в 3% случаев. Решение: CE как filter (DEC-0045).

**Pipeline v2 (RRF(60)→CE(40)→ColBERT(20)) vs v1:**

| Метрика | v1 | v2 | Delta |
|---------|----|----|-------|
| r@1 | 0.790 | 0.790 | = |
| r@2 | 0.900 | 0.920 | +0.02 |
| r@3+ | = | = | = |

**Вывод:** Minimal improvement (+0.02 r@2), не стоит усложнения pipeline.

**CE score distribution (2000 docs, relevant vs irrelevant):**

| | Relevant (n=143) | Irrelevant (n=1857) |
|---|---|---|
| median | **8.35** | **-1.11** |
| min | -7.20 | -12.33 |
| max | 10.78 | 10.68 |

При filter_threshold=0.0: keep 92% relevant, remove 55% irrelevant.

**Coverage cosine-based (legacy) vs nugget-based (LANCER):**

| Метрика | Cosine-based | Nugget-based |
|---------|-------------|--------------|
| Coverage median | 0.69 | 1.0 (simple) / 0.8 (complex) |
| Refinements при threshold | 45% (threshold 0.65) | ~5% (threshold 0.75) |
| Latency impact | +15-20s per refinement | eliminated for most queries |

**Smoke tests после всех изменений:**

| Query | Latency before | Latency after | Refinements |
|-------|---------------|---------------|-------------|
| q01 (entity) | 55s | 33s | 0 (was 1-2) |
| q02 (product) | 76s | 28s | 0 (was 2) |
| q03 (fact) | 41-62s | 14.5s | 0 (was 1-2) |
| q04 (cross-channel) | 60-75s | 49s | 0 (was 1) |

### Корневые проблемы

| # | Проблема | Статус |
|---|----------|--------|
| 1 | Embedding anisotropy (cosine 0.78-0.83) | Решено: ColBERT per-token MaxSim |
| 2 | Attractor documents | Решено: Weighted RRF 3:1 + ColBERT + channel dedup |
| 3 | Single-query search bug | Решено: Multi-query + round-robin merge |
| 4 | System prompt хардкодил tool name | Решено: Data-driven routing |
| 5 | Strict doc matching = ложная метрика | Решено: Manual judge = primary metric |
| 6 | LLM не знает entity names (Vera Rubin) | Открыто: prompt engineering |
| 7 | Pre-search false refusal (Qwen3) | Частично: forced search работает, LLM игнорирует результаты |

---

## Что дальше

### Ближайшие приоритеты

1. **CE filter_threshold калибровка** — поставить 0.0 (logit boundary), smoke test
2. **Full eval 36 Qs** — baseline после всех pipeline changes
3. **fetch_docs chunk stitching** — длинные посты (>1500 chars) не собираются обратно
4. **P1 fixes** — q33 monthly hot_topics + q36 channel_expertise routing
5. **Data refresh** — re-ingest свежих постов, пересчёт weekly digests
6. **Unit tests cleanup** — удалить мёртвые тесты, покрыть coverage/state machine

### Backlog (исследовано, не реализовано)

| Technique | Expected impact | Effort | Reference |
|-----------|----------------|--------|-----------|
| Fine-tune CE reranker | Reduce 3% degradation on Russian AI/ML domain | 1-2 days + data | DEC-0045, calibration data |
| NLI citation faithfulness | Hallucination / grounding metric | 1-2 days | R19 |
| NDR / RSR / ROR robustness | Production confidence in retrieval | 1 day + compute | R20 |
| Prompt injection defense | Security hardening | 0.5 day | R25 |
| RAG necessity classifier | Latency / avoid unnecessary retrieval | 0.5 day | R21 |
| ColBERT as independent retrieval path | Bypass BM25+Dense failures | Blocked: latency on 13K docs | — |
| CRAG full implementation | T5-large relevance evaluator | Low priority — nugget coverage + CE filter cover same use cases | DEC-0045 |

### Abandoned / superseded

| Technique | Why abandoned | Date |
|-----------|---------------|------|
| CE reranking after ColBERT | r@3 degrades 0.97→0.94, marginal r@1 gain. Replaced with CE filter | 2026-04-01 |
| Pipeline v2 (RRF→CE→ColBERT) | Only +0.02 r@2, not worth complexity | 2026-04-01 |
| Cosine-based coverage (DEC-0018) | Not calibrated for pplx-embed, 45% false refinements | 2026-04-01 |
| Lost-in-middle mitigation | Docs already reranked by ColBERT, reorder hurts citation consistency | 2026-03-31 |
| GPT-4o comparison | Deprioritized — focus on pipeline quality first | 2026-03-31 |

Подробные описания техник: [experiment_history.md](experiment_history.md) → Tier 1/2/3 секции.
