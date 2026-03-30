# Retrieval Improvement Playbook

> Операционный документ. "Что попробовали, где мы сейчас, что дальше."
> Последнее обновление: 2026-03-30

---

## Текущее состояние

**Основная метрика** — manual judge (strict recall@5 ненадёжен для temporal/analytics запросов).

| Eval | Factual | Useful | KTA | Questions | Report |
|------|---------|--------|-----|-----------|--------|
| **Qwen3.5 + observability (consensus, 2026-03-30)** | **0.833** | **1.611** | **0.970** | 36 | [claude_judge_verdicts.md](../../results/eval_qwen35_langfuse/claude_judge_verdicts.md) |
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

1. **P1 fixes** — q33 monthly hot_topics + q36 channel_expertise routing
2. **P2 fix** — q21 deterministic out-of-range refusal
3. **Clean baseline rerun** — full 36 Qs after P1+P2, зафиксировать как canonical
4. **Data refresh** — re-ingest свежих постов, пересчёт weekly digests и channel profiles
5. **Eval expansion** — 36→100+ вопросов (decompose required_claims, add navigation/refusal)
6. **Unit tests cleanup** — удалить мёртвые тесты, покрыть analytics/state machine
7. **Ablation study** — ColBERT / reranker / RRF weights

### Backlog (исследовано, не реализовано)

| Technique | Expected impact | Effort | Reference |
|-----------|----------------|--------|-----------|
| NLI citation faithfulness | Hallucination / grounding metric | 1-2 days | R19 |
| NDR / RSR / ROR robustness | Production confidence in retrieval | 1 day + compute | R20 |
| CRAG-lite quality gate | Better fail-safe behavior | 2-3 days | R25 + CRAG |
| Prompt injection defense | Security hardening | 0.5 day | R25 |
| GPT-4o comparison | External baseline / credibility | 0.5 day | R25 |
| Health/readiness endpoints | Ops hygiene | 10-30 min | R25 |
| RAG necessity classifier | Latency / avoid unnecessary retrieval | 0.5 day | R21 |

Подробные описания техник: [experiment_history.md](experiment_history.md) → Tier 1/2/3 секции.
