# Retrieval Improvement Playbook

> Операционный документ. "Что попробовали, где мы сейчас, что дальше."
> Последнее обновление: 2026-03-25

---

## Текущее состояние

**Основная метрика** — manual judge (strict recall@5 ненадёжен для temporal/analytics запросов).

| Eval | Factual | Useful | KTA | Questions | Report |
|------|---------|--------|-----|-----------|--------|
| **Golden v1 + SPEC-RAG-15** | **1.79/2** | **1.72/2** | 0.926 | 30 | [eval_judge_20260325_spec15.md](../../results/reports/eval_judge_20260325_spec15.md) |
| Agent v1 (legacy) | — | — | — | 10, recall@5=0.76 | [details](experiment_history.md#agent-eval-v1) |
| Agent v2 (legacy) | — | — | — | 10, recall@5=0.685 | [details](experiment_history.md#agent-eval-v2) |
| Retrieval-only | — | — | — | 100, recall@5=0.73 | [details](experiment_history.md#retrieval-eval) |

**Текущий pipeline**: 13 tools, dynamic visibility (max 5), data-driven routing (`datasets/tool_keywords.json`).
**Analytics tools** (SPEC-RAG-15): entity_tracker (top/timeline/compare/co_occurrence) + arxiv_tracker (top/lookup). KTA=100%, Facet <100ms.

### Что реализовано

| Phase | Что | Spec | Результат |
|-------|-----|------|-----------|
| 3.1 | Payload enrichment (NER, arxiv, temporal) | SPEC-RAG-12 | 16 payload indexes, 91 entities |
| 3.2 | Simple tools (list_channels, related, compare, summarize) | SPEC-RAG-13 | 11 tools, dynamic visibility |
| 3.3 | Eval pipeline v2 | SPEC-RAG-14 | Golden dataset, tool tracking, failure attribution |
| 3.4 | Entity analytics tools | SPEC-RAG-15 | 13 tools, analytics state machine, data-driven routing |

### Remaining issues

| # | Проблема | Impact | Fix direction |
|---|----------|--------|---------------|
| 1 | q01: Qwen3 false refusal (refuses after seeing docs) | Factual 0/2 | LLM-level: prompt engineering или SFT. Forced search уже работает — LLM не использует результаты |
| 2 | q27: "С согласия NVIDIA" — phrasing | Useful 1/2 | Qwen3 generation quality, не tools |
| 3 | Strict recall@5 = 0.342 | Misleading metric | Analytics Qs не имеют source_post_ids. Manual judge = primary |
| 4 | Unit tests для analytics | Tech debt | entity normalization, arxiv dedup, state machine tests |

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

1. **Unit tests** для analytics state machine (entity normalization, arxiv dedup, visibility)
2. **q01 false refusal** — исследовать: prompt tuning, reasoning budget, альтернативный LLM
3. **Soft recall metric** — заменить strict document matching на semantic overlap

### Backlog (исследовано, не реализовано)

| Technique | Expected impact | Effort | Reference |
|-----------|----------------|--------|-----------|
| Contextual Retrieval | +10-20% recall | 3-5 days | Anthropic technique, R11 |
| Fine-tune Qwen3-Embedding | +5-15% recall | 3-7 days | Hard negatives from our data |
| CRAG (Corrective RAG) | +10-15% на сложных | 2-3 days | R11 §3 |
| BM25-based diversity | +2-4% recall | 1 day | R11 §2 |
| Reranker-as-fusion | Unknown | 1-2 days | R11 §2 |
| Hot topics (BERTopic) | New capability | 2-3 days | R17 T2 |
| Channel expertise | New capability | 1-2 days | R17 T5 |
| Arxiv full-text ingest | New capability | 2-3 days | SPEC-RAG-15 future ideas |

Подробные описания техник: [experiment_history.md](experiment_history.md) → Tier 1/2/3 секции.
