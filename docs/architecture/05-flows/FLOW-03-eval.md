## FLOW-03: Evaluation Pipeline V2 (SPEC-RAG-14)

### Problem
Количественное измерение качества агента: tool selection, answer quality, refusal behavior, retrieval grounding.
Evaluation должна быть воспроизводимой, сравнимой между версиями, и учитывать multi-criteria quality.

### Три режима evaluation

**1. Agent Eval V2** — полный pipeline через LLM (~30-40с/запрос):
```
python scripts/evaluate_agent.py \
  --dataset datasets/eval_golden_v1.json \
  --skip-judge \
  --api-key $TOKEN
```

**2. Agent Eval V2 + LLM Judge** — с Claude API judge:
```
EVAL_JUDGE_API_KEY=sk-ant-... python scripts/evaluate_agent.py \
  --dataset datasets/eval_golden_v1.json \
  --judge claude \
  --api-key $TOKEN
```

**3. Retrieval Eval** — прямые Qdrant queries, без LLM (~5с/запрос):
```
python scripts/evaluate_retrieval.py \
  --dataset datasets/eval_retrieval_100.json \
  --collection news_colbert_v2
```

### Agent Eval V2 Sequence

```
Operator → evaluate_agent.py(dataset, --judge, --api-key)
  ↓
Load dataset (auto-detect golden vs legacy format)
  ↓
For each question:
  POST /v1/agent/stream (SSE)
    Parse: step_started → visible_tools tracking
    Parse: tool_invoked → tools_invoked list
    Parse: citations → citation_hits, coverage
    Parse: final → answer text
  ↓
  Compute: recall@5 (fuzzy ±5/±50)
  Compute: key_tool_accuracy (binary whitelist vs forbidden)
  Compute: failure_type (tool_hidden/wrong/failed, retrieval_empty, generation_wrong, refusal_wrong)
  ↓
  If --judge claude:
    Claude API: factual_correctness (0.0/0.5/1.0)
    Claude API: usefulness (0/1/2)
  ↓
  POST /v1/qa (baseline comparison)
  ↓
Aggregate: recall, key_tool, factual, useful, coverage, latency, failure_breakdown, by_category
  ↓
Output: unified JSON (eval_metadata + aggregate + per_question) + Markdown report
```

### Метрики V2

| Метрика | Источник | Описание |
|---------|----------|---------|
| `recall@5` | Программный | Fuzzy match expected_documents vs citation_hits (±5/±50 по категории) |
| `key_tool_accuracy` | SSE tool_invoked | Binary: agent вызвал key_tool ∪ alternatives, не вызвал forbidden |
| `factual_correctness` | LLM Judge (Claude) | 0.0/0.5/1.0 — фактическая корректность vs expected_answer |
| `usefulness` | LLM Judge (Claude) | 0/1/2 — полезность ответа |
| `failure_type` | Программный | tool_hidden/tool_wrong/tool_failed/retrieval_empty/generation_wrong/refusal_wrong/judge_uncertain |
| `coverage` | SSE citations | Взвешенная сумма cosine-сигналов (0–1) |
| `latency` | Программный | Время от запроса до final события |

### Текущие результаты (2026-03-24)

| Dataset | Recall@5 | Key Tool | Coverage | Factual* | Useful* | Тип |
|---------|----------|----------|----------|----------|---------|-----|
| v1 (10 Qs) | **0.76** | — | 0.86 | — | — | Agent eval legacy |
| v2 (10 Qs) | **0.685** | — | 0.80 | — | — | Agent eval legacy |
| **golden_v1 (25 Qs)** | **~0.43** | **0.955** | 0.66 | **0.52** | **1.14/2** | Agent eval v2 |
| 100 Qs, RRF+ColBERT | **0.73** | — | — | — | — | Retrieval eval |

*Manual judge (консенсус Claude + Codex). Strict recall@5 на golden_v1 занижен — смешивает dataset strictness и real retrieval misses.

### Датасеты

| Файл | Формат | Вопросов | Описание |
|------|--------|----------|----------|
| `eval_golden_v1.json` | **Golden** | 25 | 6 categories, key_tools, forbidden_tools, calibration, future_baseline |
| `eval_dataset_quick.json` | Legacy | 10 | factual, temporal, channel, comparative, multi_hop, negative |
| `eval_dataset_quick_v2.json` | Legacy | 10 | entity, product, fact_check, cross_channel, recency |
| `eval_dataset_v3.json` | Legacy | 30 | temporal, channel, entity, broad, negative |
| `eval_retrieval_100.json` | Retrieval | 100 | Auto-generated, 35 каналов |

### Failure Attribution (P0-P1.5)

| Failure Type | Описание | Trigger |
|-------------|----------|---------|
| `tool_hidden` | Key tool не был в visible set | SSE step_started visible_tools |
| `tool_selected_wrong` | Agent вызвал не тот tool | key_tools/forbidden mismatch |
| `tool_execution_failed` | Runtime error / 400 в answer | Error markers в final answer |
| `retrieval_empty` | Поиск не вернул citations | citation_hits пуст |
| `generation_wrong` | Docs найдены, ответ плохой | factual < 0.5 |
| `refusal_wrong` | Должен отказать, но ответил | expected_refusal vs actual |

### Следующие шаги (Phase 3.4+)

- Audit zero-recall cases: разделить true miss / dataset too strict / alternative valid evidence
- Soft metric для compare/summarize (channel-level matching)
- Stochastic refusal hardening (q19/q20)
- Release-grade eval: 450-500 Qs, Qwen local judge, robustness (NDR/RSR/ROR), ablation
- Подробный blueprint: `docs/research/reports/R18-deep-evaluation-methodology-dataset.md`
