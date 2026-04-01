# SPEC-RAG-20d: Pipeline Cleanup — Context Budget, Coverage, Observability

> **Status:** In Progress (Phase 1-4 done, Phase 5 calibrating)
> **Created:** 2026-03-31
> **Parent:** SPEC-RAG-20
> **Context:** Audit выявил 30-47% потерю контекста, дублирование текстов в history, некалиброванный coverage, слабую observability. Codex audit нашёл 8 findings (P0-P2). Claude obs audit нашёл 15 findings.

---

## Выполненные исправления

### Phase 1: Context budget ✅
1. ✅ compose_context budget: 1200 → 4000 tokens
2. ✅ serialize_tool_payload: tool_name-aware, compose prompt НЕ обрезается (effective_limit=24000)
3. ✅ search hits в history: полные тексты стрипаются, остаётся id+score+snippet
4. ✅ Smoke test: prompt 10-15K chars доходит до LLM полностью

### Phase 2: Codex audit fixes ✅
5. ✅ trim_messages: atomic blocks (assistant+tool пары), pin compose_context
6. ✅ LLM 400 retry: сохраняет compose_context pair вместо слепой обрезки
7. ✅ Temporal guard: для всех tools с date_from/date_to (не только temporal_search)
8. ✅ Coverage: по фактически включённым docs после truncation
9. ✅ Lost-in-middle: отключён (docs уже reranked ColBERT)
10. ✅ k_per_query vs k_total: cap поднят до min(k*num_queries, 30)
11. ✅ QA fallback: использует agent context вместо legacy QAService
12. ⏳ fetch_docs chunk stitching: отложено (нужна спека на ingest pipeline)

### Phase 3: Observability ✅
13. ✅ Double JSON encoding fix в trace attributes
14. ✅ LLM spans: metadata (message_count, tool_calls, finish_reason) + token aggregation
15. ✅ Root trace: plan, route, strategy, tokens, citations_count, refinement_count
16. ✅ Tool spans: rich output (hits_count, coverage, prompt_len, answer_len) + error marking
17. ✅ search_execution span: strategy, routing, filters, candidates_total
18. ✅ compose_execution span: included/dropped docs, coverage, prompt_chars
19. ✅ LLM step names: phase-aware (pre_search/post_search/final)
20. ✅ ColBERT vs RRF-only distinction, reranker_type metadata
21. ✅ gpu_server.py: guard против пустых строк в tokenizer

### Phase 4: Coverage + Reranker redesign ✅
22. ✅ LANCER-style nugget coverage (services/agent/coverage.py)
    - query_plan subqueries = nuggets
    - Implicit nuggets из search subqueries (если query_plan не вызывался)
    - coverage = доля покрытых nuggets
    - threshold: 0.75 (3/4 nuggets)
23. ✅ Targeted refinement по uncovered nuggets (SEAL-RAG style)
    - Вместо repeat search(original_query) → search(missing_nuggets)
    - max_refinements: 2 → 1 (targeted достаточно одного)
24. ✅ Cross-encoder → CRAG-style confidence filter
    - НЕ reranking (ColBERT порядок сохраняется)
    - Фильтрация: docs с CE score < threshold отсекаются
    - filter_threshold: 0.1 (мягкий, pending калибровка)
25. ✅ Refinement plan-aware: использует plan subqueries + metadata_filters

### Phase 5: Калибровка (в процессе)
26. ✅ Retrieval calibration dataset: 100 queries из текущей коллекции (hand-crafted)
27. ✅ calibrate_coverage.py: recall@1-20, coverage, CE scores, monotonicity check
28. ✅ Baseline замеры:
    - Retrieval: r@1=0.80, r@3=0.97, r@5=0.97, r@20=0.98 (monotonic)
    - CE reranker поверх ColBERT: r@1=0.81(+0.01), r@3=0.94(-0.03) — marginal, иногда вредит
    - Pipeline v2 (RRF→CE→ColBERT): r@2=0.92(+0.02), остальное =. Не стоит усложнения.
    - Dense cosine (pplx-embed): median 0.47, range 0.07-0.86
    - Coverage (legacy cosine): median 0.69, 45% refinements при threshold 0.65
29. ⏳ CE score distribution для filter_threshold (прогон запущен)

---

## Текущий pipeline

```
Query → LLM step 1
  → query_plan (опционально, LLM решает) → subqueries (nuggets)
  → search (BM25 top-100 + Dense top-40 → weighted RRF 3:1 → ColBERT MaxSim → top-20)
  → rerank (CE confidence filter: score < threshold → отсечь, ColBERT порядок сохраняется)
  → compose_context (budget 4000 tokens, coverage по nuggets)
  → nugget coverage check:
      ≥ 0.75 → final_answer
      < 0.75 → targeted refinement (search по uncovered nuggets) → compose → final_answer
```

---

## Acceptance Criteria
- [x] Модель видит ≥90% текста найденных документов (prompt 10-15K chars)
- [x] Нет двойных копий документов в history (search texts stripped)
- [x] Нет лишних refinements (nugget coverage vs cosine: 0 refinements на simple queries)
- [x] Langfuse trace содержит plan, tokens, coverage, tool outputs
- [ ] CE filter_threshold откалиброван на real data
- [ ] Token usage на финальном шаге < 16000 (запас до 32768)
- [ ] fetch_docs chunk stitching (отдельная спека)
