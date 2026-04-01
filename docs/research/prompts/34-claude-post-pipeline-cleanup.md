# Prompt 34: Claude — Post Pipeline Cleanup Session

## Контекст

Сессия 2026-04-01 (SPEC-RAG-20d) — крупная переработка pipeline. 32 code changes, новые модули, новые скрипты. Всё задокументировано, eval прогнан, артефакты сохранены.

## Что было сделано

### Pipeline
- serialize_tool_payload: tool-aware (search texts stripped, compose prompt 24K limit)
- trim_messages: atomic blocks (assistant+tool пары), pin compose_context
- Temporal guard для всех tools с dates, не только temporal_search
- k_total = k × num_queries (до 30 кандидатов для reranker)
- Lost-in-middle отключён (docs уже reranked ColBERT)
- QA fallback: agent context вместо legacy pipeline

### Coverage (DEC-0044)
- LANCER nugget coverage в `src/services/agent/coverage.py`
- query_plan subqueries = nuggets
- Implicit nuggets из search subqueries если plan не вызывался (state.py)
- Threshold 0.75, max_refinements 1
- Targeted refinement по uncovered nuggets (SEAL-RAG style)
- Результат: latency -40-65%, 0 лишних refinements на простых вопросах

### Reranker (DEC-0045)
- Cross-encoder НЕ ранжирует — ColBERT порядок сохраняется (state.py переписан)
- CE как CRAG-style confidence filter: score < 0.0 → отсечь
- Calibration: relevant median CE=8.35, irrelevant median=-1.11
- filter_threshold=0.0 (logit boundary), keep 92% relevant

### Observability (15 findings)
- Root trace: plan, tokens, strategy, coverage, citations_count, refinement_count
- Tool spans: rich output (hits_count, coverage, prompt_len) + error marking (level=ERROR)
- search_execution + compose_execution spans
- LLM step names phase-aware (pre_search/post_search/final)
- Token aggregation в RequestContext → root trace
- gpu_server: empty text guard для tokenizer

### Retrieval Calibration
- 100 hand-crafted queries: `datasets/eval_retrieval_calibration.json`
- `scripts/calibrate_coverage.py`: recall@1-20, CE scores, pipeline v2, monotonicity
- Baseline: r@1=0.80, r@3=0.97, r@20=0.98 (monotonic OK)
- CE reranking degrades r@3 (0.97→0.94) → replaced with filter
- Pipeline v2 (RRF→CE→ColBERT): +0.02 r@2, not worth complexity

## Текущие метрики (36 Qs golden_v2, 2026-04-01)
- Factual: **0.847** (was ~0.80)
- Useful: **1.861** (was ~1.53)
- KTA: **1.000**
- Strict recall: **0.637** (was 0.461)
- Latency: **23.6s** (was 26.4s)
- 0 agent errors

## Баги пофикшены в сессии
- `_perform_refinement`: UnboundLocalError `plan` — переменная определялась только в else ветке
- `_colbert_url` AttributeError в hybrid_retriever — несуществующий атрибут в span output

## Что делать дальше (приоритеты)

### P1: Re-judge с полным артефактом
- `results/reports/full_judge_artifact_20260401.md` — 788 строк, query+expected+answer+citations
- Пользователь хочет чтобы judge смотрел на цитаты (retrieved documents), не только answer vs expected
- Это даст evidence_support score помимо factual/useful

### P1: Data refresh
- Re-ingest свежих постов (2026-03-18 → current)
- Channel profiles re-compute (techsparks не попадает в робототехнику — q24/q36 промах)
- Weekly digests re-compute

### P1: Bugs
- q27 пустой ответ (entity_tracker co_occurrence mode)
- fetch_docs chunk stitching (длинные посты >1500 chars) — нужна спека

### P2: Quality
- Fine-tune CE reranker на domain data
- NLI citation faithfulness (R19 ready)

### НЕ делать
- CRAG full — nugget coverage + CE filter покрывают
- Pipeline v2 (RRF→CE→ColBERT) — marginal
- Codex as judge — необъективен (q19/q20 refusals, числовые расхождения)

## Ключевые файлы

### Код (изменённые)
- `src/services/agent/coverage.py` — НОВЫЙ, nugget coverage
- `src/services/agent/state.py` — uncovered_nuggets, implicit nuggets, CE ColBERT order
- `src/services/agent_service.py` — nugget integration, root trace enrichment, targeted refinement, plan fix
- `src/services/agent/formatting.py` — serialize tool-aware, trim atomic blocks
- `src/services/agent/executor.py` — temporal guard all tools, CE filter threshold
- `src/services/tools/rerank.py` — CRAG filter mode
- `src/services/tools/compose_context.py` — coverage по included, lost-in-middle off
- `src/services/tools/search.py` — k_total, search_execution span
- `src/services/tools/tool_runner.py` — rich output, error marking
- `src/core/observability.py` — double encoding fix
- `src/core/settings.py` — threshold 0.75, max_refinements 1

### Скрипты
- `scripts/calibrate_coverage.py` — recall@1-20, CE scores, pipeline v2
- `scripts/generate_retrieval_dataset.py` — dataset generator из Qdrant+LLM

### Документация
- `docs/specifications/active/SPEC-RAG-20d-pipeline-cleanup.md` — полный статус
- `docs/architecture/11-decisions/decision-log.md` — DEC-0044, DEC-0045
- `docs/planning/retrieval_improvement_playbook.md` — эксперименты + метрики
- `docs/planning/project_scope.md` — Phase 3.5

### Артефакты eval
- `results/raw/eval_results_20260401-014638.json` — raw + judge scores
- `results/raw/claude_judge_20260401.json` — corrected judge verdicts
- `results/reports/full_judge_artifact_20260401.md` — полный review артефакт (788 строк)
- `results/raw/calibration_20260331-134336.json` — retrieval calibration
- `datasets/eval_retrieval_calibration.json` — 100 hand-crafted queries
