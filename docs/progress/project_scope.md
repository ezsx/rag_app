# Project Scope: rag_app

> Живой документ. История развития проекта и предстоящие задачи.
> Последнее обновление: 2026-04-10

---

## Что это

RAG-платформа с агентским ReAct-пайплайном над курированной базой из 36 русскоязычных AI/ML Telegram-каналов. Self-hosted inference на двух GPU (V100 SXM2 32GB + RTX 5060 Ti 16GB).

Закрытая база знаний (не web search) — паттерн enterprise RAG: grounding, traceability, audit trail. Telegram-каналы = аналог корпоративных источников информации.

---

## Текущее состояние (2026-04-10)

### Прогрессия agent-level метрик

| Этап | Factual | Useful | KTA | Faithfulness | Qs | Judge | Дата | Артефакт |
|------|---------|--------|-----|--------------|----|-------|------|----------|
| Golden v1 + SPEC-RAG-15 | 1.79/2 | 1.72/2 | 0.926 | — | 30 | Consensus Claude+Codex | 2026-03-25 | [eval](../../experiments/legacy/agent_eval/eval_results_20260325-192924.json), [judge](../../experiments/legacy/reports/eval_judge_20260325_spec15.md) |
| Golden v2 baseline (SPEC-RAG-18) | ~0.80 | ~1.53 | 1.000 | — | 36 | Consensus Claude+Codex | 2026-03-30 | [eval](../../experiments/legacy/agent_eval/eval_results_20260330-035118.json), [judge](../../experiments/legacy/claude_judge_golden_v2.md) |
| Qwen3.5 + Langfuse (SPEC-RAG-19) | 0.833 | 1.611 | 0.970 | — | 36 | Consensus | 2026-03-30 | [verdicts](../../experiments/legacy/agent_eval/) |
| SPEC-RAG-20d pipeline cleanup | 0.847 | 1.861 | 1.000 | — | 36 | Claude judge | 2026-04-01 | [eval](../../experiments/legacy/agent_eval/eval_results_20260401-014638.json), [report](../../experiments/legacy/reports/eval_report_20260401-014638.md) |
| + q27 SecurityManager fix | **0.875** | **1.917** | 1.000 | — | 36 | Claude judge | 2026-04-01 | [judge](../../experiments/legacy/agent_eval/claude_judge_20260401.json) |
| + NLI faithfulness (SPEC-RAG-21) | 0.842 | 1.778 | 1.000 | **0.91** | 36 | Claude + ruBERT NLI | 2026-04-01 | [eval](../../experiments/legacy/agent_eval/eval_results_20260401-091242.json), [nli](../../experiments/legacy/agent_eval/nli_scores_20260401_full.json), [analysis](../../experiments/legacy/reports/nli_faithfulness_analysis_20260401.md) |
| **Benchmark SPEC-RAG-29** | **0.84** | **1.77** | — | — | 17 | Claude Opus 4.6 | 2026-04-03 | [scores](../../experiments/legacy/benchmarks/judge_scores.md), [artifact](../../experiments/legacy/benchmarks/judge_artifact.json) |
| **RUN-008 corrected baseline** | **0.858** | **1.708** | **1.000** | **0.91** | 36 | Claude Opus (published) + manual calibration | 2026-04-08/10 | [results](../../experiments/runs/RUN-008/results.yaml), [`compute_confidence.py`](../../scripts/compute_confidence.py) |

Bootstrap CI для текущего baseline (`RUN-008`, 10K resamples): factual **0.858** with 95% CI **[0.792, 0.917]**, useful **1.708** with 95% CI **[1.606, 1.803]**. Retrieval factual: **0.888** [0.782, 0.965] on 17 retrieval questions. Analytics factual: **0.793** [0.679, 0.893] on 14 analytics questions.

Интервалы пока широкие из-за малого `n=36`. Golden v3 dataset подготовлен для следующего eval: `datasets/golden_v3/eval_golden_v3.json` — 120 reviewed questions (60 strict-anchor retrieval items, 32 analytics, 8 navigation, 15 refusal/adversarial, 7 edge/tool-boundary cases; categories overlap by eval purpose). Метрики v3 ещё не прогонялись; текущие portfolio-числа остаются RUN-008 на `eval_golden_v2_fixed.json`.

### Robustness (SPEC-RAG-23, bypass pipeline)

| Метрика | BERTScore (proxy) | **Claude Judge (final)** | Δ |
|---------|-------------------|--------------------------|---|
| **NDR** | 0.818 | **0.963** (26/27) | BERTScore занижал на 0.145 |
| **RSR** | 0.706 | **0.941** (16/17) | Ложные violations k=10→20 |
| **ROR** | 0.974 | **0.959** | Примерно корректен |
| **Composite** | 0.826 | **0.954** | |

**Вывод**: BERTScore F1 как proxy для robustness **провалился**. Semantic similarity не ловит factual correctness — "уверенный отказ" семантически похож на expected answer. Claude judge обязателен для финальных чисел.

Retrieval критически важен: avg factual k=0 = 0.10, k=20 = 0.63 (delta **+0.53**).

Raw: [ndr_rsr_ror_raw](../../experiments/legacy/robustness/ndr_rsr_ror_raw_20260402-082135.json), [report](../../experiments/legacy/robustness/ndr_rsr_ror_report_20260402-082135.md), [judge scores](../../experiments/legacy/robustness/judge_ndr_rsr_ror_final.json).

### Retrieval-level метрики

| Pipeline | R@1 | R@5 | R@20 | MRR | Датасет | Дата |
|----------|-----|-----|------|-----|---------|------|
| BM25+Dense → RRF 3:1 → ColBERT | **0.80** | **0.97** | **0.98** | — | 100 hand-crafted Qs | 2026-03-31 | [calibration](../../datasets/eval_retrieval_calibration.json) |
| + CE reranking after ColBERT | 0.81 | 0.94 | 0.98 | — | 100 hand-crafted Qs | 2026-03-31 | — |
| Benchmark: custom | 0.94 | 0.95 | 0.95 | 0.944 | 100 auto-generated Qs | 2026-04-03 | [results](../../experiments/legacy/benchmarks/retrieval_auto_generated.json) |
| Benchmark: custom | 0.78 | 0.97 | 0.98 | 0.866 | 100 hand-crafted Qs | 2026-04-03 | [results](../../experiments/legacy/benchmarks/retrieval_calibration.json) |

CE reranking **degrades** R@3 (0.97→0.94) → заменён на confidence filter ([DEC-0045](../architecture/11-decisions/decision-log.md)). CE score: relevant median=8.35, irrelevant median=-1.11.

### Автоматические proxy-метрики (evaluate_agent.py, без LLM judge)

Замеряются при каждом прогоне. Исследованы в SPEC-RAG-22 (R26). Скрипт: `scripts/evaluate_agent.py`.

| Группа | Метрика | Что измеряет |
|--------|---------|--------------|
| **Primary** | key_tool_accuracy | Правильный ли tool вызван (binary whitelist) |
| | tool_call_f1 | F1 между вызванными и ожидаемыми tools |
| **Retrieval IR** | precision@5 | Доля релевантных в top-5 citations |
| | MRR | Mean Reciprocal Rank первого правильного документа |
| | nDCG@5 | Normalized Discounted Cumulative Gain |
| | BERTScore F1 | Semantic similarity ответа с expected (ruBERT-large) |
| | SummaC faithfulness | Sentence-level NLI без claim decomposition (SPEC-RAG-22 §1.3) |
| **Grounding** | acceptable_set_hit | Найден ли хотя бы один документ из acceptable evidence sets |
| | retrieval_sufficiency | Достаточно ли документов для ответа (offline judge) |
| | evidence_support | Подтверждён ли ответ документами (offline judge) |
| **Diagnostic** | strict_anchor_recall | Exact match по expected document IDs |
| | coverage | LANCER-inspired lexical nugget coverage (query_plan subqueries) |
| | latency (agent / baseline) | Время полного pipeline + p95 |

Метрики retrieval_sufficiency и evidence_support требуют offline judge (Claude/Codex batch review). IR метрики (precision@5, MRR, nDCG@5, BERTScore, SummaC) добавлены в SPEC-RAG-22, доступны с 2026-04-01.

**Последний полный agent прогон** (36 Qs, 2026-04-01, [eval_results_20260401-091242](../../experiments/legacy/agent_eval/eval_results_20260401-091242.json), [report](../../experiments/legacy/reports/eval_report_20260401-091242.json)):

| Метрика | Значение | Scope |
|---------|----------|-------|
| key_tool_accuracy | **1.000** | 33 golden Qs |
| acceptable_set_hit | **0.471** | 17 retrieval Qs |
| strict_anchor_recall | **0.588** (9 full, 6 zero) | 17 retrieval Qs |
| coverage (LANCER) | **0.414** | 36 Qs |
| latency agent | **24.4s** mean, **65.6s** p95 | 36 Qs |
| failure_breakdown | refusal_wrong: 2 | 36 Qs |

**NLI faithfulness** (полный прогон, 36 Qs, [nli_scores_20260401_full](../../experiments/legacy/agent_eval/nli_scores_20260401_full.json)):

| Метрика | Значение |
|---------|----------|
| faithfulness (raw, lenient) | 0.792 |
| faithfulness (raw, strict) | 0.753 |
| **faithfulness (corrected)** | **~0.91** |
| citation_precision | 0.509 |
| claims verifiable | 171 |
| claims supported | 133 (78%) |
| contradictions (raw) | 19 |
| **contradictions (real)** | **0** |
| NLI pairs processed | 1 977 |

19 raw contradictions проверены вручную: 12 false positives (ruBERT fails на русских парафразах), 5 wrong-doc matches, 2 borderline. 0 реальных hallucinations. [Полный анализ](../../experiments/legacy/reports/nli_faithfulness_analysis_20260401.md)

**BERTScore proxy (robustness прогон, 151 ответов, ruBERT-large layer 18):**

Использован как scoring function для NDR/RSR/ROR. Результат: semantic similarity **не ловит factual correctness** — BERTScore занижал NDR на 0.145, показывал ложные RSR violations. Подробные числа в таблице robustness выше.

**По категориям (latency):**

| Категория | Qs | Latency (mean) |
|-----------|---:|-------:|
| analytics_channel_expertise | 3 | 11.5s |
| analytics_hot_topics | 3 | 15.2s |
| future_baseline | 8 | 10.2s |
| navigation | 2 | 11.7s |
| negative_refusal | 3 | 23.7s |
| constrained_search | 7 | 29.7s |
| broad_search | 6 | 36.6s |
| compare_summarize | 4 | 48.4s |

### Custom vs LlamaIndex benchmark (SPEC-RAG-29)

4 pipeline × 17 retrieval Qs, judge: Claude Opus 4.6.

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive (dense-only) | 0.55 | 1.04 | 0.28 |
| LI-stock (default hybrid) | 0.51 | 1.13 | 0.46 |
| LI-maxed (weighted RRF + CE) | 0.54 | 1.21 | 0.48 |
| **custom** | **0.84** | **1.77** | **0.88** |

Delta custom vs best-of-three: **+0.30 factual, +0.56 usefulness, +0.40 grounding**.
Main gain: multi-query planning + LANCER coverage + specialized tools, не reranker.

### Стек

| Компонент | Модель / Технология | Где |
|-----------|--------------------|----|
| LLM | Qwen3.5-35B-A3B Q4_K_M | V100 SXM2, llama-server.exe, порт 8080 |
| Embedding | pplx-embed-v1-0.6B (bf16, mean pooling) | RTX 5060 Ti, gpu_server.py, порт 8082 |
| Reranker | Qwen3-Reranker-0.6B-seq-cls | RTX 5060 Ti, gpu_server.py |
| ColBERT | jina-colbert-v2 (560M, 128-dim MaxSim) | RTX 5060 Ti, gpu_server.py |
| NLI | rubert-base-cased-nli-threeway (180M) | RTX 5060 Ti, gpu_server.py |
| Vector store | Qdrant (dense 1024 + sparse BM25 + ColBERT 128) | Docker CPU |
| Fusion | Weighted RRF (BM25 3:1) → ColBERT rerank → CE filter | — |
| Observability | Langfuse v3 self-hosted (7 instrumentation points) | Docker CPU |
| Agent | 15 tools, dynamic visibility (max 5), LANCER coverage | — |

---

## Коллекция данных

36 каналов отобраны из 70+ кандидатов через Deep Research ([R09](../research/reports/R09-telegram-channels-collection.md)).
Критерии: авторитетность автора, активность 2025–2026, оригинальный контент, фактологическая плотность.

| Параметр | Значение |
|----------|----------|
| Каналов | 36 |
| Тематических областей | 10 (LLM, research, applied ML, NLP, CV, DS, MLOps, open-source, индустрия, этика) |
| Период ingest | 2025-07-01 → 2026-03-18 |
| Points в Qdrant | 13 088 (`news_colbert_v2`) |
| Auxiliary коллекции | `weekly_digests`, `channel_profiles` |
| Валидация | `scripts/validate_channels.py` — 36/37 web-preview, 37/37 Telethon API |

---

## История развития

### Phase 1–2: Инфраструктура и базовый RAG

Docker compose (API + Qdrant). Self-hosted LLM на V100, embedding + reranker на RTX 5060 Ti через gpu_server.py. Telegram ingestion. FastAPI + SSE streaming + JWT auth + Web UI.

Qdrant с тремя named vectors (dense + sparse + ColBERT). Two-tier chunking. UUID5 deterministic IDs.

Native function calling agent (не regex ReAct). Forced search. Channel dedup. Context overflow protection.

### Phase 3.0: Расширение коллекции [ЗАВЕРШЕНО]

Deep Research → 32 новых канала. Валидация через Telethon API. Ingest 36 каналов, 13K+ points.

### Phase 3.1: Eval + Pipeline Optimization [ЗАВЕРШЕНО]

**Recall@5: 0.15 → 0.76** через 22 итерации с evidence.

| Milestone | Impact | Дата |
|-----------|--------|------|
| Убрали dense re-score | 0.15 → 0.33 | 2026-03-19 |
| Original query injection + Weighted RRF 3:1 | 0.33 → 0.70 | 2026-03-19 |
| ColBERT reranking (jina-colbert-v2) | R@1 +97% (retrieval eval) | 2026-03-20 |
| Multi-query search fix (critical bug) | v2: 0.46 → 0.61 (+33%) | 2026-03-20 |
| LLM tool selection + dynamic visibility | v2: 0.61 → 0.685 (+12%) | 2026-03-21 |

Протестированы и отклонены с evidence: Cosine MMR (0.70→0.11), PCA whitening (0.70→0.56), DBSF (паритет с RRF), dense re-score (убивает BM25).

Полная хронология: [experiment_log.md](experiment_log.md)

### Phase 3.2: Adaptive Retrieval + Tool Router [ЗАВЕРШЕНО]

> [SPEC-RAG-11](../specifications/completed/SPEC-RAG-11-adaptive-retrieval.md)/[13](../specifications/completed/SPEC-RAG-13-simple-tools.md). Research: [R13](../research/reports/R13-deep-tool-router-architecture.md), [R14](../research/reports/R14-deep-beyond-frameworks-techniques.md).

15 tools с phase-based dynamic visibility (max 5 одновременно). Signal + keyword routing из `datasets/tool_keywords.json`.

Phases: PRE-SEARCH → POST-SEARCH → NAV-COMPLETE → ANALYTICS-COMPLETE.
Navigation short-circuit, analytics short-circuit, refusal policy с temporal guard.

### Phase 3.3: Evaluation Pipeline V2 [ЗАВЕРШЕНО]

> [SPEC-RAG-14](../specifications/completed/SPEC-RAG-14-evaluation-pipeline.md). Research: [R18](../research/reports/R18-deep-evaluation-methodology-dataset.md).

Golden dataset v1 (25 Qs) → golden dataset v2 (36 Qs: 18 retrieval, 13 analytics, 2 navigation, 3 refusal) → golden dataset v3 (120 reviewed Qs, 2026-04-10, no judge run yet).

SSE tool tracking, key tool accuracy, failure attribution, LLM judge (Claude API), offline judge workflow (batch export для consensus review).

Метрика redesign: strict recall → diagnostic only. Primary = factual + usefulness + KTA.

### Phase 3.4: Tool Expansion + Entity Analytics + Hardening [ЗАВЕРШЕНО]

> [SPEC-RAG-12](../specifications/completed/SPEC-RAG-12-payload-enrichment.md)/[15](../specifications/completed/SPEC-RAG-15-entity-analytics-tools.md)/[16](../specifications/completed/SPEC-RAG-16-hot-topics-channel-expertise.md)/[17](../specifications/completed/SPEC-RAG-17-production-hardening.md).

**Payload enrichment** ([SPEC-RAG-12](../specifications/completed/SPEC-RAG-12-payload-enrichment.md)): entities[], arxiv_ids[], urls[], lang, year_week + 16 payload indexes. 95 AI/ML entities в словаре.

**Analytics tools** (SPEC-RAG-15): `entity_tracker` (Facet API, 4 режима) + `arxiv_tracker`. Analytics short-circuit.

**Precomputed tools** (SPEC-RAG-16): `hot_topics` (BERTopic weekly digests → `weekly_digests` коллекция) + `channel_expertise` (monthly profiles → `channel_profiles` коллекция).

**Production hardening** (SPEC-RAG-17): RequestContext + ContextVar isolation, cooperative deadline 90s, rate limiter, CORS, auth hardening.

**Golden v2 baseline** (SPEC-RAG-18): 36 Qs, consensus Claude + Codex → factual ~0.80, useful ~1.53/2, KTA 1.000.

### Phase 3.5: Pipeline Cleanup + Observability + Coverage [ЗАВЕРШЕНО]

> [SPEC-RAG-20d](../specifications/completed/SPEC-RAG-20d-pipeline-cleanup.md), [SPEC-RAG-19](../specifications/completed/SPEC-RAG-19-observability-langfuse.md).

**Pipeline cleanup**: serialize_tool_payload, atomic trim_messages, k_per_query vs k_total, LLM 400 retry, temporal guard. 32 code changes.

**Observability**: Langfuse v3 self-hosted ([DEC-0040](../architecture/11-decisions/decision-log.md)). 7 instrumentation points. Double JSON fix, phase-aware step names, token aggregation.

**Coverage redesign**: LANCER-inspired lexical nugget coverage ([DEC-0044](../architecture/11-decisions/decision-log.md)) — query_plan subqueries как nuggets, threshold 0.75, max 1 targeted refinement. Latency −40-65%.

**CE confidence filter** ([DEC-0045](../architecture/11-decisions/decision-log.md)): cross-encoder фильтрует irrelevant docs (keep 92% relevant, remove 55% irrelevant). ColBERT порядок сохраняется.

**Retrieval calibration** (100 hand-crafted queries): R@1=0.80, R@3=0.97, R@20=0.98. Monotonicity OK.

### Phase 3.6: NLI + Robustness + Code Quality [ЗАВЕРШЕНО]

> [SPEC-RAG-21](../specifications/completed/SPEC-RAG-21-nli-citation-faithfulness.md)/[22](../specifications/completed/SPEC-RAG-22-comprehensive-eval-metrics.md)/[23](../specifications/completed/SPEC-RAG-23-ndr-rsr-ror-bypass.md)/[24-28](../specifications/completed/).

**NLI citation faithfulness** (SPEC-RAG-21): ruBERT NLI pipeline (claim decomposition → per-claim verification). Faithfulness 0.91 corrected, 0 real hallucinations из 171 claims.

**Robustness baseline** (SPEC-RAG-23): bypass pipeline (прямые Qdrant + LLM, без agent). NDR 0.963, RSR 0.941, ROR 0.959, composite 0.954. Методология: simplified Cao et al. (160 LLM calls vs 55K в оригинале). BERTScore как proxy провалился — Claude judge обязателен.

**Comprehensive eval** (SPEC-RAG-22): SummaC-ZS sentence-level faithfulness + RGB noise robustness + query perturbation harness.

**Code quality** (SPEC-RAG-24-28): dead code cleanup (−544 lines), DRY extraction, type annotations, complexity reduction. Аудит: 9/10.

### Phase 3.7: Framework Benchmark [ЗАВЕРШЕНО]

> [SPEC-RAG-29](../specifications/completed/SPEC-RAG-29-framework-comparison-benchmark.md). Research: [R27](../research/reports/R27-framework-benchmark-methodology.md).

4 pipeline (naive / LI-stock / LI-maxed / custom) × 2 retrieval датасета (100 Qs each) + agent E2E (17 Qs).

Custom доминирует: +0.30 factual, +0.56 usefulness, +0.40 grounding vs best framework config.
LlamaIndex default hybrid = zero gain over dense-only. Weighted RRF = main differentiator.
Custom 7x быстрее LI-maxed (framework abstraction overhead).

### Phase 3.8: Repository Cleanup [DONE]

Реорганизация docs/ (5 planning файлов → 2), перенумерация research reports (R00-R27), удаление мёртвых скриптов (-544 lines dead code), specs active/ → completed/.

### Phase 3.9: Retrieval Ablation Study [DONE]

39+ экспериментов за 5 дней (2026-04-04..08). Retrieval R@5: 0.833 → 0.900 (+8%). Full pipeline validated: R@5 = RO, quality > RO (judge 6:1:8). Формализован experiment protocol.

Post-protocol validation (RUN-004–008, 2026-04-08..10):
- `compose_context` 1800→4000 — no regression, budget kept
- channel dedup `2→3` — adopted
- dual scoring (`norm_linear`, `rrf_ranks`) — rejected
- cosine recall guard — adopted
- corrected 36Q baseline after dataset audit: factual **0.858**, useful **1.708**, refusal **3/3**

Подробности: `docs/progress/ablation_study.md`, артефакты: `experiments/`.

---

## Оставшаяся работа

### Приоритет 1 — Data & Eval

- [ ] Re-ingest свежих постов (2026-03-18 → current)
- [ ] Weekly digests пересчёт для всех ~37 недель
- [ ] Channel profiles re-compute
- [ ] P1 fixes: q33 monthly hot_topics + q36 channel_expertise routing
- [ ] Clean baseline 36 Qs после фиксов
- [ ] Eval expansion: 36 → 100+ Qs
- [ ] Recompute bootstrap/significance after golden v3 (current CI: factual [0.792, 0.917], useful [1.606, 1.803])

### Приоритет 2 — Polish

- [ ] fetch_docs chunk stitching (длинные посты не собираются)
- [ ] q21 deterministic out-of-range refusal
- [ ] Unit tests для analytics tools и state machine
- [ ] CE filter_threshold=0.0 smoke test
- [ ] docs/architecture/ обновление (observability, API, security)

### Приоритет 3 — README & Presentation

- [ ] Mermaid диаграмма полного pipeline
- [ ] Скриншот web UI
- [ ] Design Decisions секция

### Retrieval ablation backlog (post-study)

| Experiment | Status | Result |
|-----------|--------|--------|
| ~~DBSF fusion vs RRF~~ | DONE (Phase 1) | RRF лучше на −0.8% |
| ~~BM25 weight sweep~~ | DONE (Phase 1) | [1:3] оптимально |
| ~~Multi-query benchmark~~ | DONE (Phase 2b-3) | qplan + inject + MMR = quality > RO |
| ~~Instruction prefix A/B~~ | DONE (Phase 1) | no-prefix = +5.8% R@5 |
| Adaptive filter tuning | **OPEN** | ce_neg=3.2, нужно ужесточить |
| MMR λ tuning (0.7→0.85) | **OPEN** | MRR gap −0.048 |

### Осознанно не делаем

Graph RAG, Self-RAG, semantic caching, multi-provider fallback, SFT/RLHF, advanced multi-turn memory.

---

## Исследовательская база (R00–R27)

Полные отчёты: [`docs/research/reports/`](../research/reports/)

| # | Тема | Ключевой вывод |
|---|------|----------------|
| [R00](../research/reports/R00-synthesis.md) | Synthesis (R01-R06) | Архитектурные решения, гипотезы H1-H17 |
| [R01](../research/reports/R01-qdrant-hybrid-rag.md) | Qdrant Hybrid RAG | FilterQuery, prefetch, FusionQuery |
| [R02](../research/reports/R02-llm-serving.md) | LLM Serving | vLLM vs llama.cpp, MoE на V100 |
| [R03](../research/reports/R03-model-selection.md) | Model Selection | Embedding + reranker benchmarks |
| [R04](../research/reports/R04-coverage-metrics.md) | Coverage Metrics | 5-signal composite coverage |
| [R05](../research/reports/R05-rag-evaluation.md) | RAG Evaluation | Eval methodology foundations |
| [R06](../research/reports/R06-async-architecture.md) | Async Architecture | httpx patterns, async lifecycle |
| [R07](../research/reports/R07-retrieval-agent-pipeline-quality.md) | Pipeline Quality | Retrieval + agent quality gaps |
| [R08](../research/reports/R08-llm-selection-v100-32gb.md) | LLM Selection V100 | Qwen3-30B MoE Q4_K_M fitting |
| [R09](../research/reports/R09-telegram-channels-collection.md) | Telegram Channels | 36 каналов из 70+ к��ндидатов |
| [R10](../research/reports/R10-gpu-docker-wsl2-troubleshooting.md) | GPU/Docker/WSL2 | V100 TCC блокирует NVML, WSL2 native workaround |
| [R11](../research/reports/R11-advanced-retrieval-strategies.md) | Advanced Retrieval | ColBERT, weighted RRF, whitening, entity extraction |
| [R12](../research/reports/R12-cluster-based-retrieval.md) | Cluster-based Retrieval | Отложено: effort > impact при 13K docs |
| [R13](../research/reports/R13-deep-tool-router-architecture.md) | Tool Router Architecture | 4→15 tools, dynamic visibility, rule-based hints |
| [R14](../research/reports/R14-deep-beyond-frameworks-techniques.md) | Beyond Frameworks | A-RAG, CRAG, Speculative RAG, temporal reasoning |
| [R15](../research/reports/R15-yandex-rag-conference-2026.md) | Yandex RAG Conference | "Less is More" tools, FC bottleneck, eval methodology |
| [R16](../research/reports/R16-deep-rag-agent-tools-expansion.md) | Generic RAG Tools | 5 recommended, 2 rejected, phase-based visibility |
| [R17](../research/reports/R17-deep-domain-specific-tools.md) | Domain-Specific Tools | entity_tracker, arxiv_tracker, hot_topics, channel_expertise |
| [R18](../research/reports/R18-deep-evaluation-methodology-dataset.md) | Eval Methodology | LLM judge, robustness, synthetic pipeline design |
| [R19](../research/reports/R19-deep-nli-citation-faithfulness.md) | NLI Faithfulness | Hybrid C: Qwen3 decomposition + XLM-RoBERTa NLI |
| [R20](../research/reports/R20-deep-retrieval-robustness-ndr-rsr-ror.md) | Retrieval Robustness | Cao et al. adapted: simplified NDR/RSR/ROR protocol |
| [R21](../research/reports/R21-deep-rag-necessity-classifier.md) | RAG Necessity | Rule-based tiers (<1ms), LLM classifier неоправдан |
| [R22](../research/reports/R22-deep-production-gap-analysis.md) | Production Gap Analysis | Gaps vs Perplexity/Cohere: proof layer, CRAG-lite, observability |
| [R23](../research/reports/R23-golden-v2-eval-baseline.md) | Golden v2 Baseline | Consensus judge: factual ~0.80, KTA 1.0 |
| [R24](../research/reports/R24-deep-docker-wsl2-networking-fix.md) | Docker/WSL2 Networking | Mirrored networking, relay architecture |
| [R25](../research/reports/R25-deep-observability-langfuse-phoenix-structlog.md) | Observability | Langfuse v3 vs Phoenix vs structlog |
| [R26](../research/reports/R26-deep-comprehensive-rag-eval-metrics.md) | Comprehensive Eval | SummaC, RGB noise, NDR/RSR/ROR, query perturbation |
| [R27](../research/reports/R27-framework-benchmark-methodology.md) | Framework Benchmark | LlamaIndex vs custom: 4/12 components covered |
