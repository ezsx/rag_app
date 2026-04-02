# Prompt 36: Deep Research — Comprehensive RAG Evaluation Metrics

## Цель

Получить **полную карту метрик** для оценки production-grade RAG-системы с агентским (ReAct) пайплайном. Не ограничиваться тем что мы уже знаем — собрать ВСЕ значимые метрики из литературы 2024-2026, сгруппировать, и дать конкретные рекомендации что нам нужно.

**Ключевой вопрос**: "Какие метрики существуют для оценки RAG-систем? Мы хотим максимально полный набор — чем больше осей оценки, тем лучше понимаем качество."

---

## Контекст проекта — подробный

### Что это
`rag_app` — production-grade RAG-платформа для поиска по 36 русскоязычным Telegram-каналам (AI/ML тематика). FastAPI + SSE streaming. Полностью self-hosted, без внешних API.

- **13K документов** в Qdrant (посты из Telegram-каналов, июль 2025 — март 2026)
- **ReAct агент** с 15 tools, native function calling через Qwen3.5-35B-A3B
- **Hybrid retrieval**: BM25 top-100 + dense top-20 → weighted RRF (BM25 3:1) → ColBERT rerank → cross-encoder confidence filter → channel dedup (max 2/channel)
- **Coverage**: LANCER-style nugget coverage (query_plan subqueries = nuggets, threshold 0.75, max 1 targeted refinement)

### Hardware
- LLM: **Qwen3.5-35B-A3B GGUF** Q4_K_M (V100 SXM2 32GB, llama-server.exe, порт 8080, ~100 calls/hour)
- Embedding: **pplx-embed-v1-0.6B** (bf16, mean pooling, 1024-dim, RTX 5060 Ti, gpu_server.py, порт 8082)
- Reranker: **Qwen3-Reranker-0.6B-seq-cls** (chat template, logit scoring, RTX 5060 Ti)
- ColBERT: **jina-colbert-v2** (560M, 128-dim per-token MaxSim, RTX 5060 Ti)
- NLI: **cointegrated/rubert-base-cased-nli-threeway** (180M, RTX 5060 Ti, endpoint /nli на gpu_server.py)
- Vector store: **Qdrant** (Docker CPU, dense + sparse + ColBERT named vectors)
- **NO Claude/OpenAI API** для автоматического judging — все judge scoring через manual chat (Claude Opus + Codex GPT-5.4 consensus)

### 15 agent tools
Оркестрация: ReAct loop, native function calling, dynamic tool visibility (phase-based, max 5 видимых).

| Tool | Назначение | Phase |
|------|-----------|-------|
| query_plan | Декомпозиция запроса на subqueries | pre-search |
| search | Hybrid search (BM25+dense+ColBERT) | pre-search |
| temporal_search | Search с temporal фильтром | pre-search |
| channel_search | Search по конкретному каналу | pre-search |
| cross_channel_compare | Сравнение каналов | pre-search |
| summarize_channel | Суммаризация канала | pre-search |
| list_channels | Навигация: список каналов | pre-search |
| rerank | Cross-encoder reranking | post-search |
| related_posts | Похожие посты | post-search |
| compose_context | Сборка контекста из документов | post-search |
| final_answer | Финальный ответ | post-search |
| entity_tracker | Analytics: entity popularity, timeline, comparison | post-search |
| arxiv_tracker | Analytics: arxiv paper tracking | post-search |
| hot_topics | Weekly digest из BERTopic pipeline | analytics-complete |
| channel_expertise | Per-channel expertise profile | analytics-complete |

Short-circuits: navigation → skip forced search; analytics → skip forced search + verify bypass.

### Язык данных
Русский (неформальный, Telegram-стиль). Морфология 6 падежей × 2 числа → BLEU/ROUGE/exact match плохо работают.

### Контекст использования — портфолио
Проект строится как **портфолио для позиции Applied LLM Engineer**. Это влияет на выбор метрик:
- Метрики должны демонстрировать глубокое понимание eval methodology (не "запустил RAGAS и получил числа")
- Многоосевая оценка > одна метрика. Чем больше осей качества измеряем — тем сильнее показываем экспертизу
- Credibility: hiring managers для Applied LLM ролей разбираются в eval — нужны статистически обоснованные метрики
- Self-hosted полностью: показать что работает без API ключей к GPT-4/Claude

### Judge workflow — принципиальное ограничение
У нас **нет automated API доступа** к frontier моделям. Judge scoring делается **вручную через чат** с Claude Opus 4.6 или Codex GPT-5.4 (consensus двух моделей). Qwen3.5-35B как local judge **не рассматривается** — слабоват для judge role + circular dependency (модель оценивает собственные ответы). Это означает:
- Автоматические метрики (programmatic, NLI-based) ценнее чем judge-dependent
- Judge-dependent метрики ограничены пропускной способностью ручного chat (~200 Qs за сессию)
- Фреймворки требующие API calls к GPT-4 per-question — не применимы

---

## Текущий eval pipeline — что уже есть

### Golden dataset v2: 36 questions (eval_golden_v2.json)

| Category | Count | eval_mode |
|----------|-------|-----------|
| broad_search | 6 | retrieval_evidence |
| constrained_search | 7 | retrieval_evidence |
| compare_summarize | 4 | retrieval_evidence |
| analytics_hot_topics | 3 | analytics |
| analytics_channel_expertise | 3 | analytics |
| navigation | 2 | navigation |
| negative_refusal | 3 | refusal |
| future_baseline | 8 | refusal |

Каждый вопрос содержит: expected_answer, key_tools, forbidden_tools, source_post_ids, acceptable_evidence_sets, required_claims, eval_mode, difficulty.

### Текущие метрики и как они измеряются

| Метрика | Текущее значение | Как именно измеряется | Автоматически? |
|---------|-----------------|----------------------|----------------|
| **Factual correctness** | 0.842 | Claude judge в ручном чате, шкала 0.0-1.0 с шагом 0.1. Промпт: judge_v1.md — 7 калибровочных примеров, принципы partial credit | Нет (manual) |
| **Usefulness** | 1.778/2 | Claude judge, шкала 0/1/2, тот же промпт | Нет (manual) |
| **Key Tool Accuracy (KTA)** | 1.000 | Программный: проверка что agent вызвал хотя бы один key_tool и не вызвал forbidden_tools. Binary 0/1. Tracking через SSE events (tool_invoked) | Да |
| **Faithfulness (NLI)** | 0.91 (corrected) | Pipeline: Claude decomposition (claims) → ruBERT NLI (claim × document → entailment/neutral/contradiction). Thresholds: entailment 0.45, contradiction 0.55. Lenient scoring: entailment=1.0, neutral=0.5, contradiction=0.0. 19 contradictions проверены вручную — все false positives | Частично (NLI автоматический, decomposition manual через Claude chat) |
| **Retrieval recall@k** | r@1=0.80, r@3=0.97, r@5=0.97, r@20=0.98 | 100 calibration queries, прямые Qdrant запросы (без agent). Ground truth: per-query expected post_ids | Да |
| **Agent latency** | 24.4s avg | Timestamp diff из SSE stream (first event → final event) | Да |
| **Coverage (runtime)** | ~0.75 | LANCER nugget coverage: query_plan subqueries как nuggets, keyword overlap с документами. Threshold 0.75, triggers targeted refinement | Да (runtime, не eval metric) |
| **Strict anchor recall** | ~0.43 (diagnostic) | Fuzzy match expected source_post_ids vs agent citation_hits (±5 msg_id, ±50 for temporal) | Да (но ненадёжен) |
| **Acceptable set hit** | diagnostic | Binary: хотя бы один evidence set из acceptable_evidence_sets найден | Да |
| **Failure attribution** | per-question | Программный: tool_hidden / tool_wrong / tool_failed / retrieval_empty / generation_wrong / refusal_wrong / judge_uncertain | Да |

### Judge workflow
1. `evaluate_agent.py` прогоняет 36 Qs → `eval_results.json` с offline_judge_packets
2. `--export-offline-judge` → markdown батчи по 30 Qs для ручного review
3. Claude chat: decomposition (claims) по decomposition_v1.md промпту → `claims.json`
4. Claude chat: judge scoring по judge_v1.md промпту → `judge_scores.json`
5. `scripts/run_nli.py`: claims × documents → ruBERT NLI → `nli_scores.json`
6. `scripts/merge_eval_report.py`: объединяет всё → `final_report.json` + `.md`

### Retrieval calibration dataset
Отдельно: `eval_retrieval_calibration.json` — 100 hand-crafted queries с ground truth post_ids. Запускается через `scripts/calibrate_coverage.py`. Тестирует retrieval pipeline напрямую без agent loop.

---

## Что мы знаем из наших research reports (ключевые findings)

### R05 — RAG Evaluation: RAGAS, DeepEval, LLM-judge
- **RAGAS** (v0.4.3): faithfulness через NLI-декомпозицию, answer_relevancy через reverse-question embedding, context_recall/precision. **Проблемы**: 2 breaking changes/год, NaN на vLLM, EN-only промпты.
- **DeepEval** (v3.7.6): 50+ метрик, pytest-интеграция, GEval для custom критериев, `set-local-model` для vLLM.
- **Custom LLM-judge**: полный контроль русских промптов. Prometheus-2 (7B) достигает 0.897 Pearson correlation с человеческими оценками.
- **DEC-0020**: решили использовать custom judge + DeepEval для CI/CD. RAGAS — для разовых аудитов.
- R05 предложил 3-tier метрики: retrieval (recall/precision/MRR/nDCG), generation (faithfulness/relevance/completeness/citation_accuracy), system (latency/answer_rate).

### R15 — Яндекс RAG Conference 2026 (Андрей Соколов, NLP-инженер Яндекса)
- **NDR/RSR/ROR** (robustness) — Яндекс подтвердил важность. У нас НЕ измеряем.
- **RAG necessity classifier** — skip retrieval когда parametric knowledge достаточно. -25% latency.
- Яндекс: "LLM judge не работает для factual" → нужна NLI-декомпозиция (confirm-deny на atomic claims).
- "Модели стабильно лучше отвечают при 5-7 чанках, даже с шумом" — relevance threshold.
- "Перестановка чанков не должна менять ответ" — order robustness.
- Multilingual judge consistency: Fleiss' κ ≈ 0.3 — очень низкий для русского.

### R18 — Evaluation Methodology & Dataset (полная спецификация)
- 7 секций: dataset design, scoring, tool tracking, robustness, statistical power, CI/CD.
- Decompose-then-verify: replace monolithic LLM scoring с claim-level entailment.
- NDR/RSR/ROR: compute budget ~33 часов для полного suite (1500 Qs), ~6.5 часов для 50 Qs.
- BERTScore для русского: ai-forever/ruBert-large > xlm-roberta-large > bert-base-multilingual.
- Recommended: 0-3 integer scale, continuous scoring с post-hoc binarization.
- Statistical power: N=50 → 76% power для medium effects (d=0.5). N=100 → 80% at d=0.4.

### R19 — NLI Citation Faithfulness
- Hybrid approach: LLM decomposition + NLI verification.
- XLM-RoBERTa-large-xnli **провалился** на русском (MNLI bias, near-zero entailment).
- Заменили на **rubert-base-cased-nli-threeway** — 150x improvement on Russian.
- 19 contradictions проверены вручную — все false positives (12 paraphrase failures, 5 wrong-doc).
- Faithfulness 0.91 (corrected), 0 real hallucinations.

### R20 — Retrieval Robustness NDR/RSR/ROR (Cao et al. 2025)
- NDR: retrieval ≥ no-retrieval baseline. RSR: monotonic по k. ROR: order invariant.
- Оригинальный protocol: 1500 Qs × 6 k-values × 3 orderings × 2 retrievers = 55K calls.
- Сжатый: 50 Qs × 4 k-values × 3 orderings = 650 calls (~6.5 часов).
- BERTScore для русского: ruBert-large (#1), xlm-roberta-large (#2).
- Lost-in-the-middle: inherent architectural property каузальных трансформеров.
- Document reordering: marginal benefit в production RAG (Cuconasu EMNLP 2025).
- **Key insight**: for Russian, weakest link is LLM judge (κ≈0.3), not similarity metric.

### R21 — RAG Necessity Classifier
- NDR measurement: f(q,k) ≥ f(q,0) — retrieval помогает?
- Heuristic routing: conversational → skip RAG. -25% latency без потери quality.
- Classifier на perplexity: high perplexity → needs RAG, low → parametric OK.

### R25 — Production Gap Analysis
- **Eval credibility gap**: 30 questions → uncertainty range too wide. Minimum 50, recommended 100+.
- RAGAS synthetic dataset generation для bootstrap 30→100.
- Missing: CI/CD integration, query understanding/rewriting, error recovery, graceful degradation.
- "Retrieval pipeline is production-grade. Gaps are in everything around it."

---

## Что я хочу получить

### 1. Полная таксономия RAG-метрик (2024-2026 литература)

Сгруппировать по осям:
- **Retrieval quality** (recall, precision, MRR, nDCG, MAP, coverage, context relevance, ...)
- **Answer quality** (correctness, completeness, relevance, conciseness, ...)
- **Faithfulness / Grounding** (citation accuracy, hallucination rate, attribution, claim-level verification, ...)
- **Robustness** (noise resistance, order sensitivity, retrieval size sensitivity, query perturbation stability, ...)
- **Efficiency** (latency, cost, token usage, throughput, ...)
- **Agent-specific** (tool routing accuracy, planning quality, coverage/refinement, multi-hop success, ...)
- **User-facing** (usefulness, readability, format quality, ...)
- **Другие оси** если есть (consistency across runs, safety, fairness, ...)

Для каждой метрики:
- Название и источник (paper/framework, arXiv ID если есть)
- Формула или определение
- Как измеряется (автоматически / LLM judge / human / NLI model)
- Compute cost (calls, time, какие ресурсы)
- **Применимость к нашей системе**: русский язык, агентский пайплайн, offline judge (нет API), local models

### 2. Обзор существующих eval фреймворков (2024-2026)

Покрыть минимум:
- **RAGAS** (v0.4+) — текущее состояние, что исправили из проблем v0.2-v0.3
- **RAGChecker** (ACL 2025, Zhang et al.) — claim-level diagnostic
- **TruLens** — feedback functions
- **DeepEval** (v3.7+) — 50+ метрик, GEval
- **ARES** — automated RAG eval
- **LlamaIndex evaluation module** — что предлагает
- **Новые фреймворки 2025-2026** — что появилось (включая специализированные для agents)

Для каждого:
- Полный список реализованных метрик
- Dependency на GPT-4/Claude API (мы НЕ имеем automated API access)
- Поддержка русского языка (промпты, tokenization)
- Совместимость с local LLM (Qwen3.5-35B через OpenAI-compatible API)
- Совместимость с local NLI models (ruBERT через HTTP endpoint)
- Можно ли использовать отдельные метрики без полного фреймворка

### 3. Конкретные рекомендации для нашей системы

Учитывая constraints:
- **Нет API** к frontier моделям для автоматического judging (judging через manual chat с Claude)
- **Русский язык** (морфология, неформальный стиль, Telegram)
- **Local models**: Qwen3.5-35B (judge потенциально), ruBERT NLI (faithfulness), pplx-embed-v1 (similarity)
- **36-50 golden questions**, ручная курация
- **Агентский пайплайн** с 15 tools (не простой retrieve-and-generate)
- **~100 LLM calls/hour** throughput

Ответить на:
1. Какие метрики добавить к текущему стеку? Приоритет по информативности и feasibility.
2. Какие фреймворки реально работают с нашими constraints (local models, русский, no API)?
3. NDR/RSR/ROR — оригинальный retrieval-level (vary k, shuffle docs) vs query-level perturbations? Что правильнее для агентского пайплайна?
4. Метрики которые можно вычислить **полностью автоматически** (без judge) с нашими моделями?
5. Оптимальный размер eval dataset — 36 vs 50 vs 100 vs 500? Статистическая мощность.
6. BERTScore для русского — ruBert-large vs xlm-roberta-large, какой layer, calibration?
7. При manual judge через chat (Claude/Codex) — как максимизировать информацию за минимум judge calls?
8. Claim-level vs answer-level scoring — когда что лучше?
9. Metric consistency: как измерять stability оценок при повторных прогонах?

### 4. Финальная рекомендация — Metric Stack

Составить **конкретный metric stack** для нашей системы:

**Tier 1** (must-have, внедрять сейчас — максимальный ROI):
- Что добавить
- Implementation plan: модели, скрипты, данные, compute, время

**Tier 2** (valuable, внедрить после Tier 1):
- Что добавить
- Implementation plan

**Tier 3** (nice-to-have, при наличии времени):
- Что

Для каждой метрики: **конкретная реализация** (не "используйте RAGAS", а "вызовите ruBERT на claim × doc pairs с threshold 0.45, агрегируйте через lenient mean").

---

## Формат ответа

Structured report. Таблицы для сравнений. Конкретные формулы. Ссылки на papers (arXiv IDs). Никакого fluff — только actionable information.
