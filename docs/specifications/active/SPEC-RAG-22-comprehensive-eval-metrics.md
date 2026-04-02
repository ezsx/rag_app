# SPEC-RAG-22: Comprehensive Eval Metrics

> Статус: DRAFT v3 (post-Codex review #2)
> Автор: Claude + Human
> Дата: 2026-04-01
> Research base: R29 (comprehensive RAG eval metrics), R20 (robustness NDR/RSR/ROR), R18 (eval methodology), R15 (Yandex RAG conf)
> Зависимости: evaluate_agent.py, eval_golden_v2.json, gpu_server.py, ruBERT NLI, API running
> История: v1 (query perturbation — отклонена), v2 (Codex review: 0 FAIL 10 CONCERN), v3 (fixes all concerns)

---

## Цель

Расширить eval pipeline с **6 метрик до ~15**, покрывая 6 осей оценки. Всё self-hosted, без API.

**Зачем:**
- Текущие 6 метрик (factual, useful, KTA, faithfulness, recall, latency) — одномерная картина
- Между прогонами нет автоматического сигнала quality (judge = manual)
- Нет robustness testing (Яндекс R15 highlight, Cao et al. 2025)
- Нет statistical confidence (CI ±13% при 50 Qs — бесполезно для fine-tuning decisions)
- Портфолио: многоосевая eval = экспертиза для Applied LLM Engineer

---

## Архитектура: три слоя eval

```
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 1a: ALWAYS-ON AUTOMATIC                      │
│            (каждый прогон, 0 LLM cost)                         │
│                                                                 │
│  evaluate_agent.py расширяется:                                │
│  ├─ Retrieval: Precision@5, MRR, nDCG@5 (+ existing Recall)   │
│  ├─ Answer: BERTScore (ruBert-large, layer 18, IDF)            │
│  ├─ Faithfulness: summac_faithfulness (sentence-level NLI)     │
│  └─ Agent: ToolCallF1, Trajectory Match (+ existing KTA)       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│              LAYER 1b: PERIODIC AUTOMATIC DIAGNOSTICS           │
│            (monthly / при подозрении на regression)             │
│                                                                 │
│  ├─ Stability: Cross-run variance (3×, σ)                      │
│  └─ Hallucination: LettuceDetect spike (optional, если рус ОК) │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│              LAYER 2: ROBUSTNESS                                │
│            (per release, ~520 agent calls)                      │
│                                                                 │
│  scripts/evaluate_robustness.py [NEW]:                         │
│  ├─ RGB noise robustness (inject irrelevant docs)              │
│  ├─ RGB negative rejection (all docs irrelevant)               │
│  ├─ Proxy-NDR: retrieval vs no-retrieval (BERTScore scoring)   │
│  └─ Query perturbation (noise/substitution/reorder)            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│              LAYER 3: STATISTICAL                               │
│            (post-processing, no extra calls)                    │
│                                                                 │
│  scripts/compute_confidence.py [NEW]:                          │
│  ├─ Bootstrap CIs (B=1000, BCa) + paired bootstrap для A/B     │
│  ├─ Wilson intervals для binary metrics                        │
│  └─ PPI (deferred — после валидации prerequisites)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1a: Always-On Automatic Metrics

Вычисляются на каждом eval прогоне. Zero LLM cost.

### 1.1 Retrieval: Precision@5, MRR, nDCG@5

Добавить к existing Recall@k. Используют те же `source_post_ids` / `acceptable_evidence_sets`.

**Precision@5** — доля релевантных в top-5:
```
Precision@5 = |Rel ∩ Ret@5| / 5
```

**MRR** — средняя обратная позиция первого релевантного:
```
MRR = (1/|Q|) × Σ_q 1/rank_q
```

**nDCG@5** — position-weighted graded relevance:
```
DCG@5 = Σ_{i=1}^{5} rel_i / log₂(i+1)
nDCG@5 = DCG@5 / IDCG@5
```

Grading: binary (rel=1 if doc in acceptable_evidence_sets, else 0).
Scope: только `eval_mode=retrieval_evidence` (17 Qs из 36). Analytics/navigation/refusal — skip.

**Реализация**: новые функции в `evaluate_agent.py :: _compute_metrics()`. Данные уже есть (citation_hits + source_post_ids).

### 1.2 Answer Quality: BERTScore

Semantic similarity между agent answer и expected_answer. Primary automatic answer quality signal для русского.

**Конфигурация**:
- **Модель**: `ai-forever/ruBert-large` (24 layers, Russian-native tokenizer)
- **Layer**: 18 (starting default из R29 §6; tune на dev-set 30-50 пар с human similarity при первом запуске)
- **IDF**: да, из sample 1000 docs из Qdrant (domain-specific weighting)
- **WordPiece aggregation**: объединить incomplete subword tokens в complete words, усреднить vectors (Vetrov et al. 2022, arXiv:2203.05598). Custom tokenization wrapper.
- **Baseline rescaling**: вычислить на 500 random Russian sentence pairs из корпуса (precomputed baselines не существуют для ruBert-large)

```python
from bert_score import score
P, R, F1 = score(
    cands=[agent_answer],
    refs=[expected_answer],
    model_type="ai-forever/ruBert-large",
    num_layers=18,
    idf=True,
    idf_sents=domain_corpus,
)
```

**Ожидаемая корреляция с human**: ρ ≈ 0.55-0.65 (R29 §6). Достаточно как один component в metric ensemble, **не** sole quality signal.

**Calibration**: при первом запуске — tune layer (17-20) на dev-set из 30-50 golden Qs с existing Claude judge scores. Выбрать layer с max Spearman ρ(BERTScore, judge_factual).

**Реализация**: зависимость `bert-score`. Lazy load (один раз за eval run). Поле `bertscore_f1` в metrics.

### 1.3 Faithfulness: `summac_faithfulness` (sentence-level NLI)

Полностью автоматическая faithfulness **без claim decomposition**. Дополняет existing `claim_faithfulness` из SPEC-RAG-21.

**Naming convention (fix Codex concern #3):**
- `summac_faithfulness` — sentence-level, automatic, Layer 1a (каждый прогон)
- `claim_faithfulness` — claim-level с Claude decomposition, SPEC-RAG-21, Tier 3 diagnostic
- **Release-facing metric**: `summac_faithfulness` (т.к. fully automatic и reproducible)
- **Deep diagnostic**: `claim_faithfulness` (когда нужна per-claim granularity)

**Алгоритм** (SummaC-ZS, Laban et al. TACL 2022):
1. Сегментировать answer на предложения через `razdel` (лучше для Telegram-стиля чем spaCy)
2. Сегментировать все cited documents на предложения
3. Для каждой пары (doc_sentence, answer_sentence) → ruBERT NLI → P(entailment)
4. Для каждого answer_sentence: `max_support = max(P(entailment))` across all doc_sentences
5. `summac_faithfulness = mean(max_support)` across answer_sentences
6. **Flag**: sentences с max_support < 0.4 → candidates для claim-level deep diagnostic

**Модель**: `cointegrated/rubert-base-cased-nli-threeway` (уже на gpu_server.py /nli)
**Throughput**: сотни eval/min на RTX 5060 Ti.

**Реализация**: `src/services/eval/summac.py`. Batch /nli calls. Поле `summac_faithfulness` в metrics.

### 1.4 Agent: ToolCallF1 + Trajectory Match

**ToolCallF1** — F1 по tool calls (partial credit вместо binary KTA):
```
Precision = |called ∩ key_tools| / |called − scaffold_tools|
Recall = |called ∩ key_tools| / |key_tools|
F1 = 2 × P × R / (P + R)
```
Scaffold tools (query_plan, rerank, compose_context, final_answer) — исключаются из precision denominator.
Forbidden tools penalty: если вызвал forbidden → F1 = 0.

**Trajectory Match** (LangChain AgentEvals pattern, unordered mode):
```python
def trajectory_match(called: List[str], expected: List[str], mode="unordered") -> float:
    """1.0 if called ⊇ expected (unordered), else |called ∩ expected| / |expected|"""
```

**KTA остаётся** как backward-compatible diagnostic (binary). ToolCallF1 — primary agent metric.

**Argument Correctness** (Codex suggestion #10): проверка аргументов tool calls (temporal filters, channel names). Deterministic — сравнение с golden dataset `key_tool_args` если задано. **Добавляем поле в dataset, пока optional.**

---

## Layer 1b: Periodic Automatic Diagnostics

Не на каждом прогоне, но без LLM cost. Monthly или при подозрении на regression.

### 1.5 Stability: Cross-run Variance

3 прогона на temp=0. Report σ per-question и σ aggregate для каждой метрики.

```bash
python scripts/evaluate_agent.py --dataset ... --runs 3 --output results/stability/
```

**Pass/fail**: σ(factual_BERTScore) > 0.05 на aggregate → investigate.
σ per-question > 0.1 → flag question as unstable.

### 1.6 Hallucination: LettuceDetect (optional spike)

**Status**: optional, не в MVP.
**Spike protocol**: 10 golden Qs, compare LettuceDetect output vs SummaC flags.
**Adopt threshold**: agreement > 60% с SummaC на русских ответах.
**If fails**: drop, SummaC + claim_faithfulness покрывают.

---

## Layer 2: Robustness

Новый скрипт `scripts/evaluate_robustness.py`. Per-release.

### 2.1 RGB Noise Robustness (Chen et al., AAAI 2024)

Inject irrelevant documents, проверить что quality не падает.

**Протокол** (fix Codex concern #4 — concrete rules):

1. **Sampling irrelevant docs**: random docs из Qdrant с `cosine_distance(query_embedding, doc_embedding) > 0.7` AND из другого channel чем expected. Это фильтрует accidental relevance.
2. **Injection point**: после search, перед compose_context. Mock search results с заменёнными docs.
3. **Noise levels**: 20%, 40%, 60%, 80% docs replaced (round up).
4. **Scoring**: BERTScore(noised_answer, expected_answer) как automatic proxy. Не judge.
5. **Per-question**: для каждого из 50 Qs × 4 noise levels.

**Метрика**: 
```
RGB_noise = BERTScore_mean(60% noise) / BERTScore_mean(0% noise)
```
RGB_noise ≥ 0.8 → robust. < 0.6 → fragile.

**Calls**: 50 × 4 = 200 agent calls.

### 2.2 RGB Negative Rejection

Все docs irrelevant → система должна отказать.

**Протокол**:
1. 20 answerable golden questions (broad_search + constrained_search subset)
2. Replace ALL retrieved docs → random irrelevant (same sampling as 2.1)
3. Agent прогон
4. **Pass**: answer contains refusal markers ("не найдено", "в базе нет", "не могу найти") OR answer is empty
5. **Fail**: agent generates confident answer from irrelevant docs

**Метрика**: `negative_rejection_rate = passed / 20`
**Pass/fail threshold**: ≥ 0.7 (14/20).

**Calls**: 20 agent calls.

### 2.3 Proxy-NDR (Semantic No-Degradation Rate)

**ВАЖНО**: это **не** оригинальный Cao et al. NDR (который требует multiple k + orderings + judge). Это simplified proxy: retrieval vs no-retrieval, BERTScore scoring.

**Протокол**:
1. Для каждого golden question — normal agent run → `answer_rag`
2. Agent run с `--no-retrieval` flag (search tool returns empty, force final_answer on parametric knowledge) → `answer_no_rag`
3. `score_rag = BERTScore_F1(answer_rag, expected_answer)`
4. `score_no_rag = BERTScore_F1(answer_no_rag, expected_answer)`

**Метрика**:
```
proxy_NDR = (1/|Q|) × Σ_q 𝟙[score_rag ≥ score_no_rag]
```

proxy_NDR ≥ 0.85 → retrieval consistently helps. < 0.7 → investigate.

**Scope**: retrieval_evidence questions only (17+ Qs).
**Calls**: 50 × 2 = 100 agent calls (можно reuse normal run для answer_rag).

### 2.4 Query Perturbation Robustness

Perturbed query → answer consistency.

**Perturbation types** (fix Codex concern #6 — Russian-specific):

**Noise** (программный, deterministic seed):
- Random char swap/insert/delete (~20% слов)
- Russian keyboard-neighbor typos (ё↔е, б↔ю, ж↔э)
- Latin/Cyrillic mixing (NVIDIA → НВИДИА, GPT → ГПТ)

```python
def add_noise(query: str, seed: int) -> str:
    """Russian-aware noise: char mutations + keyboard neighbors + Lat/Cyr mixing."""
```

**Substitution** (Claude-generated, 1 раз):
- Синонимы ключевых слов
- Парафраз с сохранением смысла
- Telegram shorthand ("что" → "че", "какие-нибудь" → "какие-нить")

**Reorder** (Claude-generated, 1 раз):
- Перестановка частей вопроса
- Passive ↔ active

**Метрики**:
- `perturbation_consistency = mean(BERTScore(answer_orig, answer_perturbed))` across all perturbations
- `perturbation_degradation = mean(BERTScore(answer_perturbed, expected))` — не ниже порога

**Pass/fail**: consistency ≥ 0.7, degradation не более -0.1 vs original.

**Dataset**: `datasets/eval_robustness_v1.json` — 50 Qs × 3 perturbations.
**Calls**: 50 × 3 = 150 (orig already from normal eval).

---

## Layer 3: Statistical Confidence

`scripts/compute_confidence.py`. Post-processing, zero extra calls.

### 3.1 Bootstrap CIs + Paired Bootstrap

**Per-metric CIs** (every eval):
```python
def bootstrap_ci(scores: List[float], B: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """BCa bootstrap confidence interval."""
    boot = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(B)]
    return np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
```

**Paired bootstrap для A/B** (comparing two system versions on same questions):
```python
def paired_bootstrap_test(scores_a: List, scores_b: List, B: int = 10000) -> float:
    """p-value: is system B better than A? Paired on same questions."""
    diffs = np.array(scores_b) - np.array(scores_a)
    observed = np.mean(diffs)
    boot_means = [np.mean(np.random.choice(diffs, len(diffs), replace=True)) for _ in range(B)]
    return np.mean(np.array(boot_means) <= 0)  # one-sided
```

### 3.2 Wilson Intervals для Binary Metrics

Для KTA, proxy-NDR, negative_rejection_rate, hit_rate:
```python
from statsmodels.stats.proportion import proportion_confint
lo, hi = proportion_confint(successes, total, alpha=0.05, method='wilson')
```

### 3.3 PPI (deferred)

**Status**: deferred из MVP. Требует prerequisites:
1. Synthetic dataset 200+ Qs (generation + quality filter)
2. Validation: automatic scores (BERTScore, SummaC) predictive of human labels на golden set
3. Distributional alignment: synthetic Qs ≈ golden Qs по category/difficulty

**Когда включать**: после Layer 1-2 стабилизированы и correlation BERTScore↔judge_factual подтверждена на ≥30 Qs.

**Implementation when ready**: `pip install ppi-python`, mean estimation over scalar labels.

---

## Dataset

### Golden v2: 36 Qs (existing)
Layer 1a automatic metrics — без изменений.

### Расширение до 50 Qs
+14 Qs для robustness coverage:
- 5 multi-hop (2+ subqueries)
- 3 temporal ranges
- 3 cross-channel
- 3 informal language / Telegram shorthand

### Robustness dataset: 50 Qs × 3 perturbations
`datasets/eval_robustness_v1.json`:
- Noise: программная генерация (Russian-aware: keyboard neighbors, Lat/Cyr mix)
- Substitution + reorder: Claude-generated, human-curated

---

## Compute Budget

| Layer | Calls | Time (~25s/call) | Frequency |
|-------|-------|---------|-----------|
| Layer 1a automatic | 0 LLM | ~5 min post-processing | Every eval |
| Layer 1b stability | 3 × 50 = 150 agent | ~63 min | Monthly |
| Layer 2 RGB noise | 50 × 4 = 200 agent | ~83 min | Per release |
| Layer 2 RGB rejection | 20 agent | ~8 min | Per release |
| Layer 2 proxy-NDR | 50 no-retrieval | ~21 min (reuse normal run) | Per release |
| Layer 2 query perturb | 50 × 3 = 150 agent | ~63 min | Per release |
| Layer 3 statistical | 0 | ~1 min | Every eval |
| **Total per release** | **~520 calls + 150 stability** | **~4 hours** | |

---

## Новые зависимости

| Package | Назначение | Size |
|---------|-----------|------|
| `bert-score` | BERTScore computation | ~1MB + model |
| `razdel` | Russian sentence tokenization | ~100KB |
| `ai-forever/ruBert-large` | BERTScore model | ~1.3GB |
| `statsmodels` | Wilson intervals | likely already installed |

PPI (`ppi-python`) и LettuceDetect — deferred, не в initial dependencies.

---

## Implementation Order

(По рекомендации Codex review #1, максимальный value first)

1. **Retrieval metrics + BERTScore + ToolCallF1 + report schema** — расширяем evaluate_agent.py, новые поля в aggregate report
2. **SummaC-ZS** — `src/services/eval/summac.py`, интеграция с evaluate_agent.py
3. **Bootstrap/Wilson CIs** — `scripts/compute_confidence.py`, paired bootstrap для A/B
4. **Robustness harness** — `scripts/evaluate_robustness.py`: doc injection + no-retrieval mode + checkpointing
5. **Query perturbation dataset** — Russian-specific noise generator + Claude perturbations
6. **Optional**: proxy-NDR refinement, LettuceDetect spike, PPI prerequisites

---

## Acceptance Criteria

### Layer 1a (always-on)
- [ ] evaluate_agent.py aggregate report содержит: Precision@5, MRR, nDCG@5, bertscore_f1, summac_faithfulness, tool_call_f1, trajectory_match
- [ ] BERTScore: ruBert-large загружается lazy, layer calibrated на dev-set (Spearman ρ с judge scores), IDF from 1000 domain docs, WordPiece aggregation implemented
- [ ] SummaC-ZS: razdel segmentation, batch NLI, explicit `summac_faithfulness` naming separate from `claim_faithfulness`
- [ ] ToolCallF1: scaffold tools excluded from precision, forbidden penalty = 0
- [ ] All Layer 1a metrics computed for all 36 golden Qs and reported in JSON + markdown

### Layer 1b (periodic)
- [ ] `--runs N` flag in evaluate_agent.py, per-question σ report
- [ ] Pass/fail: aggregate σ(bertscore_f1) < 0.05, per-question σ < 0.1

### Layer 2 (robustness)
- [ ] evaluate_robustness.py: RGB noise, RGB rejection, proxy-NDR, query perturbation
- [ ] RGB noise: irrelevant docs sampled with cosine_distance > 0.7 AND different channel, injection after search, 4 noise levels, BERTScore scoring
- [ ] RGB rejection: 20 answerable Qs, refusal marker detection, threshold ≥ 0.7
- [ ] Proxy-NDR: no-retrieval flag, BERTScore comparison, reported as "proxy-NDR" (NOT "Cao NDR")
- [ ] Query perturbation: noise generator with Russian keyboard neighbors + Lat/Cyr mix, consistency ≥ 0.7
- [ ] Dataset: eval_robustness_v1.json with 50 Qs × 3 perturbations
- [ ] Resume: per-question JSON checkpoints keyed by (question_id, test_type, variant)

### Layer 3 (statistical)
- [ ] compute_confidence.py: bootstrap CIs (B=1000) + Wilson for binary + paired bootstrap for A/B
- [ ] All aggregate metrics in eval report include 95% CI
- [ ] PPI: deferred, prerequisites documented

### Docs
- [ ] experiment_history.md: baseline all new metrics
- [ ] playbook: updated with new metric stack

---

## Что НЕ входит

- RSR/ROR (proxy-NDR покрывает главный вопрос, full Cao deferred)
- Full framework adoption (RAGAS/DeepEval end-to-end)
- Prometheus-2 / RAGChecker — Claude/Codex judge
- CI/CD integration — manual
- PPI — deferred до валидации prerequisites
- LettuceDetect — optional spike, не в MVP
- Расширение golden до 100 Qs — отдельная задача

---

## Ссылки

- R29: `docs/research/reports/R29-deep-comprehensive-rag-eval-metrics.md`
- R20: `docs/research/reports/R20-deep-retrieval-robustness-ndr-rsr-ror.md`
- R18: `docs/research/reports/R18-deep-evaluation-methodology-dataset.md`
- R15: `docs/research/reports/R15-yandex-rag-conference-2026.md`
- R05: `docs/research/reports/R05-rag-evaluation.md`
- Cao et al. (2025): arXiv:2505.21870
- Chen et al. RGB (2024): arXiv:2309.01431
- Laban et al. SummaC (2022): TACL
- Zhang et al. BERTScore (2020): arXiv:1904.09675
- Vetrov et al. (2022): arXiv:2203.05598
- Saad-Falcon et al. ARES/PPI (2024): arXiv:2311.09476
- SPEC-RAG-21: NLI citation faithfulness (existing)
- SPEC-RAG-14: Evaluation pipeline (existing)
