# Next Steps — Roadmap to Senior-Level Portfolio

> Scope задач после ablation study + README polish.
> **Tier 1+3 + lite Tier 2 = ready for outreach (3-4 дня).** Не блокировать job search остальными тирами.
> Tier 4-6 делать параллельно с активным поиском.
> Последнее обновление: 2026-04-09

---

## Current State

- Factual: **0.858** with **95% CI [0.792, 0.917]** (36 Qs, cross-family judge: Claude Opus + GPT-5.4 + manual calibration)
- Useful: **1.71/2** with **95% CI [1.606, 1.803]**, Faithfulness: **0.91**, Robustness: **0.954**, R@5: **0.900**
- Benchmark: **+0.30 factual** vs LlamaIndex on identical data
- 39 ablation experiments, 8 formal runs (RUN-001–RUN-008)
- README: updated with hook, latency decomposition, mermaid diagrams, security section

---

## Tier 1 — README Visual Proof (1 день)

Текст README готов. Осталось визуальное:

- [ ] **GIF SSE-streaming** (10-15 сек) — живой запрос в Web UI, SSE streaming ответа с citations
- [ ] **Скриншот Langfuse trace** — реальный retrieval query (q01 или q02), показать span tree с таймингами
- [ ] **Заменить ASCII observability** на скриншот Langfuse
- [x] **Заполнить concrete example** реальными числами из trace (RUN-008 q01)
- [x] **Создать файл LICENSE** (Apache 2.0) в корне
- [x] **Shields.io badges** для метрик в шапке README

---

## Tier 2 — Статистика (полдня, 0 compute)

Bootstrap CI и significance tests — превращает "0.858" в "0.858 ± 0.063 (95% CI [0.792, 0.917])".

**Делать дважды:** сейчас на n=36 с honest "wide CIs due to small n", потом повторить после Tier 4 на расширенном датасете. Не блокировать outreach ожиданием tight numbers.

- [x] **Bootstrap CI** для baseline judge metrics через `scripts/compute_confidence.py` (`RUN-008`: factual `0.858 ± 0.063`, useful `1.708 ± 0.099`)
- [ ] **Paired significance test** для benchmark custom vs LlamaIndex (paired bootstrap, p-value)
- [ ] **Multiple comparisons correction** в ablation (Benjamini-Hochberg на 39 экспериментов)
- [ ] **Cohen's kappa** между Claude и GPT-5.4 judge на 30-50 hard cases (`sklearn.metrics.cohen_kappa_score`). kappa > 0.7 = substantial agreement — железный аргумент про judge reliability
- [x] **Обновить README** — добавить CI к ключевым метрикам
- [ ] **Обновить ablation_study.md** — пометить какие changes значимы после correction
- [ ] **Повторить** на v3 dataset после Tier 4 — CIs сузятся вдвое

---

## Tier 3 — Production Narrative (1 день, 0 compute)

Документация демонстрирующая production thinking без production deployment.

- [ ] **docs/INCIDENTS.md** — 3 postmortems:
  - CE URL setdefault bug (14 queries timeout, 76 min wasted compute)
  - Embedding prefix not removed from production settings.py
  - CE sigmoid bug (fixed in diagnosis phase)
  - Format: что сломалось -> как обнаружили -> root cause -> systemic fix
- [ ] **docs/FAILURE_MODES.md** — what happens when:
  - gpu_server.py crashes mid-request
  - Qdrant unavailable
  - llama-server OOM
  - LLM returns malformed tool call JSON
  - Query in unsupported language
  - Rate limit exceeded
  - For each: expected behavior, graceful degradation, recovery
- [ ] **Cost model** в README или отдельным документом:
  - Self-hosted: cost per query (V100 watt-hours, electricity)
  - Managed API comparison (Claude Sonnet / GPT-4o per query)
  - Break-even calculation

---

## Tier 4 — Dataset Expansion (3-5 дней)

Расширение golden set для статистической значимости.

- [ ] **Golden set v3: 150-200 Qs** стратифицированный:
  - Retrieval: 50% (~100 Qs)
  - Analytics: 25% (~50 Qs)
  - Navigation: 5-10% (~15 Qs)
  - Refusal: 5-10% (~15 Qs)
  - Adversarial: ~10 Qs — prompt injection, tool abuse, jailbreak. Expected: жёсткий refusal
  - Edge cases: ~10 Qs — ambiguous, mixed language, multi-tool boundary, scope boundary. Expected: best-effort + caveats
- [ ] **Inter-annotator agreement** на hard cases — Claude vs GPT-5.4 independent expected answers, **Cohen's kappa** на 30-50 hard cases
- [ ] **Прогон + judge** на расширенном датасете
- [ ] **Повторить Tier 2** (bootstrap CI, significance tests) — CIs сузятся вдвое на n=150+
- [ ] **Обновить README** с новыми метриками и tight CI

---

## Tier 5 — Fine-tuning Reranker ($10-20 compute)

LoRA fine-tuning Qwen3-Reranker-0.6B на domain-specific hard negatives.

### Подготовка (0 compute, делать заранее):
- [ ] **Hard negatives** из eval logs — CE low + cosine high. Python скрипт по артефактам. Цель: 500-1000 пар
- [ ] **Positives** из successful runs — CE high + cosine high. Цель: 500-1000 пар
- [ ] **Eval baseline script** — прогоняет текущий CE на golden set, подтверждает цифры. **Зафиксировать seed + config — immutability между baseline и fine-tuned runs** (та же логика что parity check в experiment protocol)
- [ ] **Training script** — LoRA rank=16, margin loss, тест на CPU с batch=1 и n_steps=5
- [ ] **spec.yaml** по experiment protocol с гипотезой и expected outcome ДО аренды compute

### Execution ($10 compute):
- [ ] Арендовать H100 (vast.ai / runpod), ~$2-3/час
- [ ] Залить данные, запустить training (20-40 мин)
- [ ] Скачать checkpoint, освободить машину
- [ ] Eval с новым checkpoint vs baseline
- [ ] Записать результат (adopt/reject)

### Expected outcome:
- Optimistic: factual +0.02-0.03 (30% probability)
- Realistic: factual +0.01-0.015 (50% probability)
- Pessimistic: no significant improvement (20% probability — still valid as "what didn't work" entry)

---

## Tier 6 — Technical Writeup + Publication

- [ ] **Статья 2500-3500 слов** — "Building a production RAG that beats LlamaIndex: lessons from 57 experiments"
  - Hook: BERTScore failure as proxy metric
  - Core: ablation methodology, what worked and what didn't
  - Insight: cross-family judge consensus, LLM judge calibration
  - Practical: self-hosted stack tradeoffs
- [ ] **Публикация**: Habr (русскоязычная аудитория), Medium (англоязычная), r/LocalLLaMA
- [ ] **Ссылка в README** в хук сверху

---

## Ongoing — AI Workflow Governance (research track)

Отдельный research track по structured workflows для AI coding agents.

Findings так далеко:
- McKinsey QuantumBlack: deterministic orchestration + agentic execution
- cc-sdd, Pimzino spec-workflow: slash commands с validator subagents
- disler Builder/Validator: tool restrictions + Stop hooks
- systematic-debugging skill: 4 phases, Iron Law, 3-Fix Rule
- Exploration workflows: open problem

Next:
- [ ] Изучить cc-sdd и Pimzino подробнее (установить, прочитать validator промпты)
- [ ] Попробовать минимальный набор Claude Code hooks (Stop + PreToolUse)
- [ ] Решить: hooks vs slash commands vs LangGraph для нашего workflow
- [ ] Оформить как отдельную спеку если решим внедрять

---

## Done (this session, 2026-04-08/09)

- [x] RUN-004: compose_context 1800->4000, 36q judge (factual 0.83 raw → 0.858 corrected). Rejected as standalone, budget kept
- [x] RUN-005: channel dedup 2->3. Adopted — saves 3rd doc from same channel
- [x] RUN-006: dual scoring (norm_linear + rrf_ranks). Rejected — breaks CE gap detection
- [x] RUN-007: cosine recall guard. Adopted — CE precision + bi-encoder recall
- [x] RUN-008: full 36q eval with all adopted changes. Factual 0.858 (corrected), refusal 3/3
- [x] Negative intent fix — markers reduced to temporal only
- [x] Dataset audit — 7/36 open-ended questions with narrow expected answers fixed (v2_fixed)
- [x] Eval tooling — live judge_live.md, progress.log, auto judge export, rate limit bypass
- [x] README polish — hook, latency decomposition, cross-judge, mermaid diagrams, security, license
- [x] ruflo research — vaporware, не ставим
- [x] Deep research — AI workflow governance, structured workflows, McKinsey/cc-sdd/Pimzino/disler
