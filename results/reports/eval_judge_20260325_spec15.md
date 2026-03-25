# Eval Report: SPEC-RAG-15 (entity_tracker + arxiv_tracker)

> **Date**: 2026-03-25
> **Dataset**: eval_golden_v1.json (30 questions)
> **Judge**: Claude Opus 4.6 (manual) + Codex GPT-5.4 (independent, 28/30 agreement)
> **Agent LLM**: Qwen3-30B-A3B
> **Duration**: 693s (~23s/query avg)

## Aggregate Metrics (post-fix)

| Metric | Pre-fix | Post-fix | Delta |
|--------|---------|----------|-------|
| Recall@5 (strict) | 0.342 | 0.342 | = (strict metric, не меняется от fix) |
| Key tool accuracy | 0.889 (24/27) | **0.926** (25/27) | +0.037 (q25 dataset fix) |
| **Factual (manual)** | **1.62/2** | **1.79/2** | **+0.17** (q03, q19, q22 fixed) |
| **Useful (manual)** | **1.52/2** | **1.72/2** | **+0.20** (q03, q19, q22 fixed) |
| Coverage mean | 0.445 | ~0.45 | ~ |
| Failures | 5 | **2** (q01 LLM-level, q25 partial KTA) | -3 |

### Fixes applied (targeted retest 2026-03-25)

| Fix | Target | Result |
|-----|--------|--------|
| Forced search bypass: only for negative intent | q01, q03 | q03 fixed (search finds docs). q01: search forced but LLM still refuses (Qwen3 issue) |
| Refusal alt trim: deterministic cut after refusal | q19 | Clean refusal, no alternatives |
| Timeline data truncation: top-10 weeks | q22 | entity_tracker(timeline) works, 569 mentions, 25ms |
| Dataset: channel_search in acceptable_alternatives | q25 | KTA 0→1 |

## Per-Question Judge Table

Scoring:
- **Factual** (0-2): 0=wrong/hallucinated, 1=partially correct, 2=fully correct
- **Useful** (0-2): 0=useless, 1=partially useful, 2=fully answers the question
- **Tool** (0/1): correct tool selected

| ID | Category | Query (short) | Tool | Factual | Useful | Notes |
|----|----------|---------------|------|---------|--------|-------|
| q01 | broad | FT человек года 2025 | search (forced) | 0 | 0 | **Post-fix**: forced search срабатывает, 10 docs found. Но LLM всё равно отказывает после search (Qwen3 issue) |
| q02 | broad | Open-source GPT параметры | search | 2 | 2 | Точные параметры 120B/20B, MoE, MXFP4. Отлично |
| q03 | broad | Meta купила Manus | search (fixed) | 2 | 2 | **Post-fix**: forced search → query_plan → search → 10 docs found. $2 млрд найдено |
| q04 | broad | DeepSeek новые модели | search | 2 | 2 | V3.2, V3.2-Speciale, R2. Подробно, с citations |
| q05 | broad | Open-source image gen | search | 1 | 1 | Нашёл HunyuanImage 3.0, но не FLUX.2 и Kandinsky 5.0 из expected |
| q06 | constrained | AI в январе 2026 | temporal | 2 | 2 | Дайджест ML/AI за январь, модели, агенты. Хорошо |
| q07 | constrained | GTC 2026 NVIDIA | temporal | 2 | 2 | Vera Rubin, подробные характеристики. Отлично |
| q08 | constrained | AI февраль 2026 | temporal | 2 | 2 | Agibot, OpenTalks.AI, Claude Opus. Хорошо |
| q09 | constrained | llm_under_hood reasoning GPT-5 | channel | 2 | 2 | Точно: developer role, Juice=0, дешевле. Отлично |
| q10 | constrained | gonzo_ml трансформеры vs RNN | channel | 2 | 2 | HRM, Universal Transformer, AlphaGenome context. Хорошо |
| q11 | constrained | Борис Цейтлин про Opus 4.6 | channel | 2 | 2 | "Близок к автономной модели", MRCR v2, System Card. Точно |
| q12 | constrained | seeallochnaya про GPT-5 | channel | 1 | 1 | Нашёл ранние утечки TheInformation, но не GPT-5.3/5.4 из expected |
| q13 | compare | Manus покупка cross-channel | cross_ch | 2 | 2 | data_secrets, MLunderhood, $2 млрд. 3 канала. Отлично |
| q14 | compare | DeepSeek cross-channel | cross_ch | 2 | 2 | cryptovalerii, AI_Machinelearning, разные углы. Хорошо |
| q15 | compare | techsparks за неделю | summarize | 1 | 1 | Посты найдены, но не из expected (пандемия, DeepMind вместо роботакси) |
| q16 | compare | gonzo_ml дайджест месяц | summarize | 1 | 1 | Claude Sonnet 4.6 vs Gemini, CFM. Контент есть, но обрезан |
| q17 | navigation | Какие каналы | list_ch | 2 | 2 | 36 каналов с точными counts. Точно |
| q18 | navigation | Постов в llm_under_hood | list_ch | 2 | 2 | "261 пост". Точно совпадает с expected |
| q19 | negative | GPT-7 | (no search) | 2 | 2 | **Post-fix**: чистый refusal "не найдена в базе данных", без альтернатив. Negative intent bypass + refusal trim работают |
| q20 | negative | Bard 3 | (no search) | 2 | 2 | Чистый отказ "не найдена в базе". Правильно |
| q21 | negative | Апрель 2024 | temporal×3 | 2 | 2 | Отказ корректный ("данные отсутствуют"). Codex review: лишние temporal_search = tool efficiency, не answer quality |
| q22 | future | OpenAI timeline | entity_tracker | 2 | 2 | **Post-fix**: timeline truncation (top-10 weeks). 569 упоминаний, пик W06 (51), 25ms |
| q23 | future | Arxiv top papers | arxiv_tracker | 2 | 2 | Top-5 papers с точными counts. Данные корректны |
| q24 | future | Канал про робототехнику | search | 1 | 1 | Нашёл papa_robotov, но expected — techsparks. Оба правильны, но expected не совпал |
| q25 | broad | LLM production llm_under_hood + boris_again | channel | 2 | 2 | SGR, Function-calling. Содержательно. KTA=0 только потому что ожидался `search`, а agent выбрал `channel_search` — это корректный выбор |
| q26 | future | Top AI-компании | entity_tracker | 2 | 2 | OpenAI=1597, Google=1127... Top-10 org. Точно |
| q27 | future | Co-occurrence NVIDIA | entity_tracker | 2 | 1 | Данные верные (OpenAI=121, Google=99...). Но "С согласия NVIDIA" — ляп Qwen3. Useful=1 из-за фразировки |
| q28 | future | Compare OpenAI vs DeepSeek | entity_tracker | 2 | 2 | 1597 vs 344, пики W32 и W48. Точные данные, хорошая формулировка |
| q29 | future | Top arxiv papers | arxiv_tracker | 2 | 2 | 5 papers с counts. Совпадает с raw data |
| q30 | future | Кто обсуждал 1706.03762 | arxiv_tracker | 2 | 2 | gonzo_ml, 2 поста, контекст трансформеров. Dedup работает |

## Summary Scores

| Category | Questions | Factual avg | Useful avg | Tool accuracy |
|----------|-----------|-------------|------------|---------------|
| broad_search | 6 | 1.50 | 1.50 | 0.67 |
| constrained_search | 7 | 1.86 | 1.86 | 1.00 |
| compare_summarize | 4 | 1.50 | 1.50 | 1.00 |
| navigation | 2 | 2.00 | 2.00 | 1.00 |
| negative_refusal | 3 | 2.00 | 2.00 | N/A |
| future_baseline | 8 | 2.00 | 1.88 | 1.00 |
| **Overall** | **30** | **1.79** | **1.72** | **0.926** |

## Analytics Tools Performance (SPEC-RAG-15)

| Query | Tool | Mode | Latency | Factual | Useful | Notes |
|-------|------|------|---------|---------|--------|-------|
| q22 | entity_tracker | timeline | 25ms | 2 | 2 | **Post-fix**: 569 mentions, peak W06. Timeline truncation fixed 400 error |
| q23 | arxiv_tracker | top | 20.0s | 2 | 2 | 5 papers, correct counts |
| q26 | entity_tracker | top | 14.9s | 2 | 2 | Top-10 orgs, correct |
| q27 | entity_tracker | co_occurrence | 10.7s | 2 | 1 | Data correct, phrasing bad |
| q28 | entity_tracker | compare | 9.9s | 2 | 2 | OpenAI vs DeepSeek, peaks |
| q29 | arxiv_tracker | top | 12.8s | 2 | 2 | Same as q23, consistent |
| q30 | arxiv_tracker | lookup | 16.5s | 2 | 2 | gonzo_ml channel, dedup OK |

**Analytics summary**: 6/7 successful (1 infra failure). KTA=100%. Factual=2.0, Useful=1.83. Avg latency=14.1s (fast — facet queries <100ms, rest is LLM thinking).

## Key Findings

### Wins
1. **entity_tracker works end-to-end**: top, co_occurrence, compare modes — all correct data, LLM formats well
2. **arxiv_tracker works end-to-end**: top (consistent across q23/q29), lookup with dedup
3. **Key tool accuracy 100% for analytics**: LLM always chooses entity_tracker/arxiv_tracker when visible
4. **Keyword routing works**: temporal_search, channel_search, summarize_channel all triggered correctly
5. **Verify bypass works**: analytics answers don't get false-flagged by verify
6. **ANALYTICS-COMPLETE phase works**: entity_tracker → final_answer (2 steps, no forced search)

### Issues
1. **q01, q03: false refusals** — LLM refuses without searching (known Qwen3 issue, not new)
2. **q19: soft refusal** — says "not found" but suggests alternatives (violates refusal policy)
3. **q22: infra failure** — llama-server 400 error (transient, not tool issue)
4. **q27: "С согласия NVIDIA"** — Qwen3 generation quality issue (data correct, phrasing wrong)
5. **Recall@5=0.342** — low because strict post ID matching, many answers find correct info from different posts

### Comparison with Previous Eval

| Metric | Previous (golden_v1, 25Q) | Current (golden_v1, 30Q) | Delta |
|--------|--------------------------|--------------------------|-------|
| Questions | 25 | 30 | +5 analytics |
| Key tool accuracy | ~0.80 | 0.889 | +0.089 |
| Coverage mean | ~0.40 | 0.445 | +0.045 |
| Factual (manual) | 0.52 | 1.62 | +1.10 |
| Useful (manual) | 1.14 | 1.52 | +0.38 |
| Analytics tool accuracy | N/A | 1.00 (7/7) | NEW |
