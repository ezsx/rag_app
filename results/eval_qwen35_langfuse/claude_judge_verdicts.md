# Claude Judge Verdicts — Qwen3.5-35B-A3B + Langfuse run (2026-03-30)

**Judge**: Claude Opus 4.6
**Dataset**: golden_v2 (36 Qs)
**Model**: Qwen3.5-35B-A3B-Q4_K_M

## Scoring conventions
- **Factual** (0-1): 1.0=all claims correct, 0.5=partially correct/ungrounded, 0=wrong/hallucinated
- **Usefulness** (0-2): 2=complete+actionable, 1=partial/missing key info, 0=unhelpful/wrong
- Refusal questions: correct refusal = factual 1.0, useful 2.0

## Verdicts

| Q | Mode | Factual | Useful | Notes |
|---|------|---------|--------|-------|
| q01 | retrieval | 1.0 | 2.0 | Jensen Huang, NVIDIA, FT — all correct. Citations. |
| q02 | retrieval | 1.0 | 2.0 | 120B/20B params, MoE, active params — all correct. |
| q03 | retrieval | 1.0 | 2.0 | $2B acquisition — correct. Details about financing round. |
| q04 | retrieval | 1.0 | 2.0 | V3.1, mHC mentioned. Comprehensive. |
| q05 | retrieval | 0.5 | 1.0 | HunyuanImage 3.0 found but missed FLUX.2 and Kandinsky 5.0 from expected. Partial. |
| q06 | retrieval | 0.5 | 1.0 | Good coverage but missed gonzo_ml specific topics (agents, Frankenstein effect). Missed boris_again Gemini 3. |
| q07 | retrieval | 1.0 | 2.0 | GTC 2026, Vera Rubin — correct. Rich details. |
| q08 | retrieval | 0.5 | 1.0 | OpenTalks.AI mentioned, but missed Opus 4.6, Sebrant, Agibot robots from expected claims. |
| q09 | retrieval | 1.0 | 2.0 | Developer role instruction, reasoning channels, "juice" concept — all correct. |
| q10 | retrieval | 0.5 | 1.0 | Discusses transformers vs recurrent but doesn't mention AlphaGenome specifically. |
| q11 | retrieval | 1.0 | 2.0 | "Самое близкое к автономной модели" — exact quote. Citations. |
| q12 | retrieval | 0.5 | 1.0 | Mentions GPT-5 discussions but doesn't specifically state "GPT-5.3 and GPT-5.4 released 2 days apart". |
| q13 | retrieval | 1.0 | 2.0 | 3+ channels discussed, $2B, detailed per-channel analysis. |
| q14 | retrieval | 1.0 | 2.0 | Multi-channel comparison. Technical details and mHC mentioned. |
| q15 | retrieval | 1.0 | 2.0 | Recent techsparks posts, OpenAI hiring, specific dates. |
| q16 | retrieval | 1.0 | 2.0 | Good monthly digest, multiple topics covered. |
| q17 | navigation | 0.5 | 0.0 | Didn't use list_channels. Found some channels via search but incomplete list, no post counts. Wrong approach. |
| q18 | navigation | 1.0 | 2.0 | 275 posts (expected 261 — data may have been refreshed). Correct tool and format. |
| q19 | refusal | 1.0 | 2.0 | Correct refusal. "GPT-7 не найдена". |
| q20 | refusal | 1.0 | 2.0 | Correct refusal. "Bard 3 отсутствует". Clean. |
| q21 | refusal | 1.0 | 1.0 | Correct refusal ("апрель 2024 вне базы") but did unnecessary search first. -1 useful for wasted step. |
| q22 | analytics | 1.0 | 2.0 | 466 mentions, weekly dynamics, peak W06. Good analytics. |
| q23 | analytics | 1.0 | 2.0 | arxiv:2502.13266 top (4 mentions). Correct data. |
| q24 | analytics | 0.5 | 1.0 | Used channel_expertise but said "no channel specializes exclusively". Missed techsparks. |
| q25 | retrieval | 0.5 | 1.0 | Good general answer but missed specific Schema-Guided Reasoning and any2json benchmark. |
| q26 | analytics | 1.0 | 2.0 | OpenAI 1662, Google 1171, Anthropic — correct ranking. |
| q27 | analytics | 1.0 | 2.0 | OpenAI 125, Google 101, Microsoft 50 co-occurrences — specific numbers. |
| q28 | analytics | 1.0 | 2.0 | OpenAI >> DeepSeek, specific counts and peaks. |
| q29 | analytics | 1.0 | 2.0 | Same as q23. Correct arxiv data. |
| q30 | analytics | 1.0 | 2.0 | gonzo_ml, 2 posts, specific date. Correct lookup. |
| q31 | analytics | 1.0 | 2.0 | GPT-5, Apple — correct hot topics W10. 325 posts. |
| q32 | analytics | 1.0 | 2.0 | Hot topics → final_answer. Fixed: no unnecessary search pipeline. |
| q33 | analytics | 1.0 | 2.0 | GPT-5, Apple products — correct March topics. |
| q34 | analytics | 1.0 | 2.0 | xor_journal top NLP channel (0.77 authority). |
| q35 | analytics | 1.0 | 2.0 | channel_expertise only. Fixed: no unnecessary search pipeline. |
| q36 | analytics | 1.0 | 2.0 | xor_journal for robotics. Correct channel_expertise data. |

## Consensus Metrics (Claude + Codex, after q24/q33 rerun)

Conservative consensus: min(Claude, Codex) per question.

Factual consensus (min of two judges):
- 1.0: q01,q02,q03,q07,q09,q11,q13,q18,q19,q20,q21,q22,q23,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36 = 24
- 0.5: q04,q05,q06,q08,q10,q12,q14,q15,q16,q17,q24,q25 = 12
- 0.0: 0

**Consensus Factual: (24×1.0 + 12×0.5) / 36 = 30/36 = 0.833**

Useful consensus (min of two judges):
- 2.0: q01,q02,q03,q07,q09,q11,q13,q18,q19,q20,q22,q23,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36 = 23
- 1.0: q04,q05,q06,q08,q10,q12,q14,q15,q16,q21,q24,q25 = 12
- 0.0: q17 = 1

**Consensus Useful: (23×2 + 12×1 + 1×0) / 36 = 58/72 = 1.611**

## Summary

| Metric | Qwen3 baseline | Qwen3.5 + fixes (consensus) |
|--------|----------------|-------------------------------|
| Factual | ~0.80 | **0.833** (+4%) |
| Usefulness | ~1.53 | **1.611** (+5%) |
| KTA | 1.000 | **0.970** (35/36) |
| Mean Latency | N/A (timeout'ы) | **26.4s** |
| P95 Latency | N/A | **47.5s** |
| Eval run time | ~40+ min | **15.8 min** |
| Hallucinations | unknown | **0** |
| Planner works | ❌ (39s timeout) | ✅ (4s) |
| Subqueries | ❌ (fallback 1q) | ✅ (3-5 subqueries) |
| Observability | ❌ none | ✅ Langfuse traces |

## Key improvements vs previous run
1. Zero hallucinations across all 36 questions
2. Planner 10x faster (chat_completion fix, 39s → 4s)
3. Multi-query retrieval (3-5 subqueries vs 1 fallback)
4. Eval run 2.5x faster (15.8 min vs ~40+ min)
5. Full Langfuse observability with parent-child trace trees
6. Analytics tools work correctly (month aggregation, channel expertise)

## Remaining issues
1. q17: list_channels tool hidden — visibility routing issue (1 KTA miss)
2. q24: model hedges on channel_expertise output (partial pass)
3. 12 partial answers (factual 0.5) — missing specific expected claims, not hallucination
4. Codex stricter on retrieval claims coverage (q04,q14,q15,q16)
