## Codex Judge Verdicts

Source artifact judged independently: `src/results/eval_qwen35_langfuse/raw/eval_results_20260330-120258.json`

Aggregate:

- Factual: `0.792`
- Useful: `1.611 / 2`

| Q | Mode | Factual | Useful | Notes |
|---|------|---------|--------|-------|
| golden_q01 | retrieval_evidence | 1.0 | 2.0 | Correct and grounded; required FT/Jensen Huang claim is covered. |
| golden_q02 | retrieval_evidence | 1.0 | 2.0 | Fully covers GPT-OSS-120B/20B, MoE active params, MXFP4 and Apache 2.0. |
| golden_q03 | retrieval_evidence | 1.0 | 2.0 | Correctly answers `$2B` acquisition and gives grounded supporting details. |
| golden_q04 | retrieval_evidence | 0.5 | 1.0 | On-topic but misses required V3+mHC framing; mostly shifts to V3.1/V3.2/Speciale. |
| golden_q05 | retrieval_evidence | 0.5 | 1.0 | Covers Kandinsky 5.0 but misses required FLUX.2 mention. |
| golden_q06 | retrieval_evidence | 0.5 | 1.0 | January summary is grounded but misses expected gonzo_ml and any2json/Gemini claims. |
| golden_q07 | retrieval_evidence | 1.0 | 2.0 | Clearly covers GTC 2026 and Vera Rubin as the main NVIDIA announcement. |
| golden_q08 | retrieval_evidence | 0.5 | 1.0 | Includes Opus 4.6 and Agibot, but misses Sebrant AI-trends angle from expected claims. |
| golden_q09 | retrieval_evidence | 1.0 | 2.0 | Correctly describes disabling GPT-5 reasoning via developer-role instruction and Juice. |
| golden_q10 | retrieval_evidence | 0.5 | 1.0 | Correct on RNN limitations vs transformers, but required AlphaGenome claim is absent. |
| golden_q11 | retrieval_evidence | 1.0 | 2.0 | Directly covers Boris Zeitlin's “closest to autonomous model” claim. |
| golden_q12 | retrieval_evidence | 0.5 | 1.0 | Discusses GPT-5 broadly, but misses required GPT-5.3/GPT-5.4 two-day release detail. |
| golden_q13 | retrieval_evidence | 1.0 | 2.0 | Strong comparative answer across multiple channels with the expected Manus angles. |
| golden_q14 | retrieval_evidence | 0.5 | 1.0 | Broad comparison is useful, but misses required seeallochnaya V3/mHC and security-attack framing. |
| golden_q15 | retrieval_evidence | 0.5 | 1.0 | Valid weekly digest, but not the expected AI-trends/robotaxi/marketing emphasis. |
| golden_q16 | retrieval_evidence | 0.5 | 1.0 | Monthly digest is grounded, but misses required transformers/RNN/AlphaGenome focus. |
| golden_q17 | navigation | 0.5 | 1.0 | Answers with some channels, but uses retrieval path instead of navigation/meta capability and is not comprehensive. |
| golden_q18 | navigation | 1.0 | 2.0 | Correct navigation path via `list_channels`; answer matches current tool output. |
| golden_q19 | refusal | 1.0 | 2.0 | Correct refusal for out-of-database question. |
| golden_q20 | refusal | 1.0 | 2.0 | Correct refusal for out-of-database question. |
| golden_q21 | refusal | 1.0 | 2.0 | Correct refusal for out-of-timerange query; answer explicitly states dataset range. |
| golden_q22 | analytics | 1.0 | 2.0 | Correct analytics path via `entity_tracker`; timeline answer matches tool output. |
| golden_q23 | analytics | 1.0 | 2.0 | Correct `arxiv_tracker` top-list answer with expected sparse counts. |
| golden_q24 | analytics | 0.0 | 0.0 | Tool output finds relevant channels, but final answer incorrectly says no specialized channel was found. |
| golden_q25 | retrieval_evidence | 0.5 | 1.0 | Covers SGR and production patterns, but misses expected any2json and Technical AI Safety specifics from `boris_again`. |
| golden_q26 | analytics | 1.0 | 2.0 | Correct top-company ranking from `entity_tracker`. |
| golden_q27 | analytics | 1.0 | 2.0 | Correct co-occurrence answer from `entity_tracker`; NVIDIA pairings are consistent with tool output. |
| golden_q28 | analytics | 1.0 | 2.0 | Correct OpenAI vs DeepSeek comparison with counts and peak timing. |
| golden_q29 | analytics | 1.0 | 2.0 | Correct `arxiv_tracker` ranking answer. |
| golden_q30 | analytics | 1.0 | 2.0 | Correct `arxiv_tracker` lookup answer for paper `1706.03762`. |
| golden_v2_q31 | analytics | 1.0 | 2.0 | Correct weekly `hot_topics` answer for `2026-W10`, with multiple real topics from digest output. |
| golden_v2_q32 | analytics | 1.0 | 2.0 | Correct `hot_topics`-driven weekly answer for `2026-W11`; grounded and detailed. |
| golden_v2_q33 | analytics | 0.0 | 1.0 | Uses `hot_topics`, but tool output resolves to `2026-W10` instead of month aggregation for March 2026. |
| golden_v2_q34 | analytics | 1.0 | 2.0 | Correct `channel_expertise` topic-mode answer; top NLP channels are aligned with tool output. |
| golden_v2_q35 | analytics | 1.0 | 2.0 | Correct profile-style expertise answer for `gonzo_ml`, with analytics lookup plus grounding. |
| golden_v2_q36 | analytics | 1.0 | 2.0 | Correct `channel_expertise` topic-mode answer for robotics/robotaxi channels. |
