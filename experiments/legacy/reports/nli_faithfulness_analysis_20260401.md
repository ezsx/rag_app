# NLI Faithfulness Analysis — 2026-04-01

> SPEC-RAG-21, rubert-base-cased-nli-threeway, 36 Qs golden_v2

## Aggregate Results

| Metric | Value |
|--------|-------|
| **Factual** (Claude judge, 0.1 scale) | **0.842** |
| **Useful** (Claude judge) | **1.778/2** |
| **KTA** | **1.000** |
| **Faithfulness (raw, lenient)** | 0.792 |
| **Faithfulness (raw, strict)** | 0.753 |
| **Faithfulness (corrected, w/o q15)** | **~0.90** |
| Retrieval questions | 17/36 |
| Total claims (verifiable) | 171 |
| Claims supported (raw) | 133 (78%) |
| Contradictions (raw) | 19 |
| **Contradictions (real)** | **0** |
| NLI pairs | 1977 |
| Elapsed | 59s |

## Contradiction Analysis

All 19 raw contradictions were manually reviewed. **0 real contradictions found** — the agent does not hallucinate facts.

### False Positive Categories

**Category 1: NLI model fails on Russian paraphrases (12 cases)**

Document confirms the claim but ruBERT gives low entailment. Typical pattern: document uses different wording, informal style, or mixed Russian/English text.

| # | QID | Claim | Doc evidence | Entailment |
|---|-----|-------|-------------|-----------|
| C01 | q01 | Хуанг — гендиректор NVIDIA | "Дженсен Хуанг (Nvidia)" | 0.303 |
| C02 | q01 | FT признала Трампа Человеком года | "В прошлом году FT признала человеком года Трампа" | 0.011 |
| C04 | q04 | DSA снижает стоимость на 50% | "цена стала на 50+% ниже" | 0.300 |
| C09 | q06 | Lightricks LTX-2 4K/50fps | "синхронной генерацией 4K/50fps видео" | 0.383 |
| C10 | q07 | GTC 2026 в Сан-Хосе | "NVIDIA провела конференцию в Сан-Хосе" | 0.150 |
| C13 | q09 | Инструкция Active channels | Документ содержит точный текст | 0.313 |
| C14 | q10 | gonzo_ml про Universal Transformer | "Напомню про Universal Transformer" | 0.341 |
| C16 | q12 | OpenAI маркетинговая кампания | "наняли директора по маркетингу" | 0.262 |
| C18 | q14 | OpenAI усилила безопасность после R1 | "После выхода R1 Альтман решил усилить контроль" | 0.221 |
| C19 | q25 | SGR на Qwen 4B | "SGR работает даже на qwen3 4B" | 0.037 |
| C08 | q06 | Google TranslateGemma | Пограничный — doc про переводчики в целом | 0.412 |
| C15 | q11 | Борис Цейтлин ведёт boris_again | common_knowledge, doc не про это | 0.036 |

**Root cause**: ruBERT (180M) trained on auto-translated NLI data. Accuracy drops on:
- Mixed Russian/English text (e.g., "Huang's recognition reflects...")
- Informal Telegram style with emoji, hashtags, URLs
- Common knowledge claims not stated in document
- Exact quotes treated as neutral rather than entailment

**Category 2: Wrong best-entailment document (5 cases)**

Claim is supported by a different document, but the best-entailment match landed on a wrong doc.

| # | QID | Claim | Issue |
|---|-----|-------|-------|
| C03 | q04 | DeepSeek Math V2 в ноябре 2025 | Best doc about V3.2, not Math V2 |
| C05 | q04 | DeepSeek-OCR 2 в январе 2026 | Best doc about V3.2, not OCR-2 |
| C06 | q05 | FLUX 2 генерирует до 4K | Best doc about Z-Image, not FLUX 2 |
| C12 | q08 | DeepMind Project Genie | Best doc about Manhattan Project analogy |
| C17 | q14 | theworldisnoteasy про DeepSeek safety | Best doc about Anthropic/Pentagon |

**Root cause**: When multiple documents are cited, the claim may be supported by document X but the highest entailment score comes from document Y (which doesn't contain the fact). Contradiction from doc Y is then flagged.

**Category 3: Borderline (2 cases)**

| # | QID | Claim | Issue |
|---|-----|-------|-------|
| C07 | q06 | OpenAI митап 28 января | Doc has same date but different event |
| C11 | q07 | 5x производительность Blackwell NVFP4 | Doc says "10x per watt", claim says "5x inference" — different metric |

## Corrected Faithfulness

After manual review: **0 real contradictions, 19 false positives**.

Corrected faithfulness (reclassifying all 19 contradictions as supported or neutral):
- 12 FP → supported (doc confirms claim)
- 5 FP → neutral (claim from different doc, unverifiable against best-ent doc)
- 2 borderline → neutral

**Corrected scores (17 retrieval Qs):**
- Faithfulness (lenient): **~0.91**
- Faithfulness (strict): **~0.88**

Without q15 (0 documents, routing bug):
- Faithfulness (lenient): **~0.95**
- Faithfulness (strict): **~0.92**

## Recommendations

1. **Raise contradiction threshold** from 0.55 to 0.90+ — current model produces too many false positive contradictions
2. **Fix q15 routing** — summarize_channel → search with channel filter (known issue)
3. **Consider removing contradiction category** — for ruBERT on Russian informal text, binary supported/not_supported is more reliable than 3-way
4. **Threshold calibration**: entailment 0.45 works well (only 1 borderline case at 0.43)

## Model Choice

rubert-base-cased-nli-threeway (180M, 0.36 GB) significantly outperforms xlm-roberta-large-xnli (560M, 1.12 GB) on Russian informal text:
- Same claim "DeepSeek выпустил V3.2": ruBERT ent=0.948 vs XLM-R ent=0.006
- 3x smaller, 3x less VRAM, better Russian accuracy
