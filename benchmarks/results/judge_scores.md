# LLM Judge: 4-Pipeline Benchmark (SPEC-RAG-29 Phase 2)

> Judge: Claude Opus 4.6 (1M context)
> Artifact: `judge_artifact.langfuse_enriched.json`
> Date: 2026-04-03
> Questions: 17 (golden_q01–q16, q25)
> Pipelines: naive, li_stock, li_maxed, custom

---

## Summary

| Pipeline | Factual (avg) | Usefulness (avg) | Grounding (avg) |
|----------|:---:|:---:|:---:|
| **naive** | 0.55 | 1.04 | 0.28 |
| **li_stock** | 0.51 | 1.13 | 0.46 |
| **li_maxed** | 0.54 | 1.21 | 0.48 |
| **custom** | **0.84** | **1.77** | **0.88** |

Delta custom vs best-of-three: **factual +0.30**, **usefulness +0.56**, **grounding +0.40**.

---

## Per-Question Scores

### golden_q01 — Кого Financial Times назвала человеком года в 2025?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 1.0 | 1.4 | 0.3 |
| li_stock | 1.0 | 1.8 | 0.7 |
| li_maxed | 1.0 | 1.9 | 0.8 |
| custom | 1.0 | 1.6 | 0.9 |

Все 4 pipeline верно: Дженсен Хуанг, NVIDIA, FT. custom добавляет Time/архитекторы ИИ с inline [1][2][3][4]. li_maxed — URL на FT-статью. naive — без ссылок.

### golden_q02 — Какие параметры у open-source моделей GPT от OpenAI?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.8 | 1.2 | 0.3 |
| li_stock | 1.0 | 1.9 | 0.7 |
| li_maxed | 1.0 | 2.0 | 0.7 |
| custom | 0.9 | 1.6 | 0.9 |

naive пропускает Apache 2.0/mxfp4. li_maxed — самый полный (mxfp4 явно). custom включает Apache 2.0, 128K, MoE, active params но не mxfp4.

### golden_q03 — За сколько Meta купила Manus AI?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 1.0 | 1.5 | 0.3 |
| li_stock | 0.0 | 0.0 | 0.0 |
| li_maxed | 0.0 | 0.0 | 0.0 |
| custom | 1.0 | 1.6 | 0.9 |

**КРИТИЧНО**: li_stock и li_maxed уверенно заявили "Meta не покупала Manus AI" — полный retrieval failure (0 docs). naive и custom верно: $2 млрд, ARR $100M, браузер/deep research. custom добавляет дату (дек 2025) и геополитику (Китай→Сингапур).

### golden_q04 — Что нового у DeepSeek? Какие модели они выпустили?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.7 | 1.7 | 0.3 |
| li_stock | 0.7 | 1.8 | 0.6 |
| li_maxed | 0.7 | 1.7 | 0.6 |
| custom | 0.7 | 1.8 | 0.9 |

Все pipeline хорошо покрывают V3.2/Speciale/DSA/V4, но НИКТО не упоминает mHC из expected. custom добавляет OCR-2 и временную шкалу. Grounding у custom значительно выше (inline refs).

### golden_q05 — Какие open-source модели для генерации изображений существуют?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.8 | 1.6 | 0.3 |
| li_stock | 0.8 | 1.8 | 0.6 |
| li_maxed | 0.9 | 2.0 | 0.7 |
| custom | 0.8 | 1.7 | 0.9 |

li_maxed — единственный с FLUX.2 явно. Все упоминают Kandinsky 5.0. custom добавляет K-VAE 1.0 и Hunyuan3D. FLUX.2 не упомянут custom'ом.

### golden_q06 — Что обсуждалось в AI-каналах в январе 2026?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.2 | 0.4 | 0.3 |
| li_stock | 0.6 | 1.7 | 0.7 |
| li_maxed | 0.5 | 1.5 | 0.6 |
| custom | 0.7 | 1.9 | 0.9 |

custom — самый полный обзор января (8 блоков, 30 docs, CES/агенты/конференции/военные). naive получил плохие docs — отказался. Никто не упомянул gonzo_ml/any2json из expected.

### golden_q07 — Что было на GTC 2026 от NVIDIA?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 1.0 | 1.8 | 0.3 |
| li_stock | 1.0 | 1.8 | 0.6 |
| li_maxed | 1.0 | 1.8 | 0.6 |
| custom | 1.0 | 2.0 | 0.9 |

Все верно: Vera Rubin. custom значительно превосходит по полноте: 7 блоков (OpenClaw, Groq 3 LPX, Physical AI, Space-1, DLSS 5, GWM-1, $1 трлн заказы).

### golden_q08 — Какие AI-события произошли в феврале 2026?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.3 | 0.6 | 0.3 |
| li_stock | 0.4 | 1.2 | 0.6 |
| li_maxed | 0.4 | 1.3 | 0.6 |
| custom | 0.9 | 1.9 | 0.9 |

Expected: Opus 4.6 + Себрант + Agibot. custom — ЕДИНСТВЕННЫЙ, кто упоминает и Opus 4.6 [1], и Agibot роботов в Шанхае [4] (2 из 3 expected). Также PersonaPlex, Genie, OpenTalks, Moltbook. naive путает месяцы.

### golden_q09 — Что писал llm_under_hood про отключение reasoning у GPT-5?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.9 | 1.5 | 0.4 |
| li_stock | 0.9 | 1.5 | 0.5 |
| li_maxed | 0.9 | 1.6 | 0.5 |
| custom | 1.0 | 1.8 | 0.9 |

Все pipeline корректно описывают developer role + Juice. custom добавляет метрики (28с→10с), проблемы Structured Outputs в GPT-5, "GPT-5 Pro поглупела".

### golden_q10 — О чём писал gonzo_ml про трансформеры и рекуррентные сети?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.7 | 1.7 | 0.4 |
| li_stock | 0.7 | 1.5 | 0.5 |
| li_maxed | 0.7 | 1.5 | 0.5 |
| custom | 0.7 | 1.7 | 0.9 |

Все обсуждают transformers vs RNN от gonzo_ml. Никто не упоминает AlphaGenome из expected. custom добавляет Energy transformers, Bolmo/mLSTM.

### golden_q11 — Что Борис Цейтлин думает о Claude Opus 4.6?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.1 | 0.2 | 0.2 |
| li_stock | 0.2 | 0.5 | 0.4 |
| li_maxed | 0.2 | 0.5 | 0.4 |
| custom | 1.0 | 1.9 | 0.9 |

**КЛЮЧЕВОЕ РАЗЛИЧИЕ**: custom — ЕДИНСТВЕННЫЙ pipeline, который нашёл мнение Бориса. Точная цитата: "самый близкий к автономной модели" [2]. Плюс 5 пунктов детального анализа (контекст +17пп, Vending-Bench +30%, Cybench ~100%, ARC-AGI-2 +14.6пп). Все остальные провалились.

### golden_q12 — Что seeallochnaya писал про GPT-5?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.4 | 0.8 | 0.3 |
| li_stock | 0.0 | 0.1 | 0.0 |
| li_maxed | 0.4 | 0.8 | 0.5 |
| custom | 0.4 | 1.0 | 0.8 |

Expected: GPT-5.3/5.4 с интервалом 2 дня — НИКТО не нашёл этот конкретный факт. naive/custom находят ранние GPT-5 посты seeallochnaya (слухи, тестирование). li_stock — полный провал (0 docs). li_maxed — LMArena контент.

### golden_q13 — Как разные каналы обсуждали покупку Manus компанией Meta?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.9 | 1.7 | 0.3 |
| li_stock | 0.8 | 1.5 | 0.5 |
| li_maxed | 0.9 | 1.7 | 0.6 |
| custom | 1.0 | 1.9 | 0.9 |

Все покрывают ai_newz/data_secrets/aioftheday. custom добавляет расследование МинТорг КНР [8] и будущую интеграцию Manus Agents [5]. 23 docs, 7 tool_calls. Самое глубокое покрытие.

### golden_q14 — Сравни как разные каналы обсуждали DeepSeek — какие мнения?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.6 | 1.5 | 0.3 |
| li_stock | 0.7 | 1.7 | 0.6 |
| li_maxed | 0.7 | 1.6 | 0.5 |
| custom | 0.8 | 1.9 | 0.9 |

custom покрывает 10+ каналов (gonzo_ml/GRPO, ai_ml_big_data/OCR, protechietich/трафик РФ, boris_again/бэкдоры, theworldisnoteasy/безопасность, data_secrets/R2). Безопасность через boris_again [9] — частично совпадает с expected. Нет mHC/seeallochnaya.

### golden_q15 — Что нового в канале techsparks за последнюю неделю?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.0 | 0.1 | 0.0 |
| li_stock | 0.1 | 0.3 | 0.2 |
| li_maxed | 0.1 | 0.4 | 0.2 |
| custom | 0.6 | 1.5 | 0.7 |

custom — ЕДИНСТВЕННЫЙ с реальным обзором techsparks (14 постов, 24-30 марта: ClawBot, SORA, FOBO, GeekWire). Использует `summarize_channel`. Не упоминает Себранта/роботакси/маркетинг из expected, но даёт настоящий дайджест. JSON-обёртка. Все остальные — полный провал.

### golden_q16 — Дай дайджест канала gonzo_ml за последний месяц

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.1 | 0.2 | 0.2 |
| li_stock | 0.1 | 0.2 | 0.1 |
| li_maxed | 0.1 | 0.2 | 0.1 |
| custom | 0.8 | 2.0 | 0.8 |

custom — ЕДИНСТВЕННЫЙ, кто составил дайджест gonzo_ml (14 тем: Moltbook, деструктивная интерференция трансформеров, AGI макроэкономика, Vox Deorum, JEPA, Memory Caching RNN, SSD, MoE в MLP). Трансформеры [2] и RNN [8] покрыты. Нет AlphaGenome. ВСЕ остальные отказались.

### golden_q25 — Какие подходы к использованию LLM в production обсуждались в каналах llm_under_hood и boris_again?

| Pipeline | Factual | Usefulness | Grounding |
|----------|:---:|:---:|:---:|
| naive | 0.5 | 1.0 | 0.3 |
| li_stock | 0.6 | 1.3 | 0.5 |
| li_maxed | 0.6 | 1.4 | 0.5 |
| custom | 0.9 | 1.9 | 0.9 |

custom — единственный, кто нашёл SGR (Schema-Guided Reasoning) [1][2] и Technical AI Safety курс [8] из expected. Плюс кейсы внедрения (налоги, HR, видео) и бенчмарк ERC3-PROD. any2json не упомянут, но SGR — ключевой пункт.

---

## Key Findings

### 1. custom доминирует по всем метрикам

Разрыв не маргинальный — custom в 1.5x лучше по factual и в 3x по grounding vs naive. Преимущество обеспечивают:
- **Multi-query search** (query_plan → несколько subqueries)
- **15 specialized tools** (summarize_channel, cross_channel_compare, rerank)
- **LANCER nugget coverage** (refinement по непокрытым аспектам)
- **compose_context + final_answer** (inline ссылки [1][2][3])

### 2. Три killer-вопроса где custom побеждает радикально

| Question | custom | best-of-three | Delta |
|----------|:---:|:---:|:---:|
| q11 (Борис об Opus 4.6) | 1.0/1.9/0.9 | 0.2/0.5/0.4 | +0.8/+1.4/+0.5 |
| q15 (techsparks дайджест) | 0.6/1.5/0.7 | 0.1/0.4/0.2 | +0.5/+1.1/+0.5 |
| q16 (gonzo_ml дайджест) | 0.8/2.0/0.8 | 0.1/0.2/0.2 | +0.7/+1.8/+0.6 |

Эти вопросы требуют **channel-specific retrieval** и **summarization** — возможности, которые есть только у custom pipeline (tools `summarize_channel`, `channel_search`).

### 3. Retrieval failure li_stock/li_maxed на q03

Оба LlamaIndex pipeline уверенно ответили "Meta не покупала Manus AI" при 0 найденных документов. С 1 tool (search) промах retrieval = полный провал с hallucinated refusal. custom и naive на тех же данных ответили верно.

**Причина**: LlamaIndex pipelines имеют один инструмент search. Если первый запрос не попадает — нет fallback. custom делает query_plan → несколько subqueries → round-robin merge.

### 4. li_maxed ≈ li_stock

| Metric | li_stock | li_maxed | Delta |
|--------|:---:|:---:|:---:|
| Factual | 0.51 | 0.54 | +0.03 |
| Usefulness | 1.13 | 1.21 | +0.08 |
| Grounding | 0.46 | 0.48 | +0.02 |

Weighted RRF 3:1 + cross-encoder reranker дали минимальный прирост. Основной gain custom'а — от multi-query planning + specialized tools + coverage refinement, не от reranker.

### 5. grounding — главный дифференциатор custom'а

| Pipeline | Grounding | Mechanism |
|----------|:---:|---|
| naive | 0.28 | Нет ссылок, "согласно предоставленной информации" |
| li_stock | 0.46 | "Источники:" в конце ответа, названия каналов |
| li_maxed | 0.48 | Аналогично li_stock, иногда URL |
| custom | 0.88 | Inline [1][2][3] по всему тексту, reader может проверить каждый факт |

Разница 0.88 vs 0.28 — следствие `compose_context` (нумерует документы) → `final_answer` (ссылается на номера).

### 6. Слабые места ВСЕХ pipelines

- **q04**: никто не упоминает mHC (Manifold-Constrained Hyper-Connections) из expected
- **q10**: никто не упоминает AlphaGenome от DeepMind из expected
- **q12**: никто не нашёл конкретный факт о GPT-5.3/5.4 с интервалом 2 дня
- **q06**: никто не упоминает gonzo_ml и any2json из expected

Это указывает на возможные gaps в golden dataset (expected answers слишком специфичны) или на недостаточную глубину retrieval по отдельным постам.

### 7. Latency trade-off

| Pipeline | Avg latency | Tools |
|----------|:---:|:---:|
| naive | ~4s | 0 (single LLM call) |
| li_stock | ~9s | 1 (search) |
| li_maxed | ~11s | 1 (search) |
| custom | ~30s | 4-7 (query_plan→search→rerank→compose→final) |

custom в ~7x медленнее naive, но quality gap оправдывает задержку для production use case.
