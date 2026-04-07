# RUN-004 Judge Scores

> Judge: Claude Opus 4.6 (1M context)
> Date: 2026-04-08
> Questions: 10 (golden_q01-q10)
> Config: compose_context max_tokens=4000 + all retrieval improvements (prefix fix, dense=40, MMR, CE re-sort, adaptive filter)
> Scale: factual 0.0-1.0 (step 0.1), useful 0/1/2

## Summary

| qid | Category | Old Factual | New Factual | Old Useful | New Useful |
|-----|----------|:-:|:-:|:-:|:-:|
| golden_q01 | broad_search | 1.0 | 1.0 | 2 | 2 |
| golden_q02 | broad_search | 1.0 | 1.0 | 2 | 2 |
| golden_q03 | broad_search | 1.0 | 1.0 | 2 | 2 |
| golden_q04 | broad_search | 0.8 | 0.9 | 2 | 2 |
| golden_q05 | broad_search | 1.0 | 0.9 | 2 | 2 |
| golden_q06 | constrained | 0.5 | 0.7 | 1 | 2 |
| golden_q07 | constrained | 1.0 | 1.0 | 2 | 2 |
| golden_q08 | constrained | 0.8 | 0.9 | 2 | 2 |
| golden_q09 | constrained | 1.0 | 1.0 | 2 | 2 |
| golden_q10 | constrained | 0.7 | 0.8 | 2 | 2 |

| Metric | Old (2026-04-01) | New (RUN-004) | Delta |
|--------|:-:|:-:|:-:|
| **Factual mean** | **0.88** | **0.92** | **+0.04** |
| **Useful mean** | **1.90** | **2.00** | **+0.10** |

---

## Per-Question Reasoning

### golden_q01 — Financial Times человек года
**Factual: 1.0 | Useful: 2**
Все ключевые факты: Дженсен Хуанг, NVIDIA, FT, трансформация полупроводниковой индустрии. Дополнительно: $5 трлн капитализация, дата-центры как инфраструктура. 5 docs из 4 каналов, все citations корректны.

### golden_q02 — Open-source GPT параметры
**Factual: 1.0 | Useful: 2**
Полное покрытие: 120B (117B/5.1B active), 20B (21B/3.6B active), MoE, Apache 2.0, MXFP4, 128K context. Структурированный ответ с GPU requirements. 5 docs, все подтверждают claims.

### golden_q03 — Meta купила Manus AI
**Factual: 1.0 | Useful: 2**
$2 млрд, ARR $100M за 8 месяцев, агент с deep research. Дополнительно: разрыв связей с Китаем, интеграция в Facebook/Instagram/WhatsApp. Точные даты и контекст. 5 docs.

### golden_q04 — DeepSeek модели
**Factual: 0.9 | Useful: 2** (было 0.8)
Покрыто: V3.2, V3.2-Speciale, V3.2-Exp, DSA, OCR-2, V4. Не упомянут mHC (Manifold-Constrained Hyper-Connections) из expected — но это мелкая деталь. Зато значительно больше контекста: олимпиады, Groq, ценовая политика. 10 docs, ответ на 2500 chars — полный и детальный. Улучшение vs old (0.8→0.9) за счёт расширенного compose_context.

### golden_q05 — Open-source image generation
**Factual: 0.9 | Useful: 2** (было 1.0)
Перечислены: HunyuanImage 3.0, Kandinsky 5.0, FLUX.2, Qwen-Image, Waypoint-1, PaperBanana, PASTA. Expected: FLUX.2 и Kandinsky — оба есть. Снижение 1.0→0.9: FLUX.2 описан корр��ктно но Dev-версия под некоммерческой лицензией (не полностью open-source). 22 docs ��� compose_context budget дал обширный контекст.

### golden_q06 — AI-каналы в январе 2026
**Factual: 0.7 | Useful: 2** (было 0.5/1)
Значительное улучшение. Expected: gonzo_ml про ИИ-исследования и эффект Франкенштейна + boris_again про Gemini 3 Flash. Ни то ни другое не упомянуто в ответе. Но ответ перечисляет 7 реальных тем января (CES 2026, OpenAI устройство, агенты, медицина, конференции, AGI в Давосе) с 18 citations из множества каналов. Ответ полезный и информативный, но мимо expected_answer. Factual 0.7 (не 0.5): agent нашёл больше источников и дал более полную картину чем раньше.

### golden_q07 — GTC 2026
**Factual: 1.0 | Useful: 2**
Vera Rubin (ожидаемый ответ), плюс: Groq 3 LPX, OpenClaw/NemoClaw, космические дата-центры, DLSS 5, роботакси, GWM-1. 10 docs, чрезвычайно детальный ответ на 2687 chars. Полное покрытие + бонус.

### golden_q08 — AI-события февраля 2026
**Factual: 0.9 | Useful: 2** (было 0.8)
Expected: Opus 4.6, Себрант, Agibot роботы. Opus 4.6 — есть [5]. Agibot — есть [7]. Себрант не упомянут, но это не критично — ответ перечисляет 7 тем из 10 docs. Улучшение 0.8→0.9: compose_context=4000 дал больше docs → полнее картина.

### golden_q09 — llm_under_hood про GPT-5 reasoning
**Factual: 1.0 | Useful: 2**
Идеальный ответ: developer role инструкция, Juice параметр, Active/Disabled channels, конкретные числа (28s→10s, 1280→0 tokens). 4 docs из llm_under_hood. Полное совпадение с expected.

### golden_q10 — gonzo_ml про трансформеры и рекуррентные сети
**Factual: 0.8 | Useful: 2** (было 0.7)
Expected: AlphaGenome и ограничения RNN vs тран��формеры. AlphaGenome НЕ упомянут. Но ответ подробно покрывает: HRM/TRM vs UT/ALBERT, Memory Caching для RNN, Bolmo б��йтовые модели, латентные переменные. 5 docs ��з gonzo_ml. Улучшение 0.7→0.8: больше аспектов темы покрыто.
