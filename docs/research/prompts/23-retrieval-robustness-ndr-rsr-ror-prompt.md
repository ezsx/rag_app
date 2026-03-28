# Deep Research: Retrieval Robustness — NDR, RSR, ROR для RAG Pipeline

> **Цель**: Получить конкретный план измерения robustness метрик (NDR, RSR, ROR) для `rag_app`. Адаптировать академические протоколы (Cao et al., arXiv 2505.21870) к нашим ограничениям: 30-100 вопросов (не 500), один V100, русский язык. Результат → SPEC-RAG-19.

---

## Контекст проекта

### Что это
`rag_app` — RAG + ReAct агент для Telegram-каналов (36 каналов, 13088 points, Qdrant).

### Железо и latency
- **LLM**: Qwen3-30B-A3B на V100 (~12с на один chat completion call)
- **Retrieval**: BM25+Dense → weighted RRF 3:1 → ColBERT MaxSim → cross-encoder rerank → channel dedup
- **Полный pipeline**: ~30-45с на запрос (2-3 LLM calls)
- **Eval compute budget**: реалистично ~24-48 часов на всю серию

### Текущие метрики
- **Golden dataset**: 30 вопросов, factual 1.79/2, useful 1.72/2
- **Eval pipeline**: `scripts/evaluate_agent.py` — SSE client, tool tracking, automated LLM judge (Claude Sonnet 4.6, `claude-sonnet-4-6-20250514`). Отдельно: manual judge (Claude Opus + Codex consensus)
- **compose_context**: уже делает document reordering (попытка mitigation lost-in-the-middle), но **НЕ измеряет** помогает ли это

### Текущий retrieval pipeline
```
query_plan → search (BM25 top-100 + dense top-20 → weighted RRF 3:1 → ColBERT MaxSim)
  → rerank (bge-reranker-v2-m3 cross-encoder, top 80)
  → channel dedup (max 2/channel)
  → compose_context (6-signal coverage, threshold 0.65)
```

### Что говорят наши ресерчи

**R15 (Яндекс, 2026)**: три метрики которых нет:
- NDR: добавление контекста не должно ухудшать качество
- RSR: больше контекстов → монотонный рост, не падение
- ROR: перестановка чанков не должна менять ответ

**R18 §4**: формулы, протокол, time budget:
- NDR: `(1/Z) × Σ 𝟙[f(q,k) ≥ f(q,0)]` — binary per sample
- RSR: `𝟙[∀j<i: f(q,k_i) ≥ f(q,k_j)]` — monotonic improvement check
- ROR: `1 - 2×σ[f(q,k,o)]` — stability across orderings
- Full protocol: 6000 runs → ~67 hours (500 Qs × k=[0,3,5,10,15,20] + shuffles)
- Prioritization: RSR first → NDR second → ROR third

---

## Что я хочу получить

### 1. Адаптация протокола к нашим constraint'ам

Академический протокол (R18 §4) предполагает 500 вопросов и ~67 часов compute. У нас:
- 30 golden Qs (можно расширить до 100)
- V100 SXM2 (~12с/call), т.е. ~100 calls/hour
- Реалистичный бюджет: 24-48 часов

Мне нужен **скейлированный протокол**:
- Минимальный набор Qs для статистически значимых результатов (power analysis)
- Какие k тестировать: [3, 5, 10, 15, 20] все нужны? Или [5, 10, 20] достаточно?
- Сколько shuffles для ROR: 3 vs 6?
- Можно ли параллелизировать (наш pipeline single-threaded)?

### 2. Реализация NDR

**Ключевой вопрос**: как получить f(q, 0) — ответ без retrieval?

Варианты:
- **A**: Пустой compose_context (no docs) → LLM отвечает из parametric knowledge
- **B**: Убрать forced search → LLM может решить не искать → final_answer
- **C**: Отдельный endpoint без retrieval pipeline

Для Qwen3-30B-A3B (3B active params) parametric knowledge слабая → ожидаем высокий NDR (baseline низкий). Это интересный finding сам по себе.

### 3. Реализация RSR

**Ключевой вопрос**: как варьировать k?

Текущий pipeline: `search` возвращает top-N after RRF+ColBERT+rerank+dedup. Варьировать k нужно на уровне:
- **A**: compose_context получает top-k из уже reranked результатов (простое truncation)
- **B**: retriever возвращает разное количество (изменить `k_per_query` в HybridRetriever)
- **C**: и то и другое (full pipeline vs post-rerank truncation)

Какой вариант правильнее с точки зрения методологии?

### 4. Реализация ROR

**Текущее состояние**: compose_context.py уже делает document reordering. Мне нужно:
- Сравнить: original order vs reversed vs N random shuffles
- Метрика "ответ не изменился": LLM judge (pairwise) vs BERTScore vs embedding cosine
- Для русского: R18 рекомендует ai-forever/ruBert-large для BERTScore — подтвердить

### 5. Scoring function f(q, k)

Академический протокол использует binary correctness. У нас factual score 0-2 (manual judge). Варианты:
- **Binarize**: factual ≥ 1.5 → correct (1), else incorrect (0)
- **Continuous**: использовать factual score as-is, NDR = mean(delta ≥ 0)
- **LLM judge per run**: Claude Opus 4.6 на каждый (q, k, order) → дорого

Какой вариант балансирует accuracy vs compute cost?

### 6. Конкретные вопросы

1. **Statistical significance**: при 30-100 Qs и 3-5 k values — достаточно ли для confidence interval <0.05? Какой тест: paired bootstrap vs Wilcoxon?

2. **Lost-in-the-middle для Qwen3**: есть ли данные что MoE модели (3B active) более/менее уязвимы к positional bias чем dense models?

3. **Compose_context reordering**: текущая стратегия (`compose_context.py:168-191`) — "наиболее релевантные документы в начало и конец, менее релевантные в середину" (mitigation lost-in-the-middle). Это best practice? Или есть лучшие подходы (strict descending, random, score-weighted)?

4. **Actionability**: если RSR показывает что k=10 лучше k=20 — что конкретно менять? Просто top_k? Или reranker config? Или coverage threshold?

5. **Integration с eval pipeline**: как добавить NDR/RSR/ROR в `evaluate_agent.py`? Отдельный скрипт или расширение существующего? Формат отчёта?

6. **Ablation synergy**: RSR sweep фактически даёт данные для ablation (k sweep). Можно ли объединить RSR measurement с ablation study (ColBERT on/off × k values)?

### 7. Deliverables

- Скейлированный протокол (30-100 Qs, 24-48 часов compute)
- Конкретные k values и число shuffles
- Формулы адаптированные под наш scoring (0-2 vs binary)
- Реализация f(q, 0) для NDR
- Архитектура: отдельный скрипт vs расширение evaluate_agent.py
- Expected ranges (что считать "хорошим" NDR/RSR/ROR для нашей системы)
- Quick screen protocol: 30 Qs × 3 k values → ~4 часа → "стоит ли копать глубже?"

---

## Формат ответа

Структурированный отчёт с:
1. Адаптированный протокол (таблица: Qs × k × shuffles × total runs → часы)
2. Implementation plan (3 фазы: RSR → NDR → ROR)
3. Scoring function выбор с обоснованием
4. Expected findings (гипотезы для каждой метрики)
5. Actionable recommendations (что менять в pipeline по результатам)
6. Comparison table: academic protocol vs our pragmatic version
