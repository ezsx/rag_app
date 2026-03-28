# Deep Research: RAG Necessity Classifier — Skip Retrieval When Not Needed

> **Цель**: Получить конкретный план реализации механизма, который определяет нужен ли RAG для данного запроса. Результат → SPEC-RAG-20 (или часть ablation study).

---

## Контекст проекта

### Что это
`rag_app` — RAG + ReAct агент для Telegram-каналов (36 каналов AI/ML, 13088 points).

### Текущее поведение
Агент выполняет retrieval **почти всегда**, с тремя существующими bypass:
- **Navigation short-circuit**: `list_channels` → `navigation_answered` → skip forced search
- **Analytics short-circuit**: `entity_tracker`/`arxiv_tracker` → `analytics_done` → skip forced search
- **Negative intent bypass**: refusal markers + negative intent markers → skip forced search

Для всех остальных запросов — если LLM не вызывает tools → **forced search** (`agent_service.py:768`).
Результат: conversational queries ("привет", "объясни что такое RAG") всё равно проходят через полный pipeline → +30-45с latency, нерелевантные документы в контексте.

**Задача**: не "убрать always-on retrieval", а **расширить существующие bypass-механизмы** на новые категории запросов (conversational, definitional, meta).

### Forced search logic (текущая)
```python
# agent_service.py ~line 768
if (not tool_calls
    and agent_state.search_count == 0
    and not agent_state.navigation_answered
    and not agent_state.analytics_done
    and not skip_forced):
    # → forced search с request.query
```

### Что говорят ресерчи

**R15 (Яндекс, 2026) §7**:
- "Не всем запросам нужен RAG"
- Определяют через: перплексию, tool call решение модели, обратный сигнал (запросы где контекст ничего не меняет)
- Обучают быстрый классификатор → **до 25% экономии** + рост качества
- Рекомендация: начать с эвристики, classifier — после накопления логов

**R18 §4 (NDR metric)**:
- NDR = f(q,k) ≥ f(q,0) — если с контекстом не лучше чем без, зачем retrieval?
- По сути NDR measurement даёт ground truth для RAG necessity classifier

### Железо
- **LLM**: Qwen3-30B-A3B на V100 (~12с на один call)
- Каждый запрос: 2-3 LLM calls × 12с = 30-45с
- Skip RAG = skip 1-2 LLM calls + retrieval + reranking = -20-30с

---

## Что я хочу получить

### 1. Taxonomy запросов по RAG necessity

Для нашего домена (AI/ML новости из Telegram) — какие категории запросов?

| Категория | Пример | Нужен RAG? |
|-----------|--------|------------|
| Factual about corpus | "Что нового в DeepSeek?" | Да |
| Temporal | "Новости AI за январь 2026" | Да |
| Channel-specific | "Что пишет gonzo_ml?" | Да |
| Analytics | "Топ компаний по упоминаниям" | Да (Facet API) |
| Navigation | "Какие каналы есть?" | Уже short-circuit |
| Negative/refusal | "Существует ли GPT-7?" | Уже bypass |
| **Conversational** | "Привет", "Спасибо" | **Нет** |
| **Definitional** | "Что такое RAG?" | **Может не нужен** |
| **Meta** | "Как ты работаешь?" | **Нет** |
| **Ambiguous** | "Расскажи про трансформеры" | **Зависит** |

Мне нужна полная taxonomy для нашего домена с decision criteria.

### 2. Три подхода к RAG necessity

**Approach A — Rule-based heuristic**:
- Короткие запросы (<3 слов) + conversational patterns → skip
- Keyword matching: greetings, thanks, meta-questions
- Плюсы: zero latency, predictable, easy to debug
- Минусы: fragile, false positives on short factual queries

**Approach B — LLM self-decision (уже частично работает)**:
- LLM получает tools schema → сам решает вызывать ли search
- Сейчас: если LLM не вызывает tools → forced search override
- Изменение: убрать forced search для определённых query categories
- Плюсы: адаптивно, LLM видит полный контекст запроса
- Минусы: Qwen3-30B-A3B ненадёжно (3B active params) — иногда скипает search для factual queries

**Approach C — Lightweight classifier**:
- Отдельная модель или prompt-based classifier
- Input: query → Output: {needs_rag: true/false, confidence: 0-1}
- Можно на Qwen3 через отдельный prompt (zero-shot) или fine-tuned small model
- Плюсы: independent от основного agent loop
- Минусы: +1 LLM call если через prompt, latency overhead

Мне нужен обоснованный выбор подхода.

### 3. Связь с NDR (Track 3)

NDR measurement даёт ground truth:
- Прогнать 100 Qs с k=0 (без retrieval) и с k=10 (с retrieval)
- Queries где f(q,0) ≥ f(q,10) → RAG не нужен
- Это дообразует **training signal** для classifier (если пойдём в Approach C)

Вопрос: стоит ли делать RAG necessity **после** NDR measurement (Track 3), используя его данные? Или можно начать с heuristic (Approach A) независимо?

### 4. Конкретные вопросы

1. **Доля conversational queries**: в нашем eval dataset (30 Qs) — 0 conversational queries. В реальном использовании — сколько ожидать? Яндекс говорит ~25%. Для нашего демо-UI — скорее 5-10%?

2. **False negative risk**: если classifier говорит "skip RAG" для factual query → ответ будет из parametric knowledge → hallucination. Как митигировать? Fallback threshold? Confidence-based routing?

3. **"Что такое RAG?" problem**: definitional queries — LLM знает ответ, но наша база тоже содержит обзоры. Лучше ли ответ с контекстом или без? Это sub-question для NDR measurement.

4. **Integration с forced search**: сейчас forced search — safety net. Убирать его полностью или добавить "RAG necessity check" как **pre-filter** перед forced search?

5. **Latency budget**: если classifier добавляет +X ms — при каком X он перестаёт быть выгодным? (Учитывая что skip RAG экономит ~20-30с)

6. **Eval impact**: как добавить "RAG necessity accuracy" в eval pipeline? Precision/recall на "нужен ли RAG"?

### 5. Deliverables

- Taxonomy запросов для нашего домена (10-15 категорий с decision criteria)
- Рекомендуемый подход (A/B/C) с обоснованием
- Конкретная реализация (heuristic rules / prompt / classifier architecture)
- Integration plan с existing forced search logic
- Метрики для оценки качества classifier'а
- Связь с NDR measurement (Track 3) — sequence vs parallel
- Quick experiment: взять 30 golden Qs, классифицировать вручную, оценить ceiling

---

## Формат ответа

Структурированный отчёт с:
1. Taxonomy table
2. Approach comparison (A vs B vs C)
3. Recommended implementation
4. Integration diagram (where in agent loop)
5. Expected latency savings
6. Risk analysis (false negatives → hallucination)
7. Experiment design для валидации
