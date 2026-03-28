# Deep Research: NLI Citation Faithfulness для RAG Pipeline

> **Цель**: Получить конкретный, реализуемый план интеграции NLI-based verification в `rag_app`. Ответ агента проверяется на faithfulness: каждый claim сверяется с цитированными документами. Результат должен быть конвертируемым в SPEC-RAG-18 за 1 день.

---

## Контекст проекта

### Что это
`rag_app` — FastAPI-платформа: 36 Telegram-каналов об AI/ML → Qdrant (13088 points) → ReAct-агент с 13 tools + native function calling → SSE-ответ с цитатами.

### Железо
- **LLM**: Qwen3-30B-A3B GGUF (V100 SXM2 32GB, 3B активных параметров)
- **Embedding**: Qwen3-Embedding-0.6B (RTX 5060 Ti 16GB, ~5GB VRAM занято)
- **Reranker**: bge-reranker-v2-m3 (на том же 5060 Ti)
- **ColBERT**: jina-colbert-v2 (на том же 5060 Ti)
- **Свободно на 5060 Ti**: ~11GB VRAM

### Текущие метрики (golden_v1, 30 Qs)
- **Automated judge** (`evaluate_agent.py`): Claude Sonnet 4.6 (`claude-sonnet-4-6-20250514`), factual 0-1, usefulness 0-2
- **Manual judge** (отдельный процесс, Claude Opus + Codex GPT-5.4 consensus): factual **1.79/2**, useful **1.72/2**, KTA **0.926**
- Strict Recall@5: 0.342 (занижен — analytics Qs без source_post_ids)
- **Важно**: automated и manual judge — разные процессы. NLI verification заменит/дополнит automated judge в `evaluate_agent.py`.

### Проблема

1. **Яндекс R15 (конференция 2026)**: "LLM as Judge НЕ РАБОТАЕТ для фактической корректности" — LLM-judge видит тот же контекст что и агент, не может отличить hallucination от grounded fact.

2. **Текущий eval**: automated judge (Claude Sonnet) и manual judge (Opus + Codex consensus) — оба **reference-based** (сравнивают ответ с expected_answer), не document-based. Мы не проверяем: "поддерживают ли ЦИТИРОВАННЫЕ ДОКУМЕНТЫ утверждения в ответе?"

3. **Текущий compose_context**: считает coverage (6-signal composite), но это **retrieval quality** signal, не **generation faithfulness** signal. Ответ может иметь высокий coverage и при этом галлюцинировать.

### Что уже есть в наших ресерчах

- **R14-deep §NLI**: XLM-RoBERTa-large-xnli, decompose-then-verify, target faithfulness ≥0.92
- **R18 §3**: полная спецификация decompose-then-verify с промптами на русском (5 шагов, формулы Factual_Precision/Recall/F1)
- **R18 §3**: "0.5 penalty for extrinsic claims" (configurable strict/lenient mode)

---

## Что я хочу получить

### 1. Архитектурный анализ

Два возможных подхода к faithfulness verification:

**Approach A — LLM-based decompose-then-verify (R18 §3)**:
- Разложить ответ на atomic claims → проверить каждый claim против cited documents (NLI classification: supported/contradicted/not_mentioned)
- Используем тот же Qwen3-30B-A3B для decomposition и verification
- Pros: не нужна новая модель, работает на русском из коробки
- Cons: зависимость от того же LLM, latency +2-4 LLM calls

**Approach B — Dedicated NLI model**:
- XLM-RoBERTa-large-xnli (~1.3GB) или DeBERTa-large-MNLI (~1.5GB)
- Inference на RTX 5060 Ti (хватит VRAM)
- Pros: independent от основного LLM, быстрее (~50ms per claim), меньше bias
- Cons: отдельная модель для загрузки и обслуживания, точность на русском может быть ниже

**Approach C — Hybrid**: LLM для decomposition, NLI model для verification.

Мне нужен обоснованный выбор с цифрами: accuracy на русском тексте, latency, VRAM footprint.

### 2. Интеграция в pipeline

Два варианта интеграции:

**Вариант 1 — Runtime (перед final_answer)**:
- После compose_context, перед final_answer: проверяем каждый claim
- Если faithfulness < threshold → добавляем disclaimer или refine
- Impact на latency: +X секунд на запрос

**Вариант 2 — Eval-only (в evaluate_agent.py)**:
- Faithfulness как дополнительная метрика в eval pipeline
- Не влияет на runtime latency
- Даёт числа для ablation study и README

Мне нужен анализ: какой вариант реалистичнее для нашего hardware? Можно ли runtime на V100 вместо 5060 Ti?

### 3. Конкретные вопросы

1. **XLM-RoBERTa-large-xnli на русском**: какова реальная accuracy для NLI на русских новостных текстах? Есть ли бенчмарки XNLI-ru? Сравнение с ruBERT-based моделями?

2. **Decomposition quality**: atomic claim extraction на русском — Qwen3-30B справится? Или нужен отдельный промпт-инжиниринг? Примеры failure modes?

3. **Порог faithfulness**: R14-deep говорит ≥0.92. Это реалистично для нашего pipeline (русский текст, новостной домен, 13K docs)?

4. **Citation grounding vs factual correctness**: R18 разделяет эти два сигнала. Citation grounding = "claim подтверждается цитированным документом". Factual correctness = "claim верен относительно expected_answer". Нам нужны оба или достаточно одного?

5. **MiniCheck / SelfCheckGPT**: альтернативные подходы к faithfulness без NLI модели. Подходят ли для русского?

6. **Inference integration**: если XLM-RoBERTa — как подключить к gpu_server.py? Он сейчас обслуживает embedding + reranker + ColBERT. Ещё одна модель (~1.3GB) влезет в 11GB свободного VRAM?

### 4. Deliverables

- Выбор подхода (A/B/C) с обоснованием
- Схема интеграции (runtime vs eval-only vs both)
- Промпты для decomposition и verification (на русском)
- Пороги и формулы (Factual_Precision, Factual_Recall, F1, citation grounding ratio)
- Оценка latency overhead
- Рекомендация по модели (с учётом 5060 Ti VRAM budget)
- Пример: взять один ответ из eval_judge_20260325_spec15.md, разложить на claims, показать как каждый проверяется

---

## Формат ответа

Структурированный отчёт с:
1. Сравнительная таблица подходов (A vs B vs C)
2. Рекомендуемая архитектура
3. Step-by-step implementation plan
4. Промпты на русском
5. Expected impact на метрики
6. Риски и mitigation
