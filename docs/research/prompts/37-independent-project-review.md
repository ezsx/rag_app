# Prompt 37: Independent Project Review — Portfolio Readiness

## Задача

Ты — независимый эксперт по AI/ML инженерии и hiring для позиций Applied LLM Engineer. Тебя попросили оценить портфолио-проект кандидата.

**Твоя задача — двойная:**
1. Исследовать проект самостоятельно через MCP (repo-semantic-search) — читай код, docs, specs, decision log. Не верь промпту на слово — проверяй.
2. Дать честную, критическую оценку: для портфолио и для собеседований.

**Не приукрашивай.** Если что-то слабо — скажи прямо. Если сильно — объясни почему.

---

## Контекст кандидата

- Целевая позиция: **Applied LLM Engineer** (middle-senior)
- Опыт: backend разработка, переход в Applied LLM/AI
- Этот проект — основной в портфолио, строился ~2 месяца
- Стратегия: показать что может строить production-grade RAG системы end-to-end
- Планирует: собеседоваться в компании где нужен RAG, agents, retrieval, eval

## Проект: rag_app

Self-hosted RAG система для поиска по 36 русскоязычным AI/ML Telegram-каналам. 13K документов в Qdrant. Полностью на локальном железе, без managed API.

### Заявленные метрики

| Metric | Value | Method |
|--------|-------|--------|
| Factual correctness | 0.842 | Claude judge, 36 golden Qs |
| Usefulness | 1.778/2 | Claude judge |
| Key Tool Accuracy | 1.000 | Programmatic |
| Faithfulness | 0.91 | ruBERT NLI, 171 claims, 0 hallucinations |
| Retrieval recall@3 | 0.97 | 100 calibration queries |
| Robustness composite | 0.954 | NDR 0.963 / RSR 0.941 / ROR 0.959 |
| Latency | 24.4s | Full pipeline |

57 eval прогонов. 29 research reports. 23 спецификации. 45 ADR entries.

### Что планирует доделать

1. Расширение eval 36→100 Qs
2. UI polish
3. Unit tests cleanup
4. Fix 4 known issues (missing docs, routing confusion)
5. Re-ingest свежих данных

---

## Инструкции по исследованию

**Используй MCP tools** для проверки заявлений. Примеры:

```
hybrid_search_docs("retrieval pipeline architecture BM25 ColBERT RRF")
hybrid_search_code("agent_service ReAct tools function calling")
hybrid_search_docs("evaluation methodology factual faithfulness robustness")
hybrid_search_code("evaluate_agent metrics aggregate")
hybrid_search_docs("decision log ADR")
hybrid_search_code("hybrid_retriever BM25 dense prefetch")
```

Проверь:
- Действительно ли pipeline 4-stage (BM25→RRF→ColBERT→CE)?
- Действительно ли 15 tools с native function calling?
- Действительно ли eval methodology включает robustness (NDR/RSR/ROR)?
- Действительно ли 45 ADR entries?
- Есть ли research reports и specs в docs/?
- Как выглядит код — quality, structure, comments?
- Есть ли тесты? Какое покрытие?

---

## Что оценить

### 1. Архитектурные решения
Retrieval pipeline, agent design, model choices. Насколько это production-grade? Что бы ты изменил?

### 2. Eval methodology
Factual + faithfulness + robustness + retrieval calibration. Достаточно ли для credibility? Что не хватает? Сравни с industry standard (RAGAS, RAGChecker, Cao et al.).

### 3. Код и engineering quality
Структура проекта, quality кода, documentation-as-code, decision log. Production-ready или прототип?

### 4. Для собеседований Applied LLM Engineer
- Какие вопросы на собеседовании этот проект закрывает? (system design, retrieval, agents, eval, infra)
- Какие competencies НЕ показаны? (что нужно дополнить другим опытом)
- О чём hiring manager спросит и какие follow-up questions ожидать?

### 5. Сравнение с рынком
Как проект выглядит vs типичные портфолио для Applied LLM Engineer? Top 10%? Top 30%? Average?

### 6. Стратегия кандидата
- "Построил production-grade RAG с нуля на local hardware" — насколько это убедительно?
- Что лучше подчёркивать на собеседовании? Что лучше не упоминать?
- Какие компании/роли лучше всего подходят с этим проектом?

### 7. Что доделать (prioritized)
Из списка выше + то что ты нашёл при исследовании. Что даст максимальный ROI для портфолио?

### 8. Red flags
Что hiring manager увидит как проблему? Что вызовет сомнения?

---

## Формат ответа

Структурированный отчёт по каждому пункту. Для каждого: оценка (1-10), аргументы, конкретные рекомендации. Общий вердикт в конце: "ready / almost ready / needs work".

Пиши на русском.
