# Codex Handoff: R22 findings + scope update

> Контекст: Claude (Opus 4.6) провёл сессию исследования репозитория через MCP, подготовил и запустил Deep Research (R25), проанализировал результаты совместно с пользователем. Ниже — итоги. Твоя задача: обновить `docs/progress/project_scope.md` как source of truth и наметить конкретный план работы.

## Что сделали в этой сессии

1. **Полное MCP-исследование репозитория** — подтвердили текущее состояние: 15 tools, eval 1.79/2 factual на 30 Qs, все 5 tools из R17 реализованы, SPEC-RAG-16/17 done.

2. **Подготовили и запустили R25 (Deep Research)** — production gap analysis. Сравнение с Perplexity, Glean, Danswer, Cohere, Langdock.
   - Отчёт: `docs/research/reports/R22-deep-production-gap-analysis.md`

3. **Критический разбор R25** — не все рекомендации релевантны. Ниже что приняли, что отклонили, что скорректировали.

## R25: что подтвердил

- Retrieval pipeline на уровне production систем (Perplexity, Cohere) — менять не надо
- Custom ReAct agent без фреймворков — правильное решение, дифференциатор
- Self-hosted подход — дифференциатор
- Eval 30 Qs недостаточно — нужно 100+
- CRAG-lite — real production gap
- Ablation study — high credibility signal
- Observability — нужна (но это cosmetics, не блокер)

## R25: что отклонили (с обоснованием)

| Рекомендация R25 | Почему отклонено |
|-----------------|-----------------|
| **Query rewriting как новый gap** | У нас уже есть: `QueryPlannerService` генерирует 3-6 subqueries через LLM + GBNF, multi-query search через round-robin merge, original query injection для BM25. Это и есть query rewriting |
| **RAGAS Faithfulness вместо custom NLI** | RAGAS = LLM-as-judge (circular dependency). Яндекс конференция (R15) прямо говорит "LLM as Judge НЕ РАБОТАЕТ для фактической корректности". Наш R19 план (Qwen3 decomposition + XLM-RoBERTa NLI) — independent verification, принципиально сильнее |
| **Recall@k, Precision@k, MRR как primary metrics** | Сломаны для нашего кейса: analytics queries не имеют source_post_ids, temporal/compare queries находят правильную инфу из других постов. Strict Recall@5 = 0.342 при factual = 1.79/2. Подтверждено R15 + эмпирически |
| **NDR/RSR/ROR = academic exercise** | Делаем. R20 протокол готов. NDR отвечает на вопрос "retrieval помогает или вредит?", RSR — оптимальный top_k, ROR — lost-in-the-middle |
| **BGE-M3 вместо Qwen3-Embedding** | Qwen3-Embedding-0.6B — новейшая модель (2025), бенчмарки лучше BGE-M3 на multilingual. Мы уже проверяли при выборе |
| **Request queuing** | Не наш профиль. Мы показываем ML/RAG engineering, не MLOps scaling. Один пользователь, demo |
| **Второй проект (eval infra)** | Преждевременно. Сначала завершить rag_app. Вопрос второго проекта — на этапе маркетинга/резюме |

## R25: что добавил нового

Единственный реально новый blind spot:

1. **Prompt injection defense** — Telegram content = user-generated. Adversarial content может попасть в корпус и инжектиться в промпт через retrieval. Нужны delimiter boundaries между system instructions и retrieved content. Research цитирует снижение attack success с 73% до 9%

2. **Health check endpoints** (`/health`, `/ready`) — тривиально, 10 минут

## Что нужно обновить в scope

`docs/progress/project_scope.md` — source of truth проекта. Сейчас рассинхронизирован:

### Исправить статусы
- Phase 3.4: "В РАБОТЕ" → **DONE** (все tools реализованы, SPEC-RAG-15/16/17 done)
- Phase 3.5 research tracks: "RESEARCH DONE, IMPL NEXT" → корректно, но нужно добавить новые items
- "Исследовательская база: R01-R18 (18 отчётов)" → **R01-R21 + R25 (22+ отчёта)**

### Добавить в Phase 3.5 или создать новые phases

**Данные (блокирует всё):**
- Re-ingest свежих постов (18 марта → текущая дата)
- Weekly digests для всех ~37 недель (сейчас только W10-W11)
- Channel profiles re-compute после re-ingest

**Eval (блокирует метрики):**
- Eval expansion 30→100 Qs (hand-crafted + synthetic)
- Fresh eval прогон на 15 tools (baseline после SPEC-RAG-16/17)
- Unit tests: удалить мёртвый test_new_tools.py, покрыть analytics tools + state machine

**Substance (Phase 3.5 tracks):**
- Track 2: NLI faithfulness — R19 план (Qwen3 + XLM-RoBERTa), НЕ RAGAS
- Track 3: NDR/RSR/ROR — R20 протокол
- Track 4: CRAG-lite quality gate — LLM grades retrieval relevance → filter → fallback
- Track 5: RAG necessity classifier — R21 (low priority)
- NEW: Ablation study — ColBERT on/off, reranker on/off, RRF weights (нужен eval set 100 Qs)
- NEW: GPT-4o comparison — один прогон через API
- NEW: Prompt injection defense — delimiter boundaries в prompt template

**Production polish:**
- Observability (Langfuse или свой) — per-component latency
- Health check endpoints
- BERTopic labels cleanup

**Упаковка (в конце):**
- README + Mermaid diagram + Design Decisions + ablation table
- Docs sync (playbook, scope — привести в соответствие с кодом)

### Обновить playbook

`docs/progress/experiment_log.md` — пишет "13 tools", hot_topics в backlog. Нужно обновить до 15 tools, hot_topics/channel_expertise = done, добавить SPEC-RAG-16/17 в "Что реализовано".

## Зависимости

```
Re-ingest ──→ Weekly digests ──→ Channel profiles
                                       │
Eval expansion (100 Qs) ───────────────┼──→ Fresh eval прогон
                                       │         │
                                       ├──→ NDR/RSR/ROR (R20)
                                       ├──→ Ablation study
                                       └──→ GPT-4o comparison

NLI (R19) ──→ (independent, можно параллельно)
CRAG-lite ──→ (independent, можно параллельно)
Unit tests ──→ (independent, можно параллельно)
```

## Порядок работы

1. **Данные** — re-ingest, digests, profiles
2. **Eval** — expansion 30→100, fresh прогон, unit tests
3. **Substance** — NLI, CRAG-lite, NDR/RSR/ROR, ablation, GPT-4o (параллельно)
4. **Polish** — observability, prompt injection, health checks
5. **Упаковка** — README, docs sync, labels

## Что НЕ делать

- Graph RAG — overkill для news
- Self-RAG — нужен fine-tuning
- Semantic caching — 99% false positive rate
- Multi-provider fallback — мы self-hosted
- Custom NLI training — XLM-RoBERTa off-the-shelf достаточно
- SFT/RLHF — нет данных
- Multi-turn с hierarchical memory — sliding window достаточно если делать
