# SPEC-RAG-20: Codebase Hygiene & Refactoring

> **Status:** Draft
> **Created:** 2026-03-31
> **Context:** Audit выявил системную проблему: документация не соответствует коду, agent_service.py — монолит 838 строк в одном методе, ingest не создаёт payload indexes, устаревшие спеки в active/. Это приводит к ошибкам в каждой сессии (agent действует по устаревшим docs, забывает про indexes, не может дебажить монолит).

---

## Мотивация

После swap embedding/reranker стека (pplx-embed + Qwen3-Reranker) и добавления Langfuse observability:
- **Docs врут** — 8 файлов ссылаются на Qwen3-Embedding и bge-reranker, хотя код использует другие модели
- **CLAUDE.md и AGENTS.md врут** — агент (Claude/Codex) читает их первыми и принимает решения по устаревшей информации
- **Ingest теряет indexes** — `store.py` создаёт 4 из 16 payload indexes, `migrate_collection.py` знает все 16, но это отдельный скрипт который легко пропустить
- **agent_service.py невозможно дебажить** — `stream_agent_response` = 838 строк, 13+ обязанностей в одном методе
- **Спеки застряли в active/** — SPEC-RAG-16, 17, 18 реализованы, но не перемещены в completed/

---

## Scope

### Phase 1: Documentation Sync (блокирует всю дальнейшую работу)

**Цель:** docs отражают реальный код. Агент, читающий docs, принимает правильные решения.

| Файл | Что исправить |
|------|--------------|
| `docs/architecture/04-system/overview.md` | Embedding: pplx-embed-v1-0.6B (bf16, mean pooling). Reranker: Qwen3-Reranker-0.6B-seq-cls (chat template). Все 8 упоминаний |
| `docs/architecture/07-data-model/data-model.md` | Модели. Collection name → news_colbert_v2. 16 payload indexes (полный список) |
| `docs/architecture/05-flows/FLOW-01-ingest.md` | Embedding model. ColBERT inline (не offline). Payload indexes при создании коллекции |
| `docs/architecture/05-flows/FLOW-02-agent.md` | ANALYTICS-COMPLETE = `{final_answer}` only. Tool repeat guard. Forced completion safety net |
| `docs/architecture/03-invariants/invariants.md` | INV-06: pplx-embed. INV-08: ContextVar DONE (не техдолг). Добавить INV-14: payload indexes must match PAYLOAD_INDEXES |
| `docs/architecture/11-decisions/decision-log.md` | Добавить DEC-0042 (pplx-embed swap), DEC-0043 (Qwen3-Reranker swap). Пометить DEC-0026 и DEC-0031 как Superseded |
| `CLAUDE.md` | Модели embedding/reranker. Порты. Langfuse |
| `AGENTS.md` | Модели embedding/reranker |

**Acceptance criteria:**
- [ ] Grep по `Qwen3-Embedding` в docs/ возвращает 0 результатов (кроме decision-log где Superseded)
- [ ] Grep по `bge-reranker-v2-m3` в docs/ возвращает 0 результатов (кроме decision-log где Superseded)
- [ ] CLAUDE.md и AGENTS.md содержат pplx-embed и Qwen3-Reranker

---

### Phase 2: Ingest Safety (P0 — предотвращает повторение бага)

**Цель:** reingest не может создать коллекцию без полного набора payload indexes.

**Изменения:**

1. **Единый источник правды для indexes** — вынести `PAYLOAD_INDEXES` из `scripts/migrate_collection.py` в `src/adapters/qdrant/store.py` (или общий `src/core/qdrant_schema.py`)
2. **`store._create_payload_indices()`** — создаёт все 16 indexes (сейчас 4)
3. **`ingest_telegram.py`** — при создании новой коллекции вызывает полное создание indexes
4. **Валидация** — при старте API проверять что indexes существуют (warning log, не блокирующий)

**Acceptance criteria:**
- [ ] `_create_payload_indices()` создаёт 16 indexes
- [ ] `PAYLOAD_INDEXES` определён в одном месте и используется и в store.py, и в migrate_collection.py
- [ ] Тест: создать пустую коллекцию через store → проверить все 16 indexes
- [ ] entity_tracker(mode="top") работает без ручного создания indexes

---

### Phase 3: agent_service.py Decomposition (P2 — улучшает maintainability)

**Цель:** `stream_agent_response` разбит на понятные, тестируемые модули.

**Текущее состояние:** 838 строк, 13+ обязанностей в одном async generator.

**Предлагаемая структура:**

```
src/services/
  agent_service.py          — публичный API (stream_agent_response как оркестратор, ~100 строк)
  agent/
    __init__.py
    state.py                — AgentState, RequestContext (уже есть как dataclass, выделить)
    loop.py                 — ReAct step iteration, message management
    tool_visibility.py      — _get_step_tools(), phase logic, eviction
    guards.py               — tool repeat guard, forced search, analytics forced completion
    prompts.py              — SYSTEM_PROMPT, AGENT_TOOLS (tool schemas)
    message_history.py      — _trim_messages, _assistant_message_for_history, _tool_message_for_history
    routing.py              — _load_routing_data, _load_tool_keywords, _load_policy (data-driven routing)
```

**Правила декомпозиции:**
- Каждый модуль < 200 строк
- Каждый модуль тестируем отдельно (unit test без LLM)
- `stream_agent_response` остаётся единственным async generator, вызывает модули
- Не менять внешний контракт (SSE events, AgentRequest/AgentStepEvent)

**Acceptance criteria:**
- [ ] `stream_agent_response` < 150 строк
- [ ] Каждый модуль в `agent/` < 200 строк
- [ ] Все существующие eval тесты проходят без изменений
- [ ] SSE контракт не изменён (INV-01)

---

### Phase 4: Cleanup (P3 — гигиена)

| Задача | Описание |
|--------|----------|
| Specs → completed/ | Переместить SPEC-RAG-16, 17, 18 в `docs/specifications/completed/` |
| Eval datasets | Удалить `eval_dataset_quick.json` (superseded). DEFAULT_DATASET → eval_golden_v2.json. Убрать ссылки на несуществующие fallback файлы |
| Results directories | Консолидировать `results/` и `src/results/`. Удалить `analytics_loop_fix/` (debugging artifact) |
| Dead eval artifacts | Убрать `src/results/` — eval results должны быть только в `results/` |
| project_scope.md | Обновить метрики и текущее состояние (pplx-embed, Langfuse, Qwen3.5) |

**Acceptance criteria:**
- [ ] `docs/specifications/active/` содержит только незавершённые спеки
- [ ] `datasets/` не содержит obsolete файлов
- [ ] `results/` — единственное место для eval результатов

---

## Порядок реализации

```
Phase 1 (docs sync)     — 1 сессия, ~30 минут
  ↓
Phase 2 (ingest safety)  — 1 сессия, ~20 минут кода + тест
  ↓
Phase 3 (decomposition)  — 2-3 сессии, требует careful refactoring + полный eval прогон
  ↓
Phase 4 (cleanup)        — 1 сессия, ~15 минут
```

Phase 1 и 2 — блокирующие. Phase 3 — самая большая, но можно делать инкрементально (один модуль за сессию). Phase 4 — в любой момент.

---

## Риски

| Риск | Митигация |
|------|-----------|
| Phase 3 ломает agent loop | Полный eval прогон (36 Qs) после каждого extract. Не мержить без зелёного eval |
| Phase 3 ломает SSE контракт | Smoke test через Web UI + eval script после каждого изменения |
| Docs снова устареют | Добавить в preflight checklist: "обновить docs/architecture/ если менял модели/pipeline" |

---

## Связанные документы

- `docs/research/prompts/31-claude-agent-loop-fix.md` — контекст текущего бага
- `scripts/migrate_collection.py:54-76` — канонический список PAYLOAD_INDEXES
- `docs/architecture/00-meta/02-documentation-governance.md` — правила ведения docs
