## Open Questions

> Правило: закрытый вопрос → удаляем отсюда + добавляем DEC-xxxx в decision-log.

---

### OPEN-01: Shared state в AgentService

`AgentService._current_step` и `_current_request_id` — атрибуты экземпляра класса.
`AgentService` создаётся как singleton через `@lru_cache`.
При параллельных запросах эти поля перезаписываются.

**Решение (R06)**: `contextvars.ContextVar` для async-safe per-request state.
**Статус**: решение принято, реализация ждёт Phase 2 (вместе с vLLM / async refactor).
**Приоритет**: P1

---

### OPEN-02: Blocking LLM HTTP calls

`LlamaServerClient` использует `requests.Session.post()` — синхронный HTTP вызов,
который блокирует uvicorn event loop.
При параллельных SSE-стримах второй запрос ждёт завершения первого.

**Решение промежуточное (R06, DEC-0021)**: заменить на `httpx.AsyncClient` — минимальные изменения.
**Решение финальное**: AsyncOpenAI + vLLM после Proxmox.
**Статус**: промежуточный фикс запланирован (Phase 0), не реализован.
**Приоритет**: P1

---

### OPEN-04: Settings без Pydantic

`src/core/settings.py` использует `os.getenv()` без валидации.

**Вопрос**: мигрировать ли на `pydantic-settings.BaseSettings`?
- Опция A: да, с сохранением `update_*()` методов через `model_copy(update={})`
- Опция B: не мигрировать — текущая структура работает, риск рефакторинга не оправдан

**Приоритет**: P2 (не блокирует ничего)

---

### OPEN-05: QAService — дублирование RAG

`QAService` и `AgentService` оба делают RAG.
`AgentService` является расширенным вариантом `QAService`.

**Вопрос**: нужен ли QAService?
- Опция A: deprecated QAService, только AgentService для всего
- Опция B: QAService остаётся для простых запросов (быстрее, меньше latency)

**Приоритет**: P2

---

### OPEN-06: Eval dataset — реальные данные

`eval_dataset.json` содержит 2 фейковых примера.
Document IDs не существуют в Qdrant.

**Решение (R05)**: `generate_eval_dataset.py` — выборка из Qdrant → LLM генерирует вопросы
с принудительным распределением типов → critique-фильтр → 200 примеров.
**Статус**: скрипт спроектирован (R05), требует задеплоенного Qdrant (Phase 1).
**Приоритет**: P0 для meaningful evaluation

---

### OPEN-08: vLLM v0.15.1 + Qwen3 совместимость

Qwen3 вышел апрель 2025. vLLM v0.15.1 — до апреля 2025.
Неизвестно, поддерживает ли v0.15.1 Qwen3 архитектуру.

**Как снять**: тест при настройке vLLM после Proxmox.
Запасной вариант: ждать vLLM v0.16+ с xformers backport или llama-server + GGUF навсегда.
**Приоритет**: P1 (после Proxmox)

---

> **Закрытые вопросы** (перенесены в decision-log):
> - ~~OPEN-03: ChromaDB → Qdrant~~ → DEC-0015 ✅
> - ~~OPEN-07: Coverage metric~~ → DEC-0018, DEC-0019 ✅
