## Invariants (должно быть истинно всегда)

### INV-01: SSE Event Contract

Публичный контракт событий `/v1/agent/stream` **не изменяется без явной версии API**.

Допустимые типы событий:
- `thought` — мысль агента перед action
- `tool_invoked` — вызов инструмента + параметры
- `observation` — результат инструмента
- `citations` — список источников с coverage
- `final` — финальный ответ + step count + request_id

**Запрещено**: менять имена типов, убирать поля из `data`, без версионирования.

Клиент (evaluate_agent.py и другие) строится на этом контракте.

---

### INV-02: Coverage Threshold и Refinement

- `coverage_threshold = 0.65` (configurable через `settings.coverage_threshold`)
- `max_refinements = 2` (configurable через `settings.max_refinements`)
- Если `coverage < coverage_threshold` → дополнительный поисковый раунд
- Более двух refinement loops **запрещено** без явного изменения `max_refinements`
- При `coverage < 0.30` → abort: вернуть "insufficient information" вместо галлюцинации

**Обоснование**: R04 — false-negative (пропущенный поиск) → 66.1% галлюцинаций.
Bias toward retrieval: лишний поиск стоит 200–500ms, пропущенный → уверенный неверный ответ.
F1 растёт до 3 итераций; 2 refinements = оптимальный баланс latency/качество (DEC-0019).

---

### INV-03: Tool Execution Order

Инструменты выполняются в строгом порядке:

```
router_select → query_plan → search → rerank → compose_context → [verify] → final_answer
```

- Каждый инструмент принимает явный input schema (см. system prompt)
- `compose_context` **обязателен** перед `final_answer` (проверяется через system prompt RULE 4)
- `compose_context` принимает `query` как параметр — необходимо для composite coverage
- `verify` — опциональный шаг (управляется `settings.enable_verify_step`)
- Все инструменты имеют timeout через `ToolRunner`

---

### INV-04: Secrets / PII в логах

- JWT-токены **не логируются** нигде
- API-ключи **не логируются**
- Весь внешний input проходит через `sanitize_for_logging()` перед записью в лог
- `SecurityManager` используется для всей обработки external inputs
- Промпты с PII (имена пользователей, контакты) — не логируются полностью

---

### INV-05: lru_cache Singleton Pattern

Все сервисы (`AgentService`, `QAService`, `HybridRetriever`, `RerankerService`,
`QueryPlannerService`, LLM) создаются **один раз** через `@lru_cache` в `core/deps.py`.

- Изменение настроек требует явного `cache_clear()` через `settings.update_*()`
- Фабрика LLM (`_llm_factory`) передаётся как callable для lazy loading

---

### INV-06: Qdrant Atomic Ingest

При ingest (ingest_telegram.py) документ записывается **атомарно** в Qdrant:
- dense vector (multilingual-e5-large)
- sparse vector (Qdrant/bm25, language="russian")

Рассинхронизация dense и sparse vectors недопустима.
Qdrant upsert атомарен по точке — оба вектора записываются в одной операции.

**Docker**: Qdrant storage — **только named volumes**. Bind mounts → silent data corruption на Windows.

---

### INV-07: AgentService — единственный владелец ReAct State

`AgentService` является единственным местом, где:
- создаётся `AgentState` (per-request)
- запускается ReAct цикл
- управляется coverage и refinement_count
- отправляются SSE события

Никакой другой сервис не управляет agent state напрямую.

---

### INV-08: Per-Request Isolation

Каждый вызов `stream_agent_response()` получает свой `request_id` (uuid4) и
свой экземпляр `AgentState`. Никакого shared mutable state между параллельными запросами.

**Текущий техдолг**: `AgentService._current_step` и `_current_request_id` — атрибуты
класса, не per-request. Singleton через lru_cache → concurrent requests шарят эти поля.
Целевое решение: `contextvars.ContextVar` (OPEN-01, R06).

---

### INV-09: Thinking Mode — всегда отключён

Qwen3-8B thinking mode (`<think>...</think>` блоки) **всегда отключён** в production.

- llama-server: system prompt завершается директивой `/no_think`
- vLLM (после Proxmox): `extra_body={"enable_thinking": False}` в каждом запросе
- Причина: thinking токены ломают текущий ReAct-парсер + избыточные 250–1250 токенов на шаг
- LlamaServerClient содержит safeguard: фильтрация `<think>...</think>` из ответа

---

### INV-10: System Prompt Language

System prompt пишется **на английском**. Инструкция на выходной язык отдельная:
`"Always respond to the user in Russian."` — последняя строка system prompt.

Причина: 30–40% меньше токенов, лучше instruction following для структурных задач
(JSON tool calling, ReAct формат). Не менять без A/B теста на нашем домене.
