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

Клиент (evaluate_agent.py, Web UI) строится на этом контракте.

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

LLM tools выполняются в порядке:

```
query_plan → search → rerank → compose_context → [verify] → final_answer
```

- Agent использует **native function calling** (tools parameter), не text ReAct parsing
- `final_answer` **скрыт** до выполнения `search` (dynamic tool visibility)
- Если LLM не вызывает tools → **forced search** с оригинальным запросом
- `compose_context` **обязателен** перед `final_answer` (RULE 4 в system prompt)
- `compose_context` принимает `query` как параметр — необходимо для composite coverage
- `verify` и `fetch_docs` — системные вызовы, не LLM tools
- Все инструменты имеют timeout через `ToolRunner`

---

### INV-04: Secrets / PII в логах

- JWT-токены **не логируются** нигде
- API-ключи **не логируются**
- Весь внешний input проходит через `sanitize_for_logging()` перед записью в лог
- `SecurityManager` используется для всей обработки external inputs

---

### INV-05: lru_cache Singleton Pattern

Все сервисы (`AgentService`, `QAService`, `HybridRetriever`, `RerankerService`,
`QueryPlannerService`, LLM) создаются **один раз** через `@lru_cache` в `core/deps.py`.

- Изменение настроек требует явного `cache_clear()` через `settings.update_*()`
- Фабрика LLM (`_llm_factory`) передаётся как callable для lazy loading

---

### INV-06: Qdrant Atomic Ingest

При ingest (ingest_telegram.py) документ записывается **атомарно** в Qdrant:
- dense vector (Qwen3-Embedding-0.6B, 1024-dim)
- sparse vector (Qdrant/bm25, language="russian")

Рассинхронизация dense и sparse vectors недопустима.
Qdrant upsert атомарен по точке — оба вектора записываются в одной операции.
ColBERT vectors добавляются offline (отдельный batch-процесс).

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

### INV-09: Thinking Mode — управляется через reasoning-budget

Qwen3-30B-A3B thinking mode управляется через llama-server `--reasoning-budget 0`.

- При `reasoning-budget 0`: модель не генерирует `<think>` блоки
- LlamaServerClient содержит safeguard: фильтрация `<think>...</think>` из ответа

---

### INV-10: System Prompt Language

System prompt пишется **на английском**. Инструкция на выходной язык отдельная:
`"Always respond to the user in Russian."` — последняя строка system prompt.

Причина: 30–40% меньше токенов, лучше instruction following для структурных задач
(JSON tool calling). Не менять без A/B теста на нашем домене.

---

### INV-11: Multi-Query Search

Все subqueries из query_plan выполняются через round-robin merge:
- Каждый subquery → отдельный `search_with_plan()` вызов
- Результаты чередуются (round-robin interleaving), не сортируются по dense_score
- Оригинальный запрос пользователя **всегда** добавляется в subqueries (BM25 keyword match)
- Dedup по document ID

**Обоснование**: сортировка по dense_score re-promotes attractor documents, отменяя ColBERT ranking (DEC-0028).

---

### INV-12: Channel Dedup

После retrieval: max 2 документа из одного канала.
Prolific каналы (gonzo_ml, ai_machinelearning_big_data) монополизируют top-10 без dedup.
Запрашиваем k×2 из Qdrant, dedup сужает до k.

**Не использовать Qdrant group_by** — не работает с multi-stage prefetch.

---

### INV-13: ColBERT Fallback

Если ColBERT vectors недоступны в коллекции (или gpu_server не отвечает):
- Pipeline fallback на 2-stage: BM25+Dense → RRF → cross-encoder rerank
- Без ColBERT MaxSim rerank stage
- Логируется warning, не error
