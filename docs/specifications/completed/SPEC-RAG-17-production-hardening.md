# SPEC-RAG-17: Production Hardening — Bug Fixes & Safety

> **Status**: Draft → Review (v2, post-Codex review)
> **Найдено**: параллельный independent review Claude + Codex GPT-5.4 (2026-03-28)
> **Scope**: concrete code bugs, safety gaps, concurrency races. Без new features.

---

## Цель

Устранить конкретные баги и production gaps, найденные при independent code review двумя агентами. Каждый fix — isolated, testable, не ломает API контракт.

---

## P0 — Critical (ломает корректность)

### FIX-01: Request isolation в AgentService + collection race

**Проблема (часть A — agent state)**: `AgentService` — singleton (`@lru_cache` в `deps.py:183`), но пишет request-specific state в `self` (`agent_service.py:560-572`): `_agent_state`, `_current_request_id`, `_last_search_hits` и т.д. При двух параллельных запросах — cross-request data bleed.

**Проблема (часть B — collection mutation)**: request path мутирует глобальные settings:
- `agent.py:90` — `settings.update_collection(request.collection)`
- `qa.py:81`, `qa.py:222` — аналогично
- `update_collection()` сбрасывает process-wide caches (`settings.py:230`, `settings.py:248`)
- Если два запроса с разными коллекциями перекрываются — один видит чужую коллекцию

**Найден**: Claude (часть A) + Codex (часть B, review feedback).

**Fix (часть A)**: Вынести весь request state в dataclass `RequestContext`, передавать как локальный аргумент через цепочку вызовов. `AgentService` становится stateless — никаких `self._current_*`.

**Fix (часть B)**: Убрать `settings.update_collection()` из request path. Вместо этого — передавать `collection_name` как явный параметр в retriever/store/service stack:
```python
# Вместо:
settings.update_collection(request.collection)
agent_service = get_agent_service()  # singleton с мутированным state

# Делаем:
agent_service.stream_agent_response(request, collection_override=request.collection)
# Внутри: retriever получает collection как аргумент, не из settings
```

**Файлы**:
- `src/services/agent_service.py` (lines 560-572, все `self._current_*`, `self._last_*`)
- `src/api/v1/endpoints/agent.py:88-98` (collection switch)
- `src/api/v1/endpoints/qa.py:81,173,222,260` (collection switch)
- `src/core/settings.py:230,248` (`update_collection` cache clear)

**Acceptance**:
- Два параллельных запроса к `/v1/agent/stream` с разными queries → citations не перемешиваются
- Два параллельных запроса с разными `collection` → каждый видит свою коллекцию
- `settings.update_collection()` НЕ вызывается из request path

---

### FIX-02: Coverage metric — stop-words fallback

**Проблема**: `_query_term_coverage()` в `compose_context.py:77-78` возвращает `1.0` если все токены запроса — стоп-слова (длина <3 или в стоп-листе). Запрос "какие модели" → tokens=[] → coverage=1.0 → agent считает что всё нашёл → не делает refinement.

**Найден**: Claude.

**Fix**: `return 0.5` вместо `return 1.0` (нейтральное значение — "неизвестно", не "идеально").

**Файл**: `src/services/tools/compose_context.py:78`

**Acceptance**: запрос из одних стоп-слов → term_coverage ≤ 0.5, refinement срабатывает при низком coverage от остальных сигналов.

---

### FIX-03: Rate limiter — overwrite bug

**Проблема**: в `rate_limit.py:155-157`:
```python
request_history.append((now, endpoint))           # line 156: добавили в полный список
self.requests[client_id] = requests_last_hour     # line 157: ПЕРЕЗАПИСАЛИ отфильтрованным!
```
`request_history` (`line 113`) — полный список из `self.requests[client_id]`.
`requests_last_hour` (`line 123`) — **новый** list comprehension (отфильтрованная копия).
Line 156 добавляет запрос в `request_history` (т.е. в `self.requests[client_id]`).
Line 157 тут же **перезаписывает** `self.requests[client_id]` копией `requests_last_hour`, в которой текущего запроса нет.

**Найден**: Codex.

**Fix (каноническая правка)**:
```python
# Line 155-157 ЗАМЕНИТЬ на:
requests_last_hour.append((now, endpoint))        # добавляем в отфильтрованный список
self.requests[client_id] = requests_last_hour     # сохраняем (GC удалит старые записи)
```
Так текущий запрос учтён И старые записи (>1 часа) отфильтрованы.

**Файл**: `src/core/rate_limit.py:155-157`

**Acceptance**: unit test — 60 запросов за минуту → 61-й возвращает retry-after. Тест на burst: 10 запросов за 10 сек → 11-й блокируется.

---

## P1 — High (security / reliability)

### FIX-04: Tool name whitelist (по visible set)

**Проблема**: `_extract_tool_calls()` в `agent_service.py:1802` принимает любой `tool_name` от LLM без проверки. Если Qwen3 галлюцинирует "search_documents" — вызов уходит в ToolRunner, error, LLM может повторить N раз.

**Найден**: Claude. Уточнение от Codex: валидировать не против полного AGENT_TOOLS, а против **текущего visible set** — так phase-based visibility остаётся hard constraint.

**Fix**:
```python
# В _extract_tool_calls, добавить параметр visible_tools: Set[str]
def _extract_tool_calls(self, assistant_message, visible_tools: Set[str]):
    ...
    if tool_name not in visible_tools:
        logger.warning("LLM called %s outside visible set %s, skipping",
                       tool_name, visible_tools)
        continue
```

**Файл**: `src/services/agent_service.py:1770-1811`

**Acceptance**: LLM вызывает tool не из текущей фазы → skip с warning, не error loop.

---

### FIX-05: JWT secret — hard fail без explicit dev mode

**Проблема**: `auth.py:17` — insecure default `"your-secret-key-change-in-production"`.

**Найден**: Claude + Codex. Codex review: "soft warning" недостаточен — это security hole, нужен hard fail.

**Fix**:
```python
JWT_SECRET = os.getenv("JWT_SECRET")
_DEV_MODE = os.getenv("RAG_ENV", "production").lower() in ("dev", "development", "local")

if not JWT_SECRET:
    if _DEV_MODE:
        JWT_SECRET = "dev-only-insecure-secret"
        logger.warning("JWT_SECRET not set, using insecure dev default (RAG_ENV=%s)",
                       os.getenv("RAG_ENV"))
    else:
        raise RuntimeError(
            "JWT_SECRET environment variable is required in production. "
            "Set RAG_ENV=dev to use insecure default for local development."
        )
```

**Файл**: `src/core/auth.py:17`

**Acceptance**: без `JWT_SECRET` + `RAG_ENV=production` (default) → app не стартует. С `RAG_ENV=dev` → warning + insecure default. Docker compose dev → добавить `RAG_ENV=dev`.

---

### FIX-06: Auth на все read endpoints

**Проблема**: без auth:
- `/v1/qa` (`qa.py:59`)
- `/v1/qa/stream` (`qa.py:198`)
- `/v1/search/plan` (`search.py:100`)
- `/v1/search` (`search.py:111`)

**Найден**: Codex (исходный + расширение в review).

**Fix**: добавить `current_user: TokenData = Depends(require_read)` на все 4 endpoint'а.

**Файлы**: `src/api/v1/endpoints/qa.py`, `src/api/v1/endpoints/search.py`

**Acceptance**: запрос без Bearer token на любой из 4 endpoints → 401.

---

### FIX-07: CORS allowlist

**Проблема**: `main.py:109,113` — `allow_origins=["*"]` + `allow_credentials=True`.

**Найден**: Codex.

**Fix**:
```python
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8001").split(",")
```

**Файл**: `src/main.py:109-113`

**Acceptance**: запрос с неизвестного origin + credentials → CORS block.

---

## P2 — Medium (reliability / quality)

### FIX-08: Request-level timeout (cooperative deadline)

**Проблема**: суммарно agent может работать неограниченно (5 tools × 15s = 75s+).

**Найден**: Claude. Codex review: `asyncio.timeout()` не поможет — `_execute_action()` блокирует в `ThreadPoolExecutor` (`tool_runner.py:31`), async cancellation не прерывает blocking thread.

**Fix**: wall-clock deadline, проверяемый **между** blocking actions:
1. **Deadline check** перед каждым шагом main loop:
```python
deadline = time.monotonic() + (settings.agent_request_timeout or 90)

while step <= max_steps:
    if time.monotonic() > deadline:
        yield error_event("Request timeout")
        break
    ...
```
2. **Remaining budget** передаётся в ToolRunner: `timeout_sec=min(tool_timeout, deadline - now)` — каждый tool получает не больше остатка.

**Ограничение**: это cooperative timeout — deadline проверяется между шагами, а не прерывает blocking thread mid-execution. `ThreadPoolExecutor` в `tool_runner.py:31` и `query_planner_service.py:383` по-прежнему ждут `future.result(timeout=...)`. Жёсткий preemptive cancel (kill thread / process isolation) — отдельный scope, несоразмерный текущему spec.

**Файлы**: `src/services/agent_service.py:555`, `src/services/tools/tool_runner.py`

**Acceptance**: deadline checked between every blocking action. Worst case overshoot = один tool timeout (~15s). Тест: 4 шага по 25s каждый → request завершается после ~90s (deadline hit перед 4-м шагом), а не после 100s.

---

### FIX-09: UI — server-side demo token

**Проблема**: `index.html:81-82` — `body: '{"key":"1"}'` для получения JWT. Admin key виден в client-side коде.

**Найден**: Claude + Codex. Codex review: "читать ADMIN_KEY из env в HTML" невалидно — ключ всё равно раскрыт клиенту.

**Fix**: добавить endpoint `/v1/auth/demo`, доступный **только при явном opt-in** через `ENABLE_DEMO_AUTH=true`:

```python
_DEMO_AUTH_ENABLED = os.getenv("ENABLE_DEMO_AUTH", "").lower() in ("true", "1")

@router.post("/auth/demo")
async def get_demo_token():
    """Read-only demo token, не требует ключа. Только в demo/dev режиме."""
    if not _DEMO_AUTH_ENABLED:
        raise HTTPException(403, "Demo auth disabled")
    return {"token": create_token(sub="demo", scopes=["read"], hours=1)}
```

UI вызывает `/v1/auth/demo` вместо `/v1/auth/admin`. Docker compose dev → добавить `ENABLE_DEMO_AUTH=true`.

**Файлы**: `src/api/v1/endpoints/auth.py`, `src/static/index.html:81-82`, `deploy/compose/compose.dev.yml`

**Acceptance**: без `ENABLE_DEMO_AUTH=true` → endpoint возвращает 403. С флагом → read-only token, write-операции запрещены. В production по умолчанию отключен.

---

## Не входит в scope

- New features (hot_topics, channel_expertise)
- Research tracks (NLI, robustness, RAG necessity)
- Ablation study, eval expansion
- Refactoring agent loop (декомпозиция на plan/act/observe) — отдельный spec
- HybridRetriever thread shutdown — low risk, отдельный PR

---

## Порядок реализации

```
1. FIX-01 (request isolation + collection race)  — самый сложный (~80 строк, затрагивает agent+qa+settings)
2. FIX-02 (coverage bug)                         — 1 строка
3. FIX-03 (rate limiter)                         — 3 строки
4. FIX-04 (tool whitelist by visible set)        — ~10 строк
5. FIX-05 (JWT hard fail)                        — ~10 строк + RAG_ENV в compose
6. FIX-06 (auth endpoints)                       — ~10 строк (4 endpoints)
7. FIX-07 (CORS)                                 — ~5 строк
8. FIX-08 (request timeout, deadline-based)      — ~15 строк
9. FIX-09 (demo token endpoint)                  — ~15 строк (new endpoint + UI change)
```

Общий effort: **~150 строк кода**, 1-2 дня с тестированием.
