# SPEC-RAG-26: Code Quality — Type Hints, Complexity, Logging, Exceptions

**Status**: DRAFT v3 (updated after Codex GPT-5.4 review)
**Risk**: LOW-MEDIUM
**Estimated scope**: ~40 файлов
**Depends on**: SPEC-RAG-25
**Blocks**: nothing (финальный quality layer)

---

## Контекст

**Изменения v2→v3 (после Codex review):**
- Complexity reduction: уточнено что finalization.py, coverage.py уже extracted.
  Gains от оставшейся extraction будут скромнее.
- Protocols: помечен как LOW ROI при текущих mypy settings. Опционально.
- f-string logging: нужно CI enforcement (pylint или ruff plugin)
- Exceptions: качественный review важнее числовых таргетов
- Line endings: `git add --renormalize .` — отдельный коммит, не в основном diff

---

## 1. Type hints — tools и core

### Scope

**Priority 1 — все tool functions** (15 файлов):
Добавить типы параметров + return type `-> dict[str, Any]`.

**Priority 2 — core**:
- `src/core/deps.py` — return types для всех `get_*` factory functions
- `src/core/security.py` — SecurityManager methods
- `src/core/cache.py` — уже с типами (SPEC-RAG-25)

**Priority 3 — services**:
- `src/services/agent_service.py` — public methods
- `src/services/qa_service.py` — public methods

**НЕ трогать**: scripts/*.py, private methods с очевидными типами.

---

## 2. f-string в logging (~80 случаев)

**Проблема**: `logger.error(f"failed: {exc}")` — eager formatting.
**Правильно**: `logger.error("failed: %s", exc)` — lazy.

### Подход

Per-file review, не слепой regex replace.

**Риски**: `f"Error: {obj}"` → `"Error: %s", obj` может изменить поведение
если `obj.__str__()` raises. Проверять каждый случай.

**CI enforcement**: добавить `"G"` (flake8-logging-format) в ruff select
если доступен, или pylint W1203 как отдельный CI step.

---

## 3. Broad exception handling

### Подход: качественный review, не числовой таргет

Codex правильно заметил: числовой таргет (≤60) стимулирует косметику.
Вместо этого — осмысленный per-file review:

**Правила:**
- HTTP/network → `except (httpx.HTTPError, ConnectionError)`
- JSON parsing → `except (json.JSONDecodeError, KeyError, TypeError)`
- Qdrant → `except (QdrantException, ConnectionError)`
- File I/O → `except (OSError, IOError)`
- Tool top-level и endpoint top-level → **ОСТАВИТЬ** `except Exception`

**Действие**: пройтись по каждому файлу, заменить где exception type очевиден.
Не гнаться за числом.

---

## 4. Complexity reduction — targeted extraction

### Что уже extracted (не трогаем):
- `finalization.py` — build_final_payload, trim_refusal, verify
- `coverage.py` — compute_nugget_coverage
- `formatting.py` — format_observation, extract_tool_calls
- `visibility.py` — get_step_tools
- `state.py` — AgentState, RequestContext
- `prompts.py` — SYSTEM_PROMPT

### Что остаётся в stream_agent_response():
Orchestration glue + yield points. Основная сложность — event emission
и state choreography. Codex подтвердил: gains от дальнейшей extraction
будут **скромнее** чем заявлено.

### Реалистичные extraction targets:

**4a. compose_context + refinement block** (~80 строк)
Если есть чёткий блок: coverage check → refine → re-compose.
Извлечь как async helper возвращающий `(result, list[events])`.
Вызывающий код yield'ит events.

**4b. search.search() multi-query merge** (~40 строк)
Единственная чёткая extraction — round-robin merge logic.

**4c. НЕ пытаться split весь stream_agent_response()**
Async generator с ~15 yield points. Наивный split усложнит, а не упростит.

### Ожидаемые gains (реалистичная оценка):

| Функция | Было | Ожидаемое |
|---------|------|-----------|
| `stream_agent_response` | 75 locals, 55 branches | ~55 locals, ~40 branches |
| `search()` | 46 locals, 38 branches | ~30 locals, ~25 branches |

Скромнее чем v2, но реалистично.

---

## 5. Protocols — OPTIONAL, LOW ROI

Codex: "Low immediate ROI with permissive mypy settings."

**Решение**: создать `src/core/protocols.py` с минимальным набором (Retriever, Reranker),
но НЕ делать обязательным использование во всех signatures.
Использовать по возможности при рефакторинге в SPEC-RAG-25.

Если mypy settings ужесточатся позже — protocols станут enforcement point.

---

## 6. Mixed line endings + .gitattributes

**Действие**:
- Создать `.gitattributes`
- `git add --renormalize .` — **ОТДЕЛЬНЫЙ коммит**, не в основном diff
  (иначе diff будет шумный и сложно review'ить)

---

## Acceptance criteria

1. **Type hints**: все public functions в `src/services/tools/*.py` имеют type annotations
2. **f-string logging**: `grep -rc 'logger\.\w\+(f"' src/ | awk -F: '{s+=$2}END{print s}'` ≤ 10
3. **Exceptions**: per-file review выполнен, каждая замена осознанная (не числовой таргет)
4. **Complexity**: `search()` multi-query merge extracted в отдельную function
5. **Line endings**: `.gitattributes` существует (renormalize в отдельном коммите)
6. pytest: pass, 0 fail
7. ruff check: 0 errors
8. mypy: 0 new errors vs baseline

---

## Порядок работы

1. Type hints — tools (1 P1) — 30 мин
2. Type hints — core + services (1 P2-P3) — 30 мин
3. f-string logging (2) — 1 час (per-file review)
4. Broad exceptions (3) — 1 час (per-file review)
5. Complexity — compose extraction (4a) — 1 час
6. Complexity — search merge (4b) — 30 мин
7. Protocols optional (5) — 20 мин
8. `.gitattributes` (6) — отдельный коммит, 5 мин
9. Тесты + lint — 15 мин
