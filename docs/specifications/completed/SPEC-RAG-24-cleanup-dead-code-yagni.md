# SPEC-RAG-24: Cleanup — Dead Code, Broken Imports, Garbage Files

**Status**: DRAFT v3 (updated after Codex GPT-5.4 review)
**Risk**: LOW — удаление неиспользуемого кода, нет изменений в рабочей логике
**Estimated scope**: ~15 файлов, net -500 строк
**Depends on**: nothing
**Blocks**: SPEC-RAG-25

---

## Контекст

Аудит (pylint + ручной review) выявил мёртвый код, битые импорты, мусорные файлы.
Low-hanging fruit — убрать ДО архитектурного рефакторинга.

---

## 1. Битые импорты (pylint E0611 / E0401)

### 1.1 `collections.py` — целиком мёртвый файл (ChromaDB)

**Файл**: `src/api/v1/endpoints/collections.py` (170 строк)

Весь файл зависит от `get_chroma_client` (не существует). Все 3 endpoint'а сломаны.
Router уже закомментирован в `router.py:25`.

**Действие**:
- Удалить `src/api/v1/endpoints/collections.py` целиком
- Удалить import/include из `src/api/v1/router.py` если есть
- Удалить мёртвые Pydantic схемы в `src/schemas/qa.py`: `CollectionInfo`, `CollectionsResponse`,
  `SelectCollectionRequest`, `SelectCollectionResponse` (используются ТОЛЬКО этим файлом)

### 1.2 `dedup_diversify` — удалённый модуль

Lazy import внутри `answer_v2()`. Покрывается пунктом 2.2 (удаление answer_v2).

### 1.3 `ingest_service.py` — только битый импорт, НЕ удалять файл

**ВАЖНО**: `ingest_service.py` — ЖИВОЙ. Ingest API (`api/v1/endpoints/ingest.py:16`)
импортирует `job_manager`, `ingest_service:116-118` выполняет реальный ingest path.
Удалять файл или endpoint НЕЛЬЗЯ.

**Действие**:
- Удалить ТОЛЬКО битый импорт `release_llm_vram_temporarily` из `ingest_service.py:148`
- Удалить вызов `async with release_llm_vram_temporarily():` если есть
- Проверить остальные импорты в `_run_ingest_job` на актуальность
- НЕ удалять файл, НЕ удалять endpoint

### 1.4 `model_post_init` сигнатура

**Файл**: `src/core/settings.py`

**Действие**: заменить на `@model_validator(mode="after")` — robust для Pydantic v2.

---

## 2. YAGNI — неиспользуемый код

### 2.1 `qa_service` в AgentService — хранится, не вызывается

**Файл**: `src/services/agent_service.py:65`, `src/core/deps.py:288`

```python
self.qa_service = qa_service  # НИКОГДА не используется
```

**Действие**:
- Удалить параметр `qa_service` из `AgentService.__init__`
- В `deps.py:get_agent_service()`: убрать `qa_service = get_qa_service()` и передачу
- Убирает ненужную инициализацию QAService при старте AgentService

### 2.2 `qa_service.answer_v2()` — дублирует agent pipeline

**Файл**: `src/services/qa_service.py:361-616` (255 строк)

Endpoint `/v1/qa/answer_v2` НЕ существует (проверено). Метод мёртвый.

**Действие**: удалить `answer_v2()` и весь связанный dead code внутри.

### 2.3 `router_select` — мёртвый tool

После удаления `answer_v2()` потребителей `router_select` нет.
Tool НЕ входит ни в один phase visibility, НЕ в tool_keywords.json, agent его не вызывает.

**Действие**:
- Удалить `src/services/tools/router_select.py` целиком (нет потребителей)
- Deregister из ToolRunner в `deps.py`
- Удалить export из `src/services/tools/__init__.py`
- Удалить `src/tests/test_router_select.py`

### 2.4 MMR dead path в search endpoint

**Действие**: НЕ УДАЛЯТЬ — MMR используется в основном пути.

---

## 3. Мусорные файлы

### 3.1 Удалить

| Файл | Причина |
|------|---------|
| `nul` (root) | Windows NUL device artifact |
| `test_broad.json` (root) | Stale fixture |
| `test_channel.json` (root) | Stale fixture |
| `test_request.json` (root) | Stale fixture |

### 3.2 Добавить в .gitignore

```gitignore
# Binary models (regeneratable)
datasets/bertopic_model/

# MCP tool state
.serena/memories/
.serena/cache/
```

---

## Acceptance criteria

1. `grep -r "get_chroma_client\|dedup_diversify" src/` = 0
2. `grep -r "release_llm_vram_temporarily" src/` = 0
3. `grep -r "answer_v2" src/` = 0
4. `src/api/v1/endpoints/collections.py` не существует
5. `CollectionInfo`, `CollectionsResponse` удалены из `schemas/qa.py`
6. `src/services/tools/router_select.py` не существует
7. `src/tests/test_router_select.py` не существует
8. `grep "router_select" src/core/deps.py` = 0
9. Файл `nul` не существует
10. `datasets/bertopic_model/` и `.serena/memories/` в .gitignore
11. `ingest_service.py` СУЩЕСТВУЕТ и endpoint работает
12. pytest: pass, 0 fail
13. ruff check: 0 errors

---

## Порядок работы

1. Мусорные файлы + .gitignore (3) — 5 мин
2. `collections.py` + мёртвые схемы (1.1) — 15 мин
3. `ingest_service` — ТОЛЬКО битый импорт (1.3) — 10 мин
4. `model_post_init` (1.4) — 5 мин
5. YAGNI: qa_service в AgentService (2.1) — 10 мин
6. YAGNI: answer_v2 (2.2) — 15 мин
7. YAGNI: router_select удаление (2.3) — 10 мин
8. Тесты + lint — 5 мин
