# Tool Policy

Этот файл загружается всегда. Правила — **ЖЁСТКИЕ, не рекомендательные**.

---

## ЗАПРЕЩЕНО

1. **НИКОГДА** не начинать поиск с встроенных `Grep` / `Glob` инструментов Claude.
2. **НИКОГДА** не использовать встроенные `Grep` / `Glob` при живом MCP.
3. **НИКОГДА** не пропускать `repo-semantic-search` с отговоркой "не знаю, поднят ли" —
   сначала вызови `index_status()`, затем переходи к fallback при явной ошибке.

---

## Инструменты и назначение

### `repo-semantic-search` — ПЕРВЫЙ выбор для любого поиска

Semantic + hybrid retrieval, две коллекции: `code` и `docs`.

| Инструмент | Когда |
|-----------|-------|
| `hybrid_search_code(query, top_k, path_prefix?, chunk_types?, domain_tags?)` | Код по смыслу — **ВСЕГДА первый для code-поиска** |
| `hybrid_search_docs(query, top_k, path_prefix?, domain_tags?)` | Доки, specs, ADR — **ВСЕГДА первый для doc-поиска** |
| `hybrid_search(query, top_k, scope="all"/"code"/"docs")` | Cross-domain одновременно |
| `semantic_search_code/docs(...)` | Чисто векторный, когда hybrid даёт шум |
| `find_similar_chunk(scope, chunk_id, top_k)` | Похожие чанки к уже найденному |
| `read_chunk(scope, chunk_id)` | Полный текст чанка |
| `index_status()` | Проверить статус индекса и watcher |
| `rebuild_index()` | Полный пересброс обеих коллекций |
| `reindex_paths(paths)` | Точечная переиндексация после изменений |
| `update_include_globs(globs)` | Изменить что индексируется + rebuild; `["auto"]` — авто-детект |

**`path_prefix`** для сужения поиска: `"src/services"`, `"src/adapters"`, `"docs/ai"`, `"scripts"`, `"datasets"`

**`chunk_types`** для кода: `["class_summary","python_method","python_function","module_preamble"]`
**`chunk_types`** для структурных файлов: `["json_array_item","json_dict_key","yaml_section","toml_section"]`

**`domain_tags`** — строятся из пути автоматически:
- `src/services/agent_service.py` → `["src", "services"]`
- `src/api/v1/endpoints/agent.py` → `["src", "api"]`
- `src/core/settings.py` → `["src", "core"]`
- `src/tests/test_*.py` → `["src", "tests"]`
- `docs/ai/planning/foo.md` → `["docs"]`
- `agent_context/modules/agent.md` → `["agent_context"]`
- `scripts/evaluate_agent.py` → `["scripts"]`
- `datasets/eval_dataset.json` → `["datasets"]`

---

### `serena` — символьный анализ (после shortlist от repo-semantic)

| Инструмент | Когда |
|-----------|-------|
| `find_symbol(name_path, include_body, depth)` | Найти определение по имени |
| `get_symbols_overview(relative_path)` | Обзор символов файла/директории |
| `find_referencing_symbols(name_path)` | Все места вызова символа |
| `replace_symbol_body(...)` | Точечная замена тела функции/класса |
| `search_for_pattern(pattern, relative_path)` | Regex в конкретном файле |

---

### `ripgrep` MCP — строковый fallback

Только когда нужен точный идентификатор и repo-semantic не нашёл:
`mcp__ripgrep__search`, `mcp__ripgrep__advanced-search`

---

### `ast-grep` — структурный AST-поиск

Для паттернов вызовов и structural refactor:
`find_code(pattern, lang, path)`, `find_code_by_rule(rule)`

---

### `code-index` — альтернативный индекс

Если `repo-semantic-search` недоступен:
`search_code_advanced`, `get_symbols_overview`, `get_file_summary`

---

### `shell` (Bash) — только runtime

Только для: `docker`, `compose`, `git`, `pytest`, `python scripts/...`.
**Не использовать** для поиска кода или чтения файлов.

---

## Decision Tree

```
Найти файл/код/документ по смыслу
    → hybrid_search_code() или hybrid_search_docs()
    → затем serena.find_symbol() для точного символа

Найти все места вызова функции X
    → serena.find_referencing_symbols(X)
    → если нет → ast-grep.find_code(паттерн)

Найти файлы с точной строкой/идентификатором
    → hybrid_search_code(идентификатор)
    → если нет → ripgrep.search(идентификатор)

Понять структуру директории/пакета
    → serena.get_symbols_overview(path)

Найти cross-domain контекст (код + доки)
    → hybrid_search(scope="all", query=...)
    → read_chunk() для деталей

Найти похожие функции/файлы
    → find_similar_chunk(scope, chunk_id)
```

---

## Типовые сценарии в этом репо

| Задача | Вызов |
|--------|-------|
| Понять ReAct цикл | `hybrid_search_code("ReAct agent step loop coverage refinement")` |
| Найти инструменты агента | `hybrid_search_code("tool_runner register search compose_context", path_prefix="src/services/tools")` |
| Retrieval pipeline | `hybrid_search_code("hybrid retriever BM25 RRF fusion chroma")` |
| Evaluation скрипт | `hybrid_search_code("evaluate agent recall coverage latency", path_prefix="scripts")` |
| Спецификации | `hybrid_search_docs("agent spec evaluation plan", path_prefix="docs/ai")` |
| Настройки | `hybrid_search_code("settings LLM model key context size", path_prefix="src/core")` |
| Ingest pipeline | `hybrid_search_code("ingest telegram channel chroma BM25")` |

---

## Fallback-порядок (строгий)

```
1. repo-semantic-search  (hybrid_search_code / hybrid_search_docs)
2. serena                (find_symbol / search_for_pattern)
3. ripgrep MCP
4. ast-grep MCP
5. code-index MCP
6. встроенный Grep/Glob — ТОЛЬКО при явной ошибке всех MCP выше
```

---

## Когда подключать модули

- Для доменных правил читай только релевантный модуль из `agent_context/modules/*`.
