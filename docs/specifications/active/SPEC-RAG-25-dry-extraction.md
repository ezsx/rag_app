# SPEC-RAG-25: DRY Extraction — Cache Layer, Tool Formatters, Public API

**Status**: DRAFT v3 (updated after Codex GPT-5.4 review)
**Risk**: MEDIUM
**Estimated scope**: ~15 файлов, net -150 строк
**Depends on**: SPEC-RAG-24
**Blocks**: SPEC-RAG-26

---

## Контекст

Аудит нашёл 19 кластеров дублирования (pylint R0801).
Эта спека покрывает безопасные DRY-extraction'ы — без изменения архитектуры.

**Изменения v2→v3 (после Codex review):**
- Qdrant date filter helper: убран `hot_topics` (он работает с auxiliary collection, другой pattern)
- Tool formatters: добавлена оговорка что centralized formatting может быть чище
- Protected access: расширен scope — `_client`, `_embedding_client` тоже нужно fix
- Acceptance criteria: исправлены

---

## 1. Cache layer extraction

### Текущее состояние

Идентичные `get_from_cache()` / `save_to_cache()` в двух файлах:
- `src/api/v1/endpoints/search.py:59-97`
- `src/api/v1/endpoints/qa.py:24-52`

100% копипаста, ~30 строк × 2.

### Целевое состояние

**Новый файл**: `src/core/cache.py` (~30 строк)

```python
"""Обёртка Redis кеша для API endpoints."""

def cache_get(redis_client, key: str) -> dict[str, Any] | None: ...
def cache_set(redis_client, key: str, data: dict[str, Any], ttl: int) -> None: ...
```

**Действие**:
- Создать `src/core/cache.py`
- Заменить дублированные функции в обоих endpoints на импорт
- Удалить ~60 строк дублей

---

## 2. Tool observation formatting — centralized dispatch

### Текущее состояние

**`src/services/agent/formatting.py:format_observation()`** — 110 строк, 8 if/elif веток.

### Решение: dispatch table, НЕ перенос в tools

Codex review показал что перенос SSE/presentation логики в tool modules — не очевидно чище.
Вместо этого: оставить formatting centralized, но заменить if/elif на dispatch table:

```python
_FORMATTERS: dict[str, Callable[[dict], str]] = {
    "search": _format_search,
    "rerank": _format_rerank,
    "compose_context": _format_compose,
    ...
}

def format_observation(tool_name: str, result: dict) -> str:
    formatter = _FORMATTERS.get(tool_name, _format_default)
    return formatter(result)
```

Каждая `_format_*` — private function в том же файле.
Убирает if/elif chain, сохраняет centralized ownership.

**Действие**:
- Рефакторить `formatting.py`: if/elif → dispatch dict + private functions
- Файл остаётся ~80 строк (dispatch + functions), но структура чище

---

## 3. Qdrant date filter helper

### Реальный scope (скорректирован)

Дублирование date filter только в 2 файлах (НЕ 3 — `hot_topics` работает с другой коллекцией):
- `src/services/tools/entity_tracker.py` — `_build_period_filter()` helper
- `src/services/tools/arxiv_tracker.py` — inline date filter (строки 56-67)

~10-15 строк × 2 = ~25 строк дублей.

### Решение

При scope в 25 строк extraction в отдельный файл — overkill.

**Действие**: перенести `_build_period_filter()` из `entity_tracker` в `arxiv_tracker`
через copy (или inline). Не создавать новый файл ради 10 строк.

**Альтернативно**: оставить как есть — это diminishing returns. Решение на усмотрение
при реализации.

---

## 4. HybridRetriever — public API для tools

### Текущее состояние

Tools обращаются к private attributes HybridRetriever:
- `hybrid_retriever._run_sync(coro)` — 15 вызовов в 7 файлах
- `hybrid_retriever._store` — 12 вызовов в 7 файлах
- `hybrid_retriever._store._client` — 12 вызовов (Qdrant async client)
- `hybrid_retriever._embedding_client` — 2 вызова (cross_channel_compare)
- `hybrid_retriever._sparse_encoder` — 1 вызов (cross_channel_compare)

Всё это pylint W0212 (protected-access).

### Целевое состояние

```python
class HybridRetriever:
    def run_sync(self, coro):
        """Публичный sync→async bridge для tools."""
        ...

    @property
    def store(self) -> QdrantStore:
        """Публичный доступ к Qdrant store."""
        return self._store

    @property
    def embedding_client(self) -> TEIEmbeddingClient:
        """Публичный доступ к embedding client для tools (cross_channel)."""
        return self._embedding_client

    @property
    def sparse_encoder(self):
        """Публичный доступ к sparse encoder для tools (cross_channel)."""
        return self._sparse_encoder
```

`QdrantStore` уже expose'ит `client` property (`store.py:85`).
Поэтому `store._client` → `store.client`.

**Действие**:
- Добавить public properties/methods в HybridRetriever
- Обновить все tools: `_store` → `store`, `_run_sync` → `run_sync`,
  `_embedding_client` → `embedding_client`, `_store._client` → `store.client`
- ~40 замен в 7 файлах

---

## 5. DEFERRED items (не в scope этой спеки)

- **normalize_tool_params extraction** — circular deps risk (executor.py)
- **Search pipeline consolidation** — sync/async mismatch, QAService может быть удалён
- **Tool self-describing normalizers** — архитектурное изменение

---

## Acceptance criteria

1. **Cache**: единая реализация в `core/cache.py`
   - `grep -c "def get_from_cache\|def save_to_cache" src/api/` = 0
2. **Formatting**: `format_observation()` использует dispatch dict вместо if/elif chain
3. **Protected access**: `grep -r "_run_sync\|_store\b\|_embedding_client\|_sparse_encoder" src/services/tools/` = 0
   - Все обращения через public API
4. **Store client**: `grep "store\._client" src/services/tools/` = 0
   - Все через `store.client`
5. pytest: pass, 0 fail
6. ruff check: 0 errors

---

## Порядок работы

1. HybridRetriever public API (4) — 30 мин (это нужно первым, tools от него зависят)
2. Update tools: protected → public access (4) — 30 мин
3. Cache extraction (1) — 20 мин
4. Formatting dispatch table (2) — 30 мин
5. Тесты + lint — 15 мин
