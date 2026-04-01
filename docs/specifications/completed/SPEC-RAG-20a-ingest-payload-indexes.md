# SPEC-RAG-20a: Ingest Payload Indexes Safety

> **Status:** Draft
> **Created:** 2026-03-31
> **Parent:** SPEC-RAG-20 Phase 2
> **Context:** При reingest с новым embedding стеком коллекция пересоздана без payload indexes → entity_tracker/arxiv_tracker получали Qdrant 400 → agent loop. Root cause: `store.py` создаёт 4 из 16 indexes, остальные 12 — только в `migrate_collection.py`.

---

## Проблема

Два источника правды для payload indexes:

| Файл | Indexes | Когда вызывается |
|------|---------|------------------|
| `scripts/migrate_collection.py:54-76` | **16** (полный набор) | Вручную, перед reingest |
| `src/adapters/qdrant/store.py:126-155` | **4** (channel, date, author, message_id) | При создании коллекции через API |

Если `migrate_collection.py` не запущен → analytics tools не работают (Facet API требует index на `entities`, `year_week`, `arxiv_ids`).

---

## Решение

### 1. Единый `PAYLOAD_INDEXES` в `src/adapters/qdrant/store.py`

Перенести канонический список из `migrate_collection.py` в `store.py`. `migrate_collection.py` будет импортировать оттуда.

```python
# src/adapters/qdrant/store.py

PAYLOAD_INDEXES = [
    # Critical — используются почти во всех запросах
    ("channel", models.KeywordIndexParams(type="keyword", is_tenant=True)),
    ("date", models.DatetimeIndexParams(type="datetime", is_principal=True)),
    ("entities", models.PayloadSchemaType.KEYWORD),
    ("year_week", models.PayloadSchemaType.KEYWORD),
    # Secondary — для специализированных tools
    ("entity_orgs", models.PayloadSchemaType.KEYWORD),
    ("entity_models", models.PayloadSchemaType.KEYWORD),
    ("arxiv_ids", models.PayloadSchemaType.KEYWORD),
    ("hashtags", models.PayloadSchemaType.KEYWORD),
    ("url_domains", models.PayloadSchemaType.KEYWORD),
    ("lang", models.PayloadSchemaType.KEYWORD),
    ("forwarded_from_id", models.PayloadSchemaType.KEYWORD),
    ("year_month", models.PayloadSchemaType.KEYWORD),
    ("root_message_id", models.PayloadSchemaType.KEYWORD),
    ("author", models.PayloadSchemaType.KEYWORD),
    ("message_id", models.IntegerIndexParams(type="integer", lookup=True, range=True)),
    # Range
    ("text_length", models.IntegerIndexParams(type="integer", lookup=False, range=True)),
]
```

### 2. `_create_payload_indices()` создаёт все 16

Итерация по `PAYLOAD_INDEXES`, логирование каждого index.

### 3. `migrate_collection.py` импортирует из store.py

```python
from src.adapters.qdrant.store import PAYLOAD_INDEXES
```

Убрать дублированный список.

### 4. Startup validation (warning, не blocking)

При инициализации `QdrantStore` — проверить что indexes существуют. Если нет — warning в лог с конкретным списком отсутствующих.

---

## Файлы для изменения

| Файл | Изменение |
|------|-----------|
| `src/adapters/qdrant/store.py` | Добавить `PAYLOAD_INDEXES`, расширить `_create_payload_indices()`, добавить `_verify_payload_indices()` |
| `scripts/migrate_collection.py` | Импортировать `PAYLOAD_INDEXES` из store.py, удалить дубликат |

---

## Acceptance Criteria

- [ ] `PAYLOAD_INDEXES` определён в одном месте (`store.py`)
- [ ] `_create_payload_indices()` создаёт 16 indexes
- [ ] `migrate_collection.py` использует тот же список (без дубликата)
- [ ] Startup warning если indexes отсутствуют
- [ ] entity_tracker(mode="top") работает на свежесозданной коллекции без ручных шагов
