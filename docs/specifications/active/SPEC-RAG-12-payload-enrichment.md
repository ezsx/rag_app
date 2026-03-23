# SPEC-RAG-12: Payload Enrichment + Re-ingest

> **Статус**: Active
> **Создан**: 2026-03-23
> **Research basis**: R17-deep-domain-specific-tools §2
> **Зависимости**: нет (фундамент для SPEC-RAG-13, 14, 15)
> **Review**: GPT-5.4 review 2026-03-23 — fixes applied

---

## Цель

Текущий payload в Qdrant содержит 7 полей (text, channel, channel_id, message_id, date, url, point_id). **Ни одного payload индекса** — фильтрация по channel/date = full scan 13K docs.

Обогатить payload новыми полями, создать индексы, пере-ingest все 13124 документа. Это фундамент для всех новых tools (SPEC-RAG-13–16).

## Контекст

**Что уже есть**:
- `scripts/ingest_telegram.py` — Telethon → Qdrant pipeline
- `_build_point_docs_flat()` — строит payload из Message objects
- `gather_messages()` — получает сообщения из Telegram API
- Dense (Qwen3-Embedding-0.6B) + Sparse (BM25 fastembed) + ColBERT (jina-colbert-v2)
- Коллекция `news_colbert`: 13124 points, vectors: dense_vector(1024), colbert_vector(multi-128)
- `QdrantStore` использует `AsyncQdrantClient` с sync bridge через dedicated event loop
- `QdrantStore._create_payload_indices()` уже создаёт 4 индекса (channel, date, author, message_id) — но они отсутствуют в текущей `news_colbert` (коллекция создавалась до этого кода)

**Что отсутствует**:
- Payload indexes на существующей коллекции (код есть, но не применён)
- Поля: entities, arxiv_ids, urls, hashtags, lang, is_forward, reply_to_msg_id, media_types, year_week
- NER entity extraction
- URL/link parsing

**Критичная деталь: chunks vs posts**. Один Telegram message может разбиваться на несколько Qdrant points (chunks). Facet counts считают points, не posts. Для корректного подсчёта постов нужно поле `root_message_id` (message_id без chunk suffix) и dedup при агрегации.

---

## Что менять

### 1. Новый файл: `scripts/payload_enrichment.py`

Модуль извлечения метаданных из текста и Telethon Message.

```python
@dataclass
class EnrichedPayload:
    """Все поля обогащённого payload."""
    # Из Message object (Telethon)
    is_forward: bool
    forwarded_from_id: str | None   # str(channel_id) — keyword indexed
    forwarded_from_name: str | None # from_name if available
    reply_to_msg_id: int | None
    media_types: list[str]          # ["photo", "video", "document", ...]

    # Из текста (regex extraction)
    entities: list[str]             # Canonical entity names ["OpenAI", "GPT-5", ...]
    entity_orgs: list[str]          # ["OpenAI", "Google", "Anthropic", ...]
    entity_models: list[str]        # ["GPT-5", "Gemini-2.5", "Qwen3", ...]
    urls: list[str]                 # Все URLs из текста
    url_domains: list[str]          # ["arxiv.org", "github.com", ...]
    arxiv_ids: list[str]            # ["2602.03442", "2411.15399", ...]
    github_repos: list[str]         # ["user/repo", ...]
    hashtags: list[str]             # ["ai", "ml", "LLM", ...]

    # Derived
    year_week: str                  # "2025-W40"
    year_month: str                 # "2025-10"
    lang: str                       # "ru" | "en"
    text_length: int
    has_arxiv: bool
    has_links: bool


def extract_from_message(msg: Message) -> dict:
    """Извлечь metadata из Telethon Message object."""
    ...

def extract_from_text(text: str) -> dict:
    """Regex-based extraction из текста."""
    ...

def build_entity_dictionary() -> dict[str, str]:
    """Загрузить словарь entity aliases → canonical name."""
    ...
```

### 2. Entity dictionary: `datasets/entity_dictionary.json`

JSON файл с ~300-500 AI/ML entities и alias-ами:

```json
{
  "OpenAI": ["openai", "open ai", "Open AI", "оупенай"],
  "GPT-4o": ["gpt4o", "gpt-4o", "GPT 4o", "ГПТ-4о"],
  "GPT-5": ["gpt5", "gpt-5", "GPT 5", "ГПТ-5"],
  "NVIDIA": ["nvidia", "нвидиа", "Нвидиа"],
  "Anthropic": ["anthropic", "антропик"],
  "Claude": ["claude", "клод"],
  "DeepSeek": ["deepseek", "дипсик"],
  "Qwen": ["qwen", "квен"],
  "Gemini": ["gemini", "джемини"],
  "Meta": ["meta ai", "Meta AI"],
  "Llama": ["llama", "лама"],
  "HuggingFace": ["huggingface", "hugging face", "хаггинг"],
  "Яндекс": ["yandex", "яндекс"],
  "Сбер": ["sber", "сбер", "sberbank"]
}
```

**Построение словаря**: начать с top entities из R17 анализа (OpenAI 107, GPT 70, Google 61, NVIDIA 41, Anthropic 41...), расширить до ~500 записей покрывающих модели, компании, фреймворки, конференции.

Словарь группируется:
- `orgs`: OpenAI, Google, Anthropic, Meta, NVIDIA, Яндекс, Сбер, ...
- `models`: GPT-5, Claude, Gemini, Qwen3, Llama, DeepSeek-V3, ...
- `frameworks`: PyTorch, TensorFlow, LangChain, LlamaIndex, vLLM, ...
- `conferences`: NeurIPS, ICML, ICLR, GTC, CES, ...

### 3. Модифицировать `_build_point_docs_flat()` в `ingest_telegram.py`

Добавить вызов enrichment + `root_message_id` для dedup при facet:

```python
from scripts.payload_enrichment import extract_from_message, extract_from_text

# В _build_point_docs_flat():
payload = {
    "text": text,
    "channel": channel_name,
    "channel_id": int(message.chat_id),
    "message_id": int(message.id),
    "date": _to_utc_naive(message.date).isoformat(),
    "url": f"https://t.me/{channel_name}/{message.id}",
    "point_id": point_id,
    # root_message_id: уникальный ID поста (без chunk suffix).
    # Нужен для dedup при facet — один message = несколько points.
    "root_message_id": f"{channel_name}:{message.id}",
}

# Metadata из Message object (is_forward, reply_to_msg_id, media_types)
msg_meta = extract_from_message(message)
payload.update(msg_meta)

# Metadata из текста (entities, urls, arxiv_ids, hashtags, lang, ...)
text_meta = extract_from_text(text)
payload.update(text_meta)

# Author (уже есть)
if author:
    payload["author"] = author
```

### 4. Стратегия миграции: `news_colbert_v2` → `.env` switch

**Проблема**: `QdrantStore.ensure_collection()` создаёт только dense+sparse vectors, без ColBERT. Новая коллекция должна повторять полную vector config существующей `news_colbert`.

**Решение**: отдельный скрипт миграции `scripts/migrate_collection.py`:

1. Создать `news_colbert_v2` с **полной** vector config (dense 1024 + sparse BM25 + ColBERT multi-128)
2. Создать все payload indexes **до** загрузки данных
3. Re-ingest всех 36 каналов в `news_colbert_v2`
4. Проверить point count и smoke test
5. Переключить: обновить `QDRANT_COLLECTION` в `.env` с `news_colbert` на `news_colbert_v2`. Перезапустить API контейнер. Это единственный rollout path — без aliases, без rename.
6. Smoke test на переключённой коллекции
7. Старая `news_colbert` остаётся как rollback. Удалить только после подтверждения (не раньше 1 дня)

**Не использовать** `QdrantStore.ensure_collection()` — она не знает про ColBERT vector. Создание коллекции — ручное через скрипт миграции.

### 5. Создать payload indexes

В скрипте `scripts/migrate_collection.py`:

```python
from qdrant_client import models

COLLECTION = "news_colbert"

# Critical indexes
client.create_payload_index(COLLECTION, "channel",
    field_schema=models.KeywordIndexParams(
        type="keyword",
        is_tenant=True  # оптимизирует сегменты по каналам
    ))
client.create_payload_index(COLLECTION, "date",
    field_schema="datetime")
client.create_payload_index(COLLECTION, "entities",
    field_schema="keyword")
client.create_payload_index(COLLECTION, "year_week",
    field_schema="keyword")

# Secondary indexes
for field in ["entity_orgs", "entity_models", "arxiv_ids",
              "hashtags", "url_domains", "lang",
              "forwarded_from_id", "year_month", "root_message_id"]:
    client.create_payload_index(COLLECTION, field,
        field_schema="keyword")

# Range indexes
client.create_payload_index(COLLECTION, "text_length",
    field_schema=models.IntegerIndexParams(
        type="integer", lookup=False, range=True
    ))
```

### 5. Language detection

Простая эвристика (без модели):

```python
import re

_CYRILLIC_RE = re.compile(r'[а-яА-ЯёЁ]')

def detect_lang(text: str) -> str:
    """Быстрое определение языка по доле кириллицы."""
    if not text:
        return "unknown"
    cyrillic_count = len(_CYRILLIC_RE.findall(text))
    ratio = cyrillic_count / max(len(text), 1)
    return "ru" if ratio > 0.15 else "en"
```

### 6. Telethon Message metadata extraction

```python
def extract_from_message(msg) -> dict:
    """Извлечь metadata из Telethon Message."""
    result = {}

    # Forward info
    result["is_forward"] = bool(getattr(msg, "fwd_from", None))
    if msg.fwd_from:
        # Пробуем получить username forwarded канала, fallback на channel_id
        fwd_chat = getattr(msg.fwd_from, "from_id", None)
        if fwd_chat:
            if hasattr(fwd_chat, "channel_id"):
                result["forwarded_from_id"] = str(fwd_chat.channel_id)
            # username доступен только если чат resolved — часто None
            fwd_name = getattr(msg.fwd_from, "from_name", None)
            if fwd_name:
                result["forwarded_from_name"] = fwd_name

    # Reply — единообразное имя поля
    reply = getattr(msg, "reply_to", None)
    if reply:
        result["reply_to_msg_id"] = getattr(reply, "reply_to_msg_id", None)

    # Media types
    media_types = []
    if msg.photo:
        media_types.append("photo")
    if msg.video:
        media_types.append("video")
    if msg.document:
        media_types.append("document")
    if msg.audio:
        media_types.append("audio")
    if media_types:
        result["media_types"] = media_types

    return result
```

---

## Acceptance Criteria

1. **Коллекция `news_colbert_v2` содержит ≥13000 points** с обогащённым payload. После switch `.env` указывает на неё
2. **Vector config**: dense_vector(1024) + sparse BM25 + colbert_vector(multi-128) — идентичная `news_colbert`
3. **Payload indexes созданы**: channel (is_tenant), date (datetime), entities (keyword), year_week (keyword), root_message_id (keyword), + secondary
4. **Entity extraction**: top-20 entities (OpenAI, GPT, Google, ...) корректно извлекаются. Проверка: facet по `entities` возвращает OpenAI, GPT, Google в top-5
5. **Фильтрация по channel**: <10ms (вместо full scan)
6. **Facet по channel**: возвращает 36 каналов. Counts = point counts (документировано что это chunks, не posts). Для post counts — dedup по `root_message_id` client-side
7. **Facet по entities**: возвращает top entities с осмысленными counts
8. **Facet по year_week**: возвращает ≥36 недель
9. **Существующий pipeline не сломан**: 5 eval вопросов из v3 дают recall ≥ текущему
10. **Rollback**: старая коллекция `news_colbert` не удаляется до подтверждения. Откат = вернуть `QDRANT_COLLECTION=news_colbert` в `.env`

---

## Чеклист реализации

### Код
- [ ] `scripts/payload_enrichment.py` — модуль extraction (entities, urls, arxiv, lang, ...)
- [ ] `datasets/entity_dictionary.json` — словарь entities (seed из corpus frequency, ~300)
- [ ] `scripts/migrate_collection.py` — создание `news_colbert_v2` с full vector config + indexes
- [ ] Модифицировать `_build_point_docs_flat()` — добавить enrichment + `root_message_id`
- [ ] Обновить `QdrantStore._create_payload_indices()` — добавить новые индексы для будущих коллекций
- [ ] Smoke test: ingest 1 канала с новым payload, проверить все поля

### Миграция
- [ ] Создать `news_colbert_v2` с полной vector config (dense + sparse + ColBERT)
- [ ] Создать все payload indexes на пустой коллекции
- [ ] Re-ingest всех 36 каналов
- [ ] Проверить point count ≥ 13000
- [ ] Проверить facet API работает (channel, entities, year_week)
- [ ] Smoke test eval (5 вопросов) — recall не упал
- [ ] Switch: обновить `QDRANT_COLLECTION=news_colbert_v2` в `.env`, перезапустить API

### Валидация
- [ ] Facet по channel → 36 каналов
- [ ] Facet по entities → OpenAI, GPT, Google в top-5
- [ ] Фильтрация по channel < 10ms
- [ ] `root_message_id` присутствует во всех points

### Документация
- [ ] Обновить `always_on.md` — payload schema
- [ ] Decision log — DEC-XXXX: payload enrichment + migration rationale
