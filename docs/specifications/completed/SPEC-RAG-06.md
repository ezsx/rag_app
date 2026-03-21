# SPEC-RAG-06: Ingest Pipeline Migration

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Draft
> **Цель:** Заменить в `scripts/ingest_telegram.py` ChromaDB + BM25IndexManager + локальный SentenceTransformer на QdrantStore + TEIEmbeddingClient + fastembed SparseTextEmbedding (CPU).
> **Источники:** FLOW-01 (`docs/architecture/05-flows/FLOW-01-ingest.md`),
>               `docs/architecture/07-data-model/data-model.md`,
>               SPEC-RAG-02 (`TEIEmbeddingClient.embed_documents()`),
>               SPEC-RAG-03 (`QdrantStore.upsert()`, `PointDocument`),
>               DEC-0015, DEC-0016

---

## 0. Implementation Pointers

### 0.1 Текущие файлы

| Файл / компонент | Текущее поведение | После SPEC-RAG-06 |
|---|---|---|
| `scripts/ingest_telegram.py` | ChromaDB HTTP + SentenceTransformer (локал.) + BM25IndexManager | Qdrant + TEI HTTP + fastembed (CPU) |
| `create_chroma_collection()` | Подключается к ChromaDB, создаёт коллекцию с local embed | **Удалить**, заменить на `await qdrant_store.ensure_collection()` |
| `ingest_batches()` — core loop | ChromaDB `collection.upsert()` + BM25 `mgr.add_documents()` | TEI embed + sparse encode + `await qdrant_store.upsert()` |
| `resolve_embedding_model()` | Читает `RECOMMENDED_MODELS` из `model_downloader.py` | **Удалить** — TEI URL берётся из settings |
| `detect_optimal_device()` | CUDA/MPS/CPU autodetect для локальной модели | **Удалить** — TEI управляет GPU сам |
| `get_optimal_batch_size()` | По device рассчитывает batch size модели | **Удалить** — TEI управляет батчингом |
| `FastEmbeddingFunction` class | Оптимизированная SentenceTransformer обёртка | **Удалить** |
| `main()` | ChromaDB init + model_downloader + BM25 post-process | QdrantStore + TEIEmbeddingClient + SparseTextEmbedding |

### 0.2 Новые элементы

- `_build_point_docs()` — helper: Message[] + dense_vectors + sparse_results → `list[PointDocument]`
- Импорты: `QdrantStore`, `TEIEmbeddingClient`, `SparseTextEmbedding` (fastembed)

### 0.3 Что удалить из скрипта

```
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import torch
from sentence_transformers import SentenceTransformer
import numpy as np            # (если не используется нигде ещё)

def create_chroma_collection(...)
def resolve_embedding_model(...)
def detect_optimal_device(...)
def get_optimal_batch_size(...)
class FastEmbeddingFunction(...)
```

Из `_parse_args()` удалить аргументы:
- `--embed-model-key`
- `--embed-model`
- `--device`
- `--gpu-batch-multiplier`

---

## 1. Обзор

### 1.1 Задача

1. Убрать из `ingest_telegram.py` все ChromaDB/BM25/SentenceTransformer зависимости.
2. Переписать `ingest_batches()`: для каждого батча — TEI embed (async HTTP) + fastembed sparse (sync CPU) → `QdrantStore.upsert()`.
3. Переписать `main()`: инициализировать `TEIEmbeddingClient`, `SparseTextEmbedding`, `QdrantStore`; вызвать `ensure_collection()` перед циклом.
4. Обновить `_parse_args()`: убрать args, связанные с локальной моделью.
5. Добавить `_build_point_docs()` — маппинг Message → `PointDocument`.

### 1.2 Контекст — почему именно так

**Point ID формат (FLOW-01 инвариант):**
`"{channel_name}:{message_id}"` — детерминированный, позволяет идемпотентный upsert при повторном запуске. Если включён chunking (`--chunk-size > 0`), то `"{channel_name}:{message_id}:{chunk_idx}"` — каждый чанк отдельная точка.

**channel_name в point_id:**
`channel_hint.lstrip("@")` если `channel_hint` = `@username`, иначе `str(msg.chat_id)`. `@username` стабильнее numeric ID, который может измениться. Если канал задан числом — fallback на chat_id.

**Payload (стандарт Phase 1):**
`{text, channel, channel_id, message_id, date, author, url}` — выравниваем с `data-model.md`. Старый ключ `msg_id` заменяется на `message_id`, `channel_username` → `channel`.

**Async batch loop:**
`TEIEmbeddingClient.embed_documents()` — async HTTP, батчит все тексты за один вызов.
`SparseTextEmbedding.embed(docs)` — sync CPU (fastembed), вызывается напрямую.
`QdrantStore.upsert()` — async, батчит внутри (batch_size=64 по умолчанию).

**Batch size (`--batch-size`) остаётся:**
Теперь означает кол-во сообщений Telegram за одну итерацию сбора + одну итерацию TEI embed. TEI сам управляет GPU-батчингом.

### 1.3 Ключевые решения

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| Кол-во HTTP вызовов TEI | 1 вызов на батч | `embed_documents` принимает `list[str]` — все тексты батча за раз |
| SparseTextEmbedding в ingest | `sparse_encoder.embed(texts)` (не `query_embed`) | `embed()` — для индексации документов; `query_embed()` — для поисковых запросов (разные веса BM25) |
| Инициализация sparse encoder | В `main()`, один раз | Первый вызов скачивает модель Qdrant/bm25 (~5 МБ); синглтон на время скрипта |
| `ensure_collection()` | В `main()` перед циклом | Создаёт если нет, idempotent если уже существует (SPEC-RAG-03) |
| `collection` arg в `_parse_args` | Остаётся, по умолчанию `settings.qdrant_collection` | CLI гибкость; если не задан — из env QDRANT_COLLECTION |
| `--chunk-size` | Оставить | Полезен для длинных постов; ID становится `channel:msg_id:chunk_idx` |
| Telethon client code | Без изменений | `create_telegram_client`, `_gather_with_retries`, `gather_messages` — вне scope |

### 1.4 Что НЕ делать

- **НЕ** менять Telethon-код (авторизация, `iter_messages`, `FloodWaitError`).
- **НЕ** использовать `asyncio.run()` внутри корутины (скрипт уже `async def main`).
- **НЕ** передавать `"passage: "` prefix вручную в `ingest_batches` — `TEIEmbeddingClient.embed_documents()` добавляет его автоматически (SPEC-RAG-02).
- **НЕ** удалять `--chunk-size` и `--max-messages` аргументы.
- **НЕ** трогать `load_state`, `save_state`, `split_into_batches`, `_to_utc_naive`.

---

## 2. Обновлённые импорты

```python
"""
Telegram → Qdrant ingestor (Phase 1)
--------------------------------------
CLI-скрипт для загрузки сообщений Telegram-каналов в Qdrant.

Dense: TEI HTTP → multilingual-e5-large @ host.docker.internal:8082
Sparse: fastembed SparseTextEmbedding (Qdrant/bm25, language=russian, CPU)
Store: Qdrant (named volume qdrant_data, Docker CPU)

Запуск:
    docker compose --profile ingest run --rm ingest \
        --channel @channel_name --since 2024-01-01 --until 2024-07-01

Обязательные переменные окружения:
    TG_API_ID, TG_API_HASH          — Telegram API credentials
    QDRANT_URL                      — URL Qdrant (по умолчанию http://qdrant:6333)
    QDRANT_COLLECTION               — имя коллекции (по умолчанию news)
    EMBEDDING_TEI_URL               — URL TEI embedding service
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from dateutil import parser as date_parser, tz

from telethon import TelegramClient, errors as tg_errors
from telethon.errors import FloodWaitError, SessionPasswordNeededError
from telethon.tl.types import Message

# Добавляем src в sys.path для импорта из приложения
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adapters.qdrant.store import QdrantStore, PointDocument
from adapters.tei.embedding_client import TEIEmbeddingClient
from core.settings import get_settings

try:
    from fastembed import SparseTextEmbedding
except ImportError as _e:
    raise ImportError(
        "fastembed не установлен. Добавьте в requirements: fastembed>=0.3"
    ) from _e

load_dotenv()
logger = logging.getLogger(__name__)
```

---

## 3. _parse_args() — обновление

Удаляем `--embed-model-key`, `--embed-model`, `--device`, `--gpu-batch-multiplier`. Добавляем дефолт для `--collection` из env.

```python
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Telegram channel(s) into Qdrant")
    p.add_argument(
        "--channel", required=False, default=None, help="@username or chat id"
    )
    p.add_argument(
        "--channels",
        required=False,
        default=None,
        help="Comma-separated list of channels, e.g. @a,@b,@c",
    )
    p.add_argument("--since", required=True, help="ISO date (inclusive)")
    p.add_argument("--until", required=True, help="ISO date (exclusive)")
    p.add_argument(
        "--collection",
        required=False,
        default=None,
        help="Qdrant collection name (default: QDRANT_COLLECTION env or 'news')",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Сколько сообщений обрабатывать за один TEI embed запрос (default: 64)",
    )
    p.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Ограничить N сообщениями для отладки",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Разбивать длинные сообщения на чанки по N символов (0 = не разбивать)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()
```

---

## 4. _build_point_docs() — новый helper

```python
def _build_point_docs(
    messages: List[Message],
    dense_vectors: List[List[float]],
    sparse_results: List[Any],
    channel_name: str,
    chunk_size: int = 0,
) -> List[PointDocument]:
    """
    Маппинг Telegram Message[] + dense/sparse векторов → list[PointDocument].

    channel_name: строка без '@', используется в point_id.
    Если chunk_size > 0, каждый чанк получает отдельный PointDocument
    с id = "{channel_name}:{msg_id}:{chunk_idx}".
    Если chunk_size == 0, id = "{channel_name}:{msg_id}".

    Args:
        messages:      список Telegram Message (уже распакованных через _split_text если нужно)
        dense_vectors: list[list[float]] — от TEIEmbeddingClient.embed_documents()
        sparse_results: list[EmbeddingResult] — от SparseTextEmbedding.embed()
        channel_name:  идентификатор канала без '@' (для point_id)
        chunk_size:    > 0 → чанкование (учитывается при формировании id)

    Returns:
        list[PointDocument] — один элемент на каждую (message, chunk) пару
    """
    docs: List[PointDocument] = []

    vec_idx = 0  # индекс в dense_vectors / sparse_results
    for m in messages:
        text_full = (m.message or "").strip()
        if not text_full:
            continue

        parts = _split_text(text_full, chunk_size)  # [text] если chunk_size=0

        # author: из поля sender_id (может быть None для анонимных каналов)
        author: Optional[str] = None
        sender = getattr(m, "sender", None)
        if sender is not None:
            first = getattr(sender, "first_name", None) or ""
            last = getattr(sender, "last_name", None) or ""
            author = (first + " " + last).strip() or None

        # URL сообщения (Telegram public link)
        msg_url: Optional[str] = None
        if channel_name:
            msg_url = f"https://t.me/{channel_name}/{m.id}"

        for chunk_idx, part in enumerate(parts):
            # Детерминированный point ID (FLOW-01 инвариант)
            if chunk_size > 0:
                point_id = f"{channel_name}:{m.id}:{chunk_idx}"
            else:
                point_id = f"{channel_name}:{m.id}"

            sparse = sparse_results[vec_idx]
            payload: Dict[str, Any] = {
                "text": part,
                "channel": channel_name,
                "channel_id": int(m.chat_id),
                "message_id": int(m.id),
                "date": _to_utc_naive(m.date).isoformat(),
                "author": author,
                "url": msg_url,
            }
            # Убираем None значения (Qdrant не хранит null если не нужно)
            payload = {k: v for k, v in payload.items() if v is not None}

            docs.append(
                PointDocument(
                    point_id=point_id,
                    dense_vector=dense_vectors[vec_idx],
                    sparse_indices=sparse.indices.tolist(),
                    sparse_values=sparse.values.tolist(),
                    payload=payload,
                )
            )
            vec_idx += 1

    return docs
```

---

## 5. ingest_batches() — полный rewrite

```python
async def ingest_batches(
    messages: List[Message],
    batch_size: int,
    embedding_client: TEIEmbeddingClient,
    sparse_encoder: SparseTextEmbedding,
    qdrant_store: QdrantStore,
    channel_hint: Optional[str] = None,
    chunk_size: int = 0,
    progress_cb: Optional[Any] = None,
    log_every: int = 200,
) -> Dict[str, int]:
    """
    Основной цикл инжеста: батч Telegram сообщений → Qdrant upsert.

    Шаги для каждого батча:
    1. Распаковка сообщений → texts (с чанкованием если chunk_size > 0)
    2. TEI embed_documents(texts) → dense_vectors (async HTTP, prefix "passage: " внутри)
    3. SparseTextEmbedding.embed(texts) → sparse_results (sync CPU, BM25 Russian)
    4. _build_point_docs(messages, dense, sparse) → list[PointDocument]
    5. await qdrant_store.upsert(point_docs) → int (count upserted)

    Args:
        messages:         список Telegram Message
        batch_size:       кол-во сообщений за одну итерацию
        embedding_client: TEIEmbeddingClient (async HTTP → TEI :8082)
        sparse_encoder:   SparseTextEmbedding (fastembed, CPU)
        qdrant_store:     QdrantStore с уже инициализированной коллекцией
        channel_hint:     @username канала (используется в point_id и payload.channel)
        chunk_size:       > 0 → разбивать длинные сообщения
        progress_cb:      callback(dict) → dict | None
        log_every:        логировать каждые N сообщений

    Returns:
        dict {processed_in_channel, written_qdrant, total_in_channel}
    """
    import time as _time

    # channel_name без '@' для point_id и payload
    if channel_hint and channel_hint.startswith("@"):
        channel_name = channel_hint.lstrip("@")
    elif channel_hint:
        channel_name = channel_hint
    else:
        channel_name = "unknown"

    total = len(messages)
    processed_in_channel = 0
    total_written_qdrant = 0
    start_ts = _time.time()
    last_log_ts = start_ts

    for batch in tqdm_asyncio(
        split_into_batches(messages, batch_size),
        total=(total + batch_size - 1) // batch_size,
        desc="Ingesting",
        unit="batch",
    ):
        # Шаг 1: собираем тексты (с чанкованием)
        texts: List[str] = []
        source_messages: List[Message] = []  # один элемент на каждый чанк

        for m in batch:
            text_full = (m.message or "").strip()
            if not text_full:
                continue
            parts = _split_text(text_full, chunk_size)
            for part in parts:
                texts.append(part)
                source_messages.append(m)

        if not texts:
            processed_in_channel += len(batch)
            continue

        try:
            # Шаг 2: dense embedding через TEI (async HTTP)
            # embed_documents() добавляет "passage: " prefix автоматически (SPEC-RAG-02)
            dense_vectors: List[List[float]] = await embedding_client.embed_documents(texts)

            # Шаг 3: sparse BM25 encoding (sync CPU, fastembed)
            # Используем embed() (не query_embed()) — для индексации документов
            sparse_results = list(sparse_encoder.embed(texts))

            # Шаг 4: строим PointDocument с нужными Message объектами
            # _build_point_docs ожидает уникальные Message на каждый текст
            # source_messages[i] ↔ texts[i]
            point_docs = _build_point_docs_flat(
                source_messages=source_messages,
                texts=texts,
                dense_vectors=dense_vectors,
                sparse_results=sparse_results,
                channel_name=channel_name,
                chunk_size=chunk_size,
            )

            # Шаг 5: upsert в Qdrant (async, wait=True — идемпотентен при повторе)
            written = await qdrant_store.upsert(point_docs)
            total_written_qdrant += written

        except Exception as exc:
            logger.exception(
                "Ошибка при обработке батча channel=%s, пропускаем: %s",
                channel_hint,
                exc,
            )
            processed_in_channel += len(batch)
            continue

        processed_in_channel += len(batch)

        # Прогресс-коллбэк
        if progress_cb is not None:
            try:
                progress_cb(
                    {
                        "processed_in_channel": processed_in_channel,
                        "written_qdrant": total_written_qdrant,
                        "batch_size": len(batch),
                        "total_in_channel": total,
                    }
                )
            except Exception:
                pass

        # Периодическое логирование
        should_log = (processed_in_channel % max(1, log_every) == 0) or (
            processed_in_channel >= total
        )
        if should_log:
            now = _time.time()
            elapsed = now - start_ts
            step_elapsed = now - last_log_ts
            last_log_ts = now
            speed = (processed_in_channel / elapsed) if elapsed > 0 else 0.0
            logger.info(
                "progress channel=%s processed=%d/%d written_qdrant=%d elapsed_s=%.1f speed=%.1f msg/s",
                channel_hint or "?",
                processed_in_channel,
                total,
                total_written_qdrant,
                step_elapsed,
                speed,
            )

    return {
        "processed_in_channel": processed_in_channel,
        "written_qdrant": total_written_qdrant,
        "total_in_channel": total,
    }
```

**Вспомогательный helper для flat списка (source_messages ↔ texts — один к одному):**

```python
def _build_point_docs_flat(
    source_messages: List[Message],
    texts: List[str],
    dense_vectors: List[List[float]],
    sparse_results: List[Any],
    channel_name: str,
    chunk_size: int,
) -> List[PointDocument]:
    """
    Flat вариант _build_point_docs: source_messages[i] ↔ texts[i] (уже развёрнуто по чанкам).

    Формирует point_id с chunk_idx на основе счётчика повторений msg_id:
    если одно сообщение даёт несколько чанков — idx 0, 1, 2…
    """
    docs: List[PointDocument] = []
    msg_chunk_counter: Dict[int, int] = {}

    for i, (m, text) in enumerate(zip(source_messages, texts)):
        chunk_idx = msg_chunk_counter.get(m.id, 0)
        msg_chunk_counter[m.id] = chunk_idx + 1

        if chunk_size > 0:
            point_id = f"{channel_name}:{m.id}:{chunk_idx}"
        else:
            point_id = f"{channel_name}:{m.id}"

        author: Optional[str] = None
        sender = getattr(m, "sender", None)
        if sender is not None:
            first = getattr(sender, "first_name", None) or ""
            last = getattr(sender, "last_name", None) or ""
            author = (first + " " + last).strip() or None

        payload: Dict[str, Any] = {
            "text": text,
            "channel": channel_name,
            "channel_id": int(m.chat_id),
            "message_id": int(m.id),
            "date": _to_utc_naive(m.date).isoformat(),
        }
        if author:
            payload["author"] = author
        if channel_name:
            payload["url"] = f"https://t.me/{channel_name}/{m.id}"

        sparse = sparse_results[i]
        docs.append(
            PointDocument(
                point_id=point_id,
                dense_vector=dense_vectors[i],
                sparse_indices=sparse.indices.tolist(),
                sparse_values=sparse.values.tolist(),
                payload=payload,
            )
        )

    return docs
```

---

## 6. main() — полный rewrite

```python
async def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Собираем список каналов
    channels: List[str] = []
    if getattr(args, "channel", None):
        channels.append(args.channel)
    if getattr(args, "channels", None):
        for part in str(args.channels).split(","):
            part = part.strip()
            if part:
                channels.append(part)
    seen: set = set()
    channels = [c for c in channels if not (c in seen or seen.add(c))]
    if not channels:
        logger.error("Не указан ни один канал. Используйте --channel или --channels")
        sys.exit(2)

    # Определяем имя коллекции
    settings = get_settings()
    collection_name: str = (
        args.collection
        or os.getenv("QDRANT_COLLECTION")
        or settings.qdrant_collection
    )

    # ------------------------------------------------------------------
    # Инициализация TEI embedding client
    # ------------------------------------------------------------------
    embedding_tei_url: str = (
        os.getenv("EMBEDDING_TEI_URL") or settings.embedding_tei_url
    )
    logger.info("TEI embedding: %s", embedding_tei_url)
    embedding_client = TEIEmbeddingClient(base_url=embedding_tei_url)

    # Healthcheck TEI перед началом (не блокирует, только предупреждение)
    if not await embedding_client.healthcheck():
        logger.warning(
            "TEI embedding service недоступен: %s. Продолжаем (может восстановиться).",
            embedding_tei_url,
        )

    # ------------------------------------------------------------------
    # Инициализация sparse encoder (fastembed, CPU)
    # При первом вызове скачивает модель Qdrant/bm25 (~5 МБ кэш)
    # ------------------------------------------------------------------
    logger.info("Инициализация sparse encoder: Qdrant/bm25 (language=russian)…")
    sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")
    logger.info("Sparse encoder готов")

    # ------------------------------------------------------------------
    # Инициализация Qdrant store
    # ------------------------------------------------------------------
    qdrant_url: str = os.getenv("QDRANT_URL") or settings.qdrant_url
    logger.info("Qdrant: %s / collection=%s", qdrant_url, collection_name)
    qdrant_store = QdrantStore(url=qdrant_url, collection=collection_name)

    # Создаём коллекцию если нет (idempotent, race-safe — SPEC-RAG-03)
    await qdrant_store.ensure_collection()
    logger.info("Коллекция '%s' готова", collection_name)

    # ------------------------------------------------------------------
    # Telegram client
    # ------------------------------------------------------------------
    start_iso = _to_utc_naive(date_parser.isoparse(args.since))
    end_iso = _to_utc_naive(date_parser.isoparse(args.until))

    logger.info("Подключение к Telegram…")
    tg_client = await create_telegram_client()

    total_processed = 0
    total_written_qdrant = 0

    def _progress_cb(batch_stats: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal total_processed, total_written_qdrant
        total_processed += int(batch_stats.get("batch_size", 0))
        total_written_qdrant = int(batch_stats.get("written_qdrant", 0))
        return {
            "processed_total": total_processed,
            "written_qdrant_total": total_written_qdrant,
        }

    try:
        for ch in channels:
            logger.info(
                "▶ start channel=%s dates=%s→%s", ch, start_iso.date(), end_iso.date()
            )

            msgs = await _gather_with_retries(
                tg_client, ch, start_iso, end_iso, args.max_messages
            )
            logger.info("Получено %d сообщений для %s", len(msgs), ch)

            if not msgs:
                logger.warning("Пусто для %s — пропускаем", ch)
                continue

            stats = await ingest_batches(
                messages=msgs,
                batch_size=args.batch_size,
                embedding_client=embedding_client,
                sparse_encoder=sparse_encoder,
                qdrant_store=qdrant_store,
                channel_hint=ch if isinstance(ch, str) else None,
                chunk_size=int(getattr(args, "chunk_size", 0) or 0),
                progress_cb=_progress_cb,
                log_every=200,
            )

            logger.info(
                "✔ finish channel=%s read=%d written_qdrant=%d",
                ch,
                stats.get("processed_in_channel", 0),
                stats.get("written_qdrant", 0),
            )

        # Финальная статистика коллекции
        info = await qdrant_store.collection_info()
        logger.info(
            "Все каналы обработаны. В коллекции '%s': %d точек",
            collection_name,
            info.get("points_count", "?"),
        )

    except Exception as exc:
        logger.error("Ошибка во время инжеста: %s", exc)
        raise

    finally:
        await tg_client.disconnect()
        await embedding_client.aclose()
        await qdrant_store.aclose()
        logger.info("Соединения закрыты")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Тесты

Файл: `src/tests/test_ingest_pipeline.py`

```python
"""
Unit-тесты для ключевых функций ingest_telegram.py (Phase 1).
Мокируем TEIEmbeddingClient.embed_documents, SparseTextEmbedding.embed,
QdrantStore.upsert.
"""
from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest
import numpy as np

# Путь зависит от запуска — предполагаем что src/ в sys.path
from adapters.qdrant.store import PointDocument


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_message(msg_id: int, text: str, chat_id: int = 100) -> MagicMock:
    """Создаёт mock Telegram Message."""
    m = MagicMock()
    m.id = msg_id
    m.chat_id = chat_id
    m.message = text
    m.date = datetime(2024, 6, 1, tzinfo=timezone.utc)
    m.sender = None
    m.reply_to_msg_id = None
    m.views = None
    return m


def _make_sparse_result(n_tokens: int = 5) -> MagicMock:
    """Создаёт mock EmbeddingResult от fastembed."""
    r = MagicMock()
    r.indices = np.array(list(range(n_tokens)))
    r.values = np.array([1.0] * n_tokens)
    return r


# ------------------------------------------------------------------
# Тесты _build_point_docs_flat
# ------------------------------------------------------------------

class TestBuildPointDocsFlat:
    """Тесты helper функции, которая строит PointDocument из сообщений."""

    def _call(self, msgs, texts, dense, sparse, channel_name="chan", chunk_size=0):
        # Импортируем из скрипта
        import importlib.util, sys
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "ingest_telegram",
            Path(__file__).parent.parent.parent / "scripts" / "ingest_telegram.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Не выполняем main блок, только загружаем функции
        with patch.dict(sys.modules, {"ingest_telegram": mod}):
            spec.loader.exec_module(mod)
        return mod._build_point_docs_flat(msgs, texts, dense, sparse, channel_name, chunk_size)

    def test_single_message_no_chunk(self):
        """Одно сообщение без чанкования → один PointDocument."""
        m = _make_message(42, "hello world")
        dense = [[0.1] * 1024]
        sparse = [_make_sparse_result()]

        docs = self._call([m], ["hello world"], dense, sparse)

        assert len(docs) == 1
        d = docs[0]
        assert d.point_id == "chan:42"
        assert d.payload["message_id"] == 42
        assert d.payload["channel"] == "chan"
        assert d.payload["text"] == "hello world"
        assert len(d.dense_vector) == 1024

    def test_two_chunks_same_message(self):
        """Одно сообщение → 2 чанка → два PointDocument с id ...0 и ...1."""
        m = _make_message(7, "text")
        dense = [[0.1] * 1024, [0.2] * 1024]
        sparse = [_make_sparse_result(), _make_sparse_result()]

        docs = self._call([m, m], ["part1", "part2"], dense, sparse, chunk_size=100)

        assert len(docs) == 2
        assert docs[0].point_id == "chan:7:0"
        assert docs[1].point_id == "chan:7:1"

    def test_payload_fields(self):
        """Проверяем все обязательные поля payload."""
        m = _make_message(1, "text", chat_id=999)
        docs = self._call(
            [m], ["text"], [[0.0] * 1024], [_make_sparse_result()], channel_name="mychan"
        )
        p = docs[0].payload
        assert p["channel"] == "mychan"
        assert p["channel_id"] == 999
        assert p["message_id"] == 1
        assert "date" in p
        assert "text" in p
        # author не задан (sender=None) → не должен быть в payload
        assert "author" not in p

    def test_sparse_tolist(self):
        """sparse_indices и sparse_values должны быть list[int/float], не np.ndarray."""
        m = _make_message(1, "text")
        sparse = [_make_sparse_result(3)]
        docs = self._call([m], ["text"], [[0.0] * 1024], sparse)

        assert isinstance(docs[0].sparse_indices, list)
        assert isinstance(docs[0].sparse_values, list)


# ------------------------------------------------------------------
# Тест ingest_batches (async, моки)
# ------------------------------------------------------------------

class TestIngestBatches:
    """Интеграционный тест цикла ingest_batches с замоканными зависимостями."""

    @pytest.mark.asyncio
    async def test_calls_embed_and_upsert(self):
        """ingest_batches вызывает embed_documents и qdrant_store.upsert для каждого батча."""
        msgs = [_make_message(i, f"text {i}") for i in range(3)]

        embedding_client = AsyncMock()
        embedding_client.embed_documents = AsyncMock(
            return_value=[[0.1] * 1024 for _ in range(3)]
        )

        sparse_results = [_make_sparse_result() for _ in range(3)]
        sparse_encoder = MagicMock()
        sparse_encoder.embed = MagicMock(return_value=iter(sparse_results))

        qdrant_store = AsyncMock()
        qdrant_store.upsert = AsyncMock(return_value=3)

        # Импортируем ingest_batches
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "ingest_telegram",
            Path(__file__).parent.parent.parent / "scripts" / "ingest_telegram.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        stats = await mod.ingest_batches(
            messages=msgs,
            batch_size=10,
            embedding_client=embedding_client,
            sparse_encoder=sparse_encoder,
            qdrant_store=qdrant_store,
            channel_hint="@testchan",
        )

        embedding_client.embed_documents.assert_called_once()
        sparse_encoder.embed.assert_called_once()
        qdrant_store.upsert.assert_called_once()
        assert stats["written_qdrant"] == 3
        assert stats["processed_in_channel"] == 3

    @pytest.mark.asyncio
    async def test_skips_empty_messages(self):
        """Сообщения без текста пропускаются — embed не вызывается."""
        msgs = [_make_message(1, ""), _make_message(2, "  ")]

        embedding_client = AsyncMock()
        sparse_encoder = MagicMock()
        qdrant_store = AsyncMock()
        qdrant_store.upsert = AsyncMock(return_value=0)

        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "ingest_telegram2",
            Path(__file__).parent.parent.parent / "scripts" / "ingest_telegram.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        stats = await mod.ingest_batches(
            messages=msgs,
            batch_size=10,
            embedding_client=embedding_client,
            sparse_encoder=sparse_encoder,
            qdrant_store=qdrant_store,
        )

        # embed не должен вызываться — нет текстов
        embedding_client.embed_documents.assert_not_called()
        assert stats["written_qdrant"] == 0
```

---

## 8. Cleanup: мёртвый Phase 0 код

### 8.1 `src/services/tools/search.py` — удалить BM25 fallback

Параметр `bm25_retriever: Optional[Any] = None` (строка 25) и BM25 fallback код (строки 157–174) мертвы —
`get_agent_service()` не передаёт `bm25_retriever`. Удалить:
- Параметр `bm25_retriever` из сигнатуры `search()`
- Весь блок `if (not candidates or force_bm25) and bm25_retriever is not None:` (строки 161–174)
- Переменную `bm25_duration_ms` и связанные с ней вычисления
- Переменную `force_bm25` (строки 158–160)

### 8.2 `src/api/v1/endpoints/collections.py` — оставить как есть

Endpoint уже отключён в `router.py` (import закомментирован).
Полная переработка на Qdrant — **out of scope** для SPEC-RAG-06; оставить на Phase 2.

---

## 9. Документация

После реализации SPEC-RAG-06 обновить следующие файлы документации:

### 9.1 Обновить `docs/ai/modules/scripts/ingest_telegram.py.md`

Переписать описание модуля, отразив Phase 1 состояние:
- Qdrant вместо ChromaDB
- TEI HTTP вместо SentenceTransformer
- fastembed sparse вместо BM25IndexManager
- Новый Point ID формат: `"{channel_name}:{message_id}"`

### 9.2 Обновить `docs/ai/modules/src/services/tools/search.py.md` (если существует)

Убрать упоминания BM25 fallback — он удалён в 8.1.

### 9.3 Проверить `requirements_ingest.txt`

Убедиться что отсутствуют:
- `chromadb`
- `sentence-transformers`
- `torch`

Должны быть:
- `qdrant-client>=1.9.0`
- `fastembed>=0.3.0`
- `httpx` (для TEIEmbeddingClient)

---

## 10. Чеклист реализации

### Ingest pipeline
- [ ] Импорты обновлены: убраны `chromadb`, `torch`, `sentence_transformers`; добавлены `QdrantStore`, `TEIEmbeddingClient`, `SparseTextEmbedding`
- [ ] `_parse_args()` — убраны `--embed-model-key`, `--embed-model`, `--device`, `--gpu-batch-multiplier`; `--collection` стал optional с дефолтом из env
- [ ] `create_chroma_collection()` — функция удалена
- [ ] `resolve_embedding_model()` — функция удалена
- [ ] `detect_optimal_device()` — функция удалена
- [ ] `get_optimal_batch_size()` — функция удалена
- [ ] `FastEmbeddingFunction` class — удалён
- [ ] `_build_point_docs_flat()` — новая функция добавлена
- [ ] `ingest_batches()` — переписана: TEI embed + fastembed sparse + QdrantStore.upsert
- [ ] `main()` — переписан: `TEIEmbeddingClient` + `SparseTextEmbedding` + `QdrantStore`; `ensure_collection()` перед циклом
- [ ] `main()` finally-блок: `aclose()` на embedding_client и qdrant_store
- [ ] Point ID формат: `"{channel_name}:{msg_id}"` (no chunk) / `"{channel_name}:{msg_id}:{chunk_idx}"` (chunk) — проверить
- [ ] Payload: `text`, `channel`, `channel_id`, `message_id`, `date` — все присутствуют
- [ ] `embed_documents()` вызывается без `"passage: "` prefix (SPEC-RAG-02 добавляет его внутри)
- [ ] `sparse_encoder.embed()` (не `query_embed()`) для индексирования документов

### Cleanup
- [ ] `src/services/tools/search.py` — удалён `bm25_retriever` параметр и BM25 fallback код
- [ ] `requirements_ingest.txt` — нет `chromadb`, `sentence-transformers`, `torch`; есть `httpx`

### Документация
- [ ] `docs/ai/modules/scripts/ingest_telegram.py.md` — обновлён до Phase 1
- [ ] `docs/ai/modules/src/services/tools/search.py.md` — обновлён (если существует)

### Тесты
- [ ] `pytest src/tests/test_ingest_pipeline.py` — тесты проходят (не запускать, только создать)
