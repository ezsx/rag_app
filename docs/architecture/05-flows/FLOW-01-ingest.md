## FLOW-01: Telegram Ingest

### Problem
Сообщения из Telegram-каналов должны быть проиндексированы для последующего поиска.
Инжест должен быть идемпотентным (повторный запуск не дублирует данные).

### Contract
```
docker compose --profile ingest run --rm ingest --channel @channel_name
  [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--collection collection_name]
```

### Actors
- **Operator** — человек запускает ingest CLI
- **TelegramClient** — Telethon HTTP клиент к Telegram API
- **EmbeddingModel** — multilingual-e5-large (SentenceTransformer, RTX 5060 Ti)
- **SparseEncoder** — `Qdrant/bm25` с `language="russian"` (Snowball stemmer)
- **QdrantClient** — вектор-база с named vectors (dense + sparse)

### Sequence

```mermaid
sequenceDiagram
  autonumber
  participant Op as Operator
  participant CLI as ingest_telegram.py
  participant TG as Telegram API (Telethon)
  participant Embed as EmbeddingModel (e5-large)
  participant Sparse as SparseEncoder (Qdrant/bm25)
  participant Qdrant as QdrantClient

  Op->>CLI: docker run ingest --channel @name --since 2024-01-01
  CLI->>CLI: _parse_args() → validate args
  CLI->>TG: TelegramClient.connect() + authorize

  loop по батчам сообщений
    CLI->>TG: iter_messages(channel, offset_date=since, limit=batch_size)
    TG-->>CLI: batch of Message objects
    CLI->>CLI: _preprocess_message(msg) → text, metadata{id, date, channel, author, url}

    CLI->>Embed: encode("passage: " + text, batch_size=optimal)
    Embed-->>CLI: dense_vectors list[float] (1024-dim)

    CLI->>Sparse: encode_document(text)
    Sparse-->>CLI: sparse_vectors list[SparseVector]

    CLI->>Qdrant: upsert(collection=QDRANT_COLLECTION, points=[
      PointStruct(
        id="{channel}:{message_id}",
        vector={"dense_vector": dense, "sparse_vector": sparse},
        payload={text, channel, channel_id, message_id, date, author, url}
      )
    ])
    Note over CLI,Qdrant: id = f"{channel_name}:{message_id}" — идемпотентный upsert
    Qdrant-->>CLI: ok
  end

  CLI->>CLI: log stats: processed, upserted, skipped
  CLI-->>Op: 0 exit code + итоговый count
```

### Ключевые инварианты

- Document ID формат: `{channel_name}:{message_id}` — стабильный, позволяет upsert
- E5 prefix при индексировании: `"passage: " + text`; при поиске: `"query: " + query`
- Sparse encoder: `Qdrant/bm25` с `language="russian"` (Snowball) — **не BM42** (English-only)
- Батч размер определяется `get_optimal_batch_size(device)` — адаптивно по GPU/CPU
- Qdrant storage: **только named volume** `qdrant_data` — bind mounts → silent data corruption на Windows (INV-06, DEC-0015)
- Коллекция Qdrant: named vectors `dense_vector` (cosine, 1024-dim) + `sparse_vector`

### Техдолг

- Нет инкрементального обновления (live-инgest) — только batch по диапазону дат
- Нет мониторинга прогресса через SSE/webhook (только stdout)
