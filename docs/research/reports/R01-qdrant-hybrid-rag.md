# Миграция с ChromaDB на Qdrant для hybrid RAG

**Переход на Qdrant позволяет полностью заменить связку ChromaDB + кастомный BM25 (~400 строк кода) одной коллекцией с named vectors и нативным RRF fusion в одном API-вызове.** При масштабе в десятки тысяч сообщений (≈60 МБ) Qdrant работает без тюнинга на дефолтных параметрах, а нативная поддержка BM25 через FastEmbed с русским языком делает кастомный индекс избыточным. Qdrant уже запущен в Docker, что минимизирует инфраструктурные изменения. Главный риск — BM42 от Qdrant **не подходит для русского языка** (модель обучена на английском), поэтому рекомендуется использовать `Qdrant/bm25` с `language="russian"`.

---

## Архитектура коллекции: dense + sparse в одном месте

Qdrant поддерживает **named vectors** — несколько векторных пространств в одной коллекции. Dense-вектора (multilingual-e5-large, 1024d) и sparse-вектора (BM25) хранятся как именованные поля одной точки. Sparse-вектора представлены парами `(indices, values)` — хранятся только ненулевые элементы через **инвертированный индекс** (не HNSW). Поиск по sparse-векторам всегда точный (exact), метрика — dot product.

Создание коллекции с полной схемой для Telegram-сообщений:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

COLLECTION = "telegram_news"

client.create_collection(
    collection_name=COLLECTION,
    vectors_config={
        "dense": models.VectorParams(
            size=1024,                    # multilingual-e5-large
            distance=models.Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF,  # серверный IDF — ключевой параметр
        ),
    },
)

# Payload-индексы — создавать ДО загрузки данных
client.create_payload_index(
    COLLECTION, "channel",
    field_schema=models.KeywordIndexParams(
        type="keyword", is_tenant=True    # оптимизация для per-channel запросов
    ),
)
client.create_payload_index(
    COLLECTION, "date",
    field_schema=models.DatetimeIndexParams(
        type="datetime", is_principal=True # оптимизация для time-range запросов
    ),
)
client.create_payload_index(COLLECTION, "author", models.PayloadSchemaType.KEYWORD)
client.create_payload_index(
    COLLECTION, "message_id",
    field_schema=models.IntegerIndexParams(type="integer", lookup=True, range=True),
)
```

Структура payload каждой точки:

```json
{
    "channel": "news_channel",
    "message_id": 12345,
    "date": "2025-03-15T10:30:00Z",
    "author": "username",
    "text": "Текст сообщения из Telegram...",
    "url": "https://t.me/channel/12345"
}
```

Параметры `is_tenant=True` на поле `channel` и `is_principal=True` на поле `date` оптимизируют хранение и HNSW-граф: Qdrant группирует сегменты по tenant-полю и оптимизирует порядок обхода по principal-полю. Для полей `text` и `url` индексы не нужны — семантический поиск покрывается dense-вектором, а url редко используется в фильтрах.

---

## BM25 через FastEmbed вместо BM42 и кастомного индекса

Критически важный вывод: **BM42 не подходит для русского языка**. BM42 — экспериментальный подход от Qdrant (июль 2024), который заменяет TF-компоненту BM25 на attention-веса трансформера. Дефолтная модель `all-MiniLM-L6-v2` обучена на английском тексте — её WordPiece-токенизатор агрессивно разбивает русские слова на подтокены, генерируя некорректные attention-распределения. Сам Qdrant пометил BM42 как «experimental, requires further research before production use» и позже скорректировал бенчмарки: **BM25 (tantivy) показал recall@10 = 0.89 против 0.85 у BM42** на датасете Quora.

Рекомендуемая альтернатива — `Qdrant/bm25` через библиотеку FastEmbed с прямой поддержкой русского языка через Snowball stemmer:

```python
from fastembed import SparseTextEmbedding, TextEmbedding

# Sparse encoder — BM25 с русским языком
sparse_model = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    language="russian"               # Snowball stemmer + русские стоп-слова
)

# Dense encoder — ваш текущий multilingual-e5-large
dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

# Генерация эмбеддингов
documents = ["Курс рубля укрепился на 2%", "Новый закон принят Думой"]

sparse_embeddings = list(sparse_model.embed(documents))
dense_embeddings = list(dense_model.embed(documents))

# Для поисковых запросов — отдельный метод
query = "курс валют"
query_sparse = list(sparse_model.query_embed(query))[0]
query_dense = list(dense_model.query_embed(query))[0]
```

| Характеристика | Кастомный BM25 (400 строк) | Qdrant/bm25 через FastEmbed |
|---|---|---|
| Русский язык | Зависит от реализации | ✅ Snowball stemmer + стоп-слова |
| IDF обновление | Ручное при добавлении документов | ✅ Автоматическое (серверный `modifier=IDF`) |
| Персистентность | Кастомный файл на диске | ✅ Встроена в Qdrant |
| Hybrid search | Ручной RRF fusion в Python | ✅ Нативный RRF в одном запросе |
| Фильтрация | Отдельная логика | ✅ Общие payload-фильтры |
| Код для поддержки | ~400 строк | ~10 строк |
| Производительность | Зависит от реализации | Inverted index в Rust |
| Гибкость настроек | Полная (k1, b, любая логика) | Параметры k, b, avg_len через модель |

**Вывод: кастомный BM25 можно полностью убрать.** FastEmbed BM25 поддерживает русский, серверный IDF автоматически обновляется при добавлении/удалении документов, а hybrid search работает в одном API-вызове. Оставлять кастомный BM25 как fallback имеет смысл только на **переходный период** (1–2 недели параллельной работы), после чего его можно безопасно удалить.

---

## Hybrid search: dense + sparse + RRF в одном запросе

Qdrant реализует **нативный RRF (Reciprocal Rank Fusion) на сервере** через Query API с механизмом `prefetch`. Два суб-запроса (dense и sparse) выполняются параллельно, результаты объединяются через RRF — всё в одном HTTP-вызове. Это заменяет текущий `HybridRetriever`, который делает два отдельных запроса и сливает результаты в Python.

```python
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding

client = QdrantClient(url="http://localhost:6333")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")
dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

def hybrid_search(
    query: str,
    limit: int = 10,
    channel: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    prefetch_limit: int = 30,
) -> list:
    """Hybrid search: dense + sparse + RRF + фильтрация в одном запросе."""
    
    # 1. Генерация эмбеддингов
    query_sparse = list(sparse_model.query_embed(query))[0]
    query_dense = list(dense_model.query_embed(query))[0]
    
    # 2. Построение фильтра
    conditions = []
    if channel:
        conditions.append(models.FieldCondition(
            key="channel",
            match=models.MatchValue(value=channel),
        ))
    if date_from or date_to:
        conditions.append(models.FieldCondition(
            key="date",
            range=models.DatetimeRange(gte=date_from, lte=date_to),
        ))
    
    query_filter = models.Filter(must=conditions) if conditions else None
    
    # 3. Единый hybrid-запрос с RRF
    results = client.query_points(
        collection_name="telegram_news",
        prefetch=[
            models.Prefetch(
                query=query_sparse.as_object(),
                using="sparse",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=query_dense.tolist(),
                using="dense",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        with_payload=True,
        limit=limit,
    )
    
    return results.points


# Использование
results = hybrid_search(
    query="курс доллара к рублю",
    channel="finance_channel",
    date_from="2025-03-01T00:00:00Z",
    limit=10,
)

for r in results:
    print(f"Score: {r.score:.4f} | {r.payload['channel']} | {r.payload['text'][:80]}")
```

Начиная с **Qdrant v1.16.0** доступен параметризованный RRF с настройкой константы `k`, а с **v1.17.0** — weighted RRF, позволяющий задать разные веса для dense и sparse ветвей:

```python
# Weighted RRF: dense в 3 раза важнее sparse
results = client.query_points(
    collection_name="telegram_news",
    prefetch=[
        models.Prefetch(query=query_dense.tolist(), using="dense", limit=30),
        models.Prefetch(query=query_sparse.as_object(), using="sparse", limit=30),
    ],
    query=models.RrfQuery(rrf=models.Rrf(weights=[3.0, 1.0])),
    limit=10,
)
```

Альтернативный метод fusion — **DBSF (Distribution-Based Score Fusion)**, который нормализует скоры по распределению перед суммированием (доступен с v1.11.0): `models.FusionQuery(fusion=models.Fusion.DBSF)`.

---

## Фильтрация: must, should и payload-индексы

Qdrant предоставляет развитую систему фильтрации с рекурсивно вложенными условиями. Три базовых оператора — `must` (AND), `should` (OR), `must_not` (NOT) — комбинируются произвольно. Фильтр пропагируется через все уровни prefetch автоматически при указании `query_filter`.

Пример комплексного фильтра для Telegram-новостей:

```python
# Сообщения из каналов "news" ИЛИ "finance", за последние 7 дней,
# НЕ от автора "bot"
from datetime import datetime, timedelta, timezone

week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

complex_filter = models.Filter(
    must=[
        # Вложенный should = OR по каналам
        models.Filter(
            should=[
                models.FieldCondition(
                    key="channel", match=models.MatchValue(value="news"),
                ),
                models.FieldCondition(
                    key="channel", match=models.MatchValue(value="finance"),
                ),
            ]
        ),
        # ИЛИ короче через MatchAny:
        # models.FieldCondition(
        #     key="channel", match=models.MatchAny(any=["news", "finance"]),
        # ),
        models.FieldCondition(
            key="date", range=models.DatetimeRange(gte=week_ago),
        ),
    ],
    must_not=[
        models.FieldCondition(
            key="author", match=models.MatchValue(value="bot"),
        ),
    ],
)
```

Полный список поддерживаемых условий фильтрации: `MatchValue` (точное совпадение), `MatchAny` (IN), `MatchExcept` (NOT IN), `Range` (числовой диапазон), `DatetimeRange` (дата/время в RFC 3339), `MatchText` (полнотекстовый по токенам), `MatchPhrase` (точная фраза, с v1.15.0), `IsEmpty`, `IsNull`, `HasId`, `HasVector`, а также гео-фильтры (BoundingBox, Radius, Polygon).

**Payload-индексы критичны для производительности.** Без индекса фильтрация требует full scan. Qdrant поддерживает типы индексов: `keyword`, `integer`, `float`, `bool`, `datetime`, `text`, `uuid`, `geo`. Параметр `on_disk=True` позволяет вынести индекс на диск для экономии RAM.

---

## MMR нативно с v1.15.0 — без кода на Python

Qdrant **поддерживает MMR (Maximum Marginal Relevance) нативно** на сервере с версии 1.15.0. Это означает, что реализовывать MMR в Python не нужно — алгоритм работает в Rust на стороне Qdrant:

```python
# MMR-поиск: баланс между релевантностью и разнообразием
results = client.query_points(
    collection_name="telegram_news",
    query=models.NearestQuery(
        nearest=query_dense.tolist(),
        mmr=models.MmrParams(
            diversity=0.5,           # 0.0 = чистая релевантность, 1.0 = максимум diversity
            candidates_limit=100,    # пул кандидатов для MMR-отбора
        ),
    ),
    query_filter=channel_filter,     # фильтры совместимы с MMR
    with_payload=True,
    limit=10,
)
```

Параметр `diversity=0.5` — хорошая отправная точка. Для новостного корпуса, где важно показать разные аспекты темы (а не дубликаты из нескольких каналов), рекомендуется **diversity 0.3–0.5**. Значение `candidates_limit=100` определяет размер пула кандидатов — чем больше, тем качественнее MMR, но дороже вычисление.

Дополнительно Qdrant поддерживает **group by** — серверную группировку результатов по payload-полю через `client.query_groups()`. Это полезно для дедупликации по каналу (показывать не более N результатов из одного канала):

```python
results = client.query_groups(
    collection_name="telegram_news",
    query=query_dense.tolist(),
    using="dense",
    group_by="channel",
    limit=5,          # 5 групп (каналов)
    group_size=2,     # макс 2 сообщения на канал
    with_payload=True,
)
```

---

## Upsert: загрузка данных с dense + sparse векторами

Полный пайплайн загрузки Telegram-сообщений с обоими типами векторов:

```python
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding

client = QdrantClient(url="http://localhost:6333")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")
dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

def upsert_messages(messages: list[dict], batch_size: int = 64):
    """
    Загрузка сообщений в Qdrant с dense + sparse векторами.
    Каждое сообщение: {channel, message_id, date, author, text, url}
    """
    texts = [msg["text"] for msg in messages]
    
    # Batch-генерация эмбеддингов
    dense_embs = list(dense_model.embed(texts))
    sparse_embs = list(sparse_model.embed(texts))
    
    # Batch-upsert
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        points = []
        
        for j, msg in enumerate(batch):
            idx = i + j
            points.append(models.PointStruct(
                id=msg["message_id"],     # целочисленный ID сообщения
                vector={
                    "dense": dense_embs[idx].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_embs[idx].indices.tolist(),
                        values=sparse_embs[idx].values.tolist(),
                    ),
                },
                payload={
                    "channel": msg["channel"],
                    "message_id": msg["message_id"],
                    "date": msg["date"],
                    "author": msg.get("author"),
                    "text": msg["text"],
                    "url": msg.get("url"),
                },
            ))
        
        client.upsert(collection_name="telegram_news", points=points)

# Пример использования
messages = [
    {
        "channel": "finance_news",
        "message_id": 10001,
        "date": "2025-03-15T10:30:00Z",
        "author": "admin",
        "text": "Курс доллара вырос до 95 рублей на фоне геополитики",
        "url": "https://t.me/finance_news/10001",
    },
    # ...
]
upsert_messages(messages)
```

---

## План миграции ChromaDB → Qdrant

Миграция для десятков тысяч 1024-мерных векторов — тривиальная задача (~60 МБ данных, минуты работы). Dense-вектора переносятся напрямую без пересчёта. Sparse-вектора нужно генерировать отдельно — ChromaDB их не хранит.

### Шаг 1. Экспорт из ChromaDB

```python
import chromadb

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("telegram_messages")

# Экспорт всех данных (column-major формат)
data = collection.get(include=["embeddings", "documents", "metadatas"])
# data["ids"]        — List[str]
# data["embeddings"] — numpy array (n, 1024)
# data["documents"]  — List[str]
# data["metadatas"]  — List[dict]

print(f"Экспортировано {len(data['ids'])} документов")
```

### Шаг 2. Генерация sparse-векторов

```python
from fastembed import SparseTextEmbedding

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")
texts = data["documents"]
sparse_embeddings = list(sparse_model.embed(texts))
```

### Шаг 3. Создание коллекции Qdrant и загрузка

```python
from qdrant_client import QdrantClient, models
import uuid

client = QdrantClient(url="http://localhost:6333")

# Создать коллекцию (код из раздела «Архитектура» выше)
# Создать payload-индексы (код из раздела «Архитектура» выше)

# Загрузка батчами
BATCH = 100
for i in range(0, len(data["ids"]), BATCH):
    end = min(i + BATCH, len(data["ids"]))
    points = []
    
    for j in range(i, end):
        meta = data["metadatas"][j] or {}
        meta["text"] = data["documents"][j]
        meta["chroma_id"] = data["ids"][j]
        
        # ID: если message_id числовой — используем его;
        # иначе — детерминистический UUID из строкового ID ChromaDB
        point_id = meta.get("message_id", str(uuid.uuid5(uuid.NAMESPACE_URL, data["ids"][j])))
        
        points.append(models.PointStruct(
            id=point_id,
            vector={
                "dense": data["embeddings"][j].tolist(),
                "sparse": models.SparseVector(
                    indices=sparse_embeddings[j].indices.tolist(),
                    values=sparse_embeddings[j].values.tolist(),
                ),
            },
            payload=meta,
        ))
    
    client.upsert(collection_name="telegram_news", points=points)
```

### Шаг 4. Валидация

```python
# Сравнить количество
qdrant_info = client.get_collection("telegram_news")
assert qdrant_info.points_count == len(data["ids"]), "Расхождение в количестве точек!"

# Spot-check: поиск одного и того же запроса в обоих системах
# и сравнение top-5 результатов
```

### Шаг 5. Переключение приложения

Рекомендуется паттерн **feature flag**: обе системы работают параллельно 1–2 недели, запросы идут в Qdrant, а результаты логируются для сравнения с ChromaDB. После подтверждения качества — ChromaDB и кастомный BM25 удаляются.

### Риски и митигация

| Риск | Вероятность | Митигация |
|---|---|---|
| Несовпадение метрики расстояния (ChromaDB L2 vs Qdrant Cosine) | Высокая | Проверить `collection.metadata["hnsw:space"]` в ChromaDB и точно сопоставить |
| Коррупция данных на Windows Docker (bind mount) | Средняя | Использовать **named volumes**, Qdrant v1.15.3+ |
| Деградация качества sparse поиска vs кастомный BM25 | Низкая | A/B тестирование на реальных запросах |
| Потеря ID при миграции (ChromaDB string → Qdrant UUID) | Низкая | Сохранить original ID в payload `chroma_id` |

---

## Производительность: тюнинг не нужен

Для десятков тысяч 1024-мерных векторов Qdrant работает на дефолтных настройках без проблем. Формула оценки памяти: `vectors × dimensions × 4 bytes × 1.5` — для **50 000 × 1024 = ~300 МБ** с учётом overhead. Дефолтные HNSW-параметры (`m=16`, `ef_construct=100`) оптимальны для этого масштаба — тюнинг оправдан только начиная с **миллионов** векторов.

**Qdrant single-node** без проблем обрабатывает до миллионов точек в RAM, а с mmap и квантизацией — до миллиарда (подтверждённые тесты на 1B × 128d). Для текущего масштаба **квантизация не нужна**. Scalar quantization (int8) даёт 4× сжатие с минимальной потерей точности и рекомендуется только при >100K точек.

На **Docker Desktop под Windows** есть специфика: использовать **named volumes вместо bind mounts** (bind mounts вызывают коррупцию данных из-за не-POSIX файловой системы), `io_uring` недоступен (только Linux), WSL2 добавляет **5–15% overhead** по I/O. Для текущего масштаба это несущественно. Для продакшена Qdrant рекомендует Linux.

```bash
# Запуск Qdrant на Windows Docker — правильный способ
docker volume create qdrant-data
docker run -d --name qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v qdrant-data:/qdrant/storage \
    qdrant/qdrant:v1.15.3
```

---

## Заключение: что даёт миграция и что решить

Переход с ChromaDB + custom BM25 на Qdrant с named vectors устраняет **~400 строк кода** (BM25IndexManager, BM25Retriever, HybridRetriever с ручным RRF), заменяя их единой коллекцией и одним вызовом `query_points` с `prefetch` + `FusionQuery`. Серверный IDF автоматически обновляется при мутациях коллекции — не нужно перестраивать индекс вручную. Нативный MMR (с v1.15.0) позволяет убрать и кастомную логику diversification.

Три ключевых решения для реализации: **использовать `Qdrant/bm25` с `language="russian"` вместо BM42** (BM42 экспериментальный и не поддерживает русский); **weighted RRF** (v1.17.0+) для тонкой настройки баланса dense/sparse вместо равных весов; **BGE reranker оставить** как post-processing поверх Qdrant-результатов, поскольку cross-encoder всегда точнее bi-encoder retrieval. Полный пайплайн: `Qdrant hybrid search (BM25 sparse + e5 dense → RRF) → BGE CrossEncoder rerank → top-K результат`.