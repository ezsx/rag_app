# Telegram RAG Application - система вопросов и ответов

Система **Retrieval-Augmented Generation (RAG)** для ответов на вопросы на основе сообщений из каналов telegram.

## ✨ Возможности

- 🇷🇺 **Русскоязычная модель** - Vikhr-7B-instruct для качественных ответов
- 🚀 **GPU ускорение** - Быстрая обработка на NVIDIA GPU
- 📊 **Векторная база** - ChromaDB для поиска релевантных документов  
- 📱 **REST API v1** - Полноценный HTTP API с документацией
- 📨 **Telegram инgest REST** - Управление загрузкой через API
- 🔄 **Горячее переключение моделей** - Смена LLM/embedding без перезагрузки
- 📂 **Управление коллекциями** - Переключение между разными базами данных
- ⚡ **Redis кеширование** - Ускорение повторных запросов
- 🔍 **Семантический поиск** - Поиск документов без генерации ответов
- 🔥 **SSE стриминг** - Ответы в реальном времени через Server-Sent Events
- 🐳 **Docker готовность** - Простой запуск в контейнерах

## 🛠 Технологии

- **Backend**: FastAPI, Python 3.11+
- **Векторная БД**: ChromaDB 1.0.13
- **LLM**: Vikhr-7B-instruct (GGUF Q4_K_M)
- **Embeddings**: intfloat/multilingual-e5-large
- **ML**: PyTorch 2.2, Sentence-Transformers
- **Деплой**: Docker, NVIDIA Container Runtime

## 📋 Требования

### Минимальные
- Docker + Docker Compose
- 8 GB RAM
- 10 GB свободного места

### Рекомендуемые (для GPU)
- NVIDIA GPU с 8+ GB VRAM
- NVIDIA Container Runtime
- CUDA 12.x

## 🚀 Быстрый старт

### 1. Запуск системы
```bash
# Первый запуск (скачает и настроит модели)
docker compose --profile api up

# Ожидаем сообщение: "✅ Инициализация завершена успешно"
# API доступен по адресу: http://localhost:8000
```

### 2. Проверка работы
```bash
# Проверяем статус
curl http://localhost:8000/v1/health

# Просматриваем все endpoints
curl http://localhost:8000/

# Пробуем задать вопрос
curl -X POST "http://localhost:8000/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о системе", "include_context": false}'
```

### 3. Настройка кеширования (опционально)
```bash
# Добавьте в .env для ускорения повторных запросов
echo "REDIS_ENABLED=true" >> .env
echo "REDIS_HOST=localhost" >> .env
echo "REDIS_PORT=6379" >> .env

# Запустите Redis
docker run -d --name redis -p 6379:6379 redis:alpine
```

## 📊 Загрузка данных из Telegram

### Способ 1: REST API (рекомендуется)
```bash
# Запустите задачу через API
curl -X POST "http://localhost:8000/v1/ingest/telegram" \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "@durov",
    "since": "2024-01-01",
    "until": "2024-02-01",
    "collection": "durov_posts",
    "device": "auto"
  }'

# Отслеживайте прогресс
curl http://localhost:8000/v1/ingest/{job_id}
```

### Способ 2: CLI скрипт (legacy)
```bash
# Подготовка
TG_API_ID=your_api_id
TG_API_HASH=your_api_hash

# Запуск
docker compose run --rm ingest --channel @durov --since 2024-01-01 \
 --until 2024-02-01 --collection durov_posts
```

**Пример производительности:**
- CPU: ~10,000 сообщений за 30 минут  
- GPU: ~10,000 сообщений за 3-5 минут

## 🔧 Конфигурация (.env)

[.env.example](.env.example)

## 📡 API Endpoints

### 🏥 Системные endpoints

#### GET /v1/health
Проверка статуса системы
```bash
curl http://localhost:8000/v1/health
```

#### GET /v1/info
Информация о текущих настройках
```bash
curl http://localhost:8000/v1/info
```

### 🤖 QA - Вопросы и ответы

#### POST /v1/qa
Основной endpoint для вопросов и ответов

**Простой запрос:**
```bash
curl -X POST "http://localhost:8000/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{"query": "Что такое искусственный интеллект?"}'
```

**С контекстом и выбором коллекции:**
```bash
curl -X POST "http://localhost:8000/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "О чём говорят в новостях?", 
    "include_context": true,
    "collection": "news_demo4"
  }'
```

#### POST /v1/qa/stream 🔥 NEW
Стриминг ответов в реальном времени через Server-Sent Events

**Простой стрим:**
```bash
curl -N -X POST "http://localhost:8000/v1/qa/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "Привет!", "include_context": false}'
```

**Стрим с контекстом:**
```bash
curl -N -X POST "http://localhost:8000/v1/qa/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "Расскажи о новых технологиях",
    "include_context": true,
    "collection": "tech_news"
  }'
```

**Формат ответа (SSE):**
```
event: token
data: Привет
retry: 3000

event: token  
data: !
retry: 3000

event: end
data: [DONE]
retry: 3000
```

**JavaScript пример:**
```javascript
const eventSource = new EventSource('/v1/qa/stream', {
  method: 'POST',
  body: JSON.stringify({query: 'Привет!', include_context: false}),
  headers: {'Content-Type': 'application/json'}
});

eventSource.onmessage = function(event) {
  if (event.data === '[DONE]') {
    eventSource.close();
  } else {
    console.log('Token:', event.data);
  }
};
```

### 🔍 Search - Семантический поиск

#### POST /v1/search/plan — построение плана поиска
Пример:
```bash
curl -X POST "http://localhost:8000/v1/search/plan" \
  -H "Content-Type: application/json" \
  -d '{"query": "новости рбк за январь"}'
```

Ответ:
```json
{
  "normalized_queries": ["новости рбк", "главное за январь"],
  "must_phrases": [],
  "should_phrases": [],
  "metadata_filters": {"date_from": "2024-01-01", "date_to": "2024-01-31"},
  "k_per_query": 10,
  "fusion": "rrf"
}
```

#### POST /v1/search — выполнение поиска по плану
Объединяет результаты нескольких под‑запросов через RRF, опционально возвращает план.
```bash
curl -X POST "http://localhost:8000/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "новости рбк за январь", "plan_debug": true}'
```

### 📚 BM25 индекс и гибридный поиск

- Индекс BM25 хранится в каталоге `./bm25-index` и монтируется в контейнеры API и Ingest через volume.
- Включение гибрида: переменная окружения `HYBRID_ENABLED=true` (по умолчанию включено).
- Параметры:
  - `BM25_INDEX_ROOT` (по умолчанию `./bm25-index`)
  - `HYBRID_TOP_BM25`, `HYBRID_TOP_DENSE` — глубина выборки до RRF
  - `BM25_DEFAULT_TOP_K`, `BM25_RELOAD_MIN_INTERVAL_SEC`
- Инжест Telegram пишет документы одновременно в Chroma и BM25 (батчами, с коммитами).
- Эндпоинты `/v1/search` и `/v1/qa` при включенном гибриде используют BM25+Dense → RRF → (MMR) → (Reranker).

### 📂 Collections - Управление коллекциями

#### GET /v1/collections
Список всех коллекций
```bash
curl http://localhost:8000/v1/collections
```

#### POST /v1/collections/select
Выбор активной коллекции
```bash
curl -X POST "http://localhost:8000/v1/collections/select" \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "news_demo4"}'
```

#### GET /v1/collections/{collection_name}/info
Информация о коллекции
```bash
curl http://localhost:8000/v1/collections/news_demo4/info
```

### 🧠 Models - Управление моделями

#### GET /v1/models
Список доступных моделей
```bash
curl http://localhost:8000/v1/models
```

#### POST /v1/models/select
Переключение модели (горячая замена)
```bash
# Смена LLM модели
curl -X POST "http://localhost:8000/v1/models/select" \
  -H "Content-Type: application/json" \  
  -d '{
    "model_key": "qwen2.5-7b-instruct",
    "model_type": "llm"
  }'

# Смена embedding модели
curl -X POST "http://localhost:8000/v1/models/select" \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "bge-m3", 
    "model_type": "embedding"
  }'
```

#### GET /v1/models/{model_type}/current
Текущая активная модель
```bash
curl http://localhost:8000/v1/models/llm/current
curl http://localhost:8000/v1/models/embedding/current
```

### 📨 Ingest - Загрузка данных из Telegram

#### POST /v1/ingest/telegram
Запуск задачи загрузки (заменяет скрипт)
```bash
curl -X POST "http://localhost:8000/v1/ingest/telegram" \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "@durov",
    "since": "2024-01-01",
    "until": "2024-02-01", 
    "collection": "durov_posts",
    "device": "auto",
    "max_messages": 1000
  }'

# Ответ: {"job_id": "abc-123", "status": "queued", ...}
```

#### GET /v1/ingest/{job_id}
Статус задачи загрузки
```bash
curl http://localhost:8000/v1/ingest/abc-123
```

#### GET /v1/ingest
Список всех задач
```bash
curl http://localhost:8000/v1/ingest
```

#### DELETE /v1/ingest/{job_id}
Отмена задачи
```bash
curl -X DELETE http://localhost:8000/v1/ingest/abc-123
```


## 🔧 Расширенная конфигурация

### Переменные окружения (.env)
```bash
# Модели (горячее переключение)
LLM_MODEL_KEY=vikhr-7b-instruct           # qwen2.5-7b-instruct, saiga-mistral-7b
EMBEDDING_MODEL_KEY=multilingual-e5-large  # bge-m3, multilingual-mpnet

# ChromaDB
CHROMA_COLLECTION=news_demo4
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Redis кеширование (опционально)
REDIS_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL=3600

# Telegram API
TG_API_ID=your_api_id
TG_API_HASH=your_api_hash

# GPU настройки
LLM_GPU_LAYERS=35     # 0 для CPU
CUDA_VISIBLE_DEVICES=0

# Query Planner / Fusion
ENABLE_QUERY_PLANNER=true
FUSION_STRATEGY=rrf                # rrf|mmr (mmr зарезервировано)
K_FUSION=60
ENABLE_RERANKER=false
SEARCH_K_PER_QUERY_DEFAULT=10
MAX_PLAN_SUBQUERIES=5
ENABLE_CACHE=true
```

### Доступные модели
**LLM модели:**
- `vikhr-7b-instruct` - Русскоязычная модель (по умолчанию)
- `qwen2.5-7b-instruct` - Отличная для русского языка
- `saiga-mistral-7b` - Специально для русского
- `openchat-3.6-8b` - Универсальная модель

**Embedding модели:**
- `multilingual-e5-large` - Лучшая многоязычная (по умолчанию)
- `bge-m3` - Отличная многоязычная модель
- `multilingual-mpnet` - Быстрая многоязычная

## 💾 Структура проекта

```
rag_app/
├── 📁 src/                          # Исходный код приложения
│   ├── 📁 api/v1/                   # REST API v1
│   │   ├── 📁 endpoints/            # Отдельные endpoints
│   │   │   ├── 📄 qa.py             # Вопросы и ответы
│   │   │   ├── 📄 search.py         # Семантический поиск
│   │   │   ├── 📄 collections.py    # Управление коллекциями
│   │   │   ├── 📄 models.py         # Управление моделями
│   │   │   ├── 📄 ingest.py         # Telegram ingestion
│   │   │   └── 📄 system.py         # Системные endpoints
│   │   └── 📄 router.py             # Главный роутер v1
│   ├── 📁 adapters/chroma/          # Адаптер для ChromaDB
│   ├── 📁 core/                     # Основная логика
│   │   ├── 📄 deps.py               # Dependency Injection
│   │   └── 📄 settings.py           # Настройки с горячим переключением
│   ├── 📁 services/                 # Бизнес-логика
│   │   ├── 📄 qa_service.py         # QA сервис
│   │   └── 📄 ingest_service.py     # Управление задачами ingestion
│   ├── 📁 schemas/                  # Pydantic схемы для API
│   ├── 📁 utils/                    # Утилиты (промпты, загрузка моделей)
│   ├── 📁 tests/                    # Unit тесты
│   └── 📄 main.py                   # FastAPI приложение (factory)
├── 📁 scripts/                      # Legacy CLI скрипты
├── 🐳 docker-compose.yml            # Оркестрация контейнеров
└── 📄 .env.example                  # Пример конфигурации
```

## 📝 Лицензия

MIT License - см. файл LICENSE для деталей

## 🎯 Использование в продакшене

### Масштабирование
- Используйте Redis для кеширования 
- Настройте несколько реплик API
- Используйте внешний ChromaDB сервер
- Настройте мониторинг через `/v1/info`

### Безопасность  
- Ограничьте CORS origins в продакшене
- Используйте HTTPS
- Настройте аутентификацию для sensitive endpoints
- Ограничьте доступ к инgestам

---
**Версия**: v1.0.0 (Refactored Architecture)  
**API**: v1
