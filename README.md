# RAG App — система вопросов и ответов и ReAct Agent API по Telegram-данным

Сервис Retrieval-Augmented Generation (RAG) с гибридным поиском (BM25 + эмбеддинги), планированием запросов (Query Planner) и генерацией ответов LLM через FastAPI.

## ✨ Возможности

- **Query Planner (LLM + GBNF)**: декомпозиция пользовательского запроса в 3–6 подзапросов, фильтры, `k_per_query`, стратегия слияния. Кеширование планов и результатов fusion.
- **Гибридный поиск**: объединение Chroma (dense) и BM25 через RRF; поддержка MMR и CPU‑ререйкера (BGE v2‑m3).
- **SSE стриминг**: ответы LLM в реальном времени (`/v1/qa/stream`, `/v1/agent/stream`).
- **ReAct Agent**: пошаговое рассуждение + инструменты (`multi_query_rewrite`, `compose_context`, `fetch_docs`, `summarize`, `extract_entities`, `translate`, `fact_check_advanced`, `semantic_similarity`, `content_filter`, `export_to_formats`, и др.).
- **Безопасность**: JWT/Bearer или API‑Key аутентификация, rate limiting, TrustedHost, безопасное логирование, защита от prompt‑injection.
- **Горячая смена моделей**: переключение LLM/Embedding через API без рестартов.
- **Redis (опционально)**: кеширование ответов/поиска.
- **Docker‑готовность**: быстрый запуск и изоляция зависимостей.
- **Инфраструктура под ReAct**: Planner + Hybrid + Reranker + ToolRunner.

## 🛠 Технологии

- **Backend**: FastAPI, Python 3.11+
- **Vector DB**: ChromaDB 1.0.13
- **Retrieval**: BM25 (офлайновый индекс) + Chroma retriever, RRF/MMR
- **LLM**: `gpt-oss-20b` (по умолчанию); Planner LLM: `qwen2.5-3b-instruct` (CPU)
- **Embeddings**: `intfloat/multilingual-e5-large`
- **Reranker**: `BAAI/bge-reranker-v2-m3` (CPU)

## 🚀 Быстрый старт

1) Запуск (Docker Compose):
```bash
# Первый запуск скачает и подготовит модели при необходимости
docker compose --profile api up
# API: http://localhost:8000
```

2) Проверка:
```bash
curl http://localhost:8000/v1/health
curl http://localhost:8000/
```

3) Вопрос‑ответ (простая проверка):
```bash
curl -X POST "http://localhost:8000/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о системе", "include_context": false}'
```

4) ReAct Agent — стриминг шагов (SSE, требуется авторизация):
```bash
curl -N -X POST "http://localhost:8000/v1/agent/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "Authorization: Bearer <YOUR_JWT_TOKEN>" \
  -d '{
    "query": "Найди подтверждения из базы и сделай краткий вывод",
    "tools_allowlist": ["compose_context","fetch_docs","summarize"],
    "max_steps": 4
  }'
```

## 📡 API Endpoints

- System:
  - `GET /v1/health`
  - `GET /v1/info`
- QA:
  - `POST /v1/qa` — синхронный ответ
  - `POST /v1/qa/stream` — стриминг через SSE
- Search:
  - `POST /v1/search/plan` — построение плана
  - `POST /v1/search` — выполнение поиска (с планом внутри)
- Collections:
  - `GET /v1/collections`
  - `POST /v1/collections/select`
  - `GET /v1/collections/{collection_name}/info`
- Models:
  - `GET /v1/models`
  - `POST /v1/models/select`
  - `GET /v1/models/{model_type}/current`
 - Agent (требуется аутентификация):
  - `POST /v1/agent/stream` — ReAct-агент, стриминг шагов через SSE
  - `GET /v1/agent/tools` — список доступных инструментов агента
  - `GET /v1/agent/status` — статус и текущая конфигурация агента
- Ingest (Telegram):
  - `POST /v1/ingest/telegram`
  - `GET /v1/ingest/{job_id}`
  - `GET /v1/ingest`
  - `DELETE /v1/ingest/{job_id}`

Пример Search Plan (возвращает минимум 3 подзапроса):
```bash
curl -X POST "http://localhost:8000/v1/search/plan" \
  -H "Content-Type: application/json" \
  -d '{"query": "новости рбк за январь"}'
```
Ответ (пример):
```json
{
  "normalized_queries": [
    "новости рбк январь",
    "главные темы рбк январь",
    "итоги января рбк"
  ],
  "must_phrases": [],
  "should_phrases": [],
  "metadata_filters": {"date_from": "2024-01-01", "date_to": "2024-01-31"},
  "k_per_query": 10,
  "fusion": "rrf"
}
```

## 🔐 Аутентификация и безопасность

- Поддерживаются два способа аутентификации:
  - **Bearer JWT**: заголовок `Authorization: Bearer <JWT>`
  - **API Key**: заголовок `X-API-Key: <KEY>`
- Rate limiting активен по умолчанию. Заголовки ответа: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.
- TrustedHost/CORS настраиваются через переменные окружения.

Быстрый пример запроса с авторизацией:
```bash
curl -H "Authorization: Bearer <YOUR_JWT_TOKEN>" http://localhost:8000/v1/agent/status
```

## 🧠 Query Planner вкратце

- LLM строит структурированный JSON‑план. При включенной опции GBNF используется строгая грамматика, при неудаче — `chat_completion` с `response_format=json_schema` и пост‑парсинг.
- Пост‑валидация нормализует фразы, ограничивает длину/число подзапросов, приводит фильтры, задаёт «разумные» значения по умолчанию.
- Результаты fusion (RRF/MMR) кешируются на короткое время и переиспользуются в QA/поиске.

## 🔧 Конфигурация (.env)

Минимально полезные параметры:
```bash
# Модели (горячее переключение через API тоже доступно)
LLM_MODEL_KEY=gpt-oss-20b
EMBEDDING_MODEL_KEY=multilingual-e5-large

# Query Planner (CPU LLM)
ENABLE_QUERY_PLANNER=true
PLANNER_LLM_MODEL_KEY=qwen2.5-3b-instruct
PLANNER_LLM_DEVICE=cpu
USE_GBNF_PLANNER=true
MAX_PLAN_SUBQUERIES=5
SEARCH_K_PER_QUERY_DEFAULT=10

# Fusion / Ranking
FUSION_STRATEGY=rrf                  # rrf|mmr
K_FUSION=60
ENABLE_MMR=true
MMR_LAMBDA=0.7
MMR_TOP_N=120
MMR_OUTPUT_K=60
ENABLE_RERANKER=true
RERANKER_MODEL_KEY=BAAI/bge-reranker-v2-m3
RERANKER_TOP_N=80
RERANKER_BATCH_SIZE=16

# Hybrid / BM25
HYBRID_ENABLED=true
HYBRID_TOP_BM25=100
HYBRID_TOP_DENSE=100
BM25_INDEX_ROOT=./bm25-index
BM25_DEFAULT_TOP_K=100
BM25_RELOAD_MIN_INTERVAL_SEC=5

# ChromaDB
CHROMA_COLLECTION=news_demo4
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_PATH=/data/chroma

# Кеширование
ENABLE_CACHE=true
REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
CACHE_TTL=3600

# Пути к моделям и кэшам
MODELS_DIR=/models
TRANSFORMERS_CACHE=/models/.cache

# Параметры LLM (llama.cpp)
LLM_GPU_LAYERS=-1            # 0=CPU, >0=частично на GPU
LLM_CONTEXT_SIZE=4096
LLM_THREADS=8
LLM_BATCH=1024
CUDA_VISIBLE_DEVICES=0
```
Дополнительно поддерживаются: `AUTO_DOWNLOAD_LLM`, `AUTO_DOWNLOAD_EMBEDDING`, `AUTO_DOWNLOAD_RERANKER`, `LLM_MODEL_PATH`, `PLANNER_CHAT_FORMAT`, `PLANNER_LLM_CONTEXT_SIZE`, `PLANNER_LLM_THREADS`, `PLANNER_LLM_BATCH`, `RETRIEVER_TOP_K` и др.

Дополнительные параметры безопасности и агента:
```bash
# Security / Auth
JWT_SECRET=change-me-in-prod
JWT_EXPIRATION_HOURS=24
VALID_API_KEYS=

# Rate limiting / CORS / Hosts
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_BURST=10
ALLOWED_HOSTS=*
CORS_ORIGINS=*
DEBUG=false

# Agent / ReAct
AGENT_MAX_STEPS=4
AGENT_TOOL_TIMEOUT=5.0
AGENT_TOKEN_BUDGET=2048
```

## 💾 Структура проекта (ключевое)

```
rag_app/
├── src/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── qa.py                # /v1/qa, /v1/qa/stream
│   │   │   ├── search.py            # /v1/search/plan, /v1/search
│   │   │   ├── collections.py       # /v1/collections, select, info
│   │   │   ├── models.py            # /v1/models, select
│   │   │   ├── ingest.py            # /v1/ingest/*
│   │   │   └── system.py            # /v1/health, /v1/info
│   │   └── router.py
│   ├── core/
│   │   ├── settings.py              # все флаги и параметры
│   │   └── deps.py                  # DI-фабрики (LLM, Planner, Hybrid, Reranker)
│   ├── services/
│   │   ├── query_planner_service.py # планировщик запросов + TTL кеши
│   │   ├── qa_service.py            # сбор контекста, ответ/стриминг
│   │   └── reranker_service.py      # CPU BGE v2-m3
│   ├── adapters/
│   │   ├── chroma/                  # Chroma retriever
│   │   └── search/                  # BM25 + Hybrid retriever
│   ├── utils/                       # gbnf.py, ranking.py, prompt.py, model_downloader.py
│   └── main.py                      # FastAPI app
├── docs/ai/                         # архитектура, модули, pipeline
├── bm25-index/                      # офлайн индекс
├── chroma-data/                     # векторное хранилище
└── models/                          # GGUF и кэши HF
```

## 🧭 Архитектура и Roadmap ReAct

- Высокоуровневая диаграмма: см. `diagram.md` (блоки API → Planner → Hybrid → RRF/MMR → Reranker → LLM Answer; Roadmap: ReAct c инструментами `search() · rerank() · verify()`).
- Planner уже обеспечивает устойчивый JSON‑план и микро‑догенерацию недостающих подзапросов; гибридный ретривер готов к роли `search()` инструмента.

## 📨 Загрузка данных из Telegram

Через REST API:
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
```

## 📋 Рекомендации по продакшену

- Включите Redis и ограничьте CORS/HTTPS.
- Запускайте несколько реплик API, используйте внешний ChromaDB сервер.
- Следите за `/v1/info` и логами планировщика/гибрида/ререйкера.
- Настройте `JWT_SECRET`, `ALLOWED_HOSTS`, лимиты `RATE_LIMIT_*`, и используйте `Authorization`/`X-API-Key` во всех продакшен‑запросах.

## 📝 Лицензия

MIT License (см. LICENSE)
