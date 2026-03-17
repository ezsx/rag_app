# rag_app — Research Prompts

> **Как пользоваться**: каждый блок ниже — готовый промпт для deep research (Claude / ChatGPT / Perplexity).
> Копируешь промпт → отправляешь → сохраняешь ответ в `docs/research/rag-stack/reports/RXX-*.md`.
>
> Результаты становятся source of truth для архитектурных решений.
>
> **Порядок**: R01 и R02 — независимы, запускай параллельно. R03 — независим.
> R07 зависит от R02. R04 зависит от R03. R05 зависит от R04.

---

## R01 — Qdrant для hybrid RAG: замена ChromaDB + BM25

**Файл результата**: `reports/R01-qdrant-hybrid-rag.md`
**Зависимости**: нет
**Приоритет**: P0

```
Ты — эксперт по векторным базам данных и RAG системам.

## Контекст

У меня есть RAG система на Python/FastAPI:
- Telegram-новости индексируются в ChromaDB (dense vectors, multilingual-e5-large)
- Параллельно идёт custom BM25 index на диске (~400 строк кода: BM25IndexManager, BM25Retriever)
- HybridRetriever делает BM25 + ChromaDB → RRF fusion → результат
- BGE reranker (CrossEncoder) как post-processing
- Qdrant уже работает в Docker (используется для другого инструмента)

Объём данных: десятки тысяч сообщений из 5-10 Telegram каналов.
Язык: русский (основной) + мультиязычный.

## Вопросы для исследования

### 1. Sparse vectors в Qdrant
- Как работают sparse vectors в Qdrant (SPLADE, BM42, custom)?
- BM42 от Qdrant: что это, чем отличается от BM25, как загрузить модель?
- Как создать коллекцию с named vectors (dense + sparse) в Qdrant Python client?
- Как выполнить hybrid query (dense + sparse + RRF) в одном запросе к Qdrant?
- Нужен ли отдельный BM25 index если есть sparse vectors?

### 2. Filtering и metadata
- Как устроен payload filtering в Qdrant? Какие операторы поддерживаются?
- Как фильтровать по: channel_name, date range, author, message_type?
- Есть ли payload indexing для производительности фильтрации на больших коллекциях?
- Как использовать `must` + `should` условия в Qdrant filter?

### 3. MMR (Maximum Marginal Relevance)
- Поддерживает ли Qdrant MMR нативно или нужно реализовывать в Python?
- Как реализовать MMR поверх Qdrant search results (алгоритм + код)?
- Есть ли встроенный `diversity` или `deduplication` в Qdrant?

### 4. Миграция из ChromaDB
- Как получить все векторы из ChromaDB (collection.get(include=['embeddings', 'documents', 'metadatas']))?
- Как загрузить их в Qdrant с сохранением metadata?
- Нужно ли переиндексировать (пересчитывать embeddings) или можно перенести векторы напрямую?
- Как организовать схему коллекции Qdrant для Telegram-сообщений?
  Пример payload: {channel, message_id, date, author, text, url}

### 5. Производительность
- Qdrant Docker на Windows — есть ли ограничения по сравнению с Linux?
- Какой размер collection (points) до которого single-node Qdrant работает без проблем?
- Нужна ли оптимизация (HNSW params, quantization) для нашего масштаба?

## Ожидаемый результат
Подробный технический отчёт с:
- Архитектурным решением: какую схему коллекции использовать
- Примерами кода на Python для: создание коллекции, upsert, hybrid search, filtered search
- Сравнением BM42 vs кастомный BM25 для новостного корпуса (плюсы/минусы)
- Планом миграции: ChromaDB → Qdrant (шаги, риски, откат)
- Оценкой: стоит ли убирать кастомный BM25 полностью или оставить как fallback
```

---

## R02 — LLM Serving: vLLM vs Ollama для local models

**Файл результата**: `reports/R02-llm-serving.md`
**Зависимости**: нет
**Приоритет**: P0

```
Ты — эксперт по деплойменту LLM в production.

## Контекст

У меня ReAct агент на FastAPI. Сейчас LLM загружается через llama.cpp (GGUF файлы):
- Qwen2.5-7B-Instruct (GPU, ~8-10GB VRAM) — основной агент
- Qwen2.5-3B-Instruct (CPU) — Query Planner
- Функция get_llm() — 150 строк defensive кода (retry mmap/mlock, GGUF header check, __del__ hack)
- Вызовы LLM СИНХРОННЫЕ в async FastAPI — блокируют event loop
- Скоро переезд с RTX 5060Ti 16GB на V100 SXM2 32GB

Требования к агенту:
- Sequential generation (Thought → Action → Observation цикл, не batching)
- JSON output для tool parameters должен быть надёжным
- SSE streaming ответов пользователю
- Latency важнее throughput (один пользователь, но отзывчивость важна)

## Вопросы для исследования

### 1. vLLM
- vLLM 0.6.x — как запустить Qwen2.5-7B-Instruct через OpenAI-compatible API?
- Поддерживает ли vLLM GGUF или нужны safetensors/HF format?
- Как настроить guided decoding / structured outputs в vLLM для JSON tool параметров?
- Можно ли обслуживать два размера модели (7B + 3B) из одного vLLM инстанса?
- Streaming в vLLM: как работает `stream=True` в OpenAI API?
- V100 SXM2 32GB + vLLM: какие параметры (max_model_len, gpu_memory_utilization)?
- Как настроить Docker Compose сервис для vLLM?

### 2. Ollama
- Ollama поддерживает Qwen2.5-7B? Есть ли официальная модель в реестре?
- Ollama API vs OpenAI-compatible API — как подключить к FastAPI?
- Ollama structured outputs: поддерживается? Как настроить?
- V100 support в Ollama — есть ли ограничения?
- Сравнение vLLM vs Ollama для нашего use case: latency, простота настройки, надёжность

### 3. OpenAI-compatible client в Python
- Как переписать AgentService._generate_step() с llama_cpp → httpx/openai client?
- Пример кода: async streaming генерация через OpenAI API Python client
- Как передать system_prompt + conversation_history в chat completions format?
- Как реализовать timeout на HTTP-вызов LLM?
- Qwen2.5 chat template через API: нужно ли указывать что-то дополнительно?

### 4. Structured outputs / JSON generation
- Как обеспечить надёжный JSON output для tool parameters?
  - Grammar-based (GBNF в llama.cpp → что аналогично в vLLM)?
  - Guided decoding (outlines integration в vLLM)?
  - JSON schema в chat completions format?
- Насколько Qwen2.5-7B надёжен для function calling без constrained generation?

### 5. Производительность на V100
- Qwen2.5-7B FP16 на V100 32GB: throughput tokens/sec, latency первого токена?
- INT8 vs FP16 — стоит ли квантизировать на 32GB V100?
- Можно ли запустить 7B + embedding model + reranker одновременно?
- KV cache size для context=10000 токенов на V100?

## Ожидаемый результат
- Рекомендация: vLLM vs Ollama с обоснованием
- Docker Compose конфиг для выбранного сервера
- Python код: LLMClient абстракция совместимая с llama.cpp И vLLM API
- Пример async streaming генерации
- Параметры для V100 SXM2 32GB
```

---

## R03 — Выбор LLM модели для V100 32GB + Russian RAG + ReAct агент

**Файл результата**: `reports/R03-model-selection.md`
**Зависимости**: нет
**Приоритет**: P0

```
Ты — эксперт по LLM моделям, русскому NLP и ReAct агентам.

## Контекст

Строю RAG-агент для русскоязычных Telegram-новостей:
- Архитектура: ReAct цикл с 7 инструментами, SSE стриминг, FastAPI
- Retrieval: Qdrant hybrid search (dense + BM25 sparse) + BGE reranker
- LLM serving: vLLM v0.15.1 (OpenAI-compatible API) на V100 SXM2 32GB
- Задачи модели: планирование поиска, вызов инструментов (structured JSON),
  финальный ответ на русском языке (~500-1000 токенов)

VRAM бюджет на V100 32GB:
- V100 выделена **только под LLM** — embedding и reranker работают на отдельной GPU (RTX 5060 Ti)
- KV-кэш 8K ctx при GQA: ~0.5 GB (очень мало)
- Реальный бюджет под модель: ~28–30 GB (FP16)

Текущий setup двух моделей:
- Main LLM (Qwen2.5-7B): ReAct цикл, tool calling, финальный ответ
- Planner LLM (Qwen2.5-3B): только структурированный JSON (SearchPlan), CPU через Ollama
- Вопрос открыт: стоит ли взять одну большую модель для обоих или оставить раздельно?

Текущий формат tool calling:
- Текстовый ReAct: `Thought: ...\nAction: tool_name {"param": "val"}\nObservation: ...`
- Regex-парсер для извлечения Action из текста
- Финальный ответ: `Final Answer: текст на русском`

Текущая модель: Qwen2.5-7B-Instruct (выбрана под старое железо RTX 5060 Ti 16GB).
Теперь есть 32GB V100 — нужно выбрать лучшую модель осознанно.

## Вопросы для исследования

### 1. Какие модели влезают в 30GB V100 (FP16)?
- Составить список моделей, которые умещаются в ~28–30 GB:
  7B, 9B, 12B, 13B, 14B, 20B, 32B (Q4/Q8)
- Для каждой: VRAM FP16, VRAM INT8, поддержка в vLLM v0.15.1
- Есть ли смысл в INT8/AWQ квантизации на V100 или лучше FP16 на меньшей модели?
- Один большой LLM vs два маленьких (main 7B + planner 3B):
  стоит ли использовать одну модель для обоих задач или раздельный setup выгоднее?

### 2. Качество для русского языка (MTEB 2025/2026)
- Рейтинг моделей по русскоязычным задачам в 2025-2026:
  - Qwen2.5-14B vs Mistral Nemo 12B vs Gemma 3 12B vs LLaMA 3.3 / LLaMA 3.1 8B
  - Есть ли специализированные RU-модели в топе (GigaChat, Vikhr, Saiga)?
  - Насколько важен размер (7B vs 14B) для качества русского текста?
- Какая модель лучше держит language policy (не мешает RU/EN в ответе)?

### 3. Надёжность tool calling и structured output
- Рейтинг моделей по Berkeley Function Calling Leaderboard (BFCL) 2025:
  какие из ≤15B моделей лидируют по multi-turn agentic score?
- Нативный tool calling vs текстовый ReAct format: что лучше для выбранных моделей?
  Какой формат поддерживается в vLLM v0.15.1 для каждой модели?
- Constrained generation (xgrammar в vLLM): насколько надёжнее vs prompt-only?
- Рекомендуемая температура для structured output (tool calls) у локальных моделей.

### 4. Prompt engineering для выбранной модели
- Оптимальный system prompt для ReAct агента на выбранной модели:
  - Структура, длина, язык (RU vs EN vs смешанный)?
  - Few-shot examples нужны или достаточно zero-shot?
  - Chain of Thought: помогает или лишние токены?
- Как форматировать tool definitions для максимального следования инструкциям?
- Стратегия context management при 8K токенов: сколько документов передавать?

### 5. Надёжность parsing и обработка ошибок
- Как восстанавливаться от malformed JSON в Action?
  - Retry с повышенной температурой?
  - json-repair библиотека?
  - Constrained decoding как первая линия защиты?
- Как обнаружить hallucination loop (модель повторяет одно действие)?
- Обработка случая когда модель пропускает обязательный шаг (например, compose_context).

## Ожидаемый результат
- Рекомендация: конкретная модель (name + size + формат) для V100 32GB + Russian RAG
- Обоснование: VRAM, MTEB RU, BFCL, поддержка vLLM
- vLLM конфигурация для выбранной модели (dtype, max_model_len, tool-call-parser)
- Рекомендации по system prompt (структура, язык, длина)
- Стратегия надёжного tool calling (constrained vs prompt-only, parsing fallback)
```

---

## R04 — Coverage и Context Quality Metrics

**Файл результата**: `reports/R04-coverage-metrics.md`
**Зависимости**: R03
**Приоритет**: P1

```
Ты — эксперт по оценке качества RAG систем.

## Контекст

В моём ReAct агенте есть инструмент compose_context, который собирает контекст из
найденных документов и возвращает citation_coverage (float 0-1).
Если coverage < 0.8 → агент делает ещё один поисковый round.
Если coverage >= 0.8 → переходит к verify → final_answer.

Проблема: не ясно что именно измеряет citation_coverage.
Предположительно — это что-то вроде (количество документов в контексте / ожидаемое количество)
или heuristic по количеству цитат. Это слабый сигнал.

## Вопросы для исследования

### 1. Что значит "coverage" в RAG
- Как правильно определить "достаточность" контекста для ответа на вопрос?
- Какие метрики используются в production RAG системах для решения о refinement:
  - Семантическое покрытие: cosine sim между query и retrieved context?
  - Retrieval confidence: распределение scores от поиска?
  - LLM-as-judge: задать LLM вопрос "достаточно ли информации?"
  - Hybrid: комбинация нескольких сигналов?

### 2. Calibration threshold
- Как калибровать threshold на реальных данных?
- Что такое precision-recall tradeoff для refinement trigger:
  - False positive (делаем лишний поиск когда не нужно) vs False negative (не делаем когда нужно)?
  - Какой вред больше для пользователя?

### 3. Практические реализации
- FLARE (Forward-Looking Active REtrieval): как работает? Применимо?
- Self-RAG (Reflective RAG): подход с critique tokens — применимо к local 7B?
- Adaptive RAG: dynamic decision когда retrieval нужен — применимо?
- Есть ли simple heuristics которые работают лучше сложных подходов?

### 4. Применительно к нашей системе
- compose_context возвращает "prompt" + "citations" + "coverage".
  Как переработать чтобы coverage был осмысленным?
- Что хранить в citation metadata чтобы coverage был вычислим?
- Нужно ли передавать query в compose_context для семантического расчёта coverage?
- Retrieval мигрирует на Qdrant hybrid search (dense cosine + BM25 sparse → RRF fusion).
  Скоры из Qdrant: cosine similarity в диапазоне 0–1 (не L2 как в ChromaDB).
  Как использовать эти скоры для coverage? RRF-скор vs raw cosine — что информативнее?

## Ожидаемый результат
- Определение правильной метрики coverage для нашего агента
- Код: функция calculate_coverage(query, retrieved_docs) → float
- Обоснование порога 0.8 (или рекомендация сменить)
- Простая реализация без зависимости от внешних LLM (локальная)
```

---

## R05 — RAG Evaluation: RAGAS, DeepEval, LLM-judge

**Файл результата**: `reports/R05-rag-evaluation.md`
**Зависимости**: R03, R04
**Приоритет**: P1

```
Ты — эксперт по evaluation LLM систем и RAG.

## Контекст

У меня есть scripts/evaluate_agent.py — CLI скрипт который:
- Читает datasets/eval_dataset.json (пока 2 фейковых примера)
- Вызывает /v1/agent/stream для каждого вопроса
- Считает recall@5 (по expected_documents), coverage, latency
- Генерирует markdown отчёт

Нужно расширить до полноценной evaluation системы включая LLM-as-judge.

## Вопросы для исследования

### 1. RAGAS
- Что такое RAGAS и какие метрики он считает:
  - faithfulness (соответствие документам)
  - answer_relevancy (релевантность ответа вопросу)
  - context_recall (покрытие релевантных документов)
  - context_precision (точность retrieved документов)
- Как подключить RAGAS к нашему агенту (не стандартный chain)?
- Нужна ли reference LLM для RAGAS metrics? Как использовать локальный Qwen вместо GPT-4?
- RAGAS с русскоязычными данными — есть ли проблемы?

### 2. DeepEval
- DeepEval vs RAGAS — в чём разница? Что лучше для нашего случая?
- Какие метрики DeepEval доступны без OpenAI API?
- Как настроить custom judge model (Qwen3-8B как судья, локально через vLLM/llama-server)?

### 3. LLM-as-judge для нашего агента
- Промпт для оценки ответа агента по критериям:
  - Relevance: ответ соответствует вопросу?
  - Faithfulness: ответ не выходит за пределы источников?
  - Completeness: все аспекты вопроса покрыты?
  - Citation accuracy: цитаты корректны?
- Как нормализовать оценки: числовая шкала 1-5 или binary pass/fail?
- Consistency: как сделать судью детерминированным (temperature=0, seed)?

### 4. Датасет
- Как сгенерировать eval датасет из реальных данных Qdrant (hybrid search коллекция)?
  - Пример кода: случайная выборка точек из Qdrant → LLM генерирует вопрос → (вопрос, документы, ожидаемый ответ)
  - Как покрыть разные типы вопросов: temporal, aggregation, factual, negative?
  - Минимальный размер для статистической значимости (50? 100? 200 примеров)?
- Формат eval_dataset.json: какие поля обязательны?
- LLM для генерации вопросов: Qwen3-8B (локальный) — достаточно качественно?

### 5. Интеграция
- Как добавить LLM-judge в существующий evaluate_agent.py?
- Как сравнивать версии агента (baseline QA vs ReAct) по одному датасету?
- Как автоматизировать eval (pre-commit hook? nightly job?)?

## Ожидаемый результат
- Выбор инструмента: RAGAS vs DeepEval vs custom judge (с обоснованием)
- Промпт судьи для оценки ответов агента
- Код: генератор eval датасета из ChromaDB данных
- Расширенный план metrics: что добавить в evaluate_agent.py Phase 2
- Минимальный working пример evaluation pipeline
```

---

## R06 — Async архитектура: blocking LLM, tools, SSE

**Файл результата**: `reports/R06-async-architecture.md`
**Зависимости**: R02
**Приоритет**: P1

```
Ты — эксперт по async Python, FastAPI и production-grade web сервисам.

## Контекст

FastAPI приложение (uvicorn), LLM inference через HTTP к внешнему llama-server.
Текущий LLM-клиент: `LlamaServerClient` — тонкая обёртка над `requests.Session.post()`.
Целевой LLM-сервер: vLLM v0.15.1 (pinned для V100 xformers), доступен через OpenAI API.

Проблемы:
1. `requests.Session.post()` синхронный → блокирует uvicorn event loop при каждом LLM-вызове
2. ToolRunner использует ThreadPoolExecutor для каждого tool call
3. AgentService._current_step — instance variable, AgentService singleton через @lru_cache
   → state шарится между concurrent requests!
4. SSE streaming через sse_starlette.EventSourceResponse

Целевое состояние (после Proxmox + vLLM):
- LLM-вызовы: `AsyncOpenAI(base_url="http://vllm:8000/v1")` — нативный async
- До этого (промежуточный фикс): заменить `requests` на `httpx.AsyncClient`

## Вопросы

### 1. Промежуточный фикс: requests → httpx
- Как минимально заменить `requests.Session.post()` на `httpx.AsyncClient` в `LlamaServerClient`?
- Нужен ли `asyncio.run_in_executor` как альтернатива если httpx не подходит?
- Как правильно управлять жизненным циклом `httpx.AsyncClient` (один на приложение vs per-request)?
- Как измерить impact на latency и throughput при параллельных запросах?

### 2. Shared state в AgentService
- AgentService._current_step и _current_request_id — это instance variables но AgentService
  создаётся как singleton через @lru_cache → state шарится между concurrent requests!
- Как правильно изолировать per-request state?
  - request-scoped AgentService (создавать на каждый запрос)?
  - Передавать state через параметры?
  - contextvars.ContextVar для async-safe per-request state?

### 3. ToolRunner async
- Текущий ToolRunner.run() синхронный с ThreadPoolExecutor
- Как перевести tools на async def?
- Нужна ли asyncio.Semaphore для ограничения параллельных tool calls?

### 4. SSE и client disconnect
- Как корректно обрабатывать client disconnect во время SSE stream?
- Текущий код: `await fastapi_request.is_disconnected()` — это polling, правильно ли?
- Как прерывать LLM generation при disconnect?
- Memory leaks при незакрытых SSE соединениях?

### 5. FastAPI lifespan
- Как использовать lifespan для инициализации сервисов вместо @lru_cache?
- Пример: инициализация HybridRetriever, LLM client, BM25 при startup

## Ожидаемый результат
- Решение для blocking LLM calls (с кодом)
- Паттерн для per-request state isolation
- Рекомендации по async tools
- Правильная обработка SSE disconnect
```

---

## R07 — Proxmox + VFIO: V100 SXM2 → Linux VM для vLLM

**Файл результата**: `reports/R07-proxmox-vfio.md`
**Зависимости**: R02 (выполнен — vLLM v0.15.1 выбран)
**Приоритет**: P0

```
Ты — эксперт по Proxmox, VFIO GPU passthrough и Linux ML инфраструктуре.

## Контекст

Текущее железо (один физический сервер):
- CPU: Intel Xeon E5-2699 v4 (или аналогичный, сокет LGA2011-3 / C612 чипсет)
- GPU 1: V100 SXM2 32GB (Tesla, TCC mode, CUDA 7.0)
- GPU 2: RTX 5060 Ti 16GB (GDDR7, Blackwell, CUDA 12.x)
- OS сейчас: Windows 11 Pro
- Есть свободный NVMe SSD для Proxmox

Текущая проблема:
- V100 работает в TCC mode → недоступна из WSL2/Docker (GPU-PV не поддерживает TCC)
- Сейчас: LLM inference на V100 через llama-server.exe на Windows хосте, Docker → HTTP
- Цель: vLLM v0.15.1 на V100 в Linux VM (vLLM не работает на Windows)
- RTX 5060 Ti: остаётся для Windows (Gaming + embedding в Docker через WSL2)

Целевая архитектура после Proxmox:
- Proxmox на новом NVMe (UEFI boot)
- VM 1 (Linux): V100 passthrough → vLLM v0.15.1 + Docker (FastAPI, Qdrant, embedding)
- VM 2 (Windows): RTX 5060 Ti passthrough → игры, разработка
- Сеть между VM: bridge или внутренняя сеть Proxmox

## Вопросы

### 1. Proxmox установка на новый NVMe рядом с Windows
- Как установить Proxmox на второй NVMe не трогая Windows на первом?
- BIOS/UEFI: как настроить boot menu для выбора между Proxmox и Windows?
- Нужно ли включать IOMMU в BIOS заранее? Какие параметры (Intel VT-d)?
- После установки Proxmox: как проверить что IOMMU активен?

### 2. IOMMU группы на X99/C612 чипсете
- Типичные проблемы с IOMMU группами на X99/C612 (Xeon E5 v4)?
- Что делать если GPU попадают в одну IOMMU группу с другими устройствами?
- ACS Override Patch: что это, насколько рискованно, как применить в Proxmox?
- Как изолировать V100 и RTX 5060 Ti в отдельные IOMMU группы?

### 3. VFIO passthrough V100 → Linux VM
- Пошаговая конфигурация VFIO passthrough для V100 в Proxmox:
  - blacklist nvidia в хостовой системе Proxmox
  - vfio-pci bind для V100
  - VM конфигурация (q35, OVMF UEFI, CPU model)
- V100 в TCC mode: есть ли особенности при passthrough vs WDDM?
- Драйверы в Linux VM: nvidia-driver версия для V100 + CUDA 12.x?
- Как проверить что V100 полностью доступна в VM (nvidia-smi, CUDA sample)?

### 4. VFIO passthrough RTX 5060 Ti → Windows VM
- RTX 5060 Ti (Blackwell, очень новая) — известны ли проблемы с VFIO passthrough?
- Code 43 workaround для GeForce в Windows VM (hidden state, vendor_id spoofing)?
- Нужен ли virtio-gpu дополнительно или только passthrough GPU?
- Как передать монитор: Looking Glass vs физический переключатель DisplayPort?

### 5. Docker + vLLM в Linux VM с V100
- Рекомендуемый Linux дистрибутив для ML workloads (Ubuntu 22.04? Debian?)?
- NVIDIA Container Toolkit в VM с passthrough GPU — есть ли отличия от bare metal?
- docker-compose для: vLLM v0.15.1 + FastAPI + Qdrant + embedding сервис?
  - `VLLM_ATTENTION_BACKEND=XFORMERS` (обязательно для V100)
  - `--dtype half` (V100 нет bfloat16)
  - Volumes для моделей HuggingFace кэша
- Сеть: как обращаться из Windows VM к vLLM в Linux VM?
  (замена `host.docker.internal:8080` на IP Linux VM)

### 6. Migration checklist
- В каком порядке делать: установка Proxmox → IOMMU → Linux VM → Windows VM?
- Как сохранить доступ к Windows при проблемах (rollback план)?
- Что из текущего rag_app кода нужно поменять после переезда?
  (только `LLM_BASE_URL` в .env — остальное прозрачно через HTTP)

## Ожидаемый результат
- Пошаговый план установки Proxmox + VFIO на конкретное железо (C612/X99)
- Конкретные команды для VFIO конфигурации V100 и RTX 5060 Ti
- docker-compose.yml для vLLM v0.15.1 на V100 Linux VM
- Список рисков и способов откатиться если что-то пошло не так
```
