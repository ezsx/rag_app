# rag_app — Research Initiator

> **Цель документа**: определить ВСЕ неизвестные, которые нужно закрыть для принятия
> архитектурных решений по rag_app.
>
> Каждый пункт — конкретный вопрос. Ответы получаем через серию Research-документов
> (`01-research-prompts.md`) и сохраняем в `reports/RXX-*.md` как source of truth.

---

## Контекст проекта

### Текущее состояние (2026-03-16)

- **Стек**: FastAPI + llama-server HTTP + ChromaDB + кастомный BM25 (disk-based) + BGE reranker
- **LLM**: Qwen2.5-7B-Instruct через llama-server на хосте (V100 SXM2 32GB, 80 t/s)
- **LLM Planner**: тот же llama-server (GBNF grammar через /v1/completions)
- **Embedding**: multilingual-e5-large (HF, SentenceTransformer, RTX 5060 Ti в Docker)
- **Retrieval**: BM25 + ChromaDB → RRF fusion → MMR → BGE reranker → compose_context
- **Агент**: ReAct цикл, 7 инструментов, coverage threshold 0.8, max 1 refinement, SSE streaming
- **Железо**: V100 SXM2 32GB (хост, llama-server) + RTX 5060 Ti 16GB (Docker, embedding/reranker)
- **Данные**: Telegram-каналы → ingest_telegram.py → ChromaDB + BM25

### Что уже исправлено (с момента аудита)

- ✅ `List[str]` не был импортирован в settings.py → добавлен `from typing import List`
- ✅ `get_llm()` 150 строк defensive GGUF code → заменён на 10 строк `LlamaServerClient` HTTP
- ✅ `get_planner_llm()` CPU Llama → HTTP к тому же llama-server
- ✅ Двойной `_llm_factory` в `get_agent_service()` → убран дубль
- ✅ `get_redis_client(Depends(...))` + `@lru_cache` → убран сломанный Depends
- ✅ `release_llm_vram_temporarily()` VRAM хак → удалён (llama-server внешний)
- ✅ V100 TCC/WSL2 проблема → решена через llama-server на хосте (DEC-0014)

### Что исследовано

- ✅ **R01** (Qdrant hybrid RAG) → `reports/R01-qdrant-hybrid-rag.md`
  - BM42 ❌ для русского (English-only model) → **`Qdrant/bm25` + `language="russian"`** (Snowball)
  - Нативный RRF через prefetch+FusionQuery, нативный MMR (v1.15.0) — кастомный код не нужен
  - ~400 строк кода (BM25IndexManager + HybridRetriever) заменяются одним `client.query_points()`
  - Миграция тривиальна: dense-вектора переносятся напрямую, sparse генерируются заново
  - Windows Docker: **named volumes обязательны** (bind mounts → коррупция данных)

- ✅ **R02** (vLLM serving) → `reports/R02-llm-serving.md`
  - vLLM выигрывает у Ollama по structured output: xgrammar > GBNF
  - ⚠️ **Пинить vLLM v0.15.1** — v0.17.0 сломал xformers, а V100 (SM70) требует xformers
  - ⚠️ **vLLM работает только на Linux** → требует Proxmox + VFIO (R07) сначала
  - Производительность: 30–45 tok/s — медленнее llama-server, но async + prefix cache
  - Клиент: `AsyncOpenAI` закрывает OPEN-02 (async), но только после Proxmox

- ✅ **R03** (выбор модели) → `reports/R03-model-selection.md`
  - **Qwen3-8B FP16** (~16.4 GB) — оптимальная модель для V100 32GB + Russian RAG
  - V100 (SM7.0): AWQ/GPTQ-Marlin/FP8 **не работают** → FP16 единственный вариант в vLLM
  - Qwen3-8B ≈ Qwen2.5-14B по качеству; заменяет оба текущих LLM (7B + 3B planner)
  - CPU planner (3B Ollama) — узкое место: 5–15 tok/s. Qwen3-8B на V100: 40–60 tok/s
  - **Переход: text ReAct regex → нативный Hermes tool calling** (`--tool-call-parser hermes`)
  - `--enforce-eager` обязателен на V100 (нет FlashAttention2, SM<8.0)
  - `--dtype half --max-model-len 8192 --gpu-memory-utilization 0.92`
  - Non-thinking mode (`enable_thinking=False`) — экономит 250–1250 токенов на цепочку
  - English system prompt + "respond in Russian" — 30–40% меньше токенов, лучше instruction following
  - На будущее: **T-lite-it-2.1** (T-Bank, Qwen3-based) — превосходит по русскому, есть tool calling

- ✅ **R06** (async архитектура) → `reports/R06-async-architecture.md`

- ✅ **R05** (RAG evaluation) → `reports/R05-rag-evaluation.md`
  - **Гибрид**: custom LLM-judge (Qwen3-8B) + DeepEval как CI/CD test runner + RAGAS для бенчмаркинга
  - RAGAS: **2 breaking change за год** (v0.1→v0.2→v0.4), промпты на английском, NaN-скоры на vLLM → нестабилен для production
  - DeepEval: pytest-интеграция, `deepeval set-local-model` для vLLM, GEval с кастомными критериями на русском
  - **Custom judge промпты на русском** для 4 метрик: faithfulness (binary QAG-декомпозиция), relevance (1–5), completeness (1–5), citation accuracy (binary)
  - Qwen3-8B достаточен для binary/3-point judgments; PoLL (панель из 3 малых моделей) превосходит GPT-4 (κ: 0.763 vs 0.627)
  - **Eval датасет**: `generate_eval_dataset.py` из Qdrant → 5 типов вопросов с весами (factual 35%, temporal 20%, aggregation 20%, negative 15%, comparative 10%) → critique-фильтр → ~50% отсеивается
  - Без принудительного распределения типов **95% вопросов** — простые factual → завышение метрик
  - CI/CD: smoke 50 примеров на PR (15 мин), nightly 200 примеров (60 мин); минимум **200 примеров** для ±5.5% margin
  - Требует запущенного Qdrant и vLLM → зависит от R01 и R02 реализации

- ✅ **R04** (coverage metrics) → `reports/R04-coverage-metrics.md`
  - `citation_coverage` (документо-счётчик) **бесполезен** — relevance ≠ sufficiency
  - **RRF-score не подходит для coverage** (max ≈ 0.0328, не кросс-запросный). Нужен raw cosine sim
  - **Composite из 5 сигналов**: max_sim (0.25) + mean_top_k (0.20) + term_coverage (0.20) + doc_count_adequacy (0.15) + score_gap (0.15)
  - Порог refinement: **0.65–0.70**, не 0.80. Текущий порог 0.8 слишком агрессивен → лишние поиски
  - Asymmetric error: false-negative (пропущенный поиск) → **66.1% галлюцинаций**; false-positive → 200–500ms лейтенси. Bias toward retrieval.
  - Max refinements: **2–3**, F1 плато после 3 итераций (0.398 → 0.447 на HotpotQA)
  - Реализация: передавать `query` в `compose_context`, вычислять cosine отдельно (request `with_vectors=True`)

### Открытые проблемы

1. **ChromaDB + кастомный BM25** — ~1000 строк кода ради того, что Qdrant даёт из коробки.
   Qdrant уже в инфраструктуре (MCP). **(R01 ✅ — решение ясно, реализация ждёт окна)**
2. **vLLM vs llama-server** — текущий llama-server стабилен, но vLLM даст async, structured outputs.
   **(R02 ✅ — vLLM v0.15.1 выбран, но требует Linux → R07 Proxmox сначала)**
3. **Blocking HTTP** — `requests.Session.post()` всё ещё блокирует event loop. (OPEN-02)
   **Решение: AsyncOpenAI + vLLM — заблокировано Proxmox. Промежуточный фикс: httpx.AsyncClient.**
4. **Evaluation dataset** — 2 фейковых примера, нельзя измерить качество. (OPEN-06, R05)
5. **Coverage metric** — `citation_coverage` = эвристика, не откалибрована. **(R04 ✅ — решение ясно: composite 5-сигналов, порог 0.65–0.70, max 2 refinements; реализация ждёт)**
6. **Embedding model** — ⬜ **Пробел в research**: ни один трек R01–R06 не исследовал embedding-модели.
   DEC-0012 (multilingual-e5-large) принят временно, пересмотр запланирован.
   **(R-embed ⬜ — нужен research: Qwen3-Embedding-0.6B vs 4B vs e5-large на русском корпусе)**
   Предварительное решение: DEC-0026 (Qwen3-Embedding-0.6B). Реализация после Qdrant migration.

---

## A. Хранилище: Qdrant vs ChromaDB

### A1. Возможности Qdrant для RAG
- Как работают named vectors в Qdrant? Можно ли хранить dense + sparse в одной точке?
- Что такое sparse vectors в Qdrant? SPLADE? BM42? Как загрузить?
- Как реализовать hybrid search (dense + sparse + RRF) нативно в Qdrant без кастомного кода?
- Поддерживает ли Qdrant filtering (по каналу, дате, автору) при hybrid search?
- Как устроен Qdrant payload? Какие типы данных поддерживаются для фильтрации?
- Qdrant на Windows (Docker) — есть ли ограничения по производительности vs Linux?

### A2. Миграция ChromaDB → Qdrant
- Как перенести существующие коллекции ChromaDB в Qdrant?
- Нужно ли переиндексировать данные или можно перенести векторы напрямую?
- Как изменится схема коллекции? Что хранить в payload?
- Как изменится `HybridRetriever`: что останется, что уйдёт?
- Что происходит с BM25IndexManager — он полностью заменяется sparse vectors Qdrant?
- Нужен ли отдельный BM25 index вообще, или Qdrant sparse достаточно для retrieval качества?

### A3. Качество retrieval
- Насколько Qdrant sparse vectors (SPLADE/BM42) сравнимы с кастомным BM25 для новостного корпуса?
- Как MMR (Maximum Marginal Relevance) реализовать поверх Qdrant?
- Стоит ли использовать Qdrant's built-in `rescore` или делать RRF на уровне Python?

---

## B. LLM Serving: vLLM / Ollama vs llama.cpp

### B1. Варианты серверов
- vLLM vs Ollama vs TGI (Text Generation Inference) для Qwen2.5-7B — что выбрать для V100?
- Что даёт vLLM на V100 по throughput и latency для ReAct-агента (sequential generation)?
- Поддерживает ли vLLM structured outputs / constrained generation (JSON)? Это критично для tool calling.
- Как работает Ollama на V100 SXM2? Поддерживает ли он Qwen2.5 официально?
- Есть ли смысл держать два LLM через один vLLM (7B для агента, 3B для planner)?
- Как обеспечить очередность запросов в vLLM при параллельных SSE-стримах?

### B2. OpenAI-compatible API
- Какой OpenAI-compatible API endpoint у vLLM? Как настроить sampling params per-request?
- Как перевести `AgentService._generate_step()` с `llama_cpp.Llama()` на HTTP-вызов?
- Как реализовать streaming: `stream=True` в OpenAI API vs async generator в текущем коде?
- Нужен ли `chat_format="qwen2"` при работе через OpenAI API? Как передаётся system prompt?

### B3. VRAM планирование на V100 ✅ ОТВЕЧЕНО (R02)
- Qwen2.5-7B FP16: **~14 GB** весов. KV-кэш 10K токенов: **~0.5 GB** (GQA, 4 KV heads).
- Итого с embedding (~0.5 GB) + reranker (~0.6 GB): **~18 GB** → остаток **14 GB** свободно.
- **Не квантизировать**: V100 не имеет оптимизированных INT4 Tensor Core ядер; FP16 лучше.
- `--max-model-len 8192` достаточно для ReAct агента, экономит ~3 GB vs дефолтного 32K.
- `--gpu-memory-utilization 0.85` оставляет ~5 GB для embedding/reranker в отдельном процессе.
- 3B planner: **Ollama CPU-only** в отдельном контейнере (`NVIDIA_VISIBLE_DEVICES=`).

### B4. Переходный период (llama-server → vLLM) ✅ ВЫПОЛНЕНО (R02)
- ~~Как сделать AgentService независимым от llama.cpp~~ → **выполнено**: `LlamaServerClient` в
  `src/adapters/llm/llama_server_client.py`, AgentService и QueryPlannerService не изменились.
- Нужен ли feature flag `USE_VLLM=true/false`? → **не нужен**: меняем `LLM_BASE_URL` в env.
- GGUF или safetensors? → **HF safetensors** предпочтительнее; vLLM поддерживает GGUF, но нативно.
- ⚠️ **Пинить vLLM v0.15.1**: v0.17.0 убрал xformers → V100 (SM70) сломается. Triton-fallback не проверен.
- ⚠️ **vLLM работает только на Linux**: текущий стек (Windows) не поддерживается → Proxmox сначала.
- Производительность: **30–45 tok/s** (V100 FP16, batch=1) vs 80 tok/s llama-server. Компенсируется
  async + prefix caching + structured outputs.
- Python клиент: `AsyncOpenAI(base_url=..., api_key="EMPTY")` — одинаково работает с vLLM и Ollama.

---

## C. Качество ReAct агента

### C1. Tool calling с 7B моделями
- Qwen2.5-7B-Instruct — насколько надёжно он генерирует `Action: tool {"param": "val"}` формат?
- Есть ли официальная поддержка function calling в Qwen2.5? Как она соотносится с текущим
  ReAct текстовым форматом?
- Что лучше для 7B: текстовый ReAct формат vs JSON tool use vs Qwen's нативный function calling?
- Как влияет temperature на надёжность parsing action? (текущее: tool_temp=0.2, final_temp=0.3)
- Как обрабатывать malformed JSON в action (сейчас есть regex-парсер, но как улучшить)?

### C2. Системный промпт
- Оптимальная длина system prompt для Qwen2.5-7B при контексте 10000 токенов?
- Билингвальный промпт (EN инструкции + RU policy) — это лучшая практика или анти-паттерн?
- Как проверить, что модель следует language policy (не мешает кириллицу с латиницей)?
- Как сократить system prompt не теряя contract'а инструментов?

### C3. Refinement и coverage
- Что такое `citation_coverage` в `compose_context`? Как он вычисляется сейчас?
- Какой метрикой лучше мерить "достаточность" контекста для ответа?
  - Количество уникальных документов в контексте?
  - Cosine similarity между query и топ-N результатами?
  - Entropy распределения оценок поиска?
- Threshold 0.8 — как его калибровать на реальных данных?

---

## D. Evaluation Framework

### D1. Инструменты оценки RAG
- RAGAS (Retrieval Augmented Generation Assessment) — как применить к нашему агенту?
  - Какие метрики: faithfulness, answer relevancy, context recall, context precision?
  - Нужна ли reference LLM для оценки? Какая (GPT-4? локальная Qwen)?
- DeepEval — альтернатива RAGAS? Что лучше для нашего случая?
- Как измерить hallucination rate для реальных Telegram-новостей?
- Что такое ARES (Automated RAG Evaluation System)? Применимо ли?

### D2. Evaluation dataset
- Как сгенерировать качественный eval dataset из реальных данных ChromaDB?
- Что входит в одну eval строку: query, expected_answer, expected_documents, category, difficulty?
- Как автоматически генерировать вопросы из имеющегося корпуса (синтетический датасет)?
  - LLM-генерация вопросов → фильтрация → ручная разметка
  - Насколько большой датасет нужен для статистической значимости?
- Как покрыть разные категории запросов: temporal, author-specific, topic, aggregation, negative?

### D3. LLM-as-judge
- Как настроить LLM-judge для оценки качества финальных ответов агента?
- Промпт для судьи: что оценивать (relevance, completeness, faithfulness, citation accuracy)?
- Какая модель использовать как судью — та же Qwen2.5-7B или нужна более мощная?
- Как нормализовать оценки судьи (1-5? 0-1? pass/fail?)?

### D4. Интеграция в evaluation скрипт
- Как добавить LLM-judge в `scripts/evaluate_agent.py` (фаза 2)?
- Как сравнить baseline QA endpoint vs ReAct Agent по метрикам?
- Как автоматизировать запуск evaluation (CI/CD hook? scheduled job?)?

---

## E. Async архитектура

### E1. Blocking LLM calls
- `llama_cpp.Llama()` — синхронный вызов блокирует event loop. Как правильно обернуть?
  - `asyncio.run_in_executor(None, llm_call)` — thread pool executor?
  - Отдельный процесс для LLM (multiprocessing)?
  - Стоит ли это делать если переходим на vLLM (async HTTP)?
- Как измерить impact blocking calls на throughput при параллельных запросах?

### E2. Tool execution
- `ToolRunner` сейчас использует `ThreadPoolExecutor` — правильная ли это абстракция?
- Как сделать инструменты async-нативными (async def tools)?
- При параллельных SSE-стримах — как изолировать state между запросами?
  (`AgentState` — один на запрос, но `AgentService._current_step` — атрибут класса!)

### E3. FastAPI + SSE
- `EventSourceResponse` из `sse_starlette` — как правильно обрабатывать client disconnect?
- Как тестировать SSE endpoints?
- Есть ли memory leak при долгих SSE соединениях?

---

## F. Pydantic Settings и DI

### F1. Pydantic BaseSettings
- Как мигрировать текущий `Settings` (raw `os.getenv`) на `pydantic-settings.BaseSettings`?
- Как сохранить горячее переключение моделей (`update_llm_model()`) с Pydantic (frozen models)?
- Как организовать `lru_cache` + Settings с Pydantic (BaseSettings не кэшируется по умолчанию)?

### F2. Dependency Injection
- Как правильно организовать DI в FastAPI: `Annotated[X, Depends(get_x)]`?
- `deps.py` сейчас 579 строк — как разбить без нарушения существующих контрактов?
- Как починить `get_redis_client(settings: Settings = Depends(get_settings))` + `@lru_cache`?
- Что такое `lifespan` в FastAPI и нужно ли использовать его для инициализации сервисов?

---

## G. V100 SXM2: среда и железо

### G1. Архитектура с V100 ✅ ВЫПОЛНЕНО
- ~~Текущий docker-compose.yml что нужно изменить~~ → **выполнено**: LLM_BASE_URL добавлен,
  llama-server на хосте, extra_hosts: host-gateway добавлен (DEC-0014).
- ~~CUDA версия для V100~~ → llama-server использует CUDA 12.4 на хосте, Docker не нужно.
- ~~GGML_CUDA_FORCE_CUBLAS~~ → не актуально, llama.cpp не в контейнере.
- **Остаток**: если переходим на vLLM — нужен Linux (Proxmox/VM). На Windows vLLM не работает.

### G2. TEI (Text Embeddings Inference) на V100
- Текущий repo-semantic-search MCP использует TEI с Qwen3-Embedding-0.6B — это хорошо?
- Стоит ли переключить embedding модель для rag_app с e5-large на что-то лучше?
- MTEB benchmark 2025: какие embedding модели лидируют для русскоязычного retrieval?
- Нужен ли отдельный TEI для rag_app или можно шарить TEI с MCP?

---

## Матрица зависимостей

| Исследование | Статус | Блокирует |
|---|---|---|
| R01 (Qdrant hybrid RAG) | ✅ Выполнено | архитектурное решение по storage |
| R02 (vLLM serving) | ✅ Выполнено | реализация — заблокирована R07 |
| R03 (выбор модели) | ✅ Выполнено | — |
| R04 (coverage metrics) | ✅ Выполнено | реализация composite metric |
| R05 (evaluation) | ✅ Выполнено | реализация требует Qdrant + vLLM |
| R06 (async) | ⬜ Не начато | продакшн-готовность |
| R07 (Proxmox + VFIO) | ⬜ Не начато | R02 реализация, vLLM deployment |

> **Критический путь**: R07 (Proxmox) → R02 реализация (vLLM v0.15.1) → R06 (async) — путь к production
> **Storage трек**: R01 реализация (Qdrant migration) — независимо от Proxmox, можно начать сейчас
> **Качество**: R03 → R04 → R05 — независимо от инфра, параллельно

---

## Итого: группы исследований

| # | Трек | Статус | Приоритет | Зависимости |
|---|---|---|---|---|
| R01 | Qdrant hybrid RAG: sparse vectors, migration | ✅ Выполнено | — | — |
| R02 | LLM serving: vLLM v0.15.1 для V100 | ✅ Выполнено | — | реализация требует R07 |
| R03 | Выбор LLM модели: Qwen3-8B FP16 для V100 32GB | ✅ Выполнено | — | реализация требует R07 |
| R04 | Coverage и context quality metrics | ✅ Выполнено | — | реализация: composite metric + порог 0.65 |
| R05 | RAG evaluation: custom judge + DeepEval + RAGAS | ✅ Выполнено | — | реализация требует Qdrant (R01) + vLLM (R02→R07) |
| R06 | Async архитектура: httpx, SSE, shared state | ✅ Выполнено | — | — |
| R07 | Proxmox + VFIO: V100 → Linux VM для vLLM v0.15.1 | ⬜ Нужно | P0 | R02 (выполнено) |

Промпты для каждого трека — в `01-research-prompts.md`.
