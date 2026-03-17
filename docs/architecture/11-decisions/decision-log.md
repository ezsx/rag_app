## Decision Log (ADR-lite)

> Правило: каждое решение, влияющее на инварианты/контракты → **DEC-xxxx**.

---

### DEC-0001 — sequence-first документация
- **Status:** Accepted
- **Context:** архитектура должна отражаться через сценарии и инварианты.
- **Decision:** документируем через flows → invariants → data model.

### DEC-0002 — ReAct над vanilla RAG
- **Status:** Accepted
- **Context:** нужна возможность уточнять поиск при недостаточном контексте.
- **Decision:** ReAct цикл с coverage threshold вместо single-shot RAG.
  Дополнительная latency оправдана улучшенным качеством ответов.

### DEC-0003 — coverage_threshold = 0.8, max_refinements = 1
- **Status:** Accepted
- **Context:** нужен детерминированный выход из цикла refinement.
- **Decision:** фиксированные значения через settings. Более одного refinement
  увеличивает latency непропорционально пользе.

### DEC-0004 — llama.cpp для LLM inference
- **Status:** Accepted (техдолг, пересмотр при V100)
- **Context:** RTX 5060Ti 16GB, Windows, нет vLLM-совместимой среды.
- **Decision:** GGUF через llama_cpp. При переходе на V100 → vLLM (OPEN-02).

### DEC-0005 — ChromaDB + кастомный BM25 для hybrid retrieval
- **Status:** Accepted (техдолг, пересмотр при V100)
- **Context:** ChromaDB легко запускается в Docker; BM25 нужен для lexical search.
- **Decision:** кастомный BM25IndexManager + ChromaDB → RRF fusion в HybridRetriever.
  При переходе на V100 → возможна миграция на Qdrant (OPEN-03).

### DEC-0006 — lru_cache singleton для сервисов
- **Status:** Accepted
- **Context:** LLM и embedding модели дорогие в инициализации.
- **Decision:** все сервисы через `@lru_cache` в `deps.py`. Горячее переключение
  через `settings.update_*()` + явный `cache_clear()`.

### DEC-0007 — SSE event contract (INV-01)
- **Status:** Accepted
- **Context:** evaluate_agent.py и потенциальные клиенты строятся на именах событий.
- **Decision:** event types: thought / tool_invoked / observation / citations / final.
  Не менять без явного API версионирования.

### DEC-0008 — ToolRunner с ThreadPoolExecutor для timeout
- **Status:** Accepted
- **Context:** tools синхронные, нужен надёжный timeout механизм.
- **Decision:** `concurrent.futures.ThreadPoolExecutor(max_workers=1)` с `future.result(timeout)`.
  Это блокирующий wrapper — не идеален для async, но надёжен.

### DEC-0009 — GBNF grammar для QueryPlanner
- **Status:** Accepted
- **Context:** QueryPlannerService нужен структурированный JSON output от 3B модели.
- **Decision:** llama.cpp GBNF grammar для constrained generation SearchPlan JSON.

### DEC-0010 — JWT HS256 с ADMIN_KEY для auth
- **Status:** Accepted
- **Context:** single-user система, не нужен OAuth/OIDC.
- **Decision:** dev endpoint `/v1/auth/admin` + ADMIN_KEY env var → JWT токен.
  Достаточно для демо/одного пользователя.

### DEC-0011 — BGE reranker как CPU-only post-processing
- **Status:** Accepted
- **Context:** BGE reranker не требует GPU, но улучшает качество ranking.
- **Decision:** RerankerService на CPU с batch_size=16. Latency ~200-500ms приемлема.

### DEC-0012 — multilingual-e5-large для embedding
- **Status:** Accepted → **Superseded DEC-0026** (2026-03-17, целевая модель изменена)
- **Context:** русскоязычные Telegram-новости + мультиязычный корпус.
- **Decision:** intfloat/multilingual-e5-large (1024 dims). Остаётся текущей моделью до
  реализации DEC-0026 (Qwen3-Embedding). Пересмотр: MTEB 2025 benchmark → Qwen3-Embedding-0.6B.

### DEC-0013 — язык документации: RU-first
- **Status:** Accepted
- **Decision:** основной текст — русский; якоря `FLOW-xx`, `DEC-xxxx`, `OPEN-xx` — латиница;
  технические идентификаторы — как в коде.

### DEC-0015 — Qdrant вместо ChromaDB + кастомного BM25 (R01)
- **Status:** Accepted (2026-03-16)
- **Context:** ChromaDB + BM25IndexManager = ~400 строк кода ради того, что Qdrant даёт нативно.
  BM42 (Qdrant sparse) English-only → не подходит для русского. `Qdrant/bm25` с `language="russian"` (Snowball) — правильный выбор.
- **Decision:** Qdrant с named vectors (dense + sparse). Нативный RRF через prefetch+FusionQuery.
  Нативный MMR с v1.15.0. Одна коллекция, один вызов `query_points()`.
  Windows Docker: **named volumes обязательны** — bind mounts приводят к silent data corruption.
- **Закрывает:** OPEN-03

### DEC-0016 — Qwen3-8B GGUF как основная LLM (R03)
- **Status:** Accepted (2026-03-16)
- **Context:** Qwen2.5-7B устарел. V100 32GB позволяет запустить Qwen3-8B F16 (~16.4 GB).
  Qwen3-8B ≈ Qwen2.5-14B по качеству. Доступен через llama-server уже сейчас без Proxmox.
- **Decision:** Qwen3-8B GGUF (Q8_0 или F16) через существующий llama-server на V100.
  Заменяет оба LLM: Qwen2.5-7B (agent) и Qwen2.5-3B CPU (planner) — один endpoint.
  V100 SM7.0 ограничения (AWQ/GPTQ/FP8 не работают) не затрагивают llama.cpp GGUF.

### DEC-0017 — vLLM v0.15.1 как целевой LLM-сервер после Proxmox (R02)
- **Status:** Accepted (2026-03-16), реализация отложена до Proxmox
- **Context:** vLLM даёт xgrammar (100% valid JSON), prefix caching, нативный Hermes tool calling.
  Требует Linux. На Windows не работает.
  **Критично**: пинить v0.15.1 — v0.17.0 убрал xformers, V100 (SM7.0) требует xformers.
- **Decision:** vLLM v0.15.1 после Proxmox + VFIO (R07). Клиент: AsyncOpenAI.
  Переход: text ReAct regex → Hermes tool calling (требует AgentService rewrite).
  Риск: совместимость v0.15.1 с Qwen3 не проверена (OPEN-08).

### DEC-0018 — Composite coverage metric вместо document count (R04)
- **Status:** Accepted (2026-03-16)
- **Context:** `citation_coverage` (document count ratio) не измеряет достаточность контекста.
  RRF-скоры (max ≈ 0.0328) не пригодны для coverage — не cross-query сравнимы.
  Raw cosine similarity интерпретируема (0–1) и стабильна.
- **Decision:** Composite из 5 сигналов: `max_sim×0.25 + mean_top_k×0.20 + term_coverage×0.20 + doc_count_adequacy×0.15 + score_gap×0.15 + above_threshold_ratio×0.05`.
  Требует: `with_vectors=True` в Qdrant запросе, `query` как параметр `compose_context`.
  Закрывает: OPEN-07

### DEC-0019 — coverage_threshold = 0.65, max_refinements = 2 (R04)
- **Status:** Accepted (2026-03-16)
- **Context:** Старый threshold 0.80 слишком агрессивен с composite metric (natural score compression).
  Asymmetric error: false-negative (пропущенный поиск) → 66.1% галлюцинаций (Google ICLR 2025);
  false-positive (лишний поиск) → 200–500ms latency.
  F1 растёт от 1 до 3 итераций, plateau. 2 refinements = баланс.
- **Decision:** `coverage_threshold = 0.65`, `max_refinements = 2`. Требует калибровки
  на 30–50 размеченных примерах после получения реального eval датасета.
  Обновляет: INV-02, DEC-0003

### DEC-0020 — Eval framework: custom judge + DeepEval (R05)
- **Status:** Accepted (2026-03-16)
- **Context:** RAGAS нестабилен (2 breaking changes/год, NaN на vLLM, EN-only промпты).
  DeepEval стабилен, pytest-интеграция, GEval для кастомных критериев.
  Custom judge — единственный способ получить русскоязычные промпты.
  Qwen3-8B достаточна для binary/3-point judgments.
- **Decision:** Custom LLM-judge промпты на русском (faithfulness, relevance, completeness, citation accuracy)
  обёрнутые в DeepEval BaseMetric. RAGAS только для разовых reference-аудитов.
  Eval работает уже на llama-server (OpenAI-compatible), vLLM не нужен.

### DEC-0021 — httpx.AsyncClient → AsyncOpenAI (двухэтапный async фикс, R06)
- **Status:** Accepted (2026-03-16)
- **Context:** `requests.Session.post()` блокирует uvicorn event loop = блокирующий OPEN-02.
  Полноценный фикс (AsyncOpenAI) требует vLLM → Proxmox.
- **Decision:** Этап 1 (сейчас): `httpx.AsyncClient` в `LlamaServerClient` — минимальный фикс,
  закрывает блокировку event loop, совместим с llama-server.
  Этап 2 (после vLLM): `AsyncOpenAI(base_url=LLM_BASE_URL, api_key="EMPTY")`.
  Архитектура `LlamaServerClient` изолирует AgentService от деталей клиента.

### DEC-0022 — Thinking mode Qwen3 всегда отключён (R03)
- **Status:** Accepted (2026-03-16)
- **Context:** Qwen3 по умолчанию эмитирует `<think>...</think>` блоки.
  Эти блоки ломают текущий ReAct text regex parser и тратят 250–1250 токенов на шаг.
- **Decision:** Thinking mode ОТКЛЮЧЁН везде (INV-09).
  llama-server: `/no_think` в конце system prompt.
  vLLM: `extra_body={"enable_thinking": False}` в каждом запросе.
  LlamaServerClient содержит safeguard: фильтрация `<think>...</think>` из ответа.

### DEC-0023 — English system prompt с Russian output instruction (R03)
- **Status:** Accepted (2026-03-16)
- **Context:** English system prompt: 30–40% меньше токенов, лучше instruction following
  для структурных задач (JSON tool calling, ReAct формат).
- **Decision:** System prompt на английском. Последняя строка: `"Always respond to the user in Russian."` (INV-10).
  Не менять на русский без A/B теста на нашем домене.

### DEC-0026 — Qwen3-Embedding как целевая embedding-модель (R-embed)
- **Status:** Accepted (2026-03-17), реализация отложена до Qdrant migration
- **Context:** DEC-0012 (multilingual-e5-large) принят с пометкой "пересмотр по MTEB 2025 benchmark".
  Qwen3-Embedding — новое семейство моделей Alibaba (май 2025), специально обученных для retrieval.
  MTEB Multilingual 2025: Qwen3-Embedding-0.6B и 4B занимают лидирующие позиции, включая MIRACL (русский).
  Ни один из треков R01–R06 не исследовал embedding-модели целенаправленно — пробел в research.
- **Decision:** Целевая модель — `Qwen3/Qwen3-Embedding-0.6B`:
  - 600M параметров, 1024-dim (совместимо с текущей схемой Qdrant)
  - Лучше multilingual-e5-large по MIRACL (русский) и BEIR (English)
  - Тот же VRAM footprint (~2–2.5 GB) на RTX 5060 Ti через TEI
  - Требует instruction prefix: `query: <текст>` для запросов, `passage: <текст>` для документов
  - Только CPU (TEI image `120-1.9`) — совместимо с DEC-0024
  - При смене: пересоздать Qdrant-коллекцию + полный re-ingest (новые эмбеддинги несовместимы)
- **Трек:** Нужен R-embed research перед реализацией:
  - Сравнение Qwen3-Embedding-0.6B vs 4B vs multilingual-e5-large на русском Telegram-корпусе
  - Проверить поддержку в TEI (текущий образ `120-1.9`)
  - Измерить latency embed-запросов при ingest и query
- **Обновляет:** DEC-0012 (multilingual-e5-large → Qwen3-Embedding)

### DEC-0025 — TEI образ `120-1.9` для RTX 5060 Ti (SM 12.0 Blackwell)
- **Status:** Accepted (2026-03-17)
- **Context:** RTX 5060 Ti = Blackwell SM 12.0 (очень новая архитектура, ~2025/2026).
  TEI образ `cuda-1.9` deadlock на инициализации FlashBert: нет pre-compiled CUDA kernels для SM 12.0.
  `cuda-1.9` содержит FlashAttention 2, скомпилированный для SM 8.0–8.9 (Ampere/Ada).
  При попытке запустить на SM 12.0 зависает на `Starting FlashBert model` без timeout.
- **Decision:** Использовать `ghcr.io/huggingface/text-embeddings-inference:120-1.9`.
  Образ `120-1.9` = TEI 1.9 скомпилированный специально для SM 12.0 (Blackwell).
  Проверено: модель загружается за ~24 сек, `/embed` endpoint работает, dim=1024 корректен.
- **Дополнительно:**
  - Модели держать в Linux FS (`/home/tei-models/`) — `/mnt/c/` через 9P в 10-100x медленнее
  - `CUDA_VISIBLE_DEVICES=0` обязателен — изолирует от V100 (CUDA device 1, broken в WSL2)
  - CDI mode + `nvidia-ctk cdi generate` — правильный путь для WSL2 GPU доступа
  - CDI spec auto-detects WSL mode, использует `/dev/dxg` вместо NVML

### DEC-0024 — Embedding/Reranker как WSL2-native сервисы (RTX 5060 Ti GPU blocker)
- **Status:** Accepted (2026-03-16)
- **Context:** RTX 5060 Ti недоступна в Docker Desktop для Windows. Корневая причина:
  V100 SXM2 в TCC-режиме блокирует NVML-enumeration для **всех** GPU при инициализации
  Docker/nvidia-container-cli (не только себя). Это архитектурное ограничение WSL2 + TCC,
  не решается настройками Docker, CDI specs или device targeting.
  RTX 5060 Ti при этом **полностью доступна** в Ubuntu WSL2 нативно (GPU-PV работает).
- **Decision:** Embedding (multilingual-e5-large) и Reranker (bge-reranker-v2-m3) запускаются
  как WSL2-native процессы через TEI (text-embeddings-inference), а не внутри Docker.
  Docker-контейнеры (api, ingest) обращаются к ним через `host.docker.internal:8082/8083`.
  `gpus: all` убрано из docker-compose.yml — Docker-сервисы работают на CPU.
- **Trade-offs:** Нужно запускать два WSL2 сервиса до `docker compose up`. Без автостарта.
  Долгосрочно: Proxmox VFIO изолирует V100 → Docker Desktop снова получает 5060 Ti.
- **Порты:** TEI embedding `:8082`, TEI reranker `:8083`
- **Env vars:** `EMBEDDING_TEI_URL=http://host.docker.internal:8082`,
  `RERANKER_TEI_URL=http://host.docker.internal:8083`
- **Связан с:** DEC-0014 (тот же паттерн что V100 → WSL2-native для GPU-сервисов)

---

### DEC-0014 — LLM inference через llama-server на хосте (V100 TCC workaround)
- **Status:** Accepted (2026-03-16)
- **Context:** V100 SXM2 работает в TCC-режиме. WSL2 GPU-PV не поддерживает TCC-устройства.
  V100 физически недоступна из Docker/WSL2. Не решается настройками — архитектурное ограничение WSL2.
- **Decision:** LLM inference запускается как отдельный процесс на Windows хосте (`llama-server.exe`),
  Docker-контейнер обращается по HTTP: `http://host.docker.internal:8080/v1/completions`.
  В коде: `src/adapters/llm/llama_server_client.py` — HTTP-обёртка с интерфейсом совместимым
  с `llama_cpp.Llama`. `AgentService` и `QueryPlannerService` изменений не требовали.
- **Trade-offs:** llama-server нужно запускать вручную перед `docker compose up`.
  Смена модели = рестарт llama-server (~10-20 сек). Долгосрочная альтернатива: Proxmox + VFIO.
- **Закрывает:** OPEN-02 (частично — blocking HTTP заменил blocking llama_cpp, но async остаётся вопросом)
