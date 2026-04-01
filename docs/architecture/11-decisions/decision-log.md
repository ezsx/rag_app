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
- **Status:** Superseded by DEC-0019
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

### DEC-0011 — BGE reranker как post-processing
- **Status:** Superseded by DEC-0027
- **Context:** BGE reranker улучшает качество ranking.
- **Decision:** Изначально CPU-only. Теперь на GPU (RTX 5060 Ti) через gpu_server.py (DEC-0027).

### DEC-0012 — multilingual-e5-large для embedding
- **Status:** Superseded by DEC-0026 (реализовано 2026-03-18)
- **Context:** русскоязычные Telegram-новости + мультиязычный корпус.
- **Decision:** intfloat/multilingual-e5-large (1024 dims). Заменена на Qwen3-Embedding-0.6B.

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

### DEC-0026 — Qwen3-Embedding-0.6B как embedding-модель
- **Status:** Superseded by DEC-0042 (2026-03-31)
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
- **Decision:** Все GPU-модели запускаются как один WSL2-native процесс (`gpu_server.py`)
  на порту `:8082` (embedding + reranker + ColBERT). Не через TEI, а через custom PyTorch server.
  Docker-контейнеры обращаются через `host.docker.internal:8082`.
- **Trade-offs:** Нужно запускать gpu_server.py до `docker compose up`. Без автостарта.
- **Порт:** `:8082` (единый для всех GPU моделей)
- **Env vars:** `EMBEDDING_TEI_URL=http://host.docker.internal:8082`,
  `RERANKER_TEI_URL=http://host.docker.internal:8082`
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

---

### DEC-0027 — gpu_server.py заменяет TEI контейнеры (2026-03-18)
- **Status:** Implemented
- **Context:** TEI Docker контейнеры невозможно запустить с GPU (DEC-0024, V100 TCC блокирует NVML).
  CDI mode + TEI image 120-1.9 работает, но hang на compute (Flash Attention sm_120 bug).
  infinity-emb dependency hell в оффлайн WSL2 (VPN блокирует pip).
- **Decision:** Кастомный `scripts/gpu_server.py` — stdlib http.server + PyTorch cu128:
  - Один процесс, один порт `:8082`
  - 3 модели: Qwen3-Embedding-0.6B + bge-reranker-v2-m3 + jina-colbert-v2
  - ~4-5 GB VRAM, ~11 GB свободно на RTX 5060 Ti
  - Endpoints: `/embed`, `/v1/embeddings`, `/rerank`, `/colbert-encode`, `/health`
  - Venv: `/home/ezsx/infinity-env/` (torch 2.10.0+cu128, transformers 4.57.6)
- **Обновляет:** DEC-0024 (TEI → gpu_server.py), DEC-0011 (reranker теперь на GPU)

### DEC-0028 — Qwen3-30B-A3B вместо Qwen3-8B (2026-03-18)
- **Status:** Implemented
- **Context:** V100 32GB позволяет запустить значительно более мощную модель.
  Qwen3-30B-A3B — MoE (30B total, 3B active params), Q4_K_M = ~18 GB VRAM.
  По качеству ≈ Qwen2.5-72B при inference cost как 3B модель.
- **Decision:** Qwen3-30B-A3B-Q4_K_M через llama-server.
  `--jinja --reasoning-budget 0 --cache-type-k q8_0 --cache-type-v q8_0 -c 16384 --parallel 2`
  Native function calling вместо text ReAct parsing.
- **Обновляет:** DEC-0016 (Qwen3-8B → Qwen3-30B-A3B), DEC-0022 (thinking через --reasoning-budget)

### DEC-0029 — ColBERT reranking (jina-colbert-v2, 2026-03-20)
- **Status:** Implemented
- **Context:** Embedding anisotropy (cosine 0.78-0.83 для всех AI текстов) вызывает
  attractor documents — нерелевантные посты стабильно в top-10. Single-vector cosine
  не различает "Meta купила Manus" и "курс по трансформерам".
- **Decision:** 3-stage pipeline: BM25+Dense → RRF → ColBERT MaxSim rerank.
  jina-colbert-v2 (560M, 128-dim per token, 89 языков).
  Коллекция `news_colbert` с 3 named vectors.
  ColBERT vectors хранятся в Qdrant (multi-vector), encoding через gpu_server.py `/colbert-encode`.
  Manual linear projection 1024→128 (AutoModel не загружает linear.weight).
  Fallback на RRF-only если ColBERT недоступен.
- **Результат:** Recall@1 +97% (0.36→0.71), Recall@5 +33% (0.55→0.73) на 100 queries.
- **Latency:** +2.5с/запрос (5.0с vs 2.5с без ColBERT)

### DEC-0030 — Multi-query search с round-robin merge (2026-03-20)
- **Status:** Implemented (critical bug fix)
- **Context:** search tool использовал только ПЕРВЫЙ subquery из query_plan, игнорируя остальные.
  Bug в `search.py`: `hybrid_retriever.search_with_plan(deduped_queries[0], search_plan)`.
- **Decision:** Итерация по ВСЕМ subqueries, round-robin merge (не sort by dense_score).
  Dense sort re-promotes attractor documents, отменяя ColBERT ranking.
  Original query пользователя всегда добавляется в subqueries.
- **Результат:** v2 recall 0.46→0.61 (+33%)

### DEC-0031 — bge-reranker-v2-m3 вместо bge-m3 (2026-03-19)
- **Status:** Superseded by DEC-0043 (2026-03-31)
- **Context:** bge-m3 — bi-encoder, загруженный как seq-cls (logit gap 8).
  bge-reranker-v2-m3 — dedicated cross-encoder (logit gap 18, confidence 0.37→0.9995).
- **Decision:** Модель `/home/tei-models/reranker-v2`, gpu_server.py переключён.
  На малом датасете (10 Qs) recall не изменился, но score separation удвоился.

### DEC-0032 — Weighted RRF (BM25 3:1, 2026-03-19)
- **Status:** Implemented
- **Context:** При equal weight dense "магниты" перевешивают BM25 keyword matches.
  BM25 правильно находит документы с exact keywords, но dense re-promotes generic docs.
- **Decision:** `models.RrfQuery(rrf=models.Rrf(weights=[1.0, 3.0]))` — BM25 weight=3, dense=1.
  Asymmetric prefetch: BM25 limit=100, dense limit=20.
- **Результат:** Recall 0.33→0.59

### DEC-0033 — Channel dedup max 2/channel (2026-03-20)
- **Status:** Implemented
- **Context:** Prolific каналы (gonzo_ml, ai_machinelearning_big_data) монополизируют top-10.
- **Decision:** Post-retrieval `_channel_dedup(candidates, max_per_channel=2)`.
  Запрашиваем k×2 из Qdrant, dedup сужает до k.
  Qdrant group_by не работает с multi-stage prefetch — python post-processing.
- **Результат:** Diversity ↑, recall без изменений (bottleneck = candidate generation)

### DEC-0034 — Pre-computed analytics: hot_topics + BERTopic cron (2026-03-28)
- **Status:** Implemented (SPEC-RAG-16)
- **Context:** "Что обсуждали на этой неделе?" невозможно ответить real-time — нужна кластеризация всего корпуса.
- **Decision:** Weekly cron: BERTopic fit на 13K+ docs (pre-computed embeddings из Qdrant) → hot_score → LLM summary → auxiliary collection `weekly_digests`. Tool `hot_topics` scrolls pre-computed digests (<10ms). CountVectorizer без pymorphy2 (AI/ML текст ~50% English, BERTopic labels дополняются LLM labeling).
- **Результат:** 113 topics, 34s BERTopic fit, 90s total cron. Fallback to latest available digest.

### DEC-0035 — Pre-computed analytics: channel_expertise + monthly cron (2026-03-28)
- **Status:** Implemented (SPEC-RAG-16)
- **Context:** "Кто лучше пишет про NLP?" требует агрегации по всем 36 каналам.
- **Decision:** Monthly cron: per-channel aggregation → authority/speed/breadth/volume scores → LLM profile summary → auxiliary collection `channel_profiles` (36 points). Authority = 0.4×entity_coverage + 0.3×consistency + 0.2×uniqueness + 0.1×volume.
- **Результат:** 36 profiles за 169s. Top: ai_machinelearning_big_data (0.595), data_secrets (0.573).

### DEC-0036 — Request isolation via ContextVar (2026-03-28)
- **Status:** Implemented (SPEC-RAG-17, FIX-01)
- **Context:** AgentService — singleton через @lru_cache, но писал request state в self._*. При параллельных запросах — cross-request data bleed (citations одного запроса попадают в другой).
- **Decision:** `RequestContext` dataclass + `contextvars.ContextVar`. AgentService stateless, request state изолирован. Collection override через settings.update_collection() убран из request path.
- **Альтернативы:** Thread-local storage (не работает с async), new service instance per request (ломает DI cache).

### DEC-0037 — Production hardening: 9 fixes (2026-03-28)
- **Status:** Implemented (SPEC-RAG-17)
- **Context:** Independent review Claude + Codex GPT-5.4 нашёл 9 конкретных багов/gaps.
- **Decision:** FIX-01 request isolation, FIX-02 coverage metric bug, FIX-03 rate limiter overwrite, FIX-04 tool name whitelist by visible set, FIX-05 JWT hard fail, FIX-06 auth on qa/search, FIX-07 CORS allowlist, FIX-08 cooperative deadline (90s), FIX-09 demo auth endpoint (feature-flagged).
- **Результат:** 3 rounds Codex review, all findings resolved.

### DEC-0038 — Docker compose: relay :18082 для embedding/reranker (2026-03-29)
- **Status:** Superseded by DEC-0041
- **Context:** Docker containers не видят WSL2-native gpu_server на :8082 (mirrored networking limitation). Search failures и 400 errors в agent loop.
- **Decision:** Windows relay на :18082 проксирует к WSL-native :8082. compose.dev.yml переключён на relay URL.
- **Альтернативы:** Откат mirrored networking (ломает VPN), netsh portproxy (требует admin).

### DEC-0039 — Qwen3.5-35B-A3B вместо Qwen3-30B-A3B (2026-03-30)
- **Status:** Implemented
- **Context:** Qwen3-30B давала стохастические отказы (q01 false refusal, q21 hallucination, q33 wrong tool). Qwen3.5 (Feb 2026, Gated Delta Networks + 256 MoE experts) — drop-in замена с лучшими бенчмарками.
- **Decision:** Swap на Qwen3.5-35B-A3B-Q4_K_M. Фиксит 3/4 eval failures без изменений кода.

### DEC-0040 — Langfuse v3 для observability (SPEC-RAG-19, 2026-03-30)
- **Status:** Implemented (MVP)
- **Context:** Zero structured metrics per component. R25 top finding: невозможно определить bottleneck, token usage, tool latency.
- **Decision:** Self-hosted Langfuse v3 (6 Docker containers). 7 instrumentation points: agent root, LLM calls, tools, retrieval, rerank, planner. Lazy imports + SafeSpan для graceful degradation.
- **Альтернативы:** Phoenix (lighter, 1 container) — отклонён: нет eval integration, слабее portfolio signal. structlog — отклонён: нет UI, нет LLM-aware tracing.
- **TODO:** Parent-child span propagation через ThreadPoolExecutor, eval tagging (session_id/tags).

### DEC-0041 — WSL mirrored networking отключён по умолчанию (2026-03-30)
- **Status:** Implemented
- **Context:** `networkingMode=mirrored` в .wslconfig ломает Docker Desktop port forwarding — ни один Docker порт не виден на Windows localhost. Mirrored нужен только для VPN (AmneziaWG) в WSL.
- **Decision:** Mirrored закомментирован по умолчанию. Включать только для VPN: `scripts/wsl-vpn-on.cmd`, выключать обратно: `scripts/wsl-vpn-off.cmd`. Docker порты работают нормально без mirrored.
- **Supersedes:** DEC-0038 (relay больше не нужен для Docker↔Docker, но wsl_tei_relay.py остаётся для gpu_server WSL→Docker).

### DEC-0042 — pplx-embed-v1-0.6B вместо Qwen3-Embedding-0.6B (2026-03-31)
- **Status:** Implemented
- **Context:** Deep research (prompt 28) по embedding моделям 2025-2026. pplx-embed-v1-0.6B (Perplexity, март 2026) — bidirectional attention, MTEB +7 pts vs Qwen3-Embedding. Тот же размер (0.6B), та же dim (1024).
- **Decision:** Swap на pplx-embed-v1-0.6B. bf16 (не fp16 — overflow на длинных текстах). Mean pooling (не last_token_pool). Без instruction prefix (пустая строка). trust_remote_code=True. Полный reingest 13777 docs.
- **Supersedes:** DEC-0026 (Qwen3-Embedding-0.6B)

### DEC-0043 — Qwen3-Reranker-0.6B-seq-cls вместо bge-reranker-v2-m3 (2026-03-31)
- **Status:** Implemented
- **Context:** Tom Aarsen seq-cls conversion Qwen3-Reranker. Chat template с `<|im_start|>` маркерами, logit scoring через seq-cls head. Лучше score separation чем bge-reranker (+5-8 pts rerank score).
- **Decision:** Swap на Qwen3-Reranker-0.6B-seq-cls. padding_side="left". Chat template: system/user/assistant prefix/suffix. AutoModelForSequenceClassification. fp16.
- **Supersedes:** DEC-0031 (bge-reranker-v2-m3)

### DEC-0044 — LANCER nugget coverage вместо cosine-based (2026-03-31)
- **Status:** Implemented
- **Context:** Cosine-based coverage (DEC-0018) не калибровалась под pplx-embed — 45% запросов получали лишний refinement. Калибровка показала median cosine 0.47, coverage median 0.69. LANCER (Ju et al., 2026) предлагает nugget-based подход: subqueries = information nuggets, coverage = доля покрытых аспектов.
- **Decision:** Заменить _compute_coverage() на nugget-based в services/agent/coverage.py. query_plan subqueries = nuggets. Implicit nuggets из search subqueries если plan не вызывался. Threshold 0.75 (3/4 nuggets), max_refinements 1 (targeted по uncovered nuggets).
- **Supersedes:** DEC-0018 (composite 5-signal coverage), DEC-0019 (threshold 0.65, max_refinements 2)

### DEC-0045 — Cross-encoder как CRAG confidence filter (2026-03-31)
- **Status:** Implemented
- **Context:** Калибровка показала ColBERT r@3=0.97, cross-encoder поверх: r@3=0.94 (-0.03). Déjean et al. SIGIR 2024 подтверждает: CE поверх сильного first-stage часто вредит. "Drowning in Documents" 2024: pointwise CE robust в 23% случаев.
- **Decision:** Cross-encoder (Qwen3-Reranker-0.6B) НЕ ранжирует — ColBERT порядок сохраняется. CE используется как CRAG-style confidence filter: docs с score < threshold отсекаются перед compose_context.
- **filter_threshold = 0.0** (logit boundary). Обоснование из калибровки (100 queries, 2000 docs):
  - CE scores: relevant docs median=8.35, irrelevant median=-1.11
  - При t=0.0: сохраняем 132/143 (92%) relevant, убираем 1026/1857 (55%) irrelevant
  - 11 потерянных relevant docs — edge cases на хвосте ColBERT ranking (rank 15-20), основные expected docs (rank 1-5) имеют CE score 8+
  - Ноль = естественная граница logit-based scoring: положительный logit = "relevant"
  - Более агрессивный threshold (3.62 suggested) теряет 21 relevant (15%) — не оправдано
- **Open question:** стоит ли filter вообще полезен после ColBERT top-20 — ColBERT уже отфильтровал мусор. CE filter marginal. Потенциально убрать если fine-tune CE не планируется.
