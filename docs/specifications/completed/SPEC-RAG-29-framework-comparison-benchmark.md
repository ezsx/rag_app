# SPEC-RAG-29: Custom RAG vs LlamaIndex — benchmark

> **Статус**: Active
> **Создан**: 2026-04-03
> **Ревизия**: v3 (2026-04-03) — final cleanup after second Codex review
> **Research basis**: R-deep-framework-comparison (prompt 41), LlamaIndex API research (verified from source v0.12.52 / qdrant v0.10.0), Codex review
> **Зависимости**: evaluate_retrieval.py, evaluate_agent.py, eval_retrieval_100.json, eval_golden_v2.json

---

## 1. Цель

Объективный benchmark: наш custom RAG pipeline vs LlamaIndex (best-effort) vs naive baseline — на одних данных и вопросах. Две линии сравнения:

1. **Retrieval benchmark** — IR метрики (Recall@K, MRR, nDCG) на 100 вопросах. Изолирует качество поиска от LLM generation.
2. **Agent E2E benchmark** — factual_correctness, usefulness, BERTScore на 17 retrieval вопросах из golden v2. Полный pipeline: query → agent → answer.

Результат — ADR + числа для портфолио. Не framework bashing, а data-driven architectural decision.

---

## 2. Контекст

### Что уже есть

- `scripts/evaluate_retrieval.py` — retrieval eval, прямые Qdrant queries, Recall@1/5/10/20, latency
- `datasets/eval_retrieval_100.json` — 100 вопросов с `expected_documents` (channel:message_id)
- `scripts/evaluate_agent.py` — agent eval, SSE parsing, tool tracking, failure attribution. **NB:** `_run_judge()` возвращает `{}` — judge offline-only, новый judge нужно писать
- `datasets/eval_golden_v2.json` — 36 вопросов (17 retrieval, 8 analytics/future_baseline, 6 analytics_hot_topics/channel_expertise, 2 navigation, 3 refusal)
- `scripts/gpu_server.py` — embedding + reranker + ColBERT HTTP API (порт 8082)
- Qdrant коллекция `news_colbert_v2` с named vectors: `dense_vector`, `sparse_vector`, `colbert_vector`

### Что нужно построить

- LlamaIndex retrieval pipeline в двух конфигурациях: LI-stock и LI-maxed
- LlamaIndex agent pipeline в двух конфигурациях: LI-stock и LI-maxed
- Naive baseline (dense-only + LLM)
- Benchmark runner, объединяющий все 4 pipeline
- Новый LLM judge (evaluate_agent._run_judge = offline stub)
- ADR с результатами

### Verified LlamaIndex API (v0.12.52, qdrant v0.10.0)

| Компонент | Класс | Пакет |
|-----------|-------|-------|
| LLM | `OpenAILike(api_base=..., is_function_calling_model=True)` | `llama-index-llms-openai-like` |
| Embedding | Subclass `BaseEmbedding` (3 метода) | `llama-index-core` |
| Vector Store | `QdrantVectorStore(dense_vector_name=..., sparse_vector_name=..., enable_hybrid=True)` | `llama-index-vector-stores-qdrant` |
| Reranker | Subclass `BaseNodePostprocessor` (1 метод) | `llama-index-core` |
| Agent | `FunctionAgent(llm=..., tools=[...])` | `llama-index-core` |
| Tools | `FunctionTool.from_defaults(fn=...)` | `llama-index-core` |

**Возможности LlamaIndex (verified from source):**
- `hybrid_fusion_fn`: принимает custom callable `(dense, sparse) -> result`. Можно реализовать weighted RRF
- `initial_tool_choice`: принудительный первый tool call. **Не полный аналог** нашего forced search: наш guard срабатывает когда LLM _не вызвала_ tool, `initial_tool_choice` форсит tool на первом ходу всегда
- `tool_retriever`: dynamic tool selection per step по query (не phase-based, но есть)
- `sparse_query_fn`: custom sparse encoder для query-time BM25

**Подтверждённые ограничения (факт кода):**
- ColBERT multivector: нет параметра `colbert_vector_name`, Qdrant multivector query **невозможен**
- LANCER nugget coverage: zero framework support, нет концепта subquery-based coverage
- Phase-based tool visibility: `tool_retriever` = query-based selection, не phase/state-based. Наш approach (pre-search → post-search → analytics-complete) не воспроизводим
- Context trimming on 400: наш retry с обрезкой messages — custom logic в `llama_server_client.py`

**Зависимости LlamaIndex:** ~70 транзитивных пакетов (core 59 + Qdrant 11). У нас ~12.

---

## 3. Четыре pipeline (Naive / LI-stock / LI-maxed / Custom)

### 3.1 Naive baseline

Минимальный RAG: dense search → LLM. Показывает пол.

**Retrieval:**
```
Query → embed (gpu_server.py) → Qdrant dense_vector search top-K → results
```
Без BM25, без reranking, без ColBERT. Прямой HTTP в Qdrant.

**Agent:**
```
Query → dense search top-5 → format context → LLM single-shot answer
```
Без agent loop, без tools. Один вызов LLM.

### 3.2 LlamaIndex — два варианта

Для честности бенчмарка — **два LlamaIndex pipeline**:

#### LI-stock: только фреймворк, zero custom code

**Retrieval:**
```
Query → PplxEmbedding.embed(query) + sparse_query_fn(query)
      → QdrantVectorStore hybrid (dense + BM25, default relative_score_fusion)
      → results (без reranking)
```
Стандартный конфиг, default fusion. Показывает что фреймворк даёт **из коробки**.

**Agent:**
```
Query → FunctionAgent(llm=OpenAILike, tools=[search_tool])
      → agent вызывает search → генерирует ответ
```
Стандартный FunctionAgent, без настроек.

#### LI-maxed: фреймворк + честная кастомизация

**Retrieval:**
```
Query → PplxEmbedding.embed(query) + sparse_query_fn(query)
      → QdrantVectorStore hybrid (dense + BM25, custom hybrid_fusion_fn = weighted RRF 3:1)
      → QwenReranker (cross-encoder rerank через gpu_server.py)
      → results
```
Без ColBERT (невозможно — нет multivector API). Но weighted RRF через `hybrid_fusion_fn` + cross-encoder reranker.

> **Зачем два варианта:** LI-stock показывает out-of-box value фреймворка.
> LI-maxed показывает ceiling — что maximum вложение custom кода в фреймворк даёт
> ту же работу что и custom pipeline минус ColBERT + LANCER.
> Если кто-то скажет "а чё не добавили weighted RRF?" — мы показываем: добавили, вот результат.

**Agent (LI-maxed):**
```
Query → FunctionAgent(llm=OpenAILike, tools=[search_tool], initial_tool_choice="search")
      → search_tool (LI-maxed retrieval внутри)
      → agent генерирует ответ
```
`initial_tool_choice="search"` — форсит search на первом ходу (ближайший аналог нашего forced search, но не идентичный: наш guard детерминированно спасает кейс когда LLM не вызвала tool). Без LANCER, без phase-based visibility.

### 3.3 Custom pipeline (наш, as-is)

**Retrieval:**
```
Query → BM25 top-100 + Dense top-20 → RRF → ColBERT MaxSim rerank → results
```
Через прямые Qdrant HTTP queries (логика из `evaluate_retrieval.py`).

> **NB:** `evaluate_retrieval.py` делает BM25+Dense→RRF→ColBERT, но **без cross-encoder** stage.
> Cross-encoder (Qwen3-Reranker) используется только в agent pipeline (через `reranker_service`).
> Для retrieval benchmark Custom = RRF + ColBERT, без cross-encoder.
> Это означает что LI-maxed (RRF + cross-encoder) и Custom (RRF + ColBERT) тестируют
> **разные reranking стратегии** — что делает сравнение интереснее.

**Agent:**
```
Query → QueryPlanner → HybridRetriever → ColBERT → cross-encoder → channel dedup
      → LANCER coverage check → targeted refinement → compose_context → final_answer
```
Полный pipeline через `evaluate_agent.py` (SSE endpoint).

---

## 4. Benchmark Line 1: Retrieval

### Датасет

`datasets/eval_retrieval_100.json` — 100 вопросов с `expected_documents`.

### Метрики

- **Recall@1, @5, @10, @20** — доля expected documents найденных в top-K
- **MRR** — Mean Reciprocal Rank первого релевантного документа
- **Latency** — время на запрос (p50, p95)

MRR считается по `expected_documents`: для каждого вопроса — reciprocal rank первого совпадения.

### Конфигурация каждой системы (4 pipeline)

| Параметр | Naive | LI-stock | LI-maxed | Custom |
|----------|-------|----------|----------|--------|
| Dense search | top-20 | top-20 | top-20 | top-20 |
| BM25 search | нет | top-100 | top-100 | top-100 |
| Fusion | нет | default (relative_score_fusion) | weighted RRF 3:1 (hybrid_fusion_fn) | weighted RRF 3:1 |
| ColBERT rerank | нет | нет | нет (невозможно) | Qdrant multivector MaxSim |
| Cross-encoder | нет | нет | QwenReranker (gpu_server.py) | нет (только в agent pipeline) |
| Final top-K | 20 | 20 | 20 | 20 |

### Реализация

**Naive** (`benchmarks/naive/retriever.py`):
Dense-only search через Qdrant HTTP API. Embed query через gpu_server.py → search `dense_vector` → return top-K. ~50 LOC.

**LI-stock** (`benchmarks/llamaindex_pipeline/retriever.py`, config="stock"):
1. `QdrantVectorStore(enable_hybrid=True)` + `PplxEmbedding` + default fusion
2. Для каждого вопроса: `retriever.retrieve(query)` → return as-is
3. Без reranker, без custom fusion

**LI-maxed** (`benchmarks/llamaindex_pipeline/retriever.py`, config="maxed"):
1. `QdrantVectorStore(enable_hybrid=True, hybrid_fusion_fn=weighted_rrf)` + `PplxEmbedding`
2. Для каждого вопроса: `retriever.retrieve(query)` → `QwenReranker.postprocess(nodes)`
3. Weighted RRF + cross-encoder reranker

**Custom** (`benchmarks/custom_adapter/retriever.py`):
Самостоятельная реализация BM25+Dense→RRF→ColBERT через прямые Qdrant HTTP queries.
**Не импортирует** `evaluate_retrieval.py` (там deprecated instruction prefix + нет cross-encoder).
Логика: embed query (gpu_server.py, без prefix) → sparse encode (fastembed BM25) →
Qdrant prefetch (dense+sparse→RRF→ColBERT multivector) → parse results. ~80 LOC.

> **Embedding consistency:** `evaluate_retrieval.py` использует instruction prefix
> `"Instruct: Given a user question..."`. Продакшн pipeline (DEC-0042) — без prefix.
> В benchmark все 4 pipeline используют **одинаковый** подход: **без prefix** (DEC-0042).

**Runner:** `scripts/run_benchmark_retrieval.py` — запускает все 4 pipeline, собирает результаты, выводит сводную таблицу.

---

## 5. Benchmark Line 2: Agent E2E

### Датасет

`datasets/eval_golden_v2.json` — **17 retrieval вопросов** (определяются по explicit ID list, не по category filter).

**Exact IDs:**
```
broad_search (6):        golden_q01, golden_q02, golden_q03, golden_q04, golden_q05, golden_q25
constrained_search (7):  golden_q06, golden_q07, golden_q08, golden_q09, golden_q10, golden_q11, golden_q12
compare_summarize (4):   golden_q13, golden_q14, golden_q15, golden_q16
```

**Исключены:**
- `navigation` (q17, q18): тривиальные, не показательны
- `negative_refusal` (q19, q20, q21): не про retrieval quality
- `future_baseline` (q22-q24, q26-q30): `key_tools = entity_tracker/arxiv_tracker` — analytics, не retrieval
- `analytics_*` (q31-q36): domain-specific tools

### Метрики

- **factual_correctness** (0-1): LLM judge (offline — артефакт с ответами кидается в чат Claude/Codex, judge возвращает встречный артефакт с оценками)
- **usefulness** (0-2): LLM judge
- **BERTScore**: `ai-forever/ruBERT-large`
- **Latency**: total_time (TTFT не измеряем — у naive и LlamaIndex нет SSE streaming)
- **tool_call_count**: сколько tool calls сделал agent

### Конфигурация каждой системы (4 pipeline)

| Параметр | Naive | LI-stock | LI-maxed | Custom |
|----------|-------|----------|----------|--------|
| LLM | Qwen3.5 (llama-server) | Qwen3.5 (OpenAILike) | Qwen3.5 (OpenAILike) | Qwen3.5 (llama-server) |
| Agent loop | нет (single call) | FunctionAgent | FunctionAgent | AgentService |
| Tools | нет | search | search (initial_tool_choice) | 15 tools (dynamic visibility) |
| Retrieval | dense top-5 | hybrid, default fusion | hybrid, weighted RRF + reranker | hybrid, weighted RRF + ColBERT + reranker |
| Query planning | нет | нет | нет | QueryPlannerService |
| Coverage check | нет | нет | нет | LANCER nugget coverage |
| Forced search | нет | нет | initial_tool_choice="search" | да (deterministic) |

### Реализация

**Naive agent** (`benchmarks/naive_agent.py`):
```python
async def run(query: str, llm_url: str, embedding_url: str, qdrant_url: str) -> dict:
    """Dense search top-5 → single LLM call → answer."""
    docs = dense_search(query, embedding_url, qdrant_url, top_k=5)
    context = format_docs(docs)
    prompt = f"Контекст:\n{context}\n\nВопрос: {query}\nОтвет:"
    answer = call_llm(prompt, llm_url)
    return {"answer": answer, "docs": docs, "tool_calls": 0, "latency": ...}
```
~80 LOC.

**LlamaIndex agent** (`benchmarks/llamaindex_pipeline/agent.py`):
```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="qwen3.5-35b",                       # explicit model name
    api_base="http://localhost:8080/v1",        # llama-server на Windows хосте
    api_key="not-needed",
    is_function_calling_model=True,
    is_chat_model=True,
    context_window=32768,
    max_tokens=4096,
    temperature=0.0,
    # Qwen3 thinking mode конфликтует с tool_calls.
    # Наш llama_server_client.py отключает через enable_thinking=False.
    # OpenAILike: передаём через additional_kwargs.
    additional_kwargs={"enable_thinking": False},
)

def build_search_tool(retriever, reranker=None):
    """Фабрика search tool. reranker=None для LI-stock."""
    def search_documents(query: str) -> str:
        """Поиск релевантных постов из Telegram-каналов по запросу."""
        nodes = retriever.retrieve(query)
        if reranker is not None:
            nodes = reranker.postprocess_nodes(nodes, QueryBundle(query))
        return format_nodes(nodes[:5])
    return FunctionTool.from_defaults(
        fn=search_documents, name="search",
        description="Поиск релевантных постов из Telegram-каналов по запросу",
    )

def build_agent(llm, search_tool, *, maxed: bool = False):
    """LI-stock: default config. LI-maxed: initial_tool_choice + reranker."""
    return FunctionAgent(
        name="rag_agent", llm=llm, tools=[search_tool],
        system_prompt="Ты помощник по новостям AI/ML. Отвечай на русском, ссылайся на источники.",
        initial_tool_choice="search" if maxed else None,
    )

result = await agent.run(query)
```
~130 LOC (включая embedding, reranker, инициализацию).

> **NB:** `additional_kwargs={"enable_thinking": False}` — критично для Qwen3.5 через llama-server.
> Без этого thinking mode конфликтует с tool_calls в history (ошибка
> "Assistant response prefill is incompatible with enable_thinking").
> См. `src/adapters/llm/llama_server_client.py:127-133`.

**Custom agent:** Через существующий SSE endpoint `http://localhost:8001/v1/agent/stream` — парсинг из `evaluate_agent.py`.

**Runner:** `scripts/run_benchmark_agent.py`:
1. Загружает golden v2, фильтрует по exact ID list (17 retrieval questions)
2. Для каждого вопроса запускает naive → LI-stock → LI-maxed → custom (sequential, один LLM)
3. Собирает ответы, генерирует JSON артефакт для offline judge (через чат Claude/Codex)
4. Выводит JSON + markdown сводку

---

## 6. Структура кода

Следует существующему паттерну: `apps/{name}/` (Dockerfile + requirements) + `deploy/compose/compose.{name}.yml`.

```
apps/benchmark/                      ← Dockerfile + requirements (паттерн apps/api/, apps/ingest/)
├── Dockerfile
└── requirements.txt                 # LlamaIndex pinned + fastembed

benchmarks/                          ← Python package с pipeline кодом
├── __init__.py
├── config.py                        # URL-ы, пути, параметры (из env)
├── protocols.py                     # RetrieverProtocol, AgentProtocol
├── naive/
│   ├── __init__.py
│   ├── retriever.py                 # Dense-only Qdrant search (~50 LOC)
│   └── agent.py                     # Dense search → LLM single-shot (~80 LOC)
├── llamaindex_pipeline/
│   ├── __init__.py
│   ├── embedding.py                 # PplxEmbedding(BaseEmbedding) (~35 LOC)
│   ├── reranker.py                  # QwenReranker(BaseNodePostprocessor) (~45 LOC)
│   ├── fusion.py                    # weighted_rrf_fusion(hybrid_fusion_fn) (~25 LOC)
│   ├── retriever.py                 # QdrantVectorStore + hybrid, stock & maxed configs (~60 LOC)
│   └── agent.py                     # FunctionAgent + search tool, stock & maxed (~80 LOC)
├── custom_adapter/
│   ├── __init__.py
│   ├── retriever.py                 # BM25+Dense→RRF→ColBERT via Qdrant HTTP (~80 LOC)
│   └── agent.py                     # HTTP adapter к /v1/agent/stream (~60 LOC)
├── export_for_judge.py              # Генерация JSON артефакта для offline judge (~60 LOC)
└── results/
    └── .gitkeep

scripts/
├── run_benchmark_retrieval.py       # Runner: 4 retrieval pipelines × 100 Qs (~180 LOC)
└── run_benchmark_agent.py           # Runner: 4 agent pipelines × 17 Qs (~250 LOC)

deploy/compose/
└── compose.benchmark.yml            # Docker service для benchmark runner
```

### apps/benchmark/Dockerfile

```dockerfile
# Контекст сборки: корень репозитория (../../)
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
COPY apps/benchmark/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY benchmarks/ /workspace/benchmarks/
COPY datasets/ /workspace/datasets/
COPY scripts/run_benchmark_*.py /workspace/scripts/
```

### deploy/compose/compose.benchmark.yml

```yaml
name: rag-benchmark
services:
  benchmark:
    build:
      context: ../..
      dockerfile: apps/benchmark/Dockerfile
    environment:
      - QDRANT_URL=http://host.docker.internal:16333
      - EMBEDDING_URL=http://host.docker.internal:8082
      - LLM_URL=http://host.docker.internal:8080
      - AGENT_URL=http://host.docker.internal:8001
      - QDRANT_COLLECTION=news_colbert_v2
    volumes:
      - ../../benchmarks/results:/workspace/benchmarks/results
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### Запуск

```bash
# Retrieval benchmark
docker compose -f deploy/compose/compose.benchmark.yml run --rm benchmark \
  python scripts/run_benchmark_retrieval.py

# Agent benchmark
docker compose -f deploy/compose/compose.benchmark.yml run --rm benchmark \
  python scripts/run_benchmark_agent.py
```

### Принципы размещения

- `apps/benchmark/` — Dockerfile + requirements, паттерн `apps/api/` и `apps/ingest/`
- `deploy/compose/compose.benchmark.yml` — паттерн compose.dev/test/mcp/langfuse
- `benchmarks/` — Python package с pipeline кодом (копируется в контейнер)
- Runners в `scripts/` — консистентно с `evaluate_retrieval.py` и `evaluate_agent.py`
- Результаты в `benchmarks/results/` — volume mount, JSON файлы видны на хосте
- LlamaIndex зависимости изолированы в контейнере, не попадают в основной стек

### Не трогаем

- `src/` — zero changes в приложении
- `evaluate_retrieval.py` — reference для search logic, не модифицируем и не импортируем
- `evaluate_agent.py` — reference для SSE parsing, не модифицируем. **Judge code пишем заново** в `benchmarks/` (существующий `_run_judge()` = stub)
- `datasets/` — используем as-is

---

## 7. Protocols

```python
# benchmarks/protocols.py
from typing import Protocol

class RetrievalResult:
    """Единый формат результата retrieval."""
    doc_id: str          # "channel:message_id"
    score: float
    text: str | None     # опционально, для agent context
    channel: str
    message_id: int

class RetrieverProtocol(Protocol):
    """Интерфейс для всех retrieval pipelines."""
    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]: ...

class AgentResult:
    """Единый формат результата agent."""
    answer: str
    docs: list[RetrievalResult]
    tool_calls: list[str]    # имена вызванных tools
    latency: float           # total time seconds
    ttft: float | None       # time to first token (если доступно)

class AgentProtocol(Protocol):
    """Интерфейс для всех agent pipelines."""
    async def run(self, query: str) -> AgentResult: ...
```

---

## 8. LlamaIndex зависимости

```
# apps/benchmark/requirements.txt — pinned exact versions (verified from source)
llama-index-core==0.12.52
llama-index-vector-stores-qdrant==0.10.0
llama-index-llms-openai-like==0.4.0
fastembed>=0.4.0          # для sparse_query_fn (BM25 query encoding)
requests>=2.31.0          # HTTP calls к gpu_server, Qdrant, llama-server
```

**Установка:** изолирована в Docker контейнере (`apps/benchmark/Dockerfile`).

> **Почему pinned:** спека верифицирована на этих версиях. Broad ranges (`>=0.13.0,<0.15`)
> инвалидируют claim "verified from source". При обновлении — ре-верификация API.

Не добавлять в `apps/api/requirements.txt` или `apps/ingest/requirements.txt`.

---

## 9. Конфигурация

```python
# benchmarks/config.py
import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:16333")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8082")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8080")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8001")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_colbert_v2")

RETRIEVAL_DATASET = "datasets/eval_retrieval_100.json"
AGENT_DATASET = "datasets/eval_golden_v2.json"
RESULTS_DIR = "benchmarks/results"

# Retrieval params
DENSE_TOP_K = 20
SPARSE_TOP_K = 100
RERANK_TOP_N = 10
FINAL_TOP_K = 20

# Agent params
LLM_TEMPERATURE = 0.0     # deterministic для eval
LLM_MAX_TOKENS = 4096
LLM_CONTEXT_WINDOW = 32768
```

---

## 10. Порядок реализации

### Phase 1: Retrieval benchmark (приоритет)

1. `benchmarks/config.py` + `benchmarks/protocols.py` — базовые типы
2. `benchmarks/naive/retriever.py` — dense-only search
3. `benchmarks/llamaindex_pipeline/embedding.py` — PplxEmbedding
4. `benchmarks/llamaindex_pipeline/reranker.py` — QwenReranker
5. `benchmarks/llamaindex_pipeline/fusion.py` — weighted RRF callable для hybrid_fusion_fn
6. `benchmarks/llamaindex_pipeline/retriever.py` — QdrantVectorStore + hybrid (stock & maxed configs)
7. `benchmarks/custom_adapter/retriever.py` — BM25+Dense→RRF→ColBERT via Qdrant HTTP
8. `scripts/run_benchmark_retrieval.py` — runner для 4 pipeline + вывод результатов
9. Запуск, сбор результатов, проверка корректности

### Phase 2: Agent E2E benchmark

10. `benchmarks/export_for_judge.py` — генерация артефакта для offline judge
11. `benchmarks/naive/agent.py` — dense + single LLM call
12. `benchmarks/llamaindex_pipeline/agent.py` — FunctionAgent (stock & maxed configs)
13. `benchmarks/custom_adapter/agent.py` — HTTP adapter к SSE endpoint
14. `scripts/run_benchmark_agent.py` — runner для 4 pipeline + judge
15. Запуск, сбор результатов

### Phase 3: ADR + результаты

14. `docs/architecture/11-decisions/DEC-NNNN-framework-comparison.md` — ADR с числами
15. Сводная таблица в README или отдельный markdown

---

## 11. Acceptance criteria

### Retrieval benchmark
- [ ] Все 4 pipeline (naive, LI-stock, LI-maxed, custom) запускаются на 100 вопросах
- [ ] Метрики Recall@1/5/10/20, MRR, nDCG@5, latency (p50, p95) для каждого pipeline
- [ ] Результаты в JSON + markdown таблица
- [ ] LI-stock: `enable_hybrid=True`, default fusion, без reranker
- [ ] LI-maxed: `enable_hybrid=True`, `hybrid_fusion_fn` = weighted RRF, + QwenReranker
- [ ] Naive: только dense search, без BM25, без reranking

### Agent E2E benchmark
- [ ] Все 4 pipeline запускаются на **17** retrieval вопросах (exact IDs: q01-q16, q25)
- [ ] Метрики factual_correctness, usefulness, BERTScore для каждого pipeline
- [ ] Runner генерирует JSON артефакт с ответами всех pipeline → offline judge через чат Claude/Codex
- [ ] LI-maxed: FunctionAgent + search tool + initial_tool_choice + reranker
- [ ] LI-stock: FunctionAgent + search tool, default config
- [ ] Naive: single LLM call без agent loop
- [ ] temperature=0.0 для всех систем, `enable_thinking=False` для Qwen3.5

### Общее
- [ ] Zero changes в `src/` — приложение не затронуто
- [ ] LlamaIndex зависимости изолированы в `benchmarks/requirements.txt` (pinned versions)
- [ ] Все системы используют одинаковые: LLM, embeddings, Qdrant коллекцию, вопросы
- [ ] Embedding без instruction prefix (DEC-0042) во всех pipeline
- [ ] ADR документирует результаты, methodology, и architectural decision
- [ ] Raw tool schemas + prompts + versions залогированы для одного sample run

---

## 12. Что НЕ входит в scope

- Расширение eval dataset до 100+ agent questions (отдельная задача, после benchmark)
- Agent comparison на analytics/navigation/refusal questions
- Добавление LANCER в LlamaIndex (zero framework support)
- Phase-based tool visibility в LlamaIndex (tool_retriever = query-based, не phase-based)
- Docker/CI для benchmark (overkill для portfolio piece)
- Performance tuning LlamaIndex (beyond honest best-effort config)
- LangChain или Haystack comparison (может быть добавлено позже)

---

## 13. Риски и митигации

| Риск | Вероятность | Митигация |
|------|------------|-----------|
| LlamaIndex `enable_hybrid` не работает с нашими named vectors | Средняя | Fallback: raw Qdrant client + LlamaIndex retriever interface |
| `OpenAILike` несовместим с llama-server function calling | Средняя | Fallback: custom `CustomLLM` subclass |
| LlamaIndex reranker не принимает Qwen3 chat template format | Высокая | Уже заложен custom `QwenReranker(BaseNodePostprocessor)` |
| Разница в метриках < 5% (неубедительно) | Средняя | Naive baseline покажет пол — разница custom vs naive будет значительной |
| LlamaIndex sparse query encoding не совпадает с нашим fastembed BM25 | Средняя | Передаём тот же `fastembed.SparseTextEmbedding("Qdrant/bm25")` через `sparse_query_fn` |

---

## 14. Ожидаемый нарратив (гипотеза, до замеров)

> **Гипотеза:** Custom pipeline выиграет по retrieval quality за счёт ColBERT multivector
> (единственный компонент который LlamaIndex физически не может использовать).
> LI-maxed может приблизиться к Custom (у обоих weighted RRF, но разные reranking стратегии) — это должно быть подтверждено замерами.
> LI-stock покажет что даёт фреймворк из коробки. Naive покажет пол.
>
> **4 точки данных — 4 вывода:**
> - Naive → LI-stock = ценность фреймворка из коробки (delta A)
> - LI-stock → LI-maxed = ценность custom кода поверх фреймворка (delta B)
> - LI-maxed → Custom = **совокупная** разница от: ColBERT, query planner, LANCER coverage,
>>   phase-based visibility, custom retry/trimming, prompt scaffolding, agent semantics (delta C).
>>   **Это multi-factor delta — нельзя атрибутировать к одному компоненту без ablation runs.**
> - Naive → Custom = полная ценность custom engineering (delta A+B+C)
>
> **Если delta C значительная:** "Совокупность custom компонентов (ColBERT, query planner,
> LANCER, agent hardening) даёт измеримое преимущество. Точная атрибуция по компонентам
> потребует дополнительных ablation experiments."
>
> **Если delta C маленькая:** "Framework + custom fusion даёт ~95% результата.
> Дополнительная инженерия (ColBERT, LANCER) — marginal gain.
> Ценность custom pipeline — в flexibility и debuggability."
>
> **В обоих случаях** benchmark демонстрирует: ability to build both, measure objectively, make informed decisions.
