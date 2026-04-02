# SPEC-RAG-27: Code Quality Final — "не придраться"

**Status**: DRAFT v3 (updated after Codex GPT-5.4 review + discussion)
**Risk**: MEDIUM (architectural changes в retriever, deps, agent)
**Estimated scope**: ~30 файлов, 3 фазы
**Depends on**: SPEC-RAG-24/25/26 (выполнены)
**Blocks**: ничего (финальный quality pass перед portfolio show)

---

## Цель

Довести код до состояния "hiring manager открыл src/ — вопросов нет".
Каждое изменение мотивировано конкретным code smell.

### Что уже сделано (SPEC-RAG-24/25/26)
- Dead code removal (-550 строк)
- Pydantic BaseSettings
- DRY extraction (cache.py, dispatch table, public API)
- CI (ruff + pytest + mypy)
- 0 f-string logging, lazy % formatting
- Type hints на tools
- .gitattributes

### Что осталось (этот SPEC)
- HybridRetriever — dead _mmr_rerank, extractable pure functions
- agent_service.py — 959 строк, stream_agent_response ~880 строк
- deps.py — 12 closures в одной функции
- Нет explicit interface contracts (duck typing повсюду)

### Что НЕ входит
- **Тесты** — откладываем до стабилизации. Сейчас быстро меняемся, тесты будут фиксировать
- **QAService удаление** — НЕ legacy, это RAG baseline path. Оставляем, помечаем
- **Exception audit** — частично выполнен в SPEC-26, остальное low ROI
- **scripts/*.py** — tooling, не production code

---

## Фаза 1: HybridRetriever cleanup

**Проблема**: ~400 строк, мёртвый MMR код, extractable pure functions.

### Удалить (dead code, подтверждено Codex):
- `_mmr_rerank()` (строки ~312-366) — не вызывается нигде. MMR опробован, портил recall, отключён. Зафиксировано в experiment history.

### НЕ удалять:
- `_cosine_similarity()` — **используется** в `_to_candidates` (строки 405, 414) для dense_score
- Legacy shims (`get_context*`, `embed_texts`) — **используются** QAService и /v1/search endpoint
- Property accessors (`store`, `embedding_client`, `sparse_encoder`) — **используются** tools (SPEC-25)

### Extract (pure functions):
- `_build_filter()` → standalone function (не зависит от self)
- `_to_candidates()` → standalone function (не зависит от self, только от dense_vector)

### Ожидаемый результат
- HybridRetriever: ~400 → ~340 строк
- Удалён мёртвый `_mmr_rerank`
- `_build_filter` и `_to_candidates` — standalone, тестируемы отдельно

---

## Фаза 2: Protocol interfaces + agent_service.py split

### 2a. Protocol interfaces (core/protocols.py)

**Проблема**: контракты между слоями — implicit duck typing.

**Решение**: `src/core/protocols.py` (~40 строк). Для документации архитектуры в коде,
не для mypy enforcement.

```python
@runtime_checkable
class Retriever(Protocol):
    def search_with_plan(self, query_text: str, plan: SearchPlan) -> list[Candidate]: ...

class EmbeddingClient(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...

class RerankerClient(Protocol):
    async def rerank(self, query: str, passages: list[str], top_n: int) -> dict[str, Any]: ...

class LLMClient(Protocol):
    def chat_completion(self, messages: list, tools: list | None = None, **kwargs) -> dict[str, Any]: ...
```

### 2b. agent_service.py decomposition (959 → ~500 строк)

**Принцип**: helper functions возвращают data, yield points остаются в main loop.
НЕ разрезать async generator на sub-generators.

**Extraction targets** (подтверждены Codex как best candidates):

**1. LLM step → `agent/llm_step.py`** (~60 строк)
- LLM call + response parsing (content/tool_calls/tokens extraction)

**2. Guards → `agent/guards.py`** (~80 строк)
- `check_forced_search()` — forced search если LLM не вызвал tools
- `check_analytics_shortcircuit()` — analytics forced completion
- `check_tool_repeat()` — tool repeat guard

**3. Refinement glue** (~120 строк, строки ~557, ~697)
- Дублированный refinement flow → единый helper

**Что остаётся в stream_agent_response**:
- Init (context, messages, trace) — ~50 строк
- Main loop: call LLM → check guards → execute → yield events — ~200 строк
- Refinement yield loop — ~50 строк
- Finalization — ~100 строк
- **Total: ~400-500 строк** — orchestration + yield points

---

## Фаза 3: deps.py tool registration extraction

**Проблема**: `get_agent_service()` = ~100 строк с 12 closures (wrappers).

**ВАЖНО**: `temporal_search`/`channel_search` — virtual tools. Видны LLM в schema,
но executor маппит их на `search` с разными filters. Registry extraction должен
сохранить этот контракт (маппинг в executor.py, visibility в prompts.py).

**Решение**: `src/services/tools/registry.py`

```python
def build_tool_runner(settings, hybrid_retriever, reranker, ...) -> ToolRunner:
    runner = ToolRunner(default_timeout_sec=settings.agent_tool_timeout)
    runner.register("search", partial(search, hybrid_retriever=hybrid_retriever), ...)
    # ... 12 registrations
    # NOTE: temporal_search/channel_search — virtual tools.
    # LLM видит их в schema (prompts.py), executor маппит на search (executor.py:66).
    # Не регистрируются отдельно в ToolRunner.
    return runner
```

### QAService — RAG baseline path

QAService — **НЕ legacy**. Это прямой RAG pipeline без agent loop:
plan → search → rerank → LLM answer. Используется `/v1/qa`, `/v1/qa/stream`.

**Действие**: пометить как RAG baseline в docstring и комментариях.
Рядом потом заведём LlamaIndex baseline — тот же Qdrant, framework wrapper,
тоже без agent loop. Идеальный peer для A/B сравнения.

---

## Acceptance Criteria

### Фаза 1: HybridRetriever
- [ ] `_mmr_rerank` удалён
- [ ] `_cosine_similarity` **сохранён** (используется в _to_candidates)
- [ ] `_build_filter` и `_to_candidates` — standalone functions
- [ ] Legacy shims (`get_context*`, `embed_texts`) — сохранены
- [ ] pytest pass, ruff clean

### Фаза 2: Protocols + agent_service split
- [ ] `core/protocols.py` создан (Retriever, EmbeddingClient, RerankerClient, LLMClient)
- [ ] `agent/llm_step.py` создан — LLM call extraction
- [ ] `agent/guards.py` создан — forced search, analytics shortcircuit, tool repeat
- [ ] Refinement flow deduplicated
- [ ] agent_service.py ≤ 550 строк
- [ ] pytest pass, ruff clean

### Фаза 3: deps.py + QAService docs
- [ ] `services/tools/registry.py` создан — tool registration вынесен
- [ ] deps.py `get_agent_service()` ≤ 30 строк
- [ ] Virtual tools contract задокументирован
- [ ] QAService помечен как "RAG baseline path"
- [ ] pytest pass, ruff clean

### Общее
- [ ] CI green: ruff + pytest + mypy
- [ ] Smoke test через agent (минимум 1 вопрос)
- [ ] Каждая фаза — отдельный коммит

---

## Порядок работы

```
Фаза 1: HybridRetriever cleanup
  1.1 Удалить _mmr_rerank (dead code)
  1.2 Extract _build_filter → standalone
  1.3 Extract _to_candidates → standalone
  1.4 Smoke test + lint

Фаза 2: Protocols + agent_service split
  2.1 Создать core/protocols.py
  2.2 Создать agent/llm_step.py (LLM call extraction)
  2.3 Создать agent/guards.py (forced search, shortcircuits)
  2.4 Deduplicate refinement flow
  2.5 Smoke test + lint

Фаза 3: deps.py + QAService docs
  3.1 Создать services/tools/registry.py
  3.2 Refactor get_agent_service()
  3.3 Пометить QAService как RAG baseline
  3.4 Smoke test + lint
```
