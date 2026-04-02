# Prompt 39: SPEC-RAG-27 Implementation — Code Quality Final

## Контекст сессии

Продолжение после compact. Предыдущая сессия (2026-04-02):

### Выполнено в этой сессии
1. **SPEC-RAG-24** (commit `658305d`): Dead code cleanup — -550 строк
   - Удалён `collections.py` (170 строк ChromaDB), `answer_v2()` (255 строк), `router_select` (tool + test)
   - Убран `qa_service` из AgentService, fixed `model_post_init` → `@model_validator`
   - Fixed `ingest_service` (только битый импорт, файл ЖИВОЙ)

2. **SPEC-RAG-25** (commit `64bbfb9`→`282134a` с amend): DRY extraction
   - HybridRetriever public API: `store`, `embedding_client`, `sparse_encoder`, `run_sync()`
   - 40 protected→public замен в 7 tool файлах
   - `core/cache.py` — shared cache layer вместо дублей в endpoints
   - `formatting.py` — dispatch table `_FORMATTERS` вместо 110-строчного if/elif
   - Codex review: 0 FAIL, 2 CONCERN → пофикшены (добавлены formatters для всех 15 tools)

3. **SPEC-RAG-26** (commit `9c664da`): Code quality
   - Type hints на все tool params (`hybrid_retriever: Any`)
   - 57 f-string→lazy logging fixes, 0 remaining
   - 4 targeted exception narrowing (httpx, requests, OSError)
   - `_round_robin_merge()` extracted из search()
   - `.gitattributes` (eol=lf)
   - Codex review: 0 FAIL, 4 CONCERN → пофикшены (whitening back to broad, TypeError added, redis return type)

### Текущее состояние
- **Branch**: main, up to date with origin
- **Last commit**: `9c664da` (pushed)
- **CI**: ruff clean, pytest 74 pass / 0 fail / 1 skipped
- **Smoke tests**: 3 agent smoke tests passed (golden_q01, q10, q25), KTA 1.0, correct answers

### Ключевые решения из обсуждения
- **QAService = RAG baseline path**, НЕ legacy. Оставляем, помечаем. Рядом позже LlamaIndex baseline
- **`_cosine_similarity` НЕ мёртвый** — используется в `_to_candidates` для dense_score. Удаляем только `_mmr_rerank`
- **`temporal_search`/`channel_search`** — virtual tools. LLM видит их в schema, executor маппит на `search` с filters. Сохранять при registry extraction
- **Тесты откладываем** — проект быстро меняется, тесты будут при стабилизации
- **Exception audit** — частично выполнен (SPEC-26), остальное low ROI

---

## Задача: реализовать SPEC-RAG-27

Спецификация: `docs/specifications/active/SPEC-RAG-27-code-quality-final.md` (v3)

### Фаза 1: HybridRetriever cleanup

Файл: `src/adapters/search/hybrid_retriever.py` (~400 строк)

1. **Удалить `_mmr_rerank()`** — dead code, ~55 строк. Подтверждено Codex, подтверждено user (MMR опробован, портил recall)
2. **Extract `_build_filter()` → standalone function** — не зависит от self, pure data transformation
3. **Extract `_to_candidates()` → standalone function** — не зависит от self, принимает points + dense_vector
4. **НЕ трогать**: `_cosine_similarity` (живой), legacy shims (`get_context*`, `embed_texts`), property accessors

### Фаза 2: Protocols + agent_service.py split

1. **Создать `src/core/protocols.py`** (~40 строк) — Retriever, EmbeddingClient, RerankerClient, LLMClient
2. **Создать `src/services/agent/llm_step.py`** — LLM call + response parsing extraction
3. **Создать `src/services/agent/guards.py`** — forced search, analytics shortcircuit, tool repeat guard
4. **Deduplicate refinement flow** — строки ~557, ~697 в agent_service.py
5. **Цель**: agent_service.py 959 → ≤550 строк

### Фаза 3: deps.py + QAService docs

1. **Создать `src/services/tools/registry.py`** — `build_tool_runner()` вынесен из deps.py
2. **Refactor `get_agent_service()`** — ≤30 строк
3. **Пометить QAService** как "RAG baseline path" в docstrings
4. **Задокументировать virtual tools** contract в registry.py

---

## Smoke test процедура

После каждой фазы:

### Unit smoke (быстрый, ~5 сек):
```bash
PY="/c/Users/scdco/AppData/Local/Programs/Python/Python312/python.exe"
cd c:/llms/rag/rag_app
ruff check src/ scripts/
PYTHONPATH=src "$PY" -m pytest src/tests/ -q --override-ini="addopts="
```
Ожидание: ruff clean, 74 pass, 0 fail.

### Agent smoke (полный, ~30-60 сек):
```bash
PYTHONPATH=src "$PY" scripts/evaluate_agent.py \
  --dataset datasets/eval_golden_v2.json \
  --questions golden_q10 \
  --output-dir /tmp/smoke_spec27 \
  --skip-judge --skip-markdown
```
Ожидание: KTA 1.0, 0 agent errors, latency <40s.

**ВАЖНО**: перед agent smoke нужен rebuild Docker image (pydantic-settings):
```bash
docker compose -f deploy/compose/compose.dev.yml up -d --build api
```
Если Docker Hub недоступен — `docker restart rag-dev-api-1` после `docker exec ... pip install pydantic-settings`.

### Langfuse трейсы:
```bash
PY="/c/Users/scdco/AppData/Local/Programs/Python/Python312/python.exe"
curl -s "http://localhost:3100/api/public/traces?limit=3" \
  -H "Authorization: Basic $(echo -n 'pk-lf-rag-app-dev:sk-lf-rag-app-dev' | base64)" \
  | "$PY" -c "
import sys, json
data = json.load(sys.stdin)
for t in data.get('data', [])[:3]:
    print(f\"{t['id'][:12]}  {t.get('name','?')}  latency={t.get('latency')}s\")
"
```

Для деталей spans конкретного trace:
```bash
TRACE_ID="<first 12 chars>"
curl -s "http://localhost:3100/api/public/observations?traceId=$TRACE_ID&limit=50" \
  -H "Authorization: Basic $(echo -n 'pk-lf-rag-app-dev:sk-lf-rag-app-dev' | base64)" \
  | "$PY" -c "
import sys, json
data = json.load(sys.stdin)
for o in sorted(data.get('data', []), key=lambda x: x.get('startTime','')):
    print(f\"  {o.get('type','?'):12s}  {o.get('name','?'):35s}  {o.get('latency')}s\")
"
```

---

## Hardware (для запуска)

```powershell
# 1. LLM на V100 (PowerShell, Windows)
$env:CUDA_VISIBLE_DEVICES = "0"
C:\llm-test\llama\llama-server.exe -m models/Qwen3.5-35B-A3B-Q4_K_M.gguf --jinja --reasoning-budget 0 --cache-type-k q8_0 --cache-type-v q8_0 -c 32768 --port 8080

# 2. Embedding + Reranker на RTX 5060 Ti (WSL2)
source /home/ezsx/infinity-env/bin/activate && CUDA_VISIBLE_DEVICES=0 python /mnt/c/llms/rag/rag_app/scripts/gpu_server.py --with-nli

# 3. Docker (API + Qdrant)
docker compose -f deploy/compose/compose.dev.yml up -d
```

---

## Порядок работы

1. Прочитать SPEC-RAG-27: `docs/specifications/active/SPEC-RAG-27-code-quality-final.md`
2. Фаза 1 → unit smoke → commit
3. Фаза 2 → unit smoke → agent smoke → commit
4. Фаза 3 → unit smoke → agent smoke → commit + push
5. Codex review финального состояния
