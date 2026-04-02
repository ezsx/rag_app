# Prompt 38: Tier 1 — 100 Golden Qs + Refactoring + CI

## Контекст сессии

Это продолжение после compact. Предыдущая сессия (2026-04-02):
- Реализовали SPEC-RAG-22 (comprehensive eval metrics: BERTScore, SummaC, IR metrics, ToolCallF1)
- Реализовали SPEC-RAG-23 (NDR/RSR/ROR bypass script)
- Прогнали full NDR/RSR/ROR на 36 Qs: **NDR 0.963, RSR 0.941, ROR 0.959, composite 0.954** (Claude judge)
- BERTScore proxy провалился (занижал NDR на 0.15, ложные RSR violations) — Claude judge обязателен
- Independent review проекта: **7.5/10 pre-polish, 9/10 projected after**
- MCP repo-semantic-search починен (pplx-embed-v1, gpu_pplx profile)
- Коммит: `fe86dad` — 17 files, +3853/-1207

## Текущие метрики

| Metric | Value |
|--------|-------|
| Factual | 0.842 (Claude judge, 36 Qs) |
| Useful | 1.778/2 |
| KTA | 1.000 |
| Faithfulness | 0.91 (ruBERT NLI, 0 real hallucinations) |
| Recall@3 | 0.97 (100 calibration queries) |
| Robustness | 0.954 (NDR 0.963, RSR 0.941, ROR 0.959) |
| Latency | 24.4s |

## Tier 1 задачи (блокируют portfolio-ready)

### 1. Expand golden dataset 36 → 100 Qs

**Текущий dataset** (`datasets/eval_golden_v2.json`):
- 36 вопросов, 8 категорий
- Формат: id, version, query, expected_answer, category, difficulty, answerable, expected_refusal, key_tools, forbidden_tools, acceptable_alternatives, source_post_ids, source_channels, eval_mode, required_claims, expected_entities, expected_topics, expected_channels, acceptable_evidence_sets, strict_anchor_recall_eligible, calibration, metadata

**Текущие категории**:
- broad_search: 6
- constrained_search: 7
- compare_summarize: 4
- analytics_hot_topics: 3
- analytics_channel_expertise: 3
- navigation: 2
- negative_refusal: 3
- future_baseline: 8

**Нужно добавить 64 вопроса**. Приоритеты (из independent review + R18):
- Больше retrieval_evidence (broad + constrained): +20
- Multi-hop questions (2+ subqueries): +10
- Temporal range queries: +8
- Cross-channel comparison: +6
- Analytics (entity_tracker, arxiv_tracker): +6
- Refusal (out-of-scope, future, impossible): +6
- Informal/slang queries ("че нового по трансформерам"): +4
- Edge cases (long answers, numeric, entity disambiguation): +4

**Как генерировать**: Qwen/Claude генерирует candidates из документов в Qdrant. Human curates. Каждый вопрос: query + expected_answer + source_post_ids + key_tools + eval_mode + required_claims.

**Коллекция для source**: `news_colbert_v2`, 13777 points, 36 каналов. Можно scroll через Qdrant API для sampling документов.

**Важно**: required_claims должны быть decomposed (atomic facts). Для q01-q25 из текущего dataset required_claims = одна строка (не decomposed) — при expansion делать правильно.

### 2. Рефакторинг кода

**Independent review оценка**: code quality 3/10. Архитектура solid (layered, adapters, DI), код не оформлен.

**Settings** (`src/core/settings.py`, 260 строк): 
- Сейчас: plain `__init__` с `os.getenv()` для каждого параметра
- Нужно: Pydantic BaseSettings с валидацией, .env support, группировка по секциям

**Тесты** (20 test files в `src/tests/`):
- test_agent_endpoints.py, test_agent_function_calling.py, test_agent_service.py
- test_compose_context.py, test_hybrid_retriever.py, test_qdrant_store.py
- test_query_planner.py, test_reranker_service.py, test_tei_clients.py
- И ещё ~10
- Проблемы: некоторые stale (не работают с текущим кодом), покрытие неполное (analytics tools, coverage, NLI не покрыты)

**Код в целом** (91 .py файл):
- Docstrings на русском ✓
- Type hints частичные
- Import cleanup нужен
- Dead code (math_eval, time_now — не используются в production)

### 3. CI (GitHub Actions)

Нужно:
- lint (ruff)
- pytest
- type check (mypy или pyright, basic)

**Файл**: `.github/workflows/ci.yml`
Trigger: push + PR to main.
Тесты запускать без GPU/Qdrant (mock).

## Known Issues (из playbook)

| # | Проблема | Impact | Fix |
|---|----------|--------|-----|
| 1 | q33: monthly hot_topics fallback | Quality | Debug monthly digest |
| 2 | q36: channel_expertise routing miss | Routing | Keyword fix |
| 3 | q21: temporal refusal не работает | Refusal | Temporal guard |
| 4 | q01: Qwen false refusal (stochastic) | LLM limitation | Known, не fix |
| 5 | Stale tests | CI blocker | Delete dead, add new |
| 6 | Данные устарели | Demo reliability | Re-ingest |
| 7 | required_claims не decomposed | Eval quality | Fix при 100 Qs expansion |

## Pipeline (confirmed best, не менять)

```
BM25(100) + Dense(20) → RRF 3:1 → ColBERT → CE filter(0.0) → channel dedup(2)
```
Retrieval recall@3 = 0.97. NDR 0.963, RSR 0.941, ROR 0.959.

## Hardware

- LLM: Qwen3.5-35B-A3B GGUF Q4_K_M (V100 32GB, llama-server, порт 8080)
- GPU server: pplx-embed + Qwen3-Reranker + jina-colbert-v2 + ruBERT NLI (RTX 5060 Ti, gpu_server.py, порт 8082)
- Docker: API + Qdrant (CPU only)
- MCP repo-semantic-search: pplx-embed, gpu_pplx profile, 2733 code + 4425 docs chunks

## Файлы которые будут затронуты

### 100 Qs
- `datasets/eval_golden_v2.json` — расширить
- `scripts/evaluate_agent.py` — уже готов для любого размера dataset
- `datasets/prompts/decomposition_v1.md` — для claims decomposition

### Рефакторинг
- `src/core/settings.py` — Pydantic BaseSettings
- `src/tests/*` — cleanup + новые тесты
- Dead code removal (math_eval, time_now если не используются)
- Import cleanup across src/

### CI
- `.github/workflows/ci.yml` — новый файл

## Порядок работы

1. **CI first** — lint + pytest. Это покажет какие тесты broken и нужен cleanup
2. **Test cleanup** — fix broken tests, delete dead, add minimal coverage for uncovered
3. **Settings refactor** — Pydantic BaseSettings (один файл, isolated change)
4. **100 Qs expansion** — генерация + curation + decomposition claims
5. **Full eval run** — 100 Qs через agent + Claude judge

## Запуск

```bash
# LLM (PowerShell, V100)
$env:CUDA_VISIBLE_DEVICES = "0"
C:\llm-test\llama\llama-server.exe -m models/Qwen3.5-35B-A3B-Q4_K_M.gguf --jinja --reasoning-budget 0 --cache-type-k q8_0 --cache-type-v q8_0 -c 32768 --port 8080

# GPU server (WSL2, RTX 5060 Ti)
source /home/ezsx/infinity-env/bin/activate && CUDA_VISIBLE_DEVICES=0 python /mnt/c/llms/rag/rag_app/scripts/gpu_server.py --with-nli

# Docker
docker compose -f deploy/compose/compose.dev.yml up -d
```
