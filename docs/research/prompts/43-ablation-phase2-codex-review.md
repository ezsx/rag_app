# Prompt: Codex Review — Retrieval Ablation Phase 2

> Ты — сайдкар-ревьюер для RAG-проекта. Проанализируй результаты phase 1 ablation study и предложи эксперименты для phase 2. Отвечай конкретно: эксперимент → гипотеза → ожидаемый эффект.

## Проект

**rag_app** — FastAPI RAG платформа. Поисковик/агрегатор новостей из русскоязычных Telegram-каналов.
Telegram-каналы → ingest → Qdrant → Hybrid Retrieval → ReAct Agent → SSE ответ.

**Ключевые файлы проекта:**
- `src/adapters/search/hybrid_retriever.py` — HybridRetriever: Qdrant запросы, RRF fusion, ColBERT rerank
- `src/adapters/tei/reranker_client.py` — TEIRerankerClient: CE reranker (POST /rerank → scores)
- `src/services/query_planner_service.py` — QueryPlannerService: LLM-перефразирование в 3-6 subqueries
- `src/services/agent/executor.py` — Agent executor: CE filter (CRAG-style, threshold=0.0, строка 198)
- `src/services/agent/state.py:117-150` — CE filter logic: сохраняет ColBERT порядок, отсекает score < threshold
- `src/services/tools/search.py` — search tool: multi-query round-robin merge
- `src/services/tools/query_plan.py` — query_plan tool: вызывает QueryPlannerService
- `src/core/settings.py` — все настройки (dense_prefetch_limit=20, reranker_top_n, etc.)

## Данные

- **Qdrant коллекция**: `news_colbert_v2`, 13,777 документов, 35 каналов
- **Vectors**: dense (pplx-embed, 1024d) + sparse (BM25 fastembed) + ColBERT (jina-colbert-v2, 128d per-token)
- **Payload indexes**: channel, date, message_id, entities, arxiv_ids, lang, year_week + 10 других

## Production Pipeline

```
User query
  → LLM query_plan (Qwen3.5-35B, 3-6 subqueries + metadata filters)
  → Per subquery:
      BM25(top 100) + Dense(top 20) → RRF [1.0, 3.0] → ColBERT MaxSim rerank
  → Round-robin merge всех subquery результатов
  → Cross-encoder filter (Qwen3-Reranker-0.6B, threshold=0.0, score < 0 = remove)
  → Channel dedup (max 2 docs/channel)
  → compose_context → LLM final_answer
```

## Phase 1 Ablation — Результаты

**Артефакты:**
- Dataset: `datasets/eval_retrieval_v3.json` — 120 natural language вопросов, 6 категорий, 30 каналов
- Eval скрипт: `scripts/evaluate_retrieval.py` — параметризованный (--no-prefix, --dense-limit, --bm25-limit, --rrf-weights, --fusion, --output)
- Результаты: `results/ablation/` — 24 JSON файла + `summary.md` + `_summary.json`
- Старые datasets: `datasets/eval_retrieval_calibration.json` (50 hand-crafted Qs), `datasets/eval_retrieval_100.json` (100 auto-generated Qs)

**Что тестировали**: instruction prefix, dense limit (10/20/40/60), BM25 limit (50/100/200), RRF weights ([1:1]..[1:5]), DBSF fusion, ColBERT on/off, комбинации.

**Что НЕ тестировали**: CE filter/reranker, LLM query expansion, multi-query merge, channel dedup, ColBERT pool size отдельно от dense limit.

### Baseline vs Best

| Metric | Baseline | Best (no-prefix + dense=40 + RRF [1:3]) | Δ |
|--------|----------|------------------------------------------|---|
| R@1 | 0.708 | 0.758 | +0.050 |
| R@3 | 0.817 | 0.883 | +0.067 |
| R@5 | 0.833 | 0.900 | +0.067 |
| R@10 | 0.850 | 0.917 | +0.067 |
| R@20 | 0.867 | 0.933 | +0.067 |
| MRR@20 | 0.765 | 0.823 | +0.058 |

### Full Results Table (24 experiments)

| Experiment | R@1 | R@5 | R@20 | MRR | Δ R@5 |
|-----------|-----|-----|------|-----|-------|
| baseline (prefix ON, dense=20, BM25=100, RRF default, ColBERT ON) | 0.708 | 0.833 | 0.867 | 0.765 | — |
| **no_prefix** | **0.725** | **0.892** | **0.917** | **0.801** | **+0.058** |
| dense=40 | 0.733 | 0.867 | 0.900 | 0.793 | +0.034 |
| dense=10 | 0.675 | 0.817 | 0.842 | 0.738 | −0.017 |
| rrf [1:1] | 0.708 | 0.833 | 0.867 | 0.765 | 0.000 |
| rrf [1:2] | 0.708 | 0.833 | 0.867 | 0.765 | 0.000 |
| rrf [1:3] | 0.708 | 0.833 | 0.867 | 0.765 | 0.000 |
| rrf [1:4] | 0.692 | 0.825 | 0.858 | 0.753 | −0.008 |
| rrf [1:5] | 0.683 | 0.833 | 0.858 | 0.749 | 0.000 |
| bm25=50 | 0.708 | 0.833 | 0.867 | 0.765 | 0.000 |
| bm25=200 | 0.700 | 0.833 | 0.867 | 0.761 | 0.000 |
| DBSF | 0.675 | 0.825 | 0.850 | 0.744 | −0.008 |
| no ColBERT | 0.467 | 0.733 | 0.842 | 0.581 | −0.100 |
| no-prefix + dense=40 | 0.742 | 0.900 | 0.933 | 0.813 | +0.067 |
| **no-prefix + dense=40 + RRF [1:3]** | **0.758** | **0.900** | **0.933** | **0.823** | **+0.067** |
| no-prefix + bm25=200 + dense=40 | 0.742 | 0.900 | 0.933 | 0.813 | +0.067 |
| no-prefix + dense=60 + RRF [1:3] | 0.750 | 0.900 | 0.925 | 0.818 | +0.067 |
| no-prefix + RRF [1:3] | 0.725 | 0.892 | 0.917 | 0.802 | +0.058 |
| DBSF + no-prefix | 0.700 | 0.892 | 0.917 | 0.788 | +0.058 |
| no ColBERT + no-prefix | 0.542 | 0.850 | 0.933 | 0.681 | +0.017 |
| no ColBERT + RRF default | 0.442 | 0.733 | 0.842 | 0.570 | −0.100 |

### Per-Category (best config)

| Category | n | R@1 | R@5 | R@20 | MRR |
|----------|---|-----|-----|------|-----|
| factual | 48 | 0.85 | 0.96 | 0.98 | 0.909 |
| comparative | 12 | 0.67 | 1.00 | 1.00 | 0.808 |
| entity | 12 | 0.75 | 0.92 | 1.00 | 0.818 |
| channel_specific | 18 | 0.89 | 0.89 | 0.89 | 0.889 |
| temporal | 18 | 0.72 | 0.89 | 0.89 | 0.796 |
| edge | 12 | 0.33 | 0.58 | 0.75 | 0.442 |

### Key Findings

1. **Instruction prefix вредил в eval** — eval скрипт добавлял prefix, production (DEC-0042) — нет. Отключение = +5.8% R@5. Самый значимый одиночный параметр.
2. **Dense limit 20→40** — +3.4% R@5. Больше dense кандидатов для ColBERT. Выше 40 — diminishing returns (60 дало тот же R@5 но хуже R@1).
3. **ColBERT критичен** — без него R@1 −34%. 100% overhead по latency (4.35s vs 2.18s), но оправдан.
4. **RRF weights не влияют** — [1:1]..[1:3] идентичны при ColBERT. ColBERT полностью переранжирует.
5. **BM25 limit не влияет** — 50/100/200 одинаково.
6. **DBSF хуже RRF** на −0.8%.

### 12 Permanent Misses (R@5=0 на лучшей конфигурации)

| ID | Category | Query | Expected doc |
|----|----------|-------|-------------|
| ret_030 | factual | Какую конституцию опубликовала Anthropic? | theworldisnoteasy:2378 |
| ret_034 | factual | ИИ-модели играют в стратегические игры? | seeallochnaya:3054 |
| ret_049 | temporal | Что нового в генеративных моделях в феврале? | ai_newz:4433 |
| ret_059 | temporal | Инвестиции в ИИ-компании в начале 2026? | seeallochnaya:3270 |
| ret_068 | channel | llm_under_hood рекомендации? | llm_under_hood:752 |
| ret_075 | channel | Постнаука + gonzo_ml? | gonzo_ml:4666 |
| ret_098 | entity | Сколько OpenAI привлекла? | seeallochnaya:3427 |
| ret_109 | edge | "че там по трансформерам нового" | gonzo_ml:4567 |
| ret_110 | edge | "какие нейросетки умеют видосы делать" | neurohive:1929 |
| ret_111 | edge | Claude Code + старые игры? | denissexy:11238 |
| ret_112 | edge | нейросети заменят программистов? | techno_yandex:4978 |
| ret_113 | edge | ии-музыка через suno? | denissexy:10782 |

Паттерн: 5/12 — сленг/разговорный, 2 — слишком широкие temporal, 2 — channel name в query не в embedding space, 3 — большой семантический gap query↔document.

## Hardware

- **Embedding**: pplx-embed-v1-0.6B (bf16, mean pooling, без instruction prefix) → gpu_server.py в WSL2, RTX 5060 Ti, порт 8082
- **ColBERT**: jina-colbert-v2 (560M, 128-dim per-token MaxSim) → gpu_server.py, порт 8082
- **CE Reranker**: Qwen3-Reranker-0.6B-seq-cls (chat template, padding_side=left, logit scoring) → gpu_server.py, порт 8082
- **LLM**: Qwen3.5-35B-A3B GGUF Q4_K_M (V100 SXM2 32GB, llama-server.exe, порт 8080)
- **Qdrant**: Docker (CPU only), порт 16333

## Наши предварительные идеи для phase 2

### Блок A: CE stage
- CE filter threshold sweep: −2.0 / 0.0 / 1.0 / 2.0 на winning config
- CE как reranker (re-sort по CE score вместо filter) — может CE лучше ColBERT на R@1?

### Блок B: Dense ceiling
- dense=60/80/100 с расширенным ColBERT pool (сейчас ColBERT pool = top_k*3, может быть бутылочное горлышко)

### Блок C: Full pipeline с LLM query_plan
- LLM query_plan перефразирует запрос → multi-query search → merge → сравнить с single-query
- Особенно для edge cases (сленг) — LLM должен нормализовать

### Блок D: Cross-validation
- Calibration dataset (50 Qs) с правильным no-prefix
- Auto-generated dataset (100 Qs) с winning config

## Вопросы к тебе

1. Какие ещё эксперименты стоит провести в phase 2? Конкретно: эксперимент → гипотеза → что покажет.
2. Есть ли комбинации параметров которые мы не пробовали и стоит попробовать?
3. CE как reranker vs CE как filter — есть ли литература или интуиция что лучше для нашего случая (ColBERT уже ранжирует)?
4. Какие метрики кроме R@K и MRR стоит добавить для более полной картины?
5. Что ещё может объяснить 12 permanent misses? Есть ли техники кроме query expansion?
6. Стоит ли тестировать hybrid без BM25 (dense-only + ColBERT) — учитывая что BM25 limit не влияет?
7. Какие эксперименты из предложенных ты считаешь пустой тратой compute?
