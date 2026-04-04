# Prompt 35: Claude — Post NLI Session

## Контекст

Сессия 2026-04-01b завершена. SPEC-RAG-21 NLI реализован и обкатан. Docs актуализированы.

## Текущие метрики (36 Qs golden_v2, 2026-04-01)
- Factual: **0.842** (Claude judge, granular 0.1 scale)
- Useful: **1.778/2**
- KTA: **1.000**
- Faithfulness: **0.91** (corrected, ruBERT NLI, 0 hallucinations)
- Retrieval recall@3: **0.97** (100 calibration queries)
- Latency: **24.4s**

## Что делать в этой сессии

Строго по порядку:

### 1. Robustness NDR/RSR/ROR
- Research: R20 (`docs/research/reports/R20-deep-retrieval-robustness-ndr-rsr-ror.md`)
- Написать SPEC-RAG-22 (спека)
- Реализовать скрипт robustness тестов
- Расширить до 50 Qs для robustness (14 дополнительных, R20 рекомендация)
- Прогнать 50 Qs × 3 вариации (noise/substitution/reorder) = ~150 прогонов, ~1 час compute
- Зафиксировать baseline NDR/RSR/ROR

### 2. Expand golden dataset 36 → 100 Qs
- Claude генерирует вопросы по документам из Qdrant
- Вручную курировать — не автоматически
- 4 eval_modes, decomposed required_claims
- Покрыть edge cases: multi-hop, temporal ranges, cross-channel, numeric

### 3. Полный прогон 100 Qs
- evaluate_agent.py --dataset eval_golden_v3.json
- Claude judge (decomposition + judge в отдельном чате)
- NLI faithfulness (run_nli.py)
- Зафиксировать metrics на 100 Qs

### 4. LlamaIndex baseline (если время останется)
- Собрать LlamaIndex VectorStoreIndex + Qdrant + наш embedding
- Прогнать на golden_v2 (36 Qs)
- Честное сравнение — выжать максимум из фреймворка
- Сравнительная таблица в README

## Known issues
- q15: summarize_channel routing → нужен search + compose_context
- SecurityManager: мина, _skip_security = экстренный fix, нужен рефакторинг boundary
- NLI false positives: 19/19 FP, ruBERT слаб на cross-lingual парафразах
- Citation precision 0.51 — decomposition неполный или агент цитирует "про запас"

## Ключевые файлы
- `docs/research/reports/R20-deep-retrieval-robustness-ndr-rsr-ror.md` — research для robustness
- `src/services/eval/nli.py` — NLI verifier
- `scripts/run_nli.py` — NLI pipeline
- `scripts/evaluate_agent.py` — eval с --questions фильтром
- `datasets/eval_golden_v2.json` — текущий dataset (36 Qs)
- `datasets/prompts/decomposition_v1.md` + `judge_v1.md` — промпты для judge
- `docs/progress/experiment_log.md` — полная история
- `docs/progress/experiment_log.md` — overview + metrics
- `results/reports/nli_faithfulness_analysis_20260401.md` — NLI analysis

## Hardware (всё поднято)
- LLM: Qwen3.5-35B-A3B на V100 (порт 8080)
- GPU server: pplx-embed + Qwen3-Reranker + ColBERT + ruBERT-NLI (порт 8082, --with-nli)
- Docker: API (:8001) + Qdrant (:6333) + Langfuse (:3100)
