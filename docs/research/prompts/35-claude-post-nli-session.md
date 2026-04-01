# Prompt 35: Claude — Post NLI Session

## Контекст

Сессия 2026-04-01 (вторая). SPEC-RAG-21 NLI реализован, baseline получен. Docs обновлены.

## Текущие метрики (36 Qs golden_v2, 2026-04-01)
- Factual: **0.842** (Claude judge, granular 0.1 scale)
- Useful: **1.778/2**
- KTA: **1.000**
- Faithfulness: **0.91** (corrected, ruBERT NLI, 0 hallucinations)
- Retrieval recall@3: **0.97** (100 calibration queries)
- Latency: **24.4s**

## Что делать дальше

### P1: Robustness NDR/RSR/ROR (SPEC-RAG-22)
- R20 ready, обкатать на текущих 36 Qs
- Это вторая quality metric перед 100 Qs expansion

### P1: Fix known issues
- q15 routing: summarize_channel → search with channel filter (LLM output truncated, 0 citations)
- SecurityManager полный рефакторинг: security boundary на API level, не tool level

### P2: NLI improvements
- Contradiction threshold → 0.90+ или убрать contradiction category
- Premise cleaning улучшить для mixed RU/EN text
- A/B: поднять entailment threshold с 0.45 до 0.50 на полном прогоне

### P3: LlamaIndex/LangChain baseline
- Собрать LlamaIndex baseline на нашем Qdrant + тех же данных
- Прогнать на golden_v2 → сравнительная таблица для README "Why not frameworks?"
- Постараться выжать максимум из фреймворка (не strawman comparison)

### P4: 100 Qs + ablation (после robustness)
- Expand golden dataset 36 → 100+
- Ablation: выключаем компоненты по одному (ColBERT, RRF, coverage, CE filter)
- Несколько дней чисто на тесты

### P5: Polish (финальный этап)
- Unit tests + рефакторинг кода
- README: скриншот UI, Mermaid diagram
- UI restructure
- Хабр статья (опционально)

## Ключевые файлы
- `src/services/eval/nli.py` — NLI verifier
- `scripts/run_nli.py` — NLI pipeline
- `scripts/merge_eval_report.py` — merge report
- `scripts/gpu_server.py` — /nli endpoint (ruBERT, lazy loading)
- `datasets/prompts/decomposition_v1.md` + `judge_v1.md`
- `docs/specifications/active/SPEC-RAG-21-nli-citation-faithfulness.md`
- `results/reports/nli_faithfulness_analysis_20260401.md`
- `docs/planning/experiment_history.md` — полная история
- `docs/planning/retrieval_improvement_playbook.md` — overview + metrics
