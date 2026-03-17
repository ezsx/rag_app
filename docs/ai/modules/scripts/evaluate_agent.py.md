## scripts/evaluate_agent.py

### Назначение
CLI из `scripts/evaluate_agent.py` реализует MVP-оценку по спецификации `docs/ai/planning/agent_evaluation_spec.md`: загружает датасет, исполняет ReAct-агента (`/v1/agent/stream`) и baseline QA (`/v1/qa`), собирает метрики (latency, coverage, recall@5) и сохраняет сырые/агрегированные отчёты.

### Структура
- **Константы** — дефолтные пути (`datasets/eval_dataset.json`, `/v1/agent/stream`, `/v1/qa`), директории `results/{raw,reports}`.
- **`EvalItem`** — dataclass (id, query, category, expected_documents, answerable, expected_answer, notes).
- **`parse_args()`** — argparse CLI:
  - ключевые параметры: `--dataset`, `--output-dir`, `--agent-url`, `--qa-url`, `--collection`, `--max-steps`, `--disable-planner`, `--agent-timeout`, `--agent-retries`, `--baseline-timeout`, `--baseline-retries`, `--limit`, `--api-key`, `--skip-markdown`, `--dry-run`, `--verbose`.
- **`load_dataset()`** — читает JSON, поддерживает:
  - актуальный формат (массива объектов с полями §2 спецификации);
  - fallback на `datasets/eval_questions.json` (старый объект `{metadata, questions}`).
- **`iter_sse_events()`** — минимальный SSE-парсер (`event:`/`data:`), возвращает `(event_type, payload_str)`.
- **`extract_hit_ids()`** — берёт первые N hit-id из события `observation` `tool=search`.
- **Хелперы статистики**: `safe_mean`, `percentile`.

### `AgentEvaluationRunner`
- Конструктор принимает dataset, URL-ы, collection, max_steps, planner flag, таймауты, API key, retries (`agent_retries`, `baseline_retries`), флаги `dry_run`/`limit`.
- `run()` итерирует датасет (учитывает `--limit`), пишет лог `[idx/total]`, для каждого item формирует dict:
  - `agent`: результат `_call_agent()` или `_fake_agent_result()` (dry-run).
  - `baseline`: результат `_call_baseline()` или `_fake_baseline_result()`.
  - `metrics`: `_compute_metrics()` (latency, coverage, recall@5 + заглушки correctness).
  - `status`: `ok | agent_error | baseline_error | agent_and_baseline_error`.
- `_call_agent()`:
  - собирает payload (`query`, `max_steps`, `collection`, `planner` флаг);
  - до `agent_retries` попыток: `httpx.Client.stream`, парсит SSE, аккумулирует до 5 уникальных `top5_hits` из всех `search` наблюдений, ждёт `final`;
  - возвращает `answer`, `citations`, `coverage`, `refinements`, `verification`, `fallback`, `request_id`, `latency_sec`, `top5_hits`;
  - при ошибке/отсутствии `final` выставляет `error=True`.
- `_call_baseline()` — до `baseline_retries` дергает `/v1/qa`, измеряет latency, возвращает `answer`/`error`.
- `_compute_metrics()` — считает `recall@5` (сет `expected_documents` ∩ `top5_hits`), latency и coverage; correctness = None.

### Агрегация и вывод
- `aggregate_results()` — формирует summary:
  - количественные показатели (total/answerable/negative);
  - ошибки: количество agent/baseline/both по статусам;
  - latency mean/p95/max (agent/baseline);
  - coverage mean/min/max;
  - recall@5 mean, counts full/partial;
  - `correctness` (кол-во вручную размеченных результатов);
  - `by_category` (queries, средние latency/coverage/recall по категории).
- `build_markdown_report()` — Markdown отчёт (шапка, overall metrics, таблица по категориям, next steps).
- `ensure_dirs()` — создаёт `output_dir`, `output_dir/raw`, `output_dir/reports`.
- `main()`:
  - настраивает логирование (`--verbose`);
  - загружает датасет, запускает runner;
  - пишет `raw_results` в `output_dir/raw/eval_results_{ts}.json`, агрегат — `output_dir/reports/eval_report_{ts}.json`, Markdown (если не `--skip-markdown`);
  - печатает aggregated JSON в stdout.

### Особенности
- `--dry-run` полностью пропускает HTTP-вызовы (удобно для CI smoke-теста).
- `--limit` позволяет гонять подмножество запросов.
- `--api-key` добавляет `Authorization: Bearer ...` к обоим вызовам.
- Fallback на старый датасет позволяет плавно мигрировать к новой схеме `eval_dataset.json`.

