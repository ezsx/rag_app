# SPEC-RAG-14: Evaluation Pipeline — датасет, LLM judge, tool selection metrics

> **Статус**: Active
> **Создан**: 2026-03-24
> **Research basis**: R18-deep-evaluation-methodology-dataset §1-7, R15-yandex-rag-conference §6,10,13
> **Зависимости**: SPEC-RAG-13 (11 tools + dynamic visibility)
> **Review**: GPT-5.4 review 2026-03-24 — consensus reached (dev phase scope)

---

## 1. Цель

Заменить текущий eval (30 вопросов, только recall@5, нет LLM judge) на multi-criteria evaluation pipeline с:
- **Golden dataset** 20-30 вопросов (dev phase), расширяемый до 450-500 (release)
- **LLM judge** с 3 критериями (factual correctness, usefulness, key tool accuracy)
- **Failure attribution** в eval output для быстрого дебага
- **Forward-looking** вопросы для будущих tools (entity_tracker, arxiv_tracker)

Реализация итеративная: dev → checkpoint → release. Этот spec покрывает **dev phase**.

---

## 2. Контекст

### Что уже есть
- `scripts/evaluate_agent.py` — SSE parsing, recall@5 (fuzzy ±5/±50), coverage, latency, baseline /v1/qa
- `datasets/eval_dataset_quick.json` (v1, 10 Qs), `eval_dataset_quick_v2.json` (v2, 10 Qs), `eval_dataset_v3.json` (v3, 30 Qs)
- SSE events: `thought`, `tool_invoked`, `observation`, `citations`, `final` — всё что нужно для tool tracking
- 11 LLM tools с phase-based dynamic visibility (SPEC-RAG-13)

### Что отсутствует
- **LLM judge** — `agent_correct` / `baseline_correct` = None во всех прогонах
- **Tool selection metrics** — tool_invoked events парсятся, но не оцениваются
- **Failure attribution** — нет классификации причин ошибок
- **Покрытие tools** — v3 не содержит вопросов для cross_channel_compare, summarize_channel, list_channels, related_posts
- **Negative tool selection** — нет тестов на forbidden tools
- **Forward-looking** — нет baseline для будущих entity_tracker/arxiv_tracker

### Критичные ограничения
- V100 ~40 сек/запрос агента → 20-30 вопросов = ~15-20 мин на прогон (приемлемо)
- Judge через Claude API / Codex (не Qwen local) → ~10-25 мин на 30 Qs с 3 критериями
- Сравнимость между фазами: calibration subset 5-10 вопросов при смене judge-модели

---

## 3. Что менять

### 3.1 Новый формат датасета (`datasets/eval_golden_v1.json`)

```python
{
    "id": "golden_q01",
    "version": "1.0",
    "query": "Какие новинки были на GTC 2026 от NVIDIA?",
    "expected_answer": "На GTC 2026 NVIDIA представила платформу Vera Rubin...",
    "category": "constrained_search",      # 6 категорий dev phase
    "difficulty": "medium",                 # easy | medium | hard
    "answerable": True,
    "expected_refusal": False,
    "refusal_reason": None,

    # Tool selection (dev-level)
    "key_tools": ["temporal_search"],       # search tool который ДОЛЖЕН быть вызван
    "forbidden_tools": ["list_channels", "summarize_channel"],  # tools которые НЕ ДОЛЖНЫ
    "acceptable_alternatives": ["search"],  # допустимые замены key_tools (binary whitelist)

    # Source mapping
    "source_post_ids": ["ai_machinelearning_big_data:9678"],
    "source_channels": ["ai_machinelearning_big_data"],

    # Forward-looking
    "future_tool_flag": False,
    "future_key_tools": None,               # заполняется для entity_tracker/arxiv_tracker Qs

    # Metadata
    "metadata": {
        "created_at": "2026-03-24",
        "created_by": "human",
        "tags": ["nvidia", "gtc", "hardware"]
    }
}
```

### 3.2 Шесть категорий для dev phase

| Категория | Описание | Tools | Кол-во |
|-----------|----------|-------|--------|
| `broad_search` | Общий поиск, entity, factual | search | 5-6 |
| `constrained_search` | Temporal + channel-specific | temporal_search, channel_search | 6-8 |
| `compare_summarize` | Кросс-канальное + дайджесты | cross_channel_compare, summarize_channel | 4-5 |
| `navigation` | Метаданные, списки | list_channels | 2-3 |
| `negative_refusal` | Unanswerable + forbidden tool | (refusal / wrong tool penalty) | 3-4 |
| `future_baseline` | Baseline для entity_tracker/arxiv_tracker | search (сейчас), future tool (потом) | 2-3 |

**Итого: 22-29 вопросов.**

Типы negative вопросов (из R18):
- Out-of-database (тема отсутствует в корпусе)
- Out-of-timerange (даты вне июль 2025 — март 2026)
- False presupposition (ложное допущение)

### 3.3 LLM Judge — 3 критерия для dev phase

**Judge model**: Claude API или Codex (не Qwen local — экономим V100 для inference агента).

#### Критерий 1: Factual Correctness (шкала 0.0 — 1.0, единая)

**Для всех вопросов — direct rubric, нормализованная в 0-1:**

```
Вопрос: {question}
Ответ системы: {answer}
Эталонный ответ: {expected_answer}

Оцени фактическую корректность ответа относительно эталона:
0.0 — Содержит фактические ошибки или противоречит эталону
0.5 — Частично корректен, но упускает важные факты или содержит неточности
1.0 — Фактически корректен, соответствует эталону

JSON: {"reasoning": "...", "score": 0.0|0.5|1.0}
```

**Decompose-then-verify** (из R18 Section 3) откладывается на checkpoint phase.
В dev phase для всех вопросов используется единая 3-point rubric (0.0/0.5/1.0),
что даёт интерпретируемое среднее и единую шкалу в отчётах.

#### Критерий 2: Usefulness (0-2)

```
Вопрос: {question}
Ответ системы: {answer}

0 — Бесполезный: не содержит релевантной информации
1 — Частично полезный: некоторая информация есть, но неполный
2 — Полезный: полностью отвечает, конкретен, хорошо структурирован

JSON: {"reasoning": "...", "score": 0|1|2}
```

#### Критерий 3: Key Tool Accuracy (программный, без LLM)

Считается из SSE `tool_invoked` events:

```python
def key_tool_accuracy(predicted_tools: list, expected: dict) -> float:
    """
    Основная метрика tool selection.
    predicted_tools — из SSE tool_invoked events.
    expected — из датасета (key_tools, forbidden_tools, acceptable_alternatives).
    """
    key_tools = set(expected["key_tools"])
    forbidden = set(expected.get("forbidden_tools", []))
    alternatives = set(expected.get("acceptable_alternatives", []))
    predicted_set = set(predicted_tools)

    # Whitelist = key_tools ∪ acceptable_alternatives (binary, без весов)
    whitelist = key_tools | alternatives

    # Hit: agent вызвал хотя бы один tool из whitelist
    hit = 1.0 if (whitelist & predicted_set) else 0.0

    # Forbidden tool penalty: любое нарушение = 0
    if forbidden & predicted_set:
        return 0.0

    return hit
```

### 3.4 Failure Attribution в eval output

Для каждого вопроса классифицировать причину ошибки.

**Trigger**: failure attribution срабатывает если ЛЮБОЕ из:
- `factual_correctness < 0.5` (нормализованная шкала 0-1)
- `usefulness == 0`
- `key_tool_accuracy == 0.0`

Failure types:

```python
class FailureType(str, Enum):
    TOOL_HIDDEN = "tool_hidden"           # key tool не был в visible set
    TOOL_WRONG = "tool_selected_wrong"    # agent выбрал не тот search tool
    TOOL_FAILED = "tool_execution_failed" # tool вернул ошибку
    RETRIEVAL_EMPTY = "retrieval_empty"   # поиск не нашёл релевантных docs
    GENERATION_WRONG = "generation_wrong" # docs найдены, но ответ неверный
    REFUSAL_WRONG = "refusal_wrong"       # должен был отказать, но ответил (или наоборот)
    JUDGE_UNCERTAIN = "judge_uncertain"   # judge не смог оценить
```

**Требует нового SSE event** `step_started` с полем `visible_tools` в `agent_service.py`.
В `_get_step_tools()` уже формируется список видимых tools — нужно эмитить его через SSE
перед каждым шагом агента. Это единственное изменение в agent_service для SPEC-RAG-14.

Логика attribution (триггер: любой из `factual_correctness < 0.5` или `usefulness == 0` или `key_tool_accuracy == 0`):
1. Если key_tool не в visible_tools (из SSE `step_started`) → `TOOL_HIDDEN`
2. Если key_tool в visible, но agent вызвал другой → `TOOL_WRONG`
3. Если tool вызван, но вернул error/empty → `TOOL_FAILED` / `RETRIEVAL_EMPTY`
4. Если retrieval OK, но factual_correctness < 0.5 → `GENERATION_WRONG`
5. Если answerable=false, но agent ответил (или наоборот) → `REFUSAL_WRONG`

### 3.5 Calibration subset

5-10 вопросов помечаются `"calibration": true`. При смене judge-модели (dev→checkpoint→release) эти вопросы прогоняются обоими judge'ами для проверки drift.

### 3.6 Изменения в `evaluate_agent.py`

**Новые capabilities:**

1. **Migration path для формата датасета** — новый loader auto-detect: если есть `key_tools` → golden формат, иначе → legacy (v1/v2/v3). Legacy вопросы получают `key_tools=[], forbidden_tools=[], key_tool_accuracy=None`
2. **Tool sequence tracking** — сбор `tool_invoked` events из SSE в ordered list
3. **LLM judge** — вызов Claude API для factual + usefulness (configurable: `--judge claude|codex|local`)
4. **Key tool accuracy** — программный расчёт из SSE events vs dataset
5. **Failure attribution** — per-question failure type
6. **Новый JSON output** — расширенный формат с judge scores, tool metrics, failures
7. **Markdown report** — обновлённый с таблицей по категориям + failure breakdown
8. **CLI опции**: `--dataset`, `--judge claude|skip`, `--skip-judge` (для быстрых прогонов без judge)

### 3.7 Judge credentials и operational contract

```bash
# Env vars для Claude judge
EVAL_JUDGE_API_KEY=sk-ant-...     # Anthropic API key (из .env, не коммитить)
EVAL_JUDGE_MODEL=claude-sonnet-4-6-20250514  # Default: Sonnet (дешевле Opus для judge)

# Operational
EVAL_JUDGE_TIMEOUT=30             # секунд на один judge call
EVAL_JUDGE_MAX_RETRIES=2          # retry при 429/500
EVAL_JUDGE_RATE_LIMIT_DELAY=2     # секунд между calls (respect rate limits)
```

При `--judge skip` или отсутствии `EVAL_JUDGE_API_KEY` — judge пропускается, scores = None.
Fallback chain: `--judge claude` → API call → retry → skip with warning.

**Структура SSE parsing (уже есть, расширить):**

```python
# Текущее: собираем citation_hits, final_payload, coverage
# Добавить: tools_invoked list из event_name == "tool_invoked"
if event_name == "tool_invoked":
    tool_name = decoded.get("tool") or decoded.get("name")
    tools_invoked.append(tool_name)
```

### 3.8 Формат выходного отчёта

```json
{
    "eval_metadata": {
        "eval_id": "eval_20260324_golden_v1",
        "timestamp": "2026-03-24T15:00:00Z",
        "dataset": "datasets/eval_golden_v1.json",
        "judge_model": "claude-opus-4-6",
        "git_commit": "abc1234",
        "total_questions": 25,
        "duration_sec": 1200
    },
    "aggregate": {
        "recall_at_5": {"mean": 0.72, "by_category": {...}},
        "factual_correctness": {"mean": 0.72, "by_category": {...}},
        "usefulness": {"mean": 1.5, "by_category": {...}},
        "key_tool_accuracy": {"mean": 0.85, "by_category": {...}},
        "error_rate": 0.04,
        "latency": {"p50": 35, "p95": 52, "max": 78}
    },
    "failure_breakdown": {
        "tool_hidden": 1,
        "tool_selected_wrong": 2,
        "retrieval_empty": 1,
        "generation_wrong": 3,
        "refusal_wrong": 0
    },
    "per_question": [...]
}
```

---

## 4. Acceptance Criteria

1. **Датасет**: `eval_golden_v1.json` содержит 20-30 вопросов, покрывающих все 6 категорий, с min 2 вопроса на категорию
2. **Формат**: каждый вопрос содержит `key_tools`, `forbidden_tools`, `expected_answer`
3. **LLM judge**: factual_correctness + usefulness оцениваются через Claude API, результаты записываются в per-question results
4. **Key tool accuracy**: рассчитывается программно из SSE tool_invoked events, mean ≥ 0.70 на golden dataset
5. **Failure attribution**: каждый вопрос где `factual < 0.5` или `usefulness == 0` или `key_tool_accuracy == 0` получает failure type
6. **SSE telemetry**: agent_service эмитит `step_started` event с `visible_tools` list перед каждым шагом
7. **Recall@5**: сохраняется обратная совместимость с текущим fuzzy matching (±5/±50)
8. **Forward-looking**: 2-3 вопроса с `future_tool_flag=true` и `future_key_tools` заполнены
9. **Negative**: 3-4 unanswerable вопроса с `expected_refusal=true`, refusal accuracy ≥ 0.75
10. **Calibration**: 5-10 вопросов помечены `calibration: true`
11. **CLI**: `--dataset`, `--judge`, `--skip-judge` работают, migration path для v1/v2/v3 датасетов
12. **Report**: JSON + Markdown с failure breakdown и per-category metrics
13. **Время прогона**: full eval (25 Qs + judge) ≤ 45 мин, без judge ≤ 20 мин

---

## 5. Чеклист реализации

### Датасет
- [ ] Создать `datasets/eval_golden_v1.json` с 20-30 вопросами нового формата
- [ ] Покрыть все 6 категорий (broad, constrained, compare_summarize, navigation, negative, future)
- [ ] Включить 2-3 forward-looking вопроса (entity_tracker, arxiv_tracker baseline)
- [ ] Включить 3-4 negative/unanswerable вопроса
- [ ] Пометить 5-10 вопросов как calibration
- [ ] Валидировать source_post_ids — проверить что point_ids существуют в news_colbert_v2

### Код — agent_service.py (SSE telemetry)
- [ ] Добавить SSE event `step_started` с `visible_tools` list в `_run_agent_loop()` перед каждым шагом

### Код — evaluate_agent.py
- [ ] Migration path: auto-detect golden vs legacy формат датасета
- [ ] Сбор tool_invoked + visible_tools из SSE events
- [ ] Key tool accuracy calculation (binary whitelist, без весов)
- [ ] LLM judge integration (Claude API): factual (0.0/0.5/1.0) + usefulness (0/1/2)
- [ ] Failure attribution per question (с явными trigger thresholds)
- [ ] Judge credentials: env vars, retry, rate limit, skip fallback
- [ ] Новый JSON output формат
- [ ] Обновлённый Markdown report с failure breakdown
- [ ] CLI: `--judge claude|skip`, `--dataset path`

### Валидация
- [ ] Прогнать eval на golden_v1 с `--skip-judge` — все 25+ вопросов без ошибок
- [ ] Прогнать eval с `--judge claude` — judge scores записываются корректно
- [ ] Проверить failure attribution на 2-3 заведомо "сломанных" вопросах
- [ ] Проверить backward compatibility: `--dataset datasets/eval_dataset_v3.json` работает

### Документация
- [ ] Обновить `agent_context/modules/ingest_eval.md` — новый формат, метрики, CLI
- [ ] Обновить `docs/planning/retrieval_improvement_playbook.md` — добавить результаты golden_v1
- [ ] Запись в `docs/architecture/11-decisions/decision-log.md` — DEC-00XX: eval pipeline v2

---

## 6. Дальнейшие фазы (не в scope этого spec)

### Checkpoint phase (после SPEC-RAG-15)
- Расширить до 100-150 вопросов (hand-crafted + первый раунд synthetic)
- Добавить 10-12 категорий
- Добавить citation grounding criterion
- RSR quick check (k=3,5,10)
- Ablation quick screen (100 Qs × top configs)

### Release phase (портфолио)
- 450-500 вопросов (synthetic pipeline + 20-30% human verification)
- 17 категорий, Qwen local judge + Claude calibration
- Полный robustness suite (NDR + RSR + ROR)
- Ablation deep eval с paired bootstrap tests
- Decompose-then-verify для всех hard вопросов
