# SPEC-RAG-18: Golden v2 + Offline Judge Artifacts

> **Статус**: Draft → Review
> **Создан**: 2026-03-30
> **Research basis**: R18-deep-evaluation-methodology-dataset, R19-deep-nli-citation-faithfulness, R20-deep-retrieval-robustness-ndr-rsr-ror, R22-deep-production-gap-analysis
> **Depends on**: SPEC-RAG-14 (eval pipeline v2), SPEC-RAG-16 (hot_topics + channel_expertise), SPEC-RAG-17 (production hardening)
> **Scope**: новый golden dataset, redesign retrieval metric, offline judge packet, batch review workflow

---

## 1. Цель

Заменить текущий `golden_v1` как единственный eval baseline на **golden_v2**, который:

1. покрывает **все текущие runtime capabilities**, включая `hot_topics` и `channel_expertise`
2. перестаёт использовать `strict source_post_ids` как primary retrieval KPI
3. сохраняет **полный artifact для offline judging** в чате (Codex / Claude), без judge API
4. поддерживает review **батчами по 30 вопросов**

Ключевой принцип:

- **Primary metrics**: `key_tool_accuracy`, `factual_correctness`, `usefulness`
- **Retrieval metric**: judge-based `retrieval_sufficiency / evidence_support`
- **Legacy strict recall**: остаётся только как diagnostic metric

---

## 2. Проблема

### 2.1 Что сломано в `golden_v1`

Текущий датасет `datasets/eval_golden_v1.json` содержит 30 вопросов, но:

1. **Не покрывает новые tools**
   - нет вопросов для `hot_topics`
   - нет вопросов для `channel_expertise`

2. **`Recall@5` как primary KPI вводит в заблуждение**
   - dataset использует `source_post_ids` как жёсткий якорь
   - многие ответы можно корректно собрать по другим документам
   - часть выбранных `source_post_ids` были взяты как удобные anchor docs, а не как единственно допустимые evidence docs
   - для analytics / navigation / refusal вопросов retrieval recall либо бессмысленен, либо `N/A`

3. **Judge workflow не соответствует реальному процессу**
   - на практике review идёт не через API, а через чат с Codex / Claude
   - значит eval должен сохранять **всё необходимое для offline review**

### 2.2 Что считать правильным outcome

Для production-grade baseline нам нужен не “совпал ли агент с заранее придуманным post_id”, а:

1. выбрал ли агент правильный tool / path
2. дал ли фактически корректный ответ
3. полезен ли ответ пользователю
4. достаточно ли retrieved/cited evidence, чтобы такой ответ был обоснован

---

## 3. Решение

### 3.1 Новый metric stack

#### Primary

- `key_tool_accuracy`
- `factual_correctness`
- `usefulness`
- `failure_type`

#### Retrieval / grounding

- `retrieval_sufficiency_score` — можно ли на retrieved/cited docs дать такой answer
- `evidence_support_score` — подтверждают ли cited docs ключевые claims ответа
- `acceptable_set_hit` — если вопрос размечен через допустимые evidence sets

#### Diagnostic only

- `strict_anchor_recall`

`strict_anchor_recall` больше **не используется** как headline quality metric.

### 3.2 Eval modes

Каждый вопрос в `golden_v2` получает явный `eval_mode`.

Поддерживаемые режимы:

| eval_mode | Когда использовать | Primary checks |
|-----------|--------------------|----------------|
| `retrieval_evidence` | factual / temporal / channel factual QA | tool, factual, usefulness, retrieval_sufficiency, evidence_support |
| `analytics` | entity_tracker, arxiv_tracker, hot_topics, channel_expertise | tool, factual, usefulness, expected payload usage |
| `navigation` | list_channels / metadata routing | tool, refusal/no-search behavior, usefulness |
| `refusal` | out-of-database / false presupposition / out-of-timerange | refusal correctness, no hallucinated answer |

---

## 4. Формат датасета `golden_v2`

Новый файл: `datasets/eval_golden_v2.json`

### 4.1 Базовая схема

```json
{
  "id": "golden_v2_q01",
  "version": "2.0",
  "query": "Что обсуждали на прошлой неделе?",
  "category": "analytics_hot_topics",
  "difficulty": "medium",
  "calibration": false,
  "answerable": true,
  "expected_refusal": false,
  "eval_mode": "analytics",

  "key_tools": ["hot_topics"],
  "forbidden_tools": ["search", "temporal_search"],
  "acceptable_alternatives": [],

  "expected_answer": "На прошлой неделе активно обсуждались ...",
  "required_claims": [
    "Ответ должен описывать корректный период",
    "Ответ должен назвать 2-5 ключевых тем или сущностей"
  ],

  "expected_entities": ["OpenAI", "Claude"],
  "expected_topics": ["GPT-5.4", "Codex"],
  "expected_channels": [],

  "source_post_ids": [],
  "acceptable_evidence_sets": [],
  "strict_anchor_recall_eligible": false,

  "notes": "analytics tool, recall не считается",
  "metadata": {
    "created_at": "2026-03-30",
    "created_by": "human+codex",
    "tags": ["hot_topics", "weekly_digest"]
  }
}
```

### 4.2 Новые поля

| Поле | Смысл |
|------|-------|
| `eval_mode` | тип оценки вопроса |
| `required_claims` | ключевые утверждения, которые должен покрывать ответ |
| `expected_entities` | полезно для analytics / trend questions |
| `expected_topics` | полезно для `hot_topics` |
| `expected_channels` | полезно для `channel_expertise` / `list_channels` |
| `acceptable_evidence_sets` | 1+ допустимых наборов evidence docs |
| `strict_anchor_recall_eligible` | считать ли legacy strict recall |
| `calibration` | вопрос входит в стабильный subset для judge drift / cross-run comparison |

### 4.3 Что сохраняется из `golden_v1`

- `key_tools`
- `forbidden_tools`
- `acceptable_alternatives`
- `expected_refusal`
- `expected_answer`
- `source_post_ids`
- `calibration`

### 4.4 Что НЕ переносится из `golden_v1`

Поля `future_tool_flag` и `future_key_tools` в `golden_v2` **удаляются**.

Причина:
- они были переходными маркерами до реализации analytics/domain tools
- после SPEC-RAG-15/16/17 все runtime tools уже существуют
- в `golden_v2` их роль заменяется явными `eval_mode`, `key_tools` и `required_claims`

---

## 5. Retrieval metric redesign

### 5.1 Что удаляется из роли primary KPI

Старый `Recall@5` больше не headline-метрика в aggregate report.

Причина:
- для ряда вопросов существует несколько валидных evidence paths
- analytics/navigation/refusal не должны портить retrieval KPI

### 5.2 Что вводится вместо этого

#### A. `acceptable_set_hit`

Программная метрика.

Если `acceptable_evidence_sets` заполнен, агент получает hit, если его `citation_hits`
покрывают **любой** допустимый set.

Пример:

```json
"acceptable_evidence_sets": [
  ["ai_newz:4355"],
  ["other_channel:1234", "other_channel:1235"]
]
```

#### B. `retrieval_sufficiency_score`

Judge-based метрика.

Вопрос к judge:

> “Достаточны ли retrieved/cited документы агента, чтобы на их основе выдать такой ответ?”

Шкала:
- `0.0` — документы недостаточны
- `0.5` — частично достаточны
- `1.0` — достаточны

#### C. `evidence_support_score`

Judge-based метрика.

Judge оценивает `required_claims`:
- `supported`
- `partially_supported`
- `unsupported`

Агрегат:
- `0.0 / 0.5 / 1.0`

### 5.3 Что остаётся как diagnostic

`strict_anchor_recall`

Он считается **только** для вопросов, где:
- `strict_anchor_recall_eligible = true`
- `source_post_ids` реально meaningful

В aggregate report эта метрика должна быть вынесена в раздел `diagnostic`.

---

## 6. Offline judge workflow

### 6.1 Ограничение

Judge **не вызывается через API** как обязательная часть eval run.

Основной workflow:

1. локальный прогон агента
2. сохранение полного artifact-а
3. offline review в чате с Codex / Claude
4. review батчами по 30 вопросов

### 6.2 Что должен сохранять eval runner

На каждый вопрос сохраняется **offline judging packet**:

```json
{
  "query_id": "golden_v2_q01",
  "query": "...",
  "eval_mode": "retrieval_evidence",

  "answer": "...",
  "status": "ok",
  "latency_sec": 18.4,
  "coverage": 0.74,

  "tools_invoked": ["query_plan", "temporal_search", "rerank", "compose_context"],
  "visible_tools_history": [
    ["query_plan", "search", "temporal_search"],
    ["rerank", "compose_context", "final_answer"]
  ],
  "agent_thoughts": [
    "Сначала ограничу период мартом 2026 ...",
    "После rerank у меня достаточно контекста для ответа ..."
  ],

  "tool_observations": [
    {"tool": "temporal_search", "summary": "Found 10 docs"},
    {"tool": "compose_context", "summary": "coverage=0.74"}
  ],

  "citations": [
    {
      "channel": "ai_newz",
      "message_id": 4355,
      "url": "...",
      "text_excerpt": "Meta купила Manus AI за 2 млрд ..."
    }
  ],

  "citation_hits": ["ai_newz:4355"],
  "retrieved_docs": [
    {
      "id": "ai_newz:4355",
      "channel": "ai_newz",
      "message_id": 4355,
      "text_excerpt": "...",
      "score": 0.91
    }
  ],

  "dataset_contract": {
    "key_tools": ["search"],
    "forbidden_tools": ["list_channels"],
    "required_claims": ["..."],
    "acceptable_evidence_sets": [["ai_newz:4355"]],
    "expected_answer": "..."
  }
}
```

`agent_thoughts` сохраняются компактно:
- первые ~200 символов каждого `thought` event
- без скрытых reasoning blocks / служебного мусора

### 6.3 Что runner строит сам, без новых SSE событий

`SPEC-RAG-18` не требует расширения SSE контракта для eval.

Runner должен извлекать данные из уже существующих событий:

- `tool_observations` — из SSE `observation`
- `agent_thoughts` — из SSE `thought`
- `retrieved_docs` — из `observation` только для search-type tools:
  - `search`
  - `temporal_search`
  - `channel_search`
  - при необходимости `related_posts`

То есть:
- **новые runtime события не добавляются**
- логика извлечения живёт в `evaluate_agent.py`

### 6.4 Что judge может и не может делать

#### Можно

- смотреть `retrieved_docs`
- смотреть `citations`
- смотреть `required_claims`
- смотреть `expected_answer`

#### Нельзя

- делать новый свободный semantic search по корпусу
- придумывать новый retrieval path вместо агента

Reason:
- иначе judge оценивает уже не агента, а свои retrieval способности

### 6.5 Batch workflow

Eval run разбивается на judge batches:

- `judge_batch_01.md` / `json` — вопросы 1-30
- `judge_batch_02.md` / `json` — 31-60
- ...

Target:
- до 30 вопросов на один offline review pass

---

## 7. Изменения в `evaluate_agent.py`

### 7.1 Dataset loader

Нужно поддержать:
- `datasets/eval_golden_v1.json`
- `datasets/eval_golden_v2.json`

Loader должен понимать новые поля:
- `eval_mode`
- `required_claims`
- `expected_entities`
- `expected_topics`
- `expected_channels`
- `acceptable_evidence_sets`
- `strict_anchor_recall_eligible`

### 7.2 Что нужно дополнительно собрать из agent run

Уже есть:
- `tools_invoked`
- `visible_tools_history`
- `citation_hits`
- `coverage`

Нужно добавить:
- `tool_observations` (из existing SSE `observation`)
- `citations` с компактными excerpt-ами
- `retrieved_docs` / `reranked_docs` в компактном формате
- `dataset_contract` в per-question output
- `agent_thoughts`

Важно:
- `retrieved_docs` извлекаются только для search-type tools
- analytics/navigation/refusal вопросы могут не иметь `retrieved_docs`, и это корректно

### 7.3 Aggregate output

Новый aggregate report должен иметь 3 раздела:

1. `primary`
   - `key_tool_accuracy`
   - `factual_correctness`
   - `usefulness`
   - `failure_breakdown`

2. `retrieval_grounding`
   - `acceptable_set_hit`
   - `retrieval_sufficiency_score`
   - `evidence_support_score`

3. `diagnostic`
   - `strict_anchor_recall`
   - `coverage`
   - latency

`retrieval_sufficiency_score` и `evidence_support_score`:
- в **автоматическом прогоне** имеют значение `null`
- заполняются **только после offline judge pass**
- aggregate mean считается только по `non-null` значениям

То есть runner обязан корректно поддерживать двухфазную модель:
1. automatic run
2. offline judge enrichment

### 7.4 Offline judge export

Отдельный export mode:

```bash
python scripts/evaluate_agent.py \
  --dataset datasets/eval_golden_v2.json \
  --export-offline-judge
```

Результат:
- raw JSON report
- summary Markdown
- batch files для offline judging

---

## 8. Состав `golden_v2`

### 8.1 Минимальный объём

Phase 1 MVP:

- `golden_v2` должен содержать **не меньше 36 вопросов**

Target after next expansion step:

- **100 вопросов** после synthetic / semi-synthetic расширения и audit pass

### 8.2 Обязательное покрытие

| Блок | Кол-во |
|------|--------|
| `retrieval_evidence` | 18-20 |
| `analytics` | 8-10 |
| `navigation` | 3-4 |
| `refusal` | 5-6 |

### 8.3 Новые обязательные capability questions

Нужно минимум:

- `hot_topics`: 3 вопроса
- `channel_expertise`: 3 вопроса

Примеры:

1. “Что обсуждали на прошлой неделе?”
2. “Какие горячие темы были в марте 2026?”
3. “Какие темы были самыми горячими в неделю 2026-W12?”
4. “Какие каналы эксперты по NLP?”
5. “Кто лучше пишет про робототехнику и роботакси?”
6. “Профиль канала gonzo_ml: в чём его экспертиза?”

---

## 9. Acceptance criteria

1. Создан `datasets/eval_golden_v2.json` с минимум 36 вопросами
2. В датасете есть минимум 3 вопроса на `hot_topics`
3. В датасете есть минимум 3 вопроса на `channel_expertise`
4. `strict_anchor_recall` больше не headline metric в aggregate report
5. Введены `acceptable_evidence_sets` и `strict_anchor_recall_eligible`
6. Runner сохраняет полный offline judging packet
7. Runner умеет экспортировать judge batches по 30 вопросов
8. Judge может оценить `retrieval_sufficiency_score` и `evidence_support_score` без новых запросов в БД
9. `evaluate_agent.py` остаётся backward-compatible с `golden_v1`
10. Итоговый report разделяет `primary`, `retrieval_grounding`, `diagnostic`

---

## 10. Чеклист реализации

### Dataset
- [ ] Создать `datasets/eval_golden_v2.json`
- [ ] Перенести полезные вопросы из `golden_v1`
- [ ] Добавить новые `hot_topics` / `channel_expertise` cases
- [ ] Разметить `eval_mode`
- [ ] Разметить `required_claims`
- [ ] Добавить `acceptable_evidence_sets` там, где это возможно
- [ ] Явно пометить `strict_anchor_recall_eligible`
- [ ] Сохранить `calibration`
- [ ] Не переносить deprecated `future_tool_flag` / `future_key_tools`

### Eval runner
- [ ] Расширить dataclass `EvalItem`
- [ ] Поддержать новый loader format
- [ ] Сохранять `tool_observations` из existing SSE `observation`
- [ ] Сохранять compact `citations` with excerpts
- [ ] Сохранять compact `retrieved_docs` только для search-type tools
- [ ] Сохранять `agent_thoughts`
- [ ] Разделить aggregate metrics на 3 группы
- [ ] Добавить offline judge export / batch export
- [ ] Корректно обрабатывать `null` judge-grounding metrics до offline review pass

### Judge protocol
- [ ] Отдельный prompt template для `retrieval_sufficiency`
- [ ] Отдельный prompt template для `evidence_support`
- [ ] JSON verdict schema для offline review
- [ ] Batch size = 30

### Docs
- [ ] Обновить `docs/progress/experiment_log.md`
- [ ] Обновить `docs/progress/project_scope.md`
- [ ] Добавить запись в decision log после имплементации

---

## 11. Не входит в scope

- runtime NLI verification
- свободный judge search по векторной БД
- synthetic 500-question dataset
- RSR/NDR/ROR robustness
- GPT-4o online judge integration
- auto-labeling claims из answer на этапе runtime

Это отдельные следующие шаги после `golden_v2`.
