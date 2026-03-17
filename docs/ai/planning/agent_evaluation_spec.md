# Agent Evaluation Tool — Specification v0.1 (MVP)

**Статус:** Draft MVP  
**Версия:** 0.1  
**Дата:** 2025-11-21  
**Источник:** Based on `comprehensive_evaluation_strategy_for_the_ReAct_agent.md`

---

## 1. Goal / Scope

### 1.1. Цель MVP v0.1

Реализовать минимальный инструмент автоматизированной оценки ReAct-агента для:
- сбора базовых метрик производительности (latency, coverage, recall@5);
- сравнения агента с baseline QA API;
- подготовки структурированного отчёта для последующего анализа.

**Главное отличие MVP от полной версии:** оценка correctness (точность ответов) и faithfulness (верность цитируемым источникам) выполняется вручную на основе сохранённых результатов. Автоматизация этих метрик планируется в Phase 2.

### 1.2. Что ВХОДИТ в scope MVP v0.1

- Загрузка датасета запросов (`eval_dataset.json`)
- Выполнение запросов через `/v1/agent/stream` и `/v1/qa`
- Автоматический расчёт метрик:
  - Latency (agent, baseline)
  - Agent Coverage (из `final.coverage`)
  - Retrieval Recall@5 (если доступны `top5_hits` из `search` observation)
- Сохранение raw results в JSON для последующего ручного анализа
- Генерация простого отчёта (aggregated metrics + per-query JSON)

### 1.3. Что НЕ ВХОДИТ в scope MVP v0.1 (Phase 2)

- Автоматическая оценка correctness через LLM-judge
- Faithfulness / Citation Precision (проверка каждого утверждения на поддержку источниками)
- Conciseness metrics (длина ответа, избыточность)
- Автоматизированная оценка baseline correctness
- CI/CD интеграция, production monitoring
- Adversarial queries, stress testing

---

## 2. Dataset Schema

Датасет хранится в JSON-файле (`eval_dataset.json`), формат — массив объектов:

```json
[
  {
    "id": 1,
    "query": "Какие важные объявления делал @durov в августе 2023?",
    "category": "temporal+author",
    "expected_answer": "В августе 2023 года Павел Дуров объявил о запуске NFT-аукционов для имён пользователей и о введении истории в профилях.",
    "expected_documents": ["tg-msg-1001", "tg-msg-1005"],
    "answerable": true,
    "notes": null
  },
  {
    "id": 2,
    "query": "Был ли сбой Telegram 1 января 2020?",
    "category": "negative",
    "expected_answer": null,
    "expected_documents": [],
    "answerable": false,
    "notes": "Негативный кейс — нет данных в базе"
  }
]
```

### 2.1. Поля датасета

| Поле                  | Тип              | Обязательно | Описание                                                                                     |
|-----------------------|------------------|-------------|----------------------------------------------------------------------------------------------|
| `id`                  | int \| string    | Да          | Уникальный идентификатор запроса                                                             |
| `query`               | string           | Да          | Текст вопроса пользователя (Russian)                                                         |
| `category`            | string           | Да          | Категория запроса для группировки результатов (см. ниже)                                     |
| `expected_answer`     | string \| null   | Нет         | Референсный ответ для ручной оценки correctness; может быть `null` для сложных/негативных кейсов |
| `expected_documents`  | array[string]    | Да          | Список ID документов (Telegram сообщения), которые должна найти/процитировать система. Пустой массив для негативных кейсов |
| `answerable`          | boolean          | Да          | `true` если на вопрос есть ответ в данных, `false` для негативных кейсов                      |
| `notes`               | string \| null   | Нет         | Дополнительные примечания для ручного анализа                                                 |

### 2.2. Категории запросов

- `temporal` — запросы с временными фильтрами ("за последний месяц", "в августе 2023")
- `channel_filter` — запросы по конкретному каналу ("что обсуждали в @news")
- `author_query` — запросы по автору ("что писал @username")
- `multi-hop` — требуют нескольких шагов reasoning или объединения нескольких документов
- `simple_fact` — простые фактоидные вопросы
- `negative` — вопросы без ответа в данных (тест на "не знаю")

---

## 3. External APIs

### 3.1. Agent API — `/v1/agent/stream`

**Endpoint:** `POST /v1/agent/stream`  
**Content-Type:** `application/json`  
**Response:** Server-Sent Events (SSE) stream

#### 3.1.1. Request Schema (AgentRequest)

```json
{
  "query": "Какие важные объявления делал @durov в августе 2023?",
  "collection": null,
  "model_profile": null,
  "tools_allowlist": null,
  "planner": true,
  "max_steps": 8
}
```

| Поле             | Тип          | Обязательно | Default | Описание                                      |
|------------------|--------------|-------------|---------|-----------------------------------------------|
| `query`          | string       | Да          | —       | Вопрос пользователя (1-1000 символов)         |
| `collection`     | string\|null | Нет         | null    | Коллекция для использования                   |
| `model_profile`  | string\|null | Нет         | null    | Профиль модели                                |
| `tools_allowlist`| array\|null  | Нет         | null    | Разрешённые инструменты                       |
| `planner`        | boolean      | Нет         | true    | Использовать query planner                    |
| `max_steps`      | int          | Нет         | 8       | Макс. шагов reasoning (1-15)                  |

**Для MVP:** используем только `query`, остальные поля оставляем в default значениях (или можем передать `max_steps: 8`).

#### 3.1.2. Response — SSE Events

Агент возвращает поток событий в формате SSE. Каждое событие имеет:
- `event`: тип события
- `data`: JSON-объект с данными события

**Типы событий:**
- `step_started` — начало нового шага reasoning
- `thought` — мысль агента (текст)
- `tool_invoked` — вызов инструмента
- `observation` — результат выполнения инструмента
- `citations` — список цитат (в процессе формирования)
- `token` — стриминг токенов финального ответа
- `final` — итоговое событие с полным ответом и метаданными

#### 3.1.3. Что извлекать для MVP

**Из события `final`:**
```json
{
  "type": "final",
  "data": {
    "answer": "...",
    "citations": [
      {"id": "doc-123", "index": 1, "metadata": {...}},
      ...
    ],
    "coverage": 0.85,
    "refinements": 0,
    "verification": {...},
    "step": 6,
    "request_id": "..."
  }
}
```

Извлекаем:
- `answer` (string) — финальный ответ агента
- `citations` (array) — список процитированных документов с ID
- `coverage` (float) — citation coverage агента (docs_cited / docs_fetched)
- `refinements` (int) — количество refinement раундов (для диагностики)
- `verification` (object) — результат verify tool (опционально, можем логировать)

**Из события `observation` (для tool=search):**
```json
{
  "type": "observation",
  "data": {
    "tool": "search",
    "result": {
      "hits": [
        {"id": "doc-1", "score": 0.95, "metadata": {...}},
        {"id": "doc-2", "score": 0.89, ...},
        ...
      ],
      "took_ms": 234
    }
  }
}
```

Извлекаем:
- `hits` (array) — top-N результатов поиска, нужны первые 5 ID для расчёта `recall@5`

**Latency:**  
Считать на клиенте как `wall_clock_time = time_end - time_start` (от отправки запроса до получения `final` события).

---

### 3.2. Baseline API — `/v1/qa`

**Endpoint:** `POST /v1/qa`  
**Content-Type:** `application/json`  
**Response:** JSON

#### 3.2.1. Request Schema (QARequest)

```json
{
  "query": "Какие важные объявления делал @durov в августе 2023?",
  "include_context": false,
  "collection": null
}
```

| Поле              | Тип          | Обязательно | Default | Описание                              |
|-------------------|--------------|-------------|---------|---------------------------------------|
| `query`           | string       | Да          | —       | Вопрос пользователя                   |
| `include_context` | boolean      | Нет         | false   | Включить ли контекст в ответ          |
| `collection`      | string\|null | Нет         | null    | Коллекция                             |

**Для MVP:** используем только `query`.

#### 3.2.2. Response Schema (QAResponse)

```json
{
  "answer": "В августе 2023 года Павел Дуров...",
  "query": "Какие важные объявления делал @durov в августе 2023?"
}
```

Извлекаем:
- `answer` (string) — ответ baseline системы

**Latency:** считать на клиенте `wall_clock_time`.

---

## 4. Metrics

### 4.1. MVP v0.1 Metrics (реализуем сейчас)

#### 4.1.1. Agent Latency (`agent_latency_sec`)

**Определение:** Время отклика агента от отправки запроса до получения финального ответа (wall-clock).

**Метод:** 
- `t_start = time.time()` перед запросом
- `t_end = time.time()` после получения события `final`
- `agent_latency_sec = t_end - t_start`

**Агрегация:** mean, p95, max для всех запросов.

**Критерий успеха:** p95 < 30s, mean < 10s.

---

#### 4.1.2. Baseline Latency (`baseline_latency_sec`)

**Определение:** Время отклика baseline QA API (wall-clock).

**Метод:** аналогично agent latency.

**Агрегация:** mean, p95, max.

**Применение:** сравнение с агентом для оценки overhead.

---

#### 4.1.3. Agent Coverage (`agent_coverage`)

**Определение:** Citation coverage агента — доля процитированных документов от общего числа извлечённых документов (внутренняя метрика агента).

**Источник:** поле `coverage` из события `final`.

**Формула (считается внутри агента):**  
`coverage = len(cited_docs) / len(fetched_docs)`

**Интерпретация:**  
- Высокое значение (≥80%) — агент использовал большую часть найденных источников
- Низкое значение — агент отфильтровал часть найденных документов (возможно, нерелевантных)

**Агрегация:** mean, min, max по всем запросам.

---

#### 4.1.4. Retrieval Recall@5 (`recall_at_5`)

**Определение:** Доля ожидаемых документов (`expected_documents`), которые попали в топ-5 результатов поиска агента.

**Формула:**  
```
recall@5 = |expected_documents ∩ top5_hits| / |expected_documents|
```

**Метод:**
1. Из события `observation` для `search` tool извлечь первые 5 ID документов (`top5_hits`)
2. Сравнить с `expected_documents` из датасета
3. Посчитать пересечение

**Особенности:**
- Если `expected_documents` пуст (негативный кейс), `recall@5 = N/A`
- Если не удалось извлечь `top5_hits` из SSE, метрика помечается как `null` (можно реализовать позже)

**Агрегация:** mean по всем answerable запросам.

**Критерий успеха:** ≥80% (в идеале близко к 100%).

---

#### 4.1.5. Correctness (`agent_correct`, `baseline_correct`)

**Определение:** Бинарный флаг или скор (0/1), отражающий правильность ответа.

**Метод в MVP v0.1:** **РУЧНОЕ ЗАПОЛНЕНИЕ**.
- Скрипт сохраняет поля `agent_correct: null`, `baseline_correct: null` в raw results
- Аналитик вручную проверяет ответы и обновляет JSON:
  - `agent_correct: true` — ответ полностью верный
  - `agent_correct: false` — ответ неверный или неполный
  - `agent_correct: null` — не проверено

**Применение:** для расчёта accuracy после ручной валидации:
```
accuracy = sum(correct == true) / total_queries
```

**Phase 2:** автоматизация через LLM-judge (GPT-4 или аналог).

---

### 4.2. Phase 2 Metrics (НЕ реализуем в MVP v0.1)

Следующие метрики планируются для Phase 2 (после MVP), их **НЕ нужно реализовывать сейчас**:

#### 4.2.1. Answer Accuracy via LLM-judge

Автоматическая оценка correctness через LLM (GPT-4):
- Промпт: "Question: {Q}, Expected: {E}, Agent: {A} — Is correct?"
- Возвращает: Correct / Partially Correct / Incorrect

#### 4.2.2. Faithfulness / Citation Precision

**Формула:** `supported_claims / total_claims`

**Метод:**
1. Разбить `answer` на утверждения (claims)
2. Для каждого claim проверить поддержку в `citations` (через NLI model или substring match)
3. Посчитать долю supported

**Критерий:** ≥95% (почти все утверждения поддержаны источниками).

#### 4.2.3. Citation Coverage (Document-level)

**Формула:** `|expected_docs ∩ cited_docs| / |expected_docs|`

Доля ожидаемых документов, которые агент процитировал в финальном ответе.

#### 4.2.4. Conciseness

Метрики длины ответа (tokens/chars), проверка избыточности, сравнение с baseline.

#### 4.2.5. Advanced Diagnostics

- Количество шагов reasoning (steps)
- Использование refinements
- Верификация confidence
- Patterns по категориям запросов

---

## 5. Evaluation Pipeline (MVP v0.1)

### 5.1. Step 1: Load Dataset

**Входной файл:** `eval_dataset.json` (см. раздел 2)

**Действие:**
- Прочитать JSON
- Валидировать структуру (наличие обязательных полей)
- Загрузить в список `List[EvalItem]`

**Структура `EvalItem` (internal):**
```python
{
    "id": "1",
    "query": "...",
    "category": "temporal",
    "expected_answer": "...",
    "expected_documents": ["doc-1", "doc-2"],
    "answerable": True,
    "notes": None
}
```

---

### 5.2. Step 2: Execute Queries

**Для каждого `item` из датасета:**

#### 5.2.1. Call Agent API

1. Подготовить `AgentRequest`:
   ```python
   {
       "query": item.query,
       "max_steps": 8
   }
   ```

2. Отправить `POST /v1/agent/stream`

3. Обработать SSE stream:
   - Слушать события
   - Извлечь `observation` для `search` tool → сохранить `top5_hits`
   - Дождаться события `final` → сохранить `answer`, `citations`, `coverage`, `refinements`
   - Измерить `latency_sec`

4. Сохранить в `AgentResult`:
   ```python
   {
       "answer": "...",
       "citations": [{...}],
       "coverage": 0.85,
       "refinements": 0,
       "top5_hits": ["doc-1", "doc-3", ...],
       "latency_sec": 8.2,
       "verification": {...}  # опционально
   }
   ```

#### 5.2.2. Call Baseline API

1. Подготовить `QARequest`:
   ```python
   {"query": item.query}
   ```

2. Отправить `POST /v1/qa`

3. Получить ответ:
   ```python
   {
       "answer": "...",
       "query": "..."
   }
   ```

4. Измерить `latency_sec`

5. Сохранить в `BaselineResult`:
   ```python
   {
       "answer": "...",
       "latency_sec": 3.1
   }
   ```

#### 5.2.3. Combine Results

Сохранить в `raw_results`:
```python
{
    "query_id": item.id,
    "query": item.query,
    "category": item.category,
    "expected_documents": item.expected_documents,
    "answerable": item.answerable,
    "agent": AgentResult,
    "baseline": BaselineResult,
    "notes": item.notes
}
```

---

### 5.3. Step 3: Compute Metrics (MVP v0.1)

**Для каждого `raw_result` вычислить:**

#### 5.3.1. Latency
- `agent_latency_sec` = из `agent.latency_sec`
- `baseline_latency_sec` = из `baseline.latency_sec`

#### 5.3.2. Agent Coverage
- `agent_coverage` = из `agent.coverage`

#### 5.3.3. Recall@5
```python
if raw_result.answerable and raw_result.expected_documents:
    top5 = set(raw_result.agent.top5_hits[:5])
    expected = set(raw_result.expected_documents)
    recall_at_5 = len(expected & top5) / len(expected)
else:
    recall_at_5 = None
```

#### 5.3.4. Correctness (заглушка)
```python
agent_correct = None  # заполняется вручную
baseline_correct = None  # заполняется вручную
```

**Результат Step 3:**  
Обогащённый `raw_result` с вычисленными метриками:
```python
{
    ...raw_result,
    "metrics": {
        "agent_latency_sec": 8.2,
        "baseline_latency_sec": 3.1,
        "agent_coverage": 0.85,
        "recall_at_5": 1.0,
        "agent_correct": None,
        "baseline_correct": None
    }
}
```

---

### 5.4. Step 4: Aggregate Metrics

**Агрегация по всем запросам:**

```python
{
    "total_queries": 25,
    "answerable_queries": 23,
    "negative_queries": 2,
    
    "latency": {
        "agent": {
            "mean": 8.4,
            "p95": 12.3,
            "max": 15.0
        },
        "baseline": {
            "mean": 3.1,
            "p95": 4.5,
            "max": 5.2
        }
    },
    
    "coverage": {
        "mean": 0.87,
        "min": 0.65,
        "max": 1.0
    },
    
    "recall_at_5": {
        "mean": 0.92,
        "queries_with_full_recall": 21,
        "queries_with_partial_recall": 2
    },
    
    "correctness": {
        "agent_validated": 0,  # количество вручную проверенных
        "baseline_validated": 0
    },
    
    "by_category": {
        "temporal": {...},
        "channel_filter": {...},
        ...
    }
}
```

---

### 5.5. Step 5: Output

#### 5.5.1. Save Raw Results

**Файл:** `results/raw/eval_results_{timestamp}.json`

**Формат:** массив per-query results:
```json
[
  {
    "query_id": "1",
    "query": "...",
    "category": "temporal",
    "expected_documents": ["doc-1", "doc-2"],
    "answerable": true,
    "agent": {
      "answer": "...",
      "citations": [...],
      "coverage": 0.85,
      "top5_hits": [...],
      "latency_sec": 8.2
    },
    "baseline": {
      "answer": "...",
      "latency_sec": 3.1
    },
    "metrics": {
      "agent_latency_sec": 8.2,
      "baseline_latency_sec": 3.1,
      "agent_coverage": 0.85,
      "recall_at_5": 1.0,
      "agent_correct": null,
      "baseline_correct": null
    },
    "notes": null
  },
  ...
]
```

#### 5.5.2. Save Aggregated Report

**Файл:** `results/reports/eval_report_{timestamp}.json`

**Формат:** aggregated metrics (см. Step 4).

#### 5.5.3. (Optional) Generate Markdown Summary

**Файл:** `results/reports/eval_report_{timestamp}.md`

**Пример:**
```markdown
# Agent Evaluation Report
**Date:** 2025-11-21 15:30:00
**Dataset:** eval_dataset.json (25 queries)

## Overall Metrics
- **Agent Latency:** mean=8.4s, p95=12.3s
- **Baseline Latency:** mean=3.1s, p95=4.5s
- **Agent Coverage:** mean=87%
- **Recall@5:** mean=92%

## By Category
| Category       | Queries | Avg Latency | Avg Coverage | Recall@5 |
|----------------|---------|-------------|--------------|----------|
| temporal       | 5       | 9.2s        | 0.90         | 1.0      |
| channel_filter | 5       | 8.1s        | 0.85         | 0.8      |
| ...            | ...     | ...         | ...          | ...      |

## Next Steps
- Manually validate `agent_correct` / `baseline_correct` in raw results
- Compute accuracy after validation
- Proceed to Phase 2 implementation (LLM-judge, faithfulness)
```

---

## 6. Output Format (MVP v0.1)

### 6.1. Per-Query Result (JSON)

См. раздел 5.5.1 выше.

**Ключевые поля:**
- `query_id`, `query`, `category`, `expected_documents`, `answerable`
- `agent.*` — все данные от агента
- `baseline.*` — все данные от baseline
- `metrics.*` — вычисленные метрики MVP

**Использование:**  
Аналитик открывает JSON, просматривает `agent.answer` и `baseline.answer`, сравнивает с `expected_answer` (если есть в датасете), вручную ставит `metrics.agent_correct: true/false`.

---

### 6.2. Aggregated Report (JSON)

См. раздел 5.4 и 5.5.2 выше.

**Использование:**  
Быстрый overview производительности агента и baseline для:
- оценки latency overhead
- диагностики retrieval (recall@5)
- анализа coverage
- группировки по категориям

---

### 6.3. Markdown Summary (Optional)

Простой human-readable отчёт для команды (см. 5.5.3).

---

## 7. Implementation Notes

### 7.1. SSE Parsing

Для обработки SSE stream от `/v1/agent/stream` рекомендуется:
- Использовать библиотеку `sseclient-py` или `httpx` с `stream=True`
- Парсить события построчно:
  ```
  event: final
  data: {"answer": "..."}
  ```
- Для каждого события декодировать JSON из `data`

### 7.2. Error Handling

- Если агент вернул `fallback: true` в `final` event, пометить результат как `fallback`
- Если агент/baseline завершился с ошибкой, сохранить `error: true` и `error_message`
- Пропустить вычисление метрик для failed запросов

### 7.3. Timeouts

- Установить timeout для агента: 60s (больше чем ожидаемый p95=30s)
- Установить timeout для baseline: 30s
- При timeout пометить результат соответственно

### 7.4. Logging

Логировать:
- Начало/конец обработки каждого запроса
- Latency для каждого запроса
- Любые ошибки/warnings
- Итоговую статистику после завершения

---

## 8. Success Criteria (MVP v0.1)

### 8.1. Технические критерии

- [x] Скрипт успешно выполняется на датасете из 20+ запросов
- [x] Все метрики MVP корректно вычисляются
- [x] Raw results сохраняются в JSON
- [x] Aggregated report генерируется автоматически

### 8.2. Метрики (после ручной валидации correctness)

Эти критерии будут проверены после того, как аналитик заполнит `agent_correct` вручную:
- Agent accuracy ≥70% (target из исследования)
- Agent p95 latency <30s
- Recall@5 ≥80%

---

## 9. Next Steps (Phase 2)

После успешной реализации MVP v0.1:

1. **LLM-judge для correctness:**  
   Автоматизировать оценку `agent_correct` / `baseline_correct` через GPT-4 API

2. **Faithfulness calculation:**  
   Реализовать claim extraction + support verification (NLI model или LLM)

3. **Citation Coverage (document-level):**  
   Проверка `|expected_docs ∩ cited_docs| / |expected_docs|`

4. **Advanced diagnostics:**  
   Анализ patterns по категориям, failure mode classification

5. **CI/CD Integration:**  
   Интеграция в pipeline для регрессионного тестирования при изменениях агента

---

## 10. References

- Source research: `docs/ai/research/comprehensive_evaluation_strategy_for_the_ReAct_agent.md`
- Agent schema: `src/schemas/agent.py`
- Agent endpoint: `src/api/v1/endpoints/agent.py`
- QA schema: `src/schemas/qa.py`
- QA endpoint: `src/api/v1/endpoints/qa.py`

---

**End of Specification v0.1 (MVP)**
