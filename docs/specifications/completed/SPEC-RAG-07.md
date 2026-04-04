# SPEC-RAG-07 — Composite Coverage Metric

**Статус:** Draft
**Зависимости:** SPEC-RAG-01 (settings), SPEC-RAG-04 (Candidate.dense_score)
**Закрывает:** DEC-0018, DEC-0019, OPEN-07
**Суперсидирует:** DEC-0003 (coverage_threshold=0.8, max_refinements=1)

---

## 0. Контекст и цель

### 0.1 Проблема

Текущий `citation_coverage` в `src/services/tools/compose_context.py:85`:

```python
citation_coverage = len(citations) / len(docs) if docs else 1.0
```

Это ratio включённых документов, а не мера информационного покрытия запроса:
- Значение `1.0` достигается при любом количестве docs, даже если все они нерелевантны.
- Порог `0.8` завышен для сложных многоаспектных запросов → ложные refinement-ы.
- RRF-score (`max ≈ 0.0328`) не интерпретируем; `dense_score` — интерпретируем (0–1).

### 0.2 Решение

DEC-0018 (2026-03-16): заменить наивный ratio пятисигнальной взвешенной формулой на базе cosine similarity.
DEC-0019 (2026-03-16): понизить порог до `0.65`, увеличить `max_refinements` до `2`.

**Источники:** `docs/research/reports/R04-coverage-metrics.md`, `R00-synthesis.md` (ADR-003).

---

## 1. Формула

```
coverage = min(1.0,
    0.25 * max_sim              # максимальное cosine_sim среди всех docs
  + 0.20 * mean_top_k           # среднее cosine_sim по top-5
  + 0.20 * term_coverage        # доля терминов запроса в текстах docs
  + 0.15 * doc_count_adequacy   # min(1.0, релевантных_docs / target_k)
  + 0.15 * score_gap            # 1 - нормированный разброс в top-5
  + 0.05 * above_threshold_ratio  # доля всех docs выше relevance_threshold
)
```

Параметры по умолчанию (из R04):
- `relevance_threshold = 0.55` — порог cosine_sim для «релевантного» документа
- `target_k = 5` — ожидаемое количество результатов

**Примечание:** `coverage_threshold` (для trigger refinement) = `0.65` — это НЕ тот же параметр, что `relevance_threshold` (внутри формулы). Первый — в settings, второй — константа в `_compute_coverage`.

### 1.1 Семантика сигналов

| Сигнал | Значение |
|---|---|
| `max_sim` | Есть ли хоть один хорошо-подходящий документ? |
| `mean_top_k` | Насколько хорошо top-5 в среднем? |
| `term_coverage` | Покрыты ли ключевые термины запроса? |
| `doc_count_adequacy` | Достаточно ли число релевантных документов? |
| `score_gap` | Равномерна ли релевантность (нет ли единственного хита)? |
| `above_threshold_ratio` | Какова доля «годных» документов среди всех? |

### 1.2 Калибровочные ориентиры (R04)

| Диапазон cosine_sim | Интерпретация |
|---|---|
| ≥ 0.85 | Парафраз / прямой ответ |
| 0.75–0.85 | Сильное совпадение |
| 0.60–0.75 | Умеренное совпадение |
| 0.45–0.60 | Тангенциальное совпадение |
| < 0.45 | Нерелевантно |

### 1.3 Граничные случаи

| Условие | Поведение |
|---|---|
| `docs = []` | `coverage = 0.0` |
| `max(dense_score) < 0.30` | Аварийный возврат «недостаточно информации» без refinement |
| `coverage < 0.50` после `max_refinements` | Добавить disclaimer «ограниченная информация» к ответу |
| `dense_score` отсутствует в doc | Использовать `0.0` для этого документа (graceful degradation) |

---

## 2. Изменяемые файлы

| Файл | Характер изменения |
|---|---|
| `src/services/tools/compose_context.py` | Добавить `query` параметр, новые helper-функции, заменить naive ratio |
| `src/core/settings.py` | Обновить defaults: `coverage_threshold=0.65`, `max_refinements=2` |
| `src/services/agent_service.py` | Инжектировать `query` + `dense_score`; добавить abort guard и disclaimer |

---

## 3. `src/services/tools/compose_context.py`

### 3.1 Добавить перед `compose_context()`

```python
_STOP_WORDS: frozenset = frozenset({
    "the", "and", "for", "with", "this", "that", "are", "was", "from",
    "что", "как", "это", "для", "или", "при", "его", "её", "они", "как",
    "по", "на", "в", "к", "у", "о", "из", "за", "до", "со", "не",
})


def _query_term_coverage(query: str, docs: List[Dict[str, Any]]) -> float:
    """Доля значимых терминов запроса (≥3 символа), встречающихся в текстах документов.

    Используется как один из сигналов в _compute_coverage.
    Возвращает 1.0 если нет значимых терминов (нечего проверять).
    """
    if not query or not docs:
        return 0.0
    tokens = [
        t.lower()
        for t in query.split()
        if len(t) >= 3 and t.lower() not in _STOP_WORDS
    ]
    if not tokens:
        return 1.0
    all_text = " ".join(str(d.get("text", "")).lower() for d in docs)
    covered = sum(1 for t in tokens if t in all_text)
    return covered / len(tokens)


def _compute_coverage(
    query: str,
    docs: List[Dict[str, Any]],
    relevance_threshold: float = 0.55,
    target_k: int = 5,
) -> float:
    """5-сигнальная взвешенная метрика покрытия запроса (DEC-0018, R04).

    Args:
        query: Оригинальный запрос пользователя (для term_coverage).
        docs: Список документов. Ожидается поле dense_score (float 0–1).
              Если dense_score отсутствует, сигналы на его основе = 0.0.
        relevance_threshold: Порог cosine_sim для «релевантного» документа.
        target_k: Ожидаемое целевое число документов для doc_count_adequacy.

    Returns:
        float в [0.0, 1.0].
    """
    if not docs:
        return 0.0

    # Извлекаем cosine similarity: dense_score предпочтителен, fallback → score → 0.0
    sims: List[float] = sorted(
        [float(d.get("dense_score") or d.get("score") or 0.0) for d in docs],
        reverse=True,
    )
    top_k = sims[:target_k]

    max_sim = sims[0]
    mean_top_k = sum(top_k) / len(top_k)

    relevant_count = sum(1 for s in sims if s >= relevance_threshold)
    doc_count_adequacy = min(1.0, relevant_count / target_k)

    # score_gap: 1 − нормированный разброс top_k.
    # Высокое значение = равномерная релевантность (хорошо).
    # Низкое = только первый документ релевантен (риск).
    if max_sim > 0 and len(top_k) > 1:
        score_gap = 1.0 - (sims[0] - top_k[-1]) / sims[0]
    else:
        score_gap = 0.0

    above_threshold_ratio = relevant_count / len(sims)

    term_cov = _query_term_coverage(query, docs)

    return min(
        1.0,
        0.25 * max_sim
        + 0.20 * mean_top_k
        + 0.20 * term_cov
        + 0.15 * doc_count_adequacy
        + 0.15 * score_gap
        + 0.05 * above_threshold_ratio,
    )
```

### 3.2 Изменить сигнатуру `compose_context()`

**До:**
```python
def compose_context(
    docs: List[Dict[str, Any]],
    max_tokens_ctx: int = 1800,
    citation_format: str = "footnotes",
    enable_lost_in_middle_mitigation: bool = True,
) -> Dict[str, Any]:
```

**После:**
```python
def compose_context(
    docs: List[Dict[str, Any]],
    query: str = "",
    max_tokens_ctx: int = 1800,
    citation_format: str = "footnotes",
    enable_lost_in_middle_mitigation: bool = True,
) -> Dict[str, Any]:
```

`query` — с дефолтом `""`, поэтому все существующие вызовы без `query` остаются совместимыми.

### 3.3 Заменить naive ratio

**До (строка 85):**
```python
citation_coverage = len(citations) / len(docs) if docs else 1.0
```

**После:**
```python
citation_coverage = _compute_coverage(query, docs)
```

Переменная `docs` здесь — исходный список, переданный в функцию (не `indexed_docs`, который урезан под max_chars). Это правильно: оценка покрытия по исходному набору кандидатов.

---

## 4. `src/core/settings.py`

**До:**
```python
self.coverage_threshold: float = float(os.getenv("COVERAGE_THRESHOLD", "0.8"))
self.max_refinements: int = int(os.getenv("MAX_REFINEMENTS", "1"))
```

**После:**
```python
self.coverage_threshold: float = float(os.getenv("COVERAGE_THRESHOLD", "0.65"))
self.max_refinements: int = int(os.getenv("MAX_REFINEMENTS", "2"))
```

Значения переопределяются через `.env` без изменения кода (DEC-0019 требует калибровки после появления 30–50 labeled examples).

---

## 5. `src/services/agent_service.py`

### 5.1 `_current_query` — уже реализовано

`self._current_query` уже существует в `agent_service.py` (строки 45, 134): сохраняется в начале `stream_agent_response`, очищается при завершении. Изменений не требуется.

### 5.2 `_normalize_tool_params` — инжектировать `query` и `dense_score`

**Строка 931 (до):**
```python
params.pop("query", None)
params.pop("hits", None)
```

**После:**
```python
# query инжектируется из оригинального запроса; если LLM передал — перезаписываем
# (LLM не знает оригинальной формулировки точно)
params.pop("hits", None)
params["query"] = getattr(self, "_current_query", "")
```

**Строки 988–994 (нормализация docs) — добавить `dense_score`:**

**До:**
```python
normalized_docs.append(
    {
        "id": doc_id,
        "text": text_value,
        "metadata": doc.get("metadata") or doc.get("meta", {}),
    }
)
```

**После:**
```python
normalized_docs.append(
    {
        "id": doc_id,
        "text": text_value,
        "metadata": doc.get("metadata") or doc.get("meta", {}),
        "dense_score": doc.get("dense_score"),   # None если отсутствует → _compute_coverage деградирует к 0.0
    }
)
```

### 5.3 Abort guard и hedged disclaimer после compose_context

В блоке `if action_result.tool == "compose_context":` (строка 537), после строки:
```python
agent_state.coverage = coverage
```

Добавить:

**Abort guard (до проверки _should_attempt_refinement):**
```python
# Abort: если ни один документ не релевантен (max cosine_sim < 0.30)
# Refinement не поможет — данных нет в индексе.
max_sim = max(
    (float(d.get("dense_score") or 0.0) for d in normalized_docs_snapshot),
    default=0.0,
)
if max_sim < 0.30 and not agent_state.refinement_count:
    yield AgentStepEvent(
        type="thought",
        data={
            "content": "Insufficient information: no relevant documents found (max similarity < 0.30).",
            "step": step,
            "system_generated": True,
        },
    )
    # Прерываем refinement loop — переходим к FinalAnswer с оговоркой
    agent_state.coverage = 0.0
    # Не входим в _should_attempt_refinement
```

**Hedged disclaimer (после refinement loop, перед final answer):**

Добавить в `agent_state` флаг и использовать при формировании FinalAnswer:

```python
# Если покрытие < 0.50 после исчерпания refinements — пометить для disclaimer
if (
    coverage < 0.50
    and agent_state.refinement_count >= self.settings.max_refinements
):
    agent_state.low_coverage_disclaimer = True
```

При формировании финального ответа проверять `agent_state.low_coverage_disclaimer` и добавлять к промпту:
```
[Примечание: найдено ограниченное количество релевантной информации. Ответ может быть неполным.]
```

> **Примечание по реализации abort guard:** `normalized_docs_snapshot` — это список `normalized_docs` после строки 1033 (`params["docs"] = normalized_docs`). Для доступа к нему в блоке на строке 537 нужно либо сохранить его в `self._last_normalized_docs` в конце `_normalize_tool_params`, либо вычислить `max_sim` из `self._last_search_hits` напрямую (они те же данные). Рекомендуется второй вариант как более простой:
> ```python
> max_sim = max(
>     (float(h.get("dense_score") or 0.0) for h in self._last_search_hits),
>     default=0.0,
> )
> ```

---

## 6. Тесты

### 6.1 Unit-тесты `_compute_coverage` и `_query_term_coverage`

Файл: `src/tests/test_compose_context.py` (новый или дополнить существующий).

```python
def test_compute_coverage_empty():
    assert _compute_coverage("query", []) == 0.0

def test_compute_coverage_high_relevance():
    docs = [{"text": "Bitcoin crypto news", "dense_score": 0.82}] * 5
    result = _compute_coverage("bitcoin crypto", docs)
    assert result > 0.65

def test_compute_coverage_low_relevance():
    docs = [{"text": "unrelated text", "dense_score": 0.20}] * 3
    result = _compute_coverage("bitcoin crypto", docs)
    assert result < 0.35

def test_compute_coverage_no_dense_score_fallback():
    # graceful degradation: dense_score отсутствует → использует term_coverage
    docs = [{"text": "bitcoin price crypto market", "dense_score": None}]
    result = _compute_coverage("bitcoin", docs)
    assert 0.0 <= result <= 1.0

def test_query_term_coverage_basic():
    docs = [{"text": "bitcoin price rose sharply"}]
    assert _query_term_coverage("bitcoin price", docs) == 1.0

def test_query_term_coverage_partial():
    docs = [{"text": "bitcoin price rose"}]
    # "ethereum" не покрыт
    result = _query_term_coverage("bitcoin ethereum", docs)
    assert result == 0.5

def test_compute_coverage_is_capped_at_one():
    docs = [{"text": "test", "dense_score": 0.95}] * 10
    assert _compute_coverage("test", docs) <= 1.0
```

### 6.2 Регрессионный тест смены метрики

```python
def test_coverage_higher_than_naive_for_relevant_docs():
    """Для релевантных документов composite ≥ naive ratio."""
    docs = [{"text": f"bitcoin crypto news {i}", "dense_score": 0.75} for i in range(5)]
    naive = 1.0  # len(5)/len(5)
    composite = _compute_coverage("bitcoin crypto", docs)
    # composite должен быть в разумном диапазоне, не 1.0 при хорошем наборе
    assert 0.5 < composite < naive

def test_coverage_lower_than_naive_for_irrelevant_docs():
    """Для нерелевантных документов composite < naive ratio (naive всегда 1.0 если включены все)."""
    docs = [{"text": "unrelated stuff", "dense_score": 0.10}] * 5
    naive = 1.0
    composite = _compute_coverage("bitcoin crypto", docs)
    assert composite < naive
```

---

## 7. Совместимость

| Контракт | Статус |
|---|---|
| SSE поле `citation_coverage` | Сохраняется, значение изменится |
| API `citations` / `contexts` | Без изменений |
| LLM промпт | Без изменений |
| Tool schema `compose_context` | `query` добавлен с default `""` — backward compatible |
| `settings.coverage_threshold` env var | Читается из `COVERAGE_THRESHOLD`, дефолт меняется |

**Важно:** Пороговые значения типичных тестов могут упасть после смены метрики, если они тестировали конкретные значения `citation_coverage`. Нужно обновить assertions в `test_agent_service.py`.

---

## 8. Открытые вопросы

| ID | Вопрос | Решение |
|---|---|---|
| OQ-01 | `dense_score` в Phase 0 (ChromaDB) | ChromaDB возвращает `distance`, не `dense_score`. Текущие ChromaDB-based hits не имеют `dense_score`. Формула деградирует к `term_coverage` + `doc_count_adequacy`. Приемлемо для Phase 0. |
| OQ-02 | Abort guard реализация | Рекомендуется использовать `self._last_search_hits` напрямую (см. раздел 5.3). |
| OQ-03 | Калибровка threshold | DEC-0019: требуется 30–50 labeled examples. До тех пор — `0.65`. |
| OQ-04 | `agent_state.low_coverage_disclaimer` | Поле добавить в `AgentState.__init__`: `self.low_coverage_disclaimer: bool = False`. Текущий `AgentState` имеет только `coverage`, `refinement_count`, `max_refinements`. |

---

## 9. Порядок применения

1. `src/core/settings.py` — обновить defaults (не ломает ничего).
2. `src/services/tools/compose_context.py` — добавить helpers и обновить функцию.
3. `src/services/agent_service.py` — инжекция query/dense_score, abort guard, disclaimer.
4. `src/tests/test_compose_context.py` — добавить тесты.
5. Обновить assertions в `src/tests/test_agent_service.py` (смена значений coverage).
