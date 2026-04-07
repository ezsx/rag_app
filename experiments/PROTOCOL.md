# Experiment Protocol

> Правила проведения экспериментов (eval, ablation, A/B). ОБЯЗАТЕЛЕН для агента.
> Загружается при task type = eval / experiment / ablation.
> Аналог documentation-governance, но для экспериментов.

---

## Принципы

1. **Никакого compute без spec.** Сначала spec.yaml — потом прогон.
2. **Config из одного места.** Eval импортирует production settings, не хардкодит.
3. **Baseline заморожен.** Сравниваем с зафиксированным baseline.yaml, не с "прошлым прогоном".
4. **Parity перед запуском.** Автоматическая проверка что eval config == production config.
5. **Артефакты структурированы.** Каждый run = папка со spec + raw data + results.

---

## Структура

```
experiments/
  PROTOCOL.md         ← этот файл (правила)
  baseline.yaml       ← frozen: production config + текущие метрики
  log.md              ← summary всех runs (одна строка на run)
  runs/
    RUN-001/
      spec.yaml       ← гипотеза, config diff, acceptance criteria (ДО прогона)
      raw.jsonl        ← live данные от eval скрипта (ВО ВРЕМЯ прогона)
      results.yaml     ← метрики + интерпретация + решение (ПОСЛЕ прогона)
```

---

## Flow: шаг за шагом

### Phase 1 — Design (до compute)

1. **Загрузить контекст:**
   - Прочитать `experiments/baseline.yaml` (текущий config + метрики)
   - Прочитать `experiments/log.md` (что уже пробовали)

2. **Написать spec.yaml:**
   - `hypothesis` — что проверяем и почему
   - `what_changes` — конкретные изменения (файл, строка, значение)
   - `what_stays` — что не меняется (явно)
   - `baseline_ref` — ссылка на baseline.yaml
   - `dataset` — какой датасет, сколько queries
   - `acceptance_criteria` — числовые пороги для success
   - `risks` — что может пойти не так

3. **User ревьюит spec** — корректирует или утверждает.

### Phase 2 — Validate (до compute, автоматически)

4. **Preflight check:**
   - Все сервисы живы (qdrant, embedding, reranker, llm)
   - Dataset файл существует и валиден

5. **Parity check:**
   - Импортировать production `settings.py`
   - Сравнить каждый параметр retrieval config с `baseline.yaml`
   - Если расхождение НЕ указано в `what_changes` → СТОП + показать diff
   - Записать `parity_check: passed` в spec

6. **Config snapshot:**
   - Записать полный config snapshot в spec (git_sha + все параметры)
   - Это гарантирует reproducibility

### Phase 3 — Execute

7. **Запуск eval:**
   - Output → `runs/RUN-NNN/raw.jsonl`
   - Live flush, `tail -f` для мониторинга

8. **Early checkpoint (после 10 queries):**
   - Сравнить interim metrics с baseline
   - Если аномалия (CE fail rate > 0, latency > 2x baseline, recall < baseline - 0.1) → СТОП
   - Telegram уведомление при аномалии
   - Если норм → продолжить полный прогон

9. **Completion:**
   - Telegram уведомление с summary metrics

### Phase 4 — Analyze (после compute)

10. **Заполнить results.yaml:**
    - Все метрики (recall, MRR, mean_ce, ce_neg, n_channels, latency)
    - Сравнение с baseline (delta по каждой метрике)
    - Anomalies (что не ожидалось)

11. **Интерпретация:**
    - Agent формулирует выводы
    - User обсуждает, задаёт вопросы
    - Совместное решение: `adopt` / `reject` / `investigate`

12. **Если adopt:**
    - Обновить production config (settings.py, hybrid_retriever.py, etc.)
    - Обновить `baseline.yaml` новыми значениями и метриками
    - Записать в `log.md`
    - Commit с ссылкой на RUN-NNN

13. **Если reject:**
    - Откатить изменения если были
    - Записать в `log.md` с причиной
    - Spec + results остаются для reference

---

## Правила именования

| Артефакт | Паттерн | Пример |
|----------|---------|--------|
| Run folder | `RUN-NNN` | `RUN-003` |
| Spec | `spec.yaml` | `runs/RUN-003/spec.yaml` |
| Raw data | `raw.jsonl` | `runs/RUN-003/raw.jsonl` |
| Results | `results.yaml` | `runs/RUN-003/results.yaml` |

Нумерация последовательная, не пропускать.

---

## Parity check — критические параметры

Следующие параметры ОБЯЗАТЕЛЬНО проверяются перед каждым прогоном:

```yaml
retrieval:
  embedding_query_instruction    # prefix для embedding
  search_k_per_query_default     # docs per subquery
  hybrid_enabled                 # hybrid retriever on/off
  qdrant_collection              # collection name

fusion:
  fusion_strategy                # rrf / dbsf
  # RRF weights hardcoded в hybrid_retriever.py [1.0, 3.0]

reranker:
  enable_reranker                # CE on/off
  reranker_top_n                 # CE pool size
  reranker_tei_url               # endpoint URL

endpoints:
  qdrant_url                     # must be reachable
  embedding_tei_url              # must be reachable
  reranker_tei_url               # must be reachable
  llm_base_url                   # must be reachable
```

Расхождение в ЛЮБОМ параметре = прогон невалиден.

---

## Чего НЕ делать

1. **Не запускать eval без spec.** Даже "быстрый тест на 10 queries".
2. **Не менять eval скрипт во время серии экспериментов.** Фиксировать скрипт, менять только config.
3. **Не сравнивать runs с разными eval скриптами.** Git sha в spec для этого.
4. **Не хардкодить параметры в eval скрипте.** Всё из settings.py.
5. **Не запускать полный прогон (120 Qs) без checkpoint на 10 Qs.**
