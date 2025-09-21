
# ReAct + RAG: Итоговые выводы и практические установки (v1)

Документ — краткая шпаргалка для планировщика, разработчиков инструментов и имплементатора кода. Ориентирован на локальные модели (llama.cpp, GGUF), FastAPI, гибридный поиск (Chroma + BM25), строгий контроль структур (GBNF/Schema), SSE-стриминг. Цель — максимальная надёжность, управляемость и производительность на CPU, без тяжёлых фреймворков.

---

## 1) Архитектурные принципы ReAct

- **Лёгкий цикл.** Фиксированный Reason→Act-поток без внешних оркестраторов. Ограничение итераций: `max_iterations=3` (план → поиск/слияние → ответ). Один дополнительный раунд поиска допускается только по критериям «плохого покрытия».
- **Сериализация LLM.** Внутри процесса одна генерация LLM за раз. Лёгкие инструменты могут работать асинхронно (I/O).
- **Прозрачность без CoT-утечки.** Трассируем только события инструментов (start/end, took_ms, ok, error). Мысли модели в логи и ответы не попадают.
- **Строгие форматы на критических узлах.** План поиска и финальный ответ — под GBNF/JSON Schema. Остальные шаги — мягкая валидация + пост-обработка.

---

## 2) Контракт инструментов и трейс

- **Единый контракт.**
  - Вход: `{ "tool": "<name>", "input": { ... } }`
  - Выход: `{ "ok": true|false, "data": { ... }, "meta": { "took_ms": <int>, "error": "<optional>" } }`
- **Трассировка.** JSON-лог в stdout/файл: `{request_id, step, tool, took_ms, ok, error}`.
- **Сет инструментов (минимум MVP).**
  - `router_select`
  - `query_plan` (GBNF → SearchPlan + пост-валидация)
  - `multi_query_rewrite` (микро‑GBNF массива строк, по необходимости)
  - `bm25_search` / `dense_search` / `hybrid_search`
  - `fusion_rank` (RRF→MMR, дедуп)
  - `rerank` (опц., CPU CrossEncoder)
  - `fetch_docs`
  - `compose_context`
  - `time_normalize` (простые RU/EN диапазоны дат)

---

## 3) GBNF/Schema и decoding-профили (llama.cpp, CPU)

- **SearchPlan (GBNF).** Объект JSON с полями: `normalized_queries[3..6]`, `must_phrases[]`, `should_phrases[]`, `metadata_filters{...|null}`, `k_per_query(1..50)`, `fusion∈{"rrf","mmr"}`. Минимизировать `ws` после закрывающих скобок, избегать левой рекурсии.
- **Микро‑GBNF массив N строк.** Для догена недостающих подзапросов/перефразов: фиксированная длина (`N∈[1..3]`) в JSON‑массиве строк.
- **Decoding (grammar mode):** `temperature=0.2`, `top_p=0.9`, `top_k=40`, `repeat_penalty=1.2`, `max_tokens=256 (plan) / 128 (micro)`, `seed=42`. **Не** задавать `stop`, завершение по грамматике.
- **Финальный ответ (если JSON):** отдельная (проще) схема `{"answer": string, "sources": int[]}` с grammar/Schema. Если ответ plain‑text — без grammar, но с пост‑проверками.

---

## 4) Multi‑query и data‑fusion

- **Объём.** Целевое число запросов: **4–6** (минимум 3, максимум 6). Если план дал ≥4 разнообразных — `multi_query_rewrite` пропускаем.
- **Слияние.** Сначала **RRF (k=60)** поверх листов результатов, затем **MMR** для диверсификации (`λ=0.7`, косинус по эмбеддингам, greedy top‑N).
- **Дедупликация.** По `doc_id/URL` и лексемам; в MMR не добавлять кандидатов, схожих с уже выбранными.
- **Реранкер (опц.).** CPU CrossEncoder (например, MiniLM‑L6) на **top‑20** после RRF/MMR; итоговый top‑K=5–10. Триггер по латентности/порогам качества.

---

## 5) Compose context и цитаты

- **Бюджеты (n_ctx≈4096).** Контекст документов ≈ **1800 токенов**; ответ **600–800**; остальное — системка/подсказки.
- **Отбор фрагментов.** Чанки 200–300 токенов с overlap 50; ранжируем по итоговым score; распределяем бюджет пропорционально релевантности. Применить Lost‑in‑the‑Middle mitigation: важные спаны — в начале и конце контекста.
- **Формат цитат.** Нумерация `[1]`, `[2]` …; внизу — соответствие номер→(channel/title/url/msg_id). Эвристика контроля покрытия: в каждом смысловом предложении — хотя бы одна ссылка; при дефиците — один повторный раунд поиска.

---

## 6) Router‑select и сокращение шагов

- **Эвристики.**
  - Короткие «терминные» запросы (≤3 токена, имена/ID/годы) → **bm25**.
  - Разговорные/длинные вопросы → **hybrid** (bm25||dense) или **dense**.
  - Жёсткие фильтры (канал/даты) + короткий запрос → **bm25**; если вопрос сложный → **hybrid**.
- **Сокращение LLM‑вызовов.** Планировщик генерирует 3–6 subqueries за один проход; `multi_query_rewrite` — только при низком разнообразии. Retrieval (bm25+dense) — параллельно; RRF — on‑the‑fly.

---

## 7) Кеширование, таймауты, деградации

- **Кеши.**
  - План: TTL **10 мин** (key: md5(question)).
  - Fusion: TTL **5 мин** (key: `{route}:{md5(sorted_subqueries)}`).
  - Doc‑fetch LRU (id→текст). Опц.: memo для (query, doc_id) в rerank.
- **Таймауты (жёсткие).**
  - `query_plan`: 8–10s
  - `multi_query_rewrite`: 5s
  - `bm25/dense`: 4s (каждый)
  - `fusion/mmr`: 1s
  - `rerank`: 5–6s (батч)
  - `fetch_docs`: 2s
- **Деградации.** Любой fail → упрощение: fallback‑plan (один запрос), dense‑only, skip‑rerank. Не более **1** дополнительного поискового раунда.

---

## 8) Безопасность и санитизация

- **Анти‑SQL/код в запросах.** Инструкции планировщику + пост‑фильтр (`select|from|where`, императивы RU/EN). Обрезать запросы до 12 слов.
- **Валидация входов.** Pydantic‑модели для `.input` каждого инструмента. Белые списки значений и допустимых диапазонов.
- **Санация пользовательского ввода.** Экранировать кавычки/спецсимволы; не смешивать формат CLI CSV с API массивами.
- **Никакого CoT наружу.** Системный промпт запрещает утечки; пост‑проверка ответа на признаки внутреннего формата/служебных тегов.

---

## 9) API/схемы (кратко)

- **Эндпоинты `/v1/react/*`:**
  - `POST /answer` (sync), `GET /answer/stream` (SSE).
  - `POST /plan` (debug), `GET /trace/{id}` (debug).
- **Модели (Pydantic):**
  - `ToolRequest`, `ToolResponse`, `ToolMeta`, `AgentAction`, `Answer{answer: str, sources: [Source]}`, `Trace`.

---

## 10) Метрики и приёмка

- **Качество:** `valid_json_rate ≥ 99.5%`, `avg_subqueries ≥ 3.2`, `recall@20 ≥ baseline+10%`, `citation_coverage ≥ 0.8`.
- **Производительность:** `p95_latency ≤ baseline * 1.15`, `tool_timeout_rate < 1%`, `cache_hit_rate` растущий.
- **Тест‑набор:** RU/EN/шум, граничные случаи (3/6 subqueries), “SQL‑артефакты”, пустая выдача, длинные вопросы.

---

## 11) Готовые профили/параметры (переносимые)

### 11.1 llama.cpp (planner, grammar)
```ini
# planner (GBNF)
temperature=0.2
top_p=0.9
top_k=40
repeat_penalty=1.2
max_tokens=256  # SearchPlan JSON
seed=42
# no stop sequences in grammar mode
```

### 11.2 Multi‑query + Fusion
```yaml
multi_query:
  target_count: 5        # диапазон 4–6
  min_count: 3
  enable_rewrite: auto   # только при низком разнообразии

fusion:
  rrf_k: 60
  mmr:
    enabled: true
    lambda: 0.7
    top_n: 10
    sim: cosine
```

### 11.3 Rerank (CPU)
```yaml
rerank:
  enabled: auto          # включать по порогам качества/скорости
  model: "MiniLM-L6"     # компактный CrossEncoder
  input_top_n: 20
  output_top_n: 5
  batch_size: 16
  timeout_ms: 6000
```

### 11.4 Compose Context
```yaml
context:
  max_doc_tokens: 1800
  chunk_size: 256
  chunk_overlap: 50
  lost_in_middle_mitigation: true
  citation_format: "[{i}]"
  require_citation_each_fact: true
```

### 11.5 Router‑select
```yaml
router:
  short_query_tokens: 3
  rules:
    - if: "len<=3 or has_ids_or_years"
      route: "bm25"
    - if: "has_strict_filters and is_short"
      route: "bm25"
    - else: "hybrid"
```

### 11.6 Кеш/таймауты/повторы
```yaml
cache:
  plan_ttl_sec: 600
  fusion_ttl_sec: 300

timeouts_ms:
  plan: 10000
  rewrite: 5000
  search_each: 4000
  fusion: 1000
  rerank: 6000
  fetch: 2000

react:
  max_iterations: 3
  extra_round_on_poor_coverage: true
  max_extra_rounds: 1
```

---

## 12) Внедрение (порядок и риск‑контроль)

1. **A/B планировщик:** GBNF vs Schema; валидность/латентность → включить GBNF по флагу.
2. **Короткий путь:** план → hybrid retrieval → RRF/MMR → compose → генерация; без agent‑loop.
3. **Orchestrator:** tool_runner, трассировка, эвристики router, retry по покрытию, SSE‑совместимость.
4. **Фичфлаг и метрики:** частичный rollout, мониторинг p95/coverage, включение rerank по триггерам.
5. **Rollback:** один флаг выключает ReAct; fallback на baseline‑pipeline.

---

## 13) Quick‑wins

- Включить **multi‑query + RRF/MMR** сразу — даёт прирост recall при минимальной цене.
- GBNF на **финальном JSON‑ответе** — строгий контракт для downstream‑сервисов.
- Простые **эвристики router** и **санитизация** уже заметно снижают шум и ошибки.
- **Кеш плана** и **fusion** — дешёвая экономия латентности при повторных вопросах.

---

## 14) Инварианты проекта (напоминание)

- Один процесс; без рестартов/прогревов по умолчанию; ленивые фабрики; TTL‑кеши.
- SSE стримит только финальный ответ (без CoT и трейс‑событий).
- RU/EN; Unicode‑безопасные грамматики; CLI множественные значения — CSV (API — массивы).
- Offline‑friendly: без внешних API в MVP; модель‑агностичность (любой GGUF в llama.cpp).
