## План

### Целевое состояние
- Лёгкий, надёжный ReAct‑агент поверх текущего FastAPI RAG без тяжёлых фреймворков.
- Модель‑агностично и CPU‑дружелюбно: локальные GGUF (llama.cpp), один процесс; LLM‑вызовы сериализованы; async — для I/O/tool.
- Строгий контроль структур там, где критично: GBNF для SearchPlan (3–6), JSON‑контракты tool, пост‑валидация.
- Recall↑ за счёт multi‑query + fusion (RRF), управляемая диверсификация (MMR), опц. CPU cross‑encoder rerank.
- Прозрачность: единый контракт tool, трейс шагов (без CoT) в системный лог, кеши/таймауты/деградации.
- Совместимость: текущие эндпоинты не ломаем; SSE — стримим только финальный ответ; MVP — self‑host.

### Что уже есть
- Планировщик: `QueryPlannerService` (GBNF стабильно 3–6) + TTL‑кеш (10 мин) + пост‑валидация.
- Ретриверы: Dense (Chroma), BM25, Hybrid (с планом).
- Слияние/диверсификация: `utils.ranking.rrf_merge`, `mmr_select`.
- Ререйк: `services.reranker_service.RerankerService` (CPU CrossEncoder).
- Промпт/ответ: `utils.prompt.build_prompt`, LLM фабрика, SSE финального ответа.
- Инжест: Telegram → Chroma + BM25; грамматики: `utils/gbnf.py` (полная/микро).

### Чего не хватает (минимальные tool)
- `router_select` (эвристика bm25|dense|hybrid)
- `multi_query_rewrite` (3–6 перефразов; микро‑GBNF/Schema)
- `dedup_diversify` (MMR + дедуп id/лексем/URL)
- `fetch_docs` (батч тексты/окна по ids из Chroma)
- `compose_context` (спаны/цитаты, лимит ~1800 токенов)
- `time_normalize` (RU/EN → ISO даты)
- `tool_runner`/`trace` (единый контракт/таймауты/лог stdout+файл)

### Базовый ReAct‑флоу (MVP)
1) router_select → route∈{bm25|dense|hybrid}
2) plan (GBNF) → subqueries(3–6), k_per_query, filters; при <3 → микро‑GBNF доген N
3) (опц.) multi_query_rewrite до 4–6 при низком разнообразии плана
4) retrieval по пулу запросов (hybrid либо выбранный маршрут)
5) fusion: RRF(k=60) → (опц.) MMR(λ≈0.7, out_k)
6) (опц.) rerank (CPU CrossEncoder, top_n≤10–20)
7) fetch_docs(+windows) → compose_context(≤~1800 токенов, пронумерованные источники)
8) finish: LLM generate; SSE — только финальный ответ
Halting: при слабом покрытии — один доп. раунд 4–7; любой fail tool → ok:false и деградация (dense‑only, без rerank)

### Параметры (стартовые рекомендации)
- Планировщик (CPU, grammar): temperature≈0.2, top_p≈0.9, top_k≈40–50, repeat_penalty≈1.08–1.2, max_tokens≈256–384; без stop; фиксированный seed.
- Контекст: n_ctx≈4096; budget документов ≈1800 токенов; ответ ≈600–800 токенов.
- Fusion: RRF k=60; MMR λ≈0.7; после fusion рассматривать ≤15–20 кандидатов.
- Rerank: top_n≤10–20, батч компактный; включать по порогу/фичфлагом.
- Таймауты: tools 4–5s; планировщик 8–10s; генерация ответа — по длине.
- Кеши: план 10 мин; fusion 5 мин; ключи включают route и набор subqueries.

### Метрики приёмки
- valid_json_rate (план) ≥ 99.5%
- avg_subqueries ≥ 3.2
- Recall@20 ≥ baseline +10%
- citation‑coverage ≥ 0.8
- p95 latency ≤ baseline +15%
- tool timeout rate < 1%, cache hit‑rate разумный

### Риски и смягчения
- Latency↑ из‑за grammar/rerank → ограничить max_tokens, узкие микро‑GBNF, малый top_n rerank, включать по триггерам.
- «SQL/код» в subqueries → инструкции + фильтры символов/лексем + пост‑валидатор.
- Утечки CoT → логировать только шаги инструментов (JSON), без мыслей.
- Зацикливание → шаг‑лимит 3–4, один доп. раунд поиска максимум.

### План внедрения (итерации)
- Итерация 1 — Планировщик «прод»:
  - GBNF как основной, json_schema fallback; микро‑GBNF доген; пост‑валидатор; метрики.
- Итерация 2 — «Короткий путь» без ReAct:
  - `router_select`, `compose_context`, `fetch_docs`, `dedup_diversify`; конвейер: plan → hybrid retrieval → RRF → (опц. MMR) → (опц. rerank) → compose → answer.
- Итерация 3 — ReAct каркас:
  - Tool API (единый контракт), `tool_runner`/таймауты, trace sink; ReAct‑цикл с шаг‑лимитом; SSE без изменений.
- Итерация 4 — Наблюдаемость/качество:
  - Метрики шагов/таймаутов/кешей, A/B «короткий путь vs ReAct», тюнинг параметров; опц. `/v1/react/*` за фичфлагом.

