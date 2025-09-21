
# ReAct/RAG Survey (Исследование 2, 2023–2025): практики, ссылки, рекомендации для нашего стека

Дата: 2025-09-21  
Фокус: минималистичные ReAct/Tool‑Use/RAG‑конвейеры без тяжёлых платформ (self‑host, llama.cpp, CPU‑дружелюбно). Язык — Python; Rust допускается для ускорения некоторых инструментов (через pyo3/maturin).

---

## Резюме (коротко)
- **Структурирование вывода**: для локальных моделей под `llama.cpp` устойчивее всего GBNF‑грамматики (минимальные, целевые), JSON‑Schema — как fallback. Outlines / LM‑Format‑Enforcer полезны для не‑llama.cpp и быстрого прототипирования, но в проде лучше держать единый путь через GBNF.
- **Multi‑query и data‑fusion**: диапазон 4–6 перефразов достаточен; объединение через **RRF** (+ опц. MMR λ≈0.5). Ререйк на CPU имеет смысл для top‑20..40; быстрый CrossEncoder MiniLM даёт хороший прирост при умеренных затратах.
- **Compose‑context**: короткие окна/спаны с цитатами и смещением релевантных фрагментов к началу (см. Lost‑in‑the‑middle). Следить за **citation‑coverage** и триггерить дополнительный раунд при недостатке опор.
- **Router‑select**: эвристический выбор `{bm25|dense|hybrid}` по признакам запроса (терминность/операторы/дата/язык/длина). В сомнительных случаях — `hybrid` с RRF.
- **Устойчивость и кеши**: TTL‑кеши на план и fusion, memo для tool‑результатов по детерминированным ключам. Лимиты: tool 4–5 с, раунды ≤ 2, шаги ≤ 4; деградации: skip‑rerank → dense‑only.
- **Безопасность**: белые схемы инструментов (Pydantic), строгая валидация/санитизация, защита от prompt injection; фильтрация «SQL/код» артефактов в подзапросах.

---

## 1) Архитектуры ReAct/Tool‑Use без тяжёлых платформ

Минималистичные агентные циклы, реализуемые «чистым» Python без фреймворков:
- **Оригинальный ReAct** (семинальная работа): концепция Reason↔Act, интерливинг мыслей и действий; пригодна как идеоматика для тонкого цикла и шаг‑лимита.  
  Источники: [Google AI Blog (ReAct)](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/), [Paper page](https://react-lm.github.io/), [arXiv 2210.03629](https://huggingface.co/papers/2210.03629).

- **ReAct без фреймворков**: статьи/туториалы с циклами на чистом Python (шаг‑лимит, tool‑контракт, retry). Используем как ориентир интерфейсов и трейсинга:  
  [LangGraph “from scratch” (идея цикла)](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/),  
  [Tiny‑agents (HuggingFace) — ~70 строк](https://huggingface.co/blog/python-tiny-agents),  
  [Пример на Medium с авто‑tool‑циклом](https://medium.com/%40garland3/building-a-simple-ai-agent-with-function-calling-a-learning-in-public-project-acf4cd8f18bd).

- **Контракт инструментов**: Pydantic‑модели для входа/выхода, унифицированный JSON:  
  [Pydantic tools API](https://ai.pydantic.dev/tools/), [“Steering LLMs with Pydantic”](https://pydantic.dev/articles/llm-intro).

Практика для нас: сохраняем единый I/O‑контракт tool, шаг‑лимит 3–4, tool‑таймаут 4–5 с, трейс через JSON‑события (без CoT).

---

## 2) Строгий контроль структур (GBNF/CFG/JSON‑Schema)

- **GBNF в `llama.cpp`**: устойчивые минимальные грамматики; есть утилиты генерации из JSON‑Schema.  
  Ссылки: README по грамматикам (`llama.cpp`):  
  – [HuggingFace mirror README](https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/blob/main/llama.cpp/grammars/README.md),  
  – Скрипт `json-schema-to-grammar.py`: [репозиторий ggml‑org/llama.cpp (обсуждение/пример)](https://github.com/ggerganov/llama.cpp/discussions/3268),  
  – Альтернативный конвертер: [adrienbrault/json-schema-to-gbnf](https://github.com/adrienbrault/json-schema-to-gbnf).

- **Альтернативы**:  
  **Outlines** — быстрые regex/JSON‑guided генерации (любой провайдер/рантайм): [repo](https://github.com/dottxt-ai/outlines), [PyPI](https://pypi.org/project/outlines/0.0.24/).  
  **LM‑Format‑Enforcer** — лёгкое принуждение к JSON/моделям: [repo](https://github.com/noamgat/lm-format-enforcer).  
  **vLLM Structured Outputs** — guided JSON/grammar: [docs](https://docs.vllm.ai/en/v0.9.2/features/structured_outputs.html).  
  **XGrammar** — производительные CFG/JSON‑schema (опционально): [документация](https://xgrammar.mlc.ai/docs/api/python/grammar.html).

- **Рекомендации**: на CPU под `llama.cpp` держать основной путь через GBNF; `response_format=json_schema` использовать как fallback; избегать `stop`‑токенов при grammar‑режиме — грамматика сама ограничивает завершение.

---

## 3) Multi‑query и data‑fusion

- **Диапазон перефразов**: 4–6 достаточно для диверсификации; RAG‑Fusion популяризировал стратегию мультизапросов с последующим **RRF**.  
  Ссылки: [Azure Hybrid & RRF](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking), [Elastic RRF API](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion), обзор RAG‑Fusion: [DocsBot](https://docsbot.ai/article/advanced-rag-techniques-multiquery-and-rank-fusion), [RAG‑Fusion обзоры](https://pub.towardsai.net/not-rag-but-rag-fusion-understanding-next-gen-info-retrieval-477788da02e2).

- **MMR**: для отбора некоррелированных кандидатов после RRF; типичные λ≈0.3–0.7, top_n 20–50.  
  Ссылки: [MMR объяснение/формула](https://medium.com/%40ankitgeotek/mastering-maximal-marginal-relevance-mmr-a-beginners-guide-0f383035a985), [Elastic blog про MMR](https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results).

- **Ререйк (CPU)**: CrossEncoder MiniLM (MS‑MARCO) — разумный компромисс качество/латентность; бюджеты: top‑20..40, батчами.  
  Ссылки: [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), [Sentence‑Transformers efficiency](https://sbert.net/docs/cross_encoder/usage/efficiency.html).

---

## 4) Compose‑context и цитаты

- **Lost‑in‑the‑Middle**: релевантное с середины хуже используется; сдвигать ключевые спаны вверх, дробить на заголовки/секции.  
  Ссылки: [arXiv 2307.03172](https://arxiv.org/abs/2307.03172), PDF: [Stanford](https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf).

- **Цитаты и покрытие**: метрики **citation coverage/precision** полезны для авто‑проверок и триггера дополнительного раунда.  
  Ссылки: [AWS Bedrock eval docs](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-eval-llm-results.html).

Практика: наш бюджет 1800 токенов контекста; формировать компактные спаны с `[channel_id/msg_id:offset..offset]` и краткими заголовками; измерять coverage и при <0.8 повторять поиск.

---

## 5) Router‑select и сокращение шагов

- **Гибрид доминирует по умолчанию** в неоднозначных запросах; чистый BM25 — для терминных/операторных/дата‑heavy; чистый dense — для длинных, абстрактных, англо‑русский свободный текст.  
  Ссылки‑ориентиры по гибриду: [Pinecone intro](https://www.pinecone.io/learn/hybrid-search-intro/), [Elastic hybrid](https://www.elastic.co/docs/solutions/search/hybrid-search), [OpenSearch best practices](https://opensearch.org/blog/building-effective-hybrid-search-in-opensearch-techniques-and-best-practices/).

- **Сокращение шагов**: объединять retrieval↔fusion стримингом, избегать лишних LLM‑вызовов (rewrite только при слабом покрытии).

---

## 6) Кеширование и устойчивость

- **Ключи кешей**:  
  – План: `hash(model_id, prompt_template, query_norm, time_filter)`; TTL 10 мин.  
  – Fusion: `hash(route, subqueries_set, k_budget, filters)`; TTL 5 мин.  
  – Tool memo: `hash(tool_name, normalized_input)`; TTL 5–15 мин.

- **Деградации**: timeouts→skip‑rerank; пустой результат одного источника→RRF по оставшимся; общий пустой→fallback dense‑only с k↑; шаг‑лимит ≤4.

Практика: tool‑таймаут 4–5 с; один доп. раунд при coverage<0.8; логировать `ok/err/took_ms` на каждом шаге.

---

## 7) Безопасность/санитизация

- **Prompt‑injection & guardrails**: следовать OWASP LLM‑рекомендациям; не исполнять инструкции из пользовательских данных, чётко разделять роль/данные.  
  Ссылки: [OWASP GenAI LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/), [AWS guardrails](https://docs.aws.amazon.com/prescriptive-guidance/latest/llm-prompt-engineering-best-practices/best-practices.html).

- **Анти‑SQL/код**: Pydantic‑валидация tool‑входов, параметризованные запросы; фильтрация подзапросов по белому списку символов/регексу.  
  Ссылки: [OWASP SQL Injection Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html).

---

## 8) Эталонные open‑source/статьи (с выдержками)

- **ReAct (семинальное)** — “interleaved reasoning and acting” как общий паттерн:  
  [Google AI Blog](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/): «ReAct систематически превосходит отдельные парадигмы Reason или Act».  
  [Paper page](https://react-lm.github.io/) и [arXiv/summary](https://huggingface.co/papers/2210.03629).

- **GBNF и структурированный вывод**:  
  [llama.cpp grammars README](https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/blob/main/llama.cpp/grammars/README.md): примеры, советы, конвертация JSON‑Schema→GBNF.  
  [json‑schema‑to‑gbnf](https://github.com/adrienbrault/json-schema-to-gbnf): поддержка большей части JSON‑Schema.  
  [Outlines](https://github.com/dottxt-ai/outlines): быстрый guided‑JSON/regex.  
  [vLLM Structured Outputs](https://docs.vllm.ai/en/v0.9.2/features/structured_outputs.html).

- **RRF/MMR и RAG‑Fusion**:  
  [Azure Hybrid Ranking (RRF)](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking): встроенная RRF для гибридных запросов.  
  [Elastic RRF API](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion).  
  [Elastic MMR блог](https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results).  
  [DocsBot: Multi‑Query & RAG‑Fusion](https://docsbot.ai/article/advanced-rag-techniques-multiquery-and-rank-fusion).

- **Ререйк на CPU**:  
  [MiniLM CrossEncoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2): быстрый rerank для MS‑MARCO‑подобных задач.  
  [Sentence‑Transformers efficiency](https://sbert.net/docs/cross_encoder/usage/efficiency.html): ONNX/OpenVINO.

- **Lost‑in‑the‑Middle**:  
  [arXiv/Stanford PDF](https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf): деградация в середине контекста → мотивирует спаны и сортировку.

- **Метрики citation‑coverage**:  
  [AWS Bedrock eval](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-eval-llm-results.html): определения coverage/precision для цитат.

- **Темпоральная нормализация**:  
  [dateparser (RU/EN)](https://dateparser.readthedocs.io/): 200+ языков, относительные даты, `RELATIVE_BASE`.

---

## 9) Практические рекомендации (под наш проект)

### 9.1 Инструменты (минимальный набор)
- `router_select` → эвристика `{bm25|dense|hybrid}`.  
- `make_plan` → наш GBNF‑планировщик (TTL‑кеш 10 мин).  
- `multi_query_rewrite` → 0–3 доп. перефразов (итог 3–6), запускать при низком coverage или низкой диверсификации.  
- `retrieval_*` → bm25/dense/hybrid; суммарный бюджет `k_total≤120`.  
- `fusion` → RRF, затем опц. MMR (λ=0.5, top_n=40).  
- `rerank` (CPU CrossEncoder) → вход top‑20..40, батч размер 8–16, таймаут 2 с.  
- `fetch_docs(+windows)` → по id → спаны с цитатами.  
- `compose_context(max_ctx_tokens=1800)` → маркировать цитаты `[channel/msg:offset..]`.  
- `time_normalize` → `dateparser` (RU/EN) к ISO‑интервалам.  
- `tool_runner` → общий контракт, таймаут 4–5 с, трейс.  
- `trace_sink` → stdout/file JSON‑лог.

### 9.2 Параметры GBNF‑декодинга (llama.cpp, CPU)
- `temperature=0.2–0.4`, `top_p=0.9`, `top_k=40–100`, `repeat_penalty=1.1–1.2`, `min_p` не используем, `max_tokens` по задаче (план≤256, rewrite≤192).  
- **Без `stop`** при grammar — завершение регулирует грамматика; контролируем длину через `max_tokens` и семантику грамматики (конечные конструкции).  
- `seed` фиксировать для кеш‑стабильности в планировании.

### 9.3 RRF/MMR/ budgets
- **RRF**: `k=60` (смещения в формуле 1/(k+rank)), объединять все источники/подзапросы.  
- **MMR**: косинус по эмбеддингам, `λ=0.5`, `top_n=40`.  
- **Rerank**: запуск при `RRF_topk≥20`, батчи 8–16, `max_seq_len` 256–384; таймаут 2 с; падение в skip‑rerank при перегрузе.

### 9.4 Compose‑context
- Профиль 4096: `context=1800`, `answer=600–800`, остальное — системный/инструкции/буферы.  
- Сдвигать релевантные спаны вверх; при длинных фрагментах — брать 2–3 окна × 300–500 токенов, со сжатыми заголовками.  
- Контроль `citation‑coverage≥0.8`; иначе — второй раунд retrieval/fusion.

### 9.5 Router‑select (эвристика)
- **bm25**: короткие ≤3 токена, наличие кавычек/операторов (`AND|OR|"|"#|"from:|in:|before:|after:"`), много цифр/дат, упоминания `msg_id/channel_id`.  
- **dense**: длинные/абстрактные/смешанные RU/EN, «почему/как/объясни».  
- **hybrid**: дефолт; смешанные сигналы; многословные с терминами.

### 9.6 Кеши/устойчивость
- TTL: план 10 мин; fusion 5 мин; memo tool 5–15 мин.  
- Валидация: Pydantic на входах tool; жёсткие типы/диапазоны.  
- Деградации: `timeout→skip_rerank`; `empty_source→RRF остальных`; общий пустой→dense‑only с k↑ и без rerank.  
- Параллелить: независимые retrievers и fetch_docs; генерацию LLM сериализовать.

### 9.7 Безопасность
- Нормирующие регексы для subqueries: допускать буквы/цифры/пробел/дефис/подчёркивание/точка/запятая/кавычки/скобки/двоеточие/?/!/ ; баннить `;`, ````, `(){{}}`, `SELECT|INSERT|UPDATE`, URL‑протоколы.  
- Системные инструкции держать вне пользовательского контекста; цитаты — как данные, не инструкции.

---

## Приложение A — микро‑GBNF: массив N строк (RU/EN safe)

```bnf
root ::= ws "[" ws (string (ws "," ws string)*)? ws "]" ws
string ::= "\""" char* "\"""
char ::= utf8safe
utf8safe ::=
    /* Разрешаем буквы/цифры/пробел/дефис/подчёрк/точка/запятая/кавычки/скобки/двоеточие/?/!/ */
    [\u0020-\u007E\p{{L}}\p{{Nd}}] - [\\/<>;`$]
ws ::= [ \t\n\r]*
```

Комментарий: символы `;`, ````, `<`, `>`, `$` исключены для анти‑SQL/код артефактов. При необходимости можно сузить класс символов.

---

## Приложение B — JSON SearchPlan (упрощённая GBNF‑заготовка)

Идея: объект `{{"subqueries":[...], "k_per_query":int, "filters":{{"date_from":"YYYY-MM-DD","date_to":"YYYY-MM-DD"}}}}`. Рекомендуется генерировать GBNF из JSON‑Schema через `json-schema-to-grammar.py` или `json-schema-to-gbnf`.

Полезные ссылки:  
- `llama.cpp grammars README`: https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/blob/main/llama.cpp/grammars/README.md  
- Конвертер JSON‑Schema→GBNF: https://github.com/adrienbrault/json-schema-to-gbnf

---

## Приложение C — минимальный цикл ReAct (псевдокод Python)

```python
STEP_LIMIT = 4
for step in range(STEP_LIMIT):
    # 0) router_select (эвристика, без LLM)
    route = router_select(q)

    # 1) план (GBNF + TTL кеш)
    plan = make_plan(q)  # детерминируемый, grammar-constrained

    # 2) multi_query_rewrite (условно): добавить 0–3 перефраза
    pool = diversify(plan.subqueries, q)

    # 3) retrieval (параллельно по источникам) + RRF (+ MMR)
    cand = retrieve_and_fuse(pool, route, budget_k=120)

    # 4) optional CPU rerank (top-20..40)
    top = rerank(cand, budget_top=40, timeout_s=2) if need_rerank(cand) else cand[:20]

    # 5) fetch_docs(+windows) → compose_context(1800)
    ctx, coverage = compose_context(top, max_tokens=1800)

    if coverage >= 0.8 or step == STEP_LIMIT-1:
        break
    # иначе — один дополнительный раунд с увеличенным k, без rerank
```

---

## Check‑list интеграции
1. Включить GBNF для планировщика; настроить JSON‑fallback.  
2. Добавить `router_select`, `multi_query_rewrite`, `fusion(RRF+MMR)`, `rerank(CPU)`, `compose_context`.  
3. TTL‑кеши: план (10 мин), fusion (5 мин), memo tools (5–15 мин).  
4. Таймауты/лимиты: tool 4–5 с; шагов ≤4; 1 доп. раунд при coverage<0.8.  
5. Лог‑трейс: JSON‑события `{request_id, step, tool, took_ms, ok, error}`.  
6. Санитизация/валидация: Pydantic, фильтры символов, параметризация SQL.  
7. Метрики приёмки: `valid_json_rate≥99.5%`, `avg_subqueries≥3.2`, `recall@20 ≥ baseline +10%`, `citation-coverage≥0.8`, `p95_latency ≤ baseline+15%`.

---

## Примечание о переносе из иных стеков
Все примеры из LangGraph/LangChain/vLLM/Bedrock переносимы: интерфейсы инструментов маппятся на Pydantic‑модели; структурированный вывод заменяется на GBNF под `llama.cpp`; rerank — локальный через Sentence‑Transformers/ONNX/OpenVINO; hybrid — наш BM25 + Chroma + RRF/MMR.
