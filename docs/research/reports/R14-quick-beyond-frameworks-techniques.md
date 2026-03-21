# Deep Research: Techniques That Outperform Standard RAG Frameworks

**Для rag_app — portfolio-grade RAG система**
**Дата: 2026-03-20**

---

## 1. Agentic RAG Patterns (2025–2026): Что нового и что работает

### 1.1 A-RAG: Hierarchical Retrieval Interfaces (Feb 2026) — **самое релевантное для тебя**

Февральская статья от Du et al. (arXiv:2602.03442) — прямой ответ на твою проблему "линейного pipeline". Ключевая идея: вместо фиксированного workflow, модели дают **три retrieval tool** напрямую:

- `keyword_search` — BM25/лексический поиск
- `semantic_search` — dense embedding поиск
- `chunk_read` — прямое чтение chunk по ID

Агент сам решает какой tool использовать, в каком порядке, и когда хватит доказательств. Результаты: **94.5% на HotpotQA, 89.7% на 2WikiMultiHop** (с GPT-5-mini). Ablation показывает, что каждый tool вносит значимый вклад.

**Почему это критически важно для тебя:**
У тебя уже есть BM25 + dense + ColBERT. A-RAG показывает, что вместо hard-coded RRF pipeline, ты можешь дать агенту эти три модальности как отдельные tools и позволить ему решать. Это **принципиально** отличается от LlamaIndex — ты не настраиваешь config, а даёшь модели автономию.

**Реализация (2–3 дня):**
1. Создать 3 tool: `bm25_search(query, filters)`, `dense_search(query, filters)`, `colbert_rerank(doc_ids, query)`
2. Добавить `read_document(doc_id)` — прочитать конкретный документ
3. Агент сам делает keyword → dense → rerank или сразу dense → rerank в зависимости от запроса
4. Код и eval suite доступны: `github.com/Ayanami0730/arag`

### 1.2 CRAG (Corrective RAG) — проверенная техника с конкретными цифрами

CRAG добавляет **retrieval evaluator** между retrieval и generation. Классифицирует каждый набор документов как Correct / Incorrect / Ambiguous, и принимает разные действия:

- **Correct** → используй документы напрямую
- **Incorrect** → переформулируй запрос, ищи заново
- **Ambiguous** → расширь поиск, добавь дополнительные источники

Свежее воспроизведение (arXiv:2603.16169, март 2026) показывает: open-source CRAG с Phi-3-mini достигает **54.4% на PopQA** (оригинал 54.9%), а действие "Correct" даёт **78.1% accuracy** — на 26.7 п.п. выше vanilla RAG. Ключевой инсайт из SHAP анализа: **T5 evaluator полагается в основном на named entity alignment**, а не на семантическое сходство.

**Для твоего проекта:**
Тебе не нужен отдельный T5 evaluator — ты можешь использовать Qwen3-30B в thinking mode для оценки relevance. Паттерн:
1. Получил документы → LLM оценивает: "Эти документы отвечают на вопрос? [Correct/Ambiguous/Incorrect]"
2. Если Incorrect → перефразируй запрос и ищи снова
3. Если Ambiguous → расширь поиск (больше docs, другие subqueries)
4. Если Correct → генерируй ответ

**Overhead:** +1 LLM call на оценку, но это ~2-3 секунды на Qwen3-30B. При 40с/запрос — приемлемо.

### 1.3 Self-RAG vs Adaptive-RAG — что применимо без fine-tuning

**Self-RAG** (Asai et al.) требует **fine-tuning** для генерации reflection tokens (Retrieve, ISREL, ISSUP, ISUSE). Без fine-tuning **не применимо** напрямую к Qwen3-30B.

**Adaptive-RAG** комбинирует query analysis с self-corrective RAG. Из paper, routing идёт между:
- No retrieval (LLM знает ответ)
- Single-shot RAG
- Iterative RAG

**Что применимо без fine-tuning:**
- **Query complexity classifier** — LLM классифицирует запрос как simple/moderate/complex и выбирает стратегию. Это легко реализуется через промпт.
- **Self-corrective loop** — LLM оценивает свой собственный ответ: "Этот ответ полностью отвечает на вопрос на основе предоставленных документов? Если нет — что искать дальше?"

### 1.4 Self-Routing RAG (SR-RAG, 2025) — knowledge verbalization

SR-RAG (Wu et al., 2025) — LLM не просто решает "искать или нет", а сначала **вербализирует свои знания**, потом сравнивает с тем, что может дать retrieval. Если LLM уже знает ответ — skip retrieval. Если нет — ищет.

Для Qwen3-30B с thinking mode это интересный паттерн: LLM в thinking mode может рассуждать "Я знаю/не знаю это" перед тем как решить искать ли.

### 1.5 RouteRAG: Rule-Driven Query Routing (2025)

Bai et al. (Sep 2025) показали rule-driven routing agent с бинарными/взвешенными features из запроса (наличие цифр, вопросительная форма, временные маркеры). Результат: **97.6–100.0% EM accuracy**, CPU < 6.2%, до 10x снижение latency по сравнению с non-adaptive pipelines.

**Для тебя это прямой next step:**
Правила для routing:
- Содержит дату/период → добавь date filter
- Содержит имя канала → добавь channel filter
- Содержит "сравни"/"versus" → comparative mode (multi-source)
- Содержит конкретное имя продукта/модели → entity-focused search
- Содержит "сколько"/"когда" → fact lookup mode

---

## 2. Конкретный Plan of Attack на 3–5 дней

### Приоритет 1: Adaptive Tool Router (День 1–2) — **максимальный impact**

Это решает твою корневую проблему: "Pipeline линейный". Реализуй:

**Query Analyzer (через LLM prompt):**
```
Проанализируй запрос пользователя и определи:
1. query_type: factual | temporal | comparative | channel_specific | entity_lookup | broad_overview
2. requires_date_filter: true/false, если true → date_range
3. requires_channel_filter: true/false, если true → channel_names  
4. entity_names: список упомянутых сущностей (модели, компании, люди)
5. search_strategy: keyword_first | semantic_first | hybrid
```

**Specialized Tools:**
- `search_with_date_filter(query, date_from, date_to)` — Qdrant filter по payload.date
- `search_by_channel(query, channel_names)` — Qdrant filter по payload.channel
- `entity_search(entity_name)` — BM25-only поиск по exact entity name
- `broad_search(query)` — текущий RRF pipeline
- `read_document(doc_id)` — прочитать полный текст документа

Это **не** LlamaIndex — ты сам определяешь tools, routing logic, и fallback стратегии.

### Приоритет 2: CRAG Self-Correction Loop (День 2–3)

После retrieval, перед generation:

```
Шаг 1: Retrieval → получили Top-5 документов
Шаг 2: LLM оценивает relevance каждого документа к запросу
  → Score: relevant / partially_relevant / irrelevant
Шаг 3: Если < 2 relevant docs:
  → Query rewriting: LLM переформулирует запрос
  → Retry search с новым запросом
  → Max 2 retries
Шаг 4: Генерация ответа только из relevant документов
```

**Expected impact:** На основе CRAG benchmarks, 10–15% improvement в accuracy за счёт фильтрации irrelevant docs и retry механизма.

### Приоритет 3: Ablation Study + LlamaIndex Baseline (День 3–4)

Это то, что **впечатляет на собеседовании**. Таблица:

| Configuration | Recall@5 | Coverage | Latency | Notes |
|---|---|---|---|---|
| BM25 only | ? | ? | ? | Baseline |
| Dense only | ? | ? | ? | Baseline |
| RRF (BM25 + Dense) | 0.55 | — | ? | Без ColBERT |
| RRF + ColBERT rerank | 0.73 | — | ? | Текущий production |
| + Multi-query | 0.73+ | — | ? | Bug fix impact |
| + Adaptive routing | ? | ? | ? | Новый |
| + CRAG loop | ? | ? | ? | Новый |
| LlamaIndex baseline | ? | ? | ? | Для сравнения |

**LlamaIndex baseline (несколько часов):**
- `VectorStoreIndex` + Qdrant
- Default retriever (dense only)
- Тот же корпус, те же queries
- Покажет: "Мой custom pipeline даёт recall 0.73, LlamaIndex out-of-box — 0.XX"

### Приоритет 4: Expanded Evaluation (День 4–5)

**Scale eval до 50+ questions:**
- Используй LLM для генерации synthetic questions из документов (RAGAS подход)
- Добавь метрики: **faithfulness** (ответ следует из документов) + **latency** + **token cost**
- RAGAS можно подключить для reference-free evaluation

**Faithfulness check** (реализуемо без внешних API):
1. LLM генерирует claims из ответа
2. Для каждого claim — проверяет: "Этот claim следует из предоставленных документов?"
3. Faithfulness = supported_claims / total_claims

---

## 3. Что впечатляет на Applied LLM собеседованиях

### 3.1 "Senior-Level Thinking" vs "Собрал из компонентов"

На основе анализа interview patterns, вот что отличает:

**"Собрал из компонентов":**
- "Я взял LlamaIndex, подключил Qdrant, добавил reranker"
- Нет объяснения *почему* эти компоненты
- Нет измерения impact каждого компонента
- Один pipeline для всех запросов

**"Спроектировал систему" (что у тебя уже есть + что добавить):**

1. **Количественное обоснование каждого решения:**
   - "BM25 3:1 over dense потому что в моём корпусе коротких русско-английских постов keyword match даёт +30% recall. Вот ablation."
   - "ColBERT rerank даёт +97% recall@1 потому что per-token matching ловит partial entity mentions"

2. **Знание того, что НЕ работает и почему:**
   - "Cosine MMR упал 0.70→0.11 потому что re-promotes attractor documents"
   - "PCA whitening 1024→512 потерял recall потому что слишком агрессивный cutoff при том что dense не bottleneck"
   — Это **очень** ценится. Показывает understanding, не просто copy-paste.

3. **Адаптивность к данным:**
   - "Мой корпус — 13K коротких постов из Telegram, а не Wikipedia. Поэтому BM25 доминирует — в коротких документах keyword overlap более сигнальный"
   - "Русский + английский mixed → нужен multilingual embedding, выбрал Qwen3-Embedding-0.6B"

4. **Системное мышление:**
   - "Adaptive routing потому что разные типы запросов требуют разных стратегий — date filter для temporal, channel filter для author-specific"
   - "Self-correction loop потому что retrieval может fail, и нужен fallback"

### 3.2 Ключевые метрики которые ценятся

Помимо recall:

- **Faithfulness** (0–1): Ответ следует из найденных документов? Это #1 метрика в production RAG. RAGAS или LLM-as-judge.
- **Answer Relevance** (0–1): Ответ реально отвечает на вопрос? Не на другой вопрос.
- **Latency** (p50, p95): Критично для production. Покажи trade-off latency vs quality.
- **Cost per query** (tokens consumed): Сколько стоит обработка одного запроса.
- **Hallucination rate**: % ответов с информацией, которой нет в документах.
- **Coverage**: % запросов на которые система может дать ответ.

### 3.3 Как подать проект

**Killer narrative:** "Стандартные фреймворки (LlamaIndex/LangChain) дают recall X на моём корпусе. Мой custom pipeline даёт Y. Вот таблица с ablation каждого компонента и его вкладом."

**Обязательные элементы portfolio:**
- README с architecture diagram
- Ablation table с цифрами
- Comparison с baseline (LlamaIndex)
- Evaluation methodology описание
- "What I tried and what didn't work" section — показывает depth

---

## 4. Техники, НЕ реализуемые стандартными фреймворками

### 4.1 Adaptive Query Routing с Domain-Specific Rules

LlamaIndex имеет `RouterQueryEngine`, но он:
- Routing между готовыми index'ами, а не стратегиями
- Нет dynamic filter generation
- Нет query rewriting с awareness о корпусе

**Твой подход:** LLM анализирует запрос → генерирует Qdrant filters (date range, channel, etc.) → выбирает search strategy → fallback если мало результатов. Это custom logic, не config.

### 4.2 Corpus-Aware Query Expansion

Стандартные фреймворки делают generic multi-query. Ты можешь:
- LLM знает список каналов → если запрос про "директора Яндекса", LLM знает что это канал "gonzo_ml" и добавляет channel filter
- LLM знает временной диапазон корпуса → "недавно" = последний месяц, а не "всегда"
- Entity dictionary → mapping "Vera Rubin" → "NVIDIA GPU server" для query expansion

### 4.3 Progressive Information Gathering (A-RAG стиль)

Вместо одного поискового вызова:
1. Первый поиск → оценка → достаточно информации?
2. Если нет → уточняющий поиск (другой запрос, другие filters)
3. Repeat до получения достаточного контекста или max iterations

### 4.4 Cross-Channel Synthesis с Contradiction Detection

"Что разные каналы пишут про модель X?"
1. Поиск по каждому каналу отдельно (или с channel filter)
2. Сбор perspectives из разных источников
3. LLM: "Есть ли противоречия между этими источниками?"

Это multi-agent pattern — ни один фреймворк не делает это из коробки с Telegram-специфичным корпусом.

### 4.5 Temporal Reasoning

"Что нового за последний месяц?" → не просто date filter, а:
1. LLM определяет временной контекст запроса
2. Qdrant filter по date range
3. Ранжирование по recency (boost новых документов)
4. "Как развивалась тема X за последние 3 месяца?" → sequence extraction по дате

---

## 5. Open-Source примеры и ресурсы

### GitHub репозитории:
- **A-RAG**: `github.com/Ayanami0730/arag` — hierarchical retrieval interfaces, eval suite, MIT license. Используют Qwen3-Embedding-0.6B (как у тебя!)
- **LangGraph Adaptive RAG**: tutorial от LangChain — routing + self-correction в graph. Концепт полезен, но зависимость от OpenAI/Tavily
- **RAGAS**: `github.com/explodinggradients/ragas` — evaluation framework. Faithfulness, context recall, answer relevance. Работает с local LLM
- **CRAG (Meta)**: `github.com/facebookresearch/CRAG` — benchmark с 4,409 QA pairs. Scoring: correct (+1), missing (0), incorrect (−1)

### Benchmarks:
- **CRAG Benchmark** (NeurIPS 2024): 4,409 QA pairs, 5 domains, 8 question categories. Best industry RAG = 63% accuracy without hallucination. Лучшие LLM без RAG = 34%. RAG наивный = 44%.
- **HotpotQA / 2WikiMultiHop**: multi-hop QA. A-RAG достигает 94.5% / 89.7%.
- **RAGBench, T²-RAGBench**: general-purpose и multi-turn.

### Papers (must-read для собеседования):
1. A-RAG (Du et al., Feb 2026) — hierarchical retrieval interfaces
2. CRAG reproduction (Yalavarthi et al., Mar 2026) — open-source CRAG + SHAP анализ evaluator
3. RouteRAG survey (emergentmind.com/topics/routerag) — обзор всех routing подходов
4. Agentic RAG Survey (Singh et al., Jan 2025, arXiv:2501.09136) — таксономия agentic RAG
5. SR-RAG (Wu et al., Mar 2025) — self-routing с knowledge verbalization

---

## 6. Резюме: что делать и в каком порядке

| День | Задача | Expected Impact | Сложность |
|------|--------|-----------------|-----------|
| 1 | Query Analyzer + Route tools (date/channel filters) | Решает 40%+ failed queries | Средняя |
| 2 | Specialized tools + integration в ReAct агент | Pipeline перестаёт быть линейным | Средняя |
| 3 | CRAG self-correction loop | +10-15% accuracy | Низкая |
| 3–4 | LlamaIndex baseline + сравнительная таблица | "Мой pipeline на X% лучше" — killer для собеседования | Низкая |
| 4–5 | Ablation study (полная таблица всех конфигураций) | Показывает engineering depth | Средняя |
| 5 | Faithfulness metric + расширенный eval set | Production-grade evaluation | Средняя |

**Главный тезис для собеседования:**

> "Я построил RAG систему для 13K русско-английских Telegram постов, которая через custom adaptive routing, ColBERT reranking и CRAG self-correction достигает recall@5 = 0.7X — на Y% выше LlamaIndex baseline. Вот ablation table показывающий вклад каждого компонента. Вот что я попробовал и что не сработало (MMR, dense re-scoring, PCA whitening) и почему — с конкретными цифрами."

Это показывает: системное мышление, инженерную глубину, знание trade-offs, и способность измерять impact. Именно это отличает Applied LLM Engineer от "собрал chatbot из tutorials".
