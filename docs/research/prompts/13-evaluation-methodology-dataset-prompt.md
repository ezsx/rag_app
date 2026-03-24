# Deep Research: Evaluation Methodology, Dataset Design и Metrics для Agentic RAG

> **Цель**: Получить конкретный, реализуемый план по созданию evaluation pipeline для agentic RAG-системы с 11+ tools, включая dataset design, LLM judge (multi-criteria), robustness metrics и synthetic question generation. Результат должен быть конвертируемым в SPEC-RAG-14 за 1 день.

---

## Контекст проекта

### Что это
`rag_app` — FastAPI-платформа: 36 Telegram-каналов об AI/ML → Qdrant (13088 points, июль 2025 — март 2026) → ReAct-агент с native function calling → SSE-ответ с цитатами.

### Железо
- **LLM**: Qwen3-30B-A3B GGUF (V100 SXM2 32GB, 3B активных параметров из 30B MoE)
- **Embedding**: Qwen3-Embedding-0.6B (RTX 5060 Ti, 1024-dim)
- **Reranker**: bge-reranker-v2-m3 (cross-encoder)
- **ColBERT**: jina-colbert-v2 (560M, 128-dim per-token MaxSim)
- **Vector store**: Qdrant (dense 1024 + sparse BM25 + ColBERT 128-dim multi-vector, weighted RRF 3:1)

### Retrieval pipeline
```
Query → query_plan (LLM generates sub-queries)
  → [search tool] (BM25 top-100 + dense top-20 → weighted RRF 3:1 → ColBERT MaxSim rerank)
  → rerank (bge-reranker-v2-m3 cross-encoder)
  → channel dedup (max 2/channel)
  → compose_context → final_answer
```

### Текущие 11 LLM tools (SPEC-RAG-13, реализовано)

**Pre-search phase:**
1. `query_plan` — декомпозиция на 3-5 подзапросов
2. `search` — широкий поиск по всей базе (fallback)
3. `temporal_search` — по дате/периоду (queries + date_from + date_to)
4. `channel_search` — по конкретному каналу (queries + channel)
5. `cross_channel_compare` — сравнение каналов по теме (topic, optional dates). Qdrant query_points_groups
6. `summarize_channel` — дайджест канала за период (channel + time_range enum: day/week/month). Qdrant scroll
7. `list_channels` — навигация, количество постов. Qdrant Facet API

**Post-search phase:**
8. `rerank` — cross-encoder переранжирование
9. `compose_context` — сборка контекста с цитатами
10. `final_answer` — финальный ответ с источниками
11. `related_posts` — похожие посты к найденному. Qdrant RecommendQuery

**Системные (не в LLM schema):**
- `fetch_docs` — догрузка полных текстов
- `verify` — верификация финального ответа

### Dynamic tool visibility (phase-based)
- **Pre-search**: query_plan + search variants (max 5, фильтрация по keyword/signal routing)
- **Post-search**: rerank + compose_context + final_answer + related_posts
- Hard cap: max 5 visible tools на каждом шаге
- Signal routing: "в январе" → temporal_search, "gonzo_ml" → channel_search, "сравни" → cross_channel_compare

### Будущие tools (R17 research, ещё НЕ реализованы, но датасет должен их учитывать)

**SPEC-RAG-15 (entity analytics, следующий после eval):**
- `entity_tracker` — entity timeline/comparison/co-occurrence через Facet API на `entities[]` + `year_week`
- `arxiv_tracker` — popular papers, lookup by arxiv_id через Facet API на `arxiv_ids[]`

**SPEC-RAG-16 (pre-computed analytics, опционально):**
- `hot_topics` — trending topics из pre-computed weekly_digests (BERTopic cron)
- `channel_expertise` — authority assessment из pre-computed channel_profiles

### Enriched payload (SPEC-RAG-12, реализовано)
Каждый point в `news_colbert_v2` содержит:
```
text, channel, channel_id, message_id, date, author, url,
entities[], entity_orgs[], entity_models[],
arxiv_ids[], urls[], url_domains[], github_repos[], hashtags[],
lang, year_week, year_month, text_length,
is_forward, forwarded_from_id, forwarded_from_name, reply_to_msg_id,
media_types[], root_message_id, has_arxiv, has_links
```
16 keyword/datetime/integer payload indexes.

### Corpus statistics (актуальные)
- **13088 points** из **36 каналов**
- **Язык**: 12862 ru / 226 en
- **Период**: 2025-07 (1833) → 2025-08 (1691) → ... → 2026-03 (753)
- **Top entities**: OpenAI (1597), Google (1127), Claude (794), Gemini (713), Anthropic (701), GPT-5 (680), NVIDIA (525), Qwen (427), HuggingFace (369), Microsoft (350), DeepSeek (344), Яндекс (321)
- **Top channels by posts**: ai_machinelearning_big_data (1705), xor_journal (1346), data_secrets (1262), seeallochnaya (1029), aioftheday (970), gonzo_ml (820)

---

## Текущий eval — что есть и что НЕ работает

### Датасеты (3 штуки, все ручные)
- **v1** (10 Qs): factual ×3, temporal ×2, channel_specific ×2, comparative ×1, multi_hop ×1, negative ×1. Recall@5 = 0.76.
- **v2** (10 Qs, сложные): entity ×1, product_specific ×3, fact_check ×1, cross_channel ×1, recency ×1, numeric ×1, long_tail_channel ×1, negative ×1. Recall@5 = 0.685.
- **v3** (30 Qs): temporal ×7, channel ×7, entity ×7, broad ×5, negative ×2. LLM judge = 0.71, strict recall = 0.167.

### Формат вопроса
```json
{
  "id": "v3_q01",
  "query": "Вопрос на русском",
  "category": "temporal|channel|entity|broad|negative",
  "expected_answer": "Ожидаемый ответ (для LLM judge)",
  "expected_documents": ["channel_name:message_id"],
  "answerable": true
}
```

### Eval скрипт (evaluate_agent.py)
- SSE streaming parsing (thought/tool_invoked/observation/citations/final events)
- Recall@5 по citation_hits: fuzzy match channel:msg_id (±5 standard, ±50 для temporal)
- Coverage из SSE events
- Latency (agent + baseline /v1/qa)
- `agent_correct` / `baseline_correct` = **всегда None** (нет judge)
- Markdown + JSON отчёты

### Проблемы текущего eval (КРИТИЧЕСКИЕ)
1. **Нет LLM judge** — correctness не оценивается, agent_correct = None
2. **Strict recall неадекватен**: v3 strict recall = 0.167, LLM judge (ad-hoc) = 0.71. Проблема: expected_documents привязаны к конкретным msg_id, а агент находит информацию из ДРУГИХ постов того же канала
3. **Нет tool selection accuracy** — не измеряем правильно ли агент выбирает tool
4. **Нет robustness metrics** — NDR, RSR, order robustness отсутствуют
5. **30 вопросов мало** — недостаточно для статистически значимых выводов
6. **Категории не покрывают все tools** — нет вопросов для cross_channel_compare, summarize_channel, list_channels, related_posts
7. **Нет negative tool selection** — вопросы где агент НЕ должен использовать определённый tool
8. **Нет forward-looking вопросов** — для entity_tracker, arxiv_tracker, hot_topics, channel_expertise

---

## Что говорят ресерчи (YaC AI Meetup R15 + R13/R14)

### R15: Яндекс конференция (3 доклада, март 2026)

**Доклад 1 — RAG-системы (Андрей Соколов, NLP-инженер Яндекса):**
- **Три стадии оценки качества**: (1) бенчмарки (multihop, атрибуция, подтверждённость), (2) асессорские замеры (полезность, фактологичность — отдельно), (3) A/B эксперименты
- **"Чем больше градусников, тем лучше"** — главный тейкавей
- **Robustness-метрики**: Non-degradation rate (NDR) — добавление контекста не ухудшает; Retrieval size robustness (RSR) — больше k → монотонный рост; Retrieval order robustness — shuffle чанков не меняет ответ
- **Композитность** — изменение одного элемента ломает другой, нужен evaluation-driven подход

**Доклад 2 — MarketEye (Владислав Вихров, Яндекс.Маркет):**
- **⚠️ LLM as Judge НЕ РАБОТАЕТ для фактической корректности** — судья видит те же данные что и агент, не может проверить галлюцинации
- **Golden dataset 500 пар** (вопрос, релевантный документ), собран с операторами поддержки
- **AC1 + David-Skene** для взвешивания разметчиков — умнее majority vote
- **Synthetic Qs для embedding** — LLM генерирует вопрос к подстроке документа

**Доклад 3 — Function Calling с синтетикой (Цымбой, Латыпов):**
- **Сложность FC коррелирует с размером tool descriptions в system prompt**, а НЕ с количеством tools или длиной диалога
- **Dynamic tool visibility валидирована** — скрывать ненужные тулы = повышать accuracy
- **1.5M synthetic FC samples** (1.2M EN + 0.3M RU) → SFT Qwen 32B → **Claude 3.5 Sonnet-level FC**
- **Irrelevance subset** — тренировка на случаях когда tool вызывать НЕ нужно

### R13/R14: Tool Router Architecture + Beyond Frameworks

- **Три уровня eval**: (1) routing accuracy (правильная стратегия?), (2) retrieval quality per category (Precision@5, Recall@5, nDCG), (3) end-to-end answer quality per category (LLM-as-judge faithfulness + relevance)
- **Ablation study** — убирай компоненты по одному, показывай вклад каждого
- **20 вопросов минимум на категорию** для статистической значимости
- **T-Eval decomposition**: instruction following, planning, reasoning, retrieval, understanding, review

---

## Что я хочу получить от ресерча

### 1. Dataset design — КОНКРЕТНЫЙ формат и категории

**Мне нужно:**
- Точный JSON-формат вопроса с ВСЕМИ полями которые нужны для multi-criteria eval
- Категории вопросов которые покрывают ВСЕ 11 текущих tools + 4 будущих
- Сколько вопросов на категорию для статистической значимости (у нас 13088 постов, 36 каналов)
- Как сочетать hand-crafted questions (для critical cases) и synthetic (для масштаба)
- Формат expected_tool_sequence — какие tools должен вызвать агент для данного вопроса
- Как обрабатывать negative cases: (a) вопрос без ответа в корпусе, (b) tool НЕ должен использоваться
- Как учитывать будущие tools: вопросы для entity_tracker/arxiv_tracker/hot_topics/channel_expertise которые уже сейчас можно задать как baseline (агент справится через search), а после добавления tools — замерить дельту

**НЕ нужно:**
- Абстрактные рекомендации "добавьте больше вопросов" — нужны конкретные числа и обоснование
- Ссылки на платные инструменты (Humanloop, Weights & Biases) — у нас self-hosted всё

### 2. Synthetic question generation — pipeline для масштабирования

**Мне нужно:**
- Конкретный pipeline для генерации synthetic questions из наших 13088 постов
- Prompt template для LLM (Qwen3-30B-A3B) который генерирует вопрос + expected_answer по тексту поста
- Как контролировать diversity (не 100 factual вопросов, а равномерное распределение по категориям)
- Как верифицировать quality synthetic questions (какой % надо проверить вручную?)
- Как генерировать multi-hop questions (нужны 2+ документа из разных каналов)
- Как генерировать negative questions (вопрос ответа на который НЕТ в корпусе)
- Примеры промптов для каждого типа вопроса

**Яндекс (MarketEye) делает так:**
- Берут случайную подстроку документа → LLM генерирует вопрос
- На входе: подстрока + полный текст + путь в справке + few-shot примеры стиля
- Распределение synthetic вопросов соответствует реальным вопросам

### 3. LLM Judge — multi-criteria, с учётом ограничений

**КРИТИЧЕСКОЕ ограничение от Яндекса:** LLM as Judge НЕ работает для фактической корректности — судья видит те же данные, не может проверить галлюцинации.

**Мне нужно:**
- Multi-criteria judge с РАЗДЕЛЬНЫМИ оценками (не один общий score):
  - **Usefulness** (полезность ответа для пользователя) — LLM judge ОК
  - **Factual correctness** (факты в ответе совпадают с expected_answer) — нужна специальная методология
  - **Citation grounding** (каждое утверждение подкреплено цитатой) — LLM judge ОК
  - **Completeness** (все аспекты expected_answer покрыты) — LLM judge ОК
  - **Refusal accuracy** (для negative Qs: правильно ли отказался отвечать) — binary
- Конкретные prompt templates для каждого критерия (на русском, для Qwen3-30B-A3B или Claude API)
- Шкала оценки: binary (0/1), 3-point (0/0.5/1), 5-point? С обоснованием выбора
- Как решать проблему factual correctness без человеческой разметки:
  - Decompose-then-verify (разбить ответ на claims, проверить каждый)?
  - Cross-reference с expected_answer (keyword overlap + semantic similarity)?
  - Другие подходы?
- Как агрегировать multi-criteria scores в финальный score (weighted average? min?)
- Какой LLM использовать как judge: наш Qwen3-30B-A3B (бесплатно, но 3B active), Claude API (дорого, но точнее), или оба?

### 4. Robustness metrics — NDR, RSR, Order

**Мне нужно:**
- **Non-degradation rate (NDR)**: конкретный протокол. Как "добавлять контекст"? Это значит: прогнать тот же вопрос с k=3 и k=10 и проверить что ответ не ухудшился? Или: добавить irrelevant documents и проверить что ответ не изменился?
- **Retrieval size robustness (RSR)**: тест с k=3, 5, 10, 15, 20. Что значит "монотонный рост"? Recall@k должен расти? Или answer quality?
- **Retrieval order robustness**: shuffle чанков в compose_context. Как измерять "ответ не изменился"? Semantic similarity? Exact match? LLM judge?
- Конкретные формулы для каждой метрики
- Как интегрировать в eval pipeline (дополнительные прогоны или можно совместить?)

### 5. Tool selection accuracy — routing evaluation

**Мне нужно:**
- Как определить "правильный tool" для вопроса (expected_tool_sequence в датасете)
- Формат expected_tool_sequence: конкретный список ["query_plan", "temporal_search", "rerank", "compose_context", "final_answer"] или набор обязательных tools {"temporal_search"}?
- Как считать accuracy: exact match по всей цепочке или по ключевому search tool?
- Как учитывать что один вопрос может быть решён разными tools (channel_search vs search+filter)?
- Как считать "negative tool selection" — агент НЕ должен вызывать temporal_search для entity вопроса
- Метрика: Tool Selection Accuracy = % вопросов где primary search tool выбран правильно?

### 6. Ablation study design

**Мне нужно:**
- Список компонентов для ablation: ColBERT on/off, reranker on/off, RRF weights, forced search, dynamic visibility, original query injection, multi-query search
- Как проводить: каждый компонент отключаем по одному, прогоняем ВЕСЬ датасет
- Какие метрики сравниваем: recall@5, LLM judge score, latency, tool selection accuracy?
- Как визуализировать результаты (таблица, waterfall chart?)
- Сколько это займёт времени (13088 постов, ~40 сек/запрос для агента)

### 7. Eval pipeline integration

**Мне нужно:**
- Как модифицировать `evaluate_agent.py` для поддержки:
  - LLM judge (multi-criteria)
  - Tool selection tracking (из SSE tool_invoked events)
  - Robustness tests (multiple runs per question)
  - Ablation mode (config для отключения компонентов)
- Формат выходного JSON-отчёта
- Как сравнивать прогоны между собой (before/after feature)

---

## Что я НЕ хочу

1. **Абстрактных рекомендаций** — "используйте LLM judge" без конкретного prompt template
2. **Ссылок на платные сервисы** — всё self-hosted (Qwen3-30B-A3B или Claude API)
3. **Переусложнения** — мне НЕ нужна full RAGAS/ARES setup, нужен targeted evaluation
4. **Игнорирования наших ограничений** — 3B активных параметров у Qwen3, ~40 сек/запрос, single GPU
5. **Timeline оценок** — не нужны "3-5 дней на фазу", мне нужны конкретные артефакты

---

## Формат ответа

Структурируй отчёт по секциям:

1. **Dataset specification** — формат, категории, sizing, примеры вопросов (по 2-3 на категорию)
2. **Synthetic generation pipeline** — prompt templates, diversity control, verification
3. **LLM Judge prompts** — конкретные промпты для каждого из 5 критериев, шкала, агрегация
4. **Robustness metrics** — формулы, протоколы тестирования, интеграция
5. **Tool selection metrics** — формат, формулы, expected_tool_sequence design
6. **Ablation study protocol** — компоненты, порядок, визуализация
7. **Implementation plan** — конкретные изменения в evaluate_agent.py, dataset format migration

Каждая секция должна содержать:
- Конкретный deliverable (файл, формат, промпт)
- Обоснование решений (со ссылками на papers где возможно)
- Пример (реальный, из нашего корпуса AI/ML Telegram-каналов)

---

## Приложения (будут прикреплены к промпту)

1. **R15** — полный отчёт YaC AI Meetup (Яндекс конференция): RAG-системы, мультиагенты MarketEye, FC с синтетикой
2. **Текущие eval datasets** (v1, v2, v3) — для понимания текущего формата и уровня вопросов
