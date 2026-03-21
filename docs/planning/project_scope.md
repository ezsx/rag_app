# Project Scope: rag_app как портфолио-проект

> Живой документ. Обновляется по мере продвижения.
> Последнее обновление: 2026-03-22

---

http://localhost:8001/

## Зачем этот проект

**Цель**: визитная карточка для позиции **Applied LLM Engineer** ($2–3k/мес).

Проект должен показать интервьюеру:
1. Умение строить **production RAG pipeline** от ingestion до generation
2. Понимание **agent orchestration** (не просто "вызвал API")
3. Инженерную зрелость: evaluation, observability, trade-off analysis
4. Self-hosted inference: работа с железом, оптимизация, ограничения

**Контекст автора**:
- Backend-разработчик (Python), ML с вузовских времён
- Успешный ML-проект: обнаружение аневризм мозга (MRI), внедрён в больницах
- VPN-сервис с нуля за 2.5 мес (AWG + Xray, без готовых панелей)
- Понимание Transformer архитектуры, CUDA параллелизм на уровне концепций

---

## Что уже сделано (Phase 1–2)

### Инфраструктура
- [x] Docker compose (API + Qdrant), thin Dockerfiles
- [x] Self-hosted LLM: Qwen3-30B-A3B MoE на V100 SXM2 32GB
- [x] Self-hosted embedding: Qwen3-Embedding-0.6B через TEI (RTX 5060 Ti)
- [x] Self-hosted reranker: BGE-M3 через gpu_server.py (временно, целевой — bge-reranker-v2-m3)
- [x] Telegram ingestion: 36 каналов, 13124 точки с smart chunking
- [x] gpu_server.py: embedding + reranker в одном HTTP-сервере (PyTorch cu128, RTX 5060 Ti)

### RAG Pipeline
- [x] Qdrant: dense (1024-dim cosine) + sparse (BM25 russian) + ColBERT (128-dim multi-vector MaxSim)
- [x] Weighted RRF fusion (BM25 weight=3, dense weight=1) — dense re-score убран (убивал BM25)
- [x] bge-reranker-v2-m3 cross-encoder reranker через gpu_server.py (logit gap 8→18 vs старый bge-m3)
- [x] ColBERT reranking: jina-colbert-v2 (560M, 128-dim per-token MaxSim) — recall@1 +97%, recall@5 +33%
- [x] Two-tier chunking: posts <1500 chars целиком, >1500 recursive split
- [x] UUID5 deterministic point IDs
- [x] Оригинальный запрос пользователя всегда в subqueries (BM25 keyword match)
- [x] Multi-query search: все LLM subqueries через round-robin merge (critical bug fix, v2 recall +33%)

### Agent
- [x] Native function calling (не regex-парсинг ReAct)
- [x] 5 tools: query_plan → search → rerank → compose_context → final_answer
- [x] Dynamic tool availability: final_answer скрыт до выполнения search
- [x] Forced search: если LLM пропускает tools, принудительный search с оригинальным запросом
- [x] Coverage metric: 6-signal composite (cosine similarity based)
- [x] Auto-refinement: дополнительный поиск при coverage < 0.65
- [x] Grounding: citation-forced generation, source validation
- [x] Context overflow protection: _trim_messages
- [x] Channel dedup: max 2 docs/channel для diversity

### API & UI
- [x] FastAPI + SSE streaming (/v1/agent/stream)
- [x] JWT auth
- [x] Web UI: chat interface, real-time tool chain visualization

---

## Стратегия: закрытая база знаний (не web search)

**Решение**: RAG над закрытой курированной базой, **без web search**.

Почему это сильнее:
- В enterprise RAG **всегда** над закрытыми данными (внутр. документы, Slack, Jira, CRM)
- Web search = Perplexity/Tavily, подключил API и готово, не интересно для собеса
- Закрытая база показывает: grounding, traceability, audit trail — то что нужно бизнесу
- Telegram-каналы = аналог "корпоративных источников информации" — паттерн 1:1

---

## Коллекция Telegram-каналов

> Отобраны из 70+ кандидатов через Deep Research (отчёт: `docs/research/rag-stack/reports/compass_artifact_*.md`).
> Критерии: авторитетность автора, активность 2025–2026, оригинальный контент, фактологическая плотность.
> Валидация: `scripts/validate_channels.py` — 36/37 доступны через web-preview, 37/37 через Telethon API.

### Ingest параметры
- **Период**: 2025-07-01 → 2026-03-18 (9 месяцев)
- **Обоснование**: AI landscape меняется каждые 3 мес → данные старше 9 мес теряют релевантность;
  9 мес = 3 квартала → temporal queries ("что изменилось с лета"); ~5000–7000 постов → ~8000–12000 чанков

### Existing (5):
| Канал | Категория |
|-------|-----------|
| `@protechietich` | Tech новости |
| `@data_secrets` | Data Science |
| `@ai_machinelearning_big_data` | AI/ML общее |
| `@data_easy` | ML доступно |
| `@xor_journal` | Техно-журнал |

### New — LLM новости и релизы (3):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@ai_newz` | Артём Санакоев, PhD, ex-Meta GenAI | Основной источник LLM-новостей, 85K подписчиков |
| `@neurohive` | Команда neurohive.io | Технические обзоры SOTA-моделей, CV+NLP |
| `@denissexy` | Денис, IT-блогер | Авторские тесты моделей (уникальная методология) |

### New — Research papers (4):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@gonzo_ml` | Григорий Сапунов (CTO Intento, ex-Яндекс) | Эталон обзоров papers, макс. фактологическая плотность |
| `@seeallochnaya` | Игорь Котенков, ML/DS | 72K подписчиков, бенчмарки, аналитика |
| `@dendi_math_ai` | Денис Димитров, Sber AI Research (Kandinsky) | Инсайды российского AI research |
| `@complete_ai` | Андрей Кузнецов, к.т.н., AIRI FusionBrain | Взгляд из ведущей российской AI-лаборатории |

### New — Applied ML / production (4):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@llm_under_hood` | Ринат Абдуллин, Head of ML (Австрия) | RAG, production LLM, Enterprise кейсы |
| `@varim_ml` | ML-инженер | ML engineering, системный дизайн |
| `@boris_again` | Борис Цейтлин, ex-eBay | Обзоры, кейсы, взгляд международного практика |
| `@cryptovalerii` | Валерий Бабушкин (ex-Yandex, WhatsApp, BP) | Production DS/ML, ML System Design |

### New — Open-source (2):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@scientific_opensource` | ITMO OpenSource, Николай Никитин | Научный open-source, ML-инструменты |
| `@ruadaptnaya` | Исследователи | Адаптация LLM для русского языка |

### New — AI индустрия (5):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@techsparks` | Андрей Себрант, директор Яндекса | Индустриальный контекст от инсайдера |
| `@addmeto` | Григорий Бакунов (Bobuk), ex-Яндекс | AI/IT-контекст, быстрая реакция |
| `@aioftheday` | Александр Горный, ex-Mail.Ru Group | AI-продукты и стартапы |
| `@singularityfm` | Виталий Тарнавский, Head of AI Т-Банк | AI в fintech |
| `@oulenspiegel_channel` | Сергей Марков, автор книги об ИИ | AI-ландшафт, этика |

### New — NLP (2):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@rybolos_channel` | Татьяна Шаврина, NLP-исследователь | Собственные NLP-исследования, 18.5K подп. |
| `@stuffynlp` | NLP-команда Яндекса | Технические разборы NLP-статей |

### New — MLOps (1):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@MLunderhood` | ML-команда Яндекса | ML-инфраструктура, конференции |

### New — Computer Vision (2):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@deep_school` | Команда deepschool.ru | CV, ML/DL теория, собеседования |
| `@CVML_team` | CV-инженер | Embedded CV, edge AI |

### New — Data Science (2):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@smalldatascience` | А. Дьяконов, профессор МГУ, профессор РАН | Академический ML, математика |
| `@inforetriever` | К. Хрыльченко, R&D лид RecSys Яндекс | Information retrieval — прямая релевантность для RAG |

### New — AI этика (1):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@theworldisnoteasy` | Карелов С., исследователь | AI этика, когнитивистика, 45K подп. |

### New — Кросс-категорийные (6):
| Канал | Автор/credentials | Что даёт |
|-------|-------------------|----------|
| `@aihappens` | Алексей Хахунов, CEO Dbrain.io | AI-индустрия от CEO стартапа |
| `@AIgobrr` | Александр Курогло | Research + практика + open-source |
| `@toBeAnMLspecialist` | Михаил Крыжановский | ML-архитектуры, собеседования, подкаст |
| `@ml_product` | — | Research → продукты, стартапы |
| `@techno_yandex` | Команда Яндекса | Крупнейший корпоративный AI-канал (80K) |
| `@atmyre_channell` | Татьяна Гайнцева, Philips | AI в healthcare |

### Отклонённые каналы
| Канал | Причина |
|-------|---------|
| `@dlinnlp` | Заморожен с ~2021 |
| `@machinelearning_ru` | Агрегатор без авторского контента |
| `@ml_world` | Потерял ML-фокус |
| `@tproger_official` | Широкий IT, мало оригинального AI-контента |

**Итого: 37 каналов** (5 existing + 32 new). Покрытие 10 тематических областей.
Eval-запросы: фактические, аналитические, temporal, comparative, multi-hop — все покрыты.

---

## Стратегия конкуренции с фреймворками

### Где LlamaIndex/LangChain будут ≈ одинаковы
На **простых фактических запросах** с тем же Qdrant + reranker: примерно тот же результат.
Это нормально — фреймворки уже умеют базовый RAG.

### Где мы выигрываем: Adaptive Retrieval

Фреймворки используют **одну стратегию на все запросы**. Наш pipeline:

```
Запрос → Query Classifier (тип) → Strategy Selector → Adaptive Execution → Evidence Synthesis
```

| Тип запроса | Пример | Стратегия |
|------------|--------|-----------|
| Фактический | "Когда вышла Vera Rubin?" | Точный поиск по сущности, малый top_k |
| Аналитический | "Тренды в open-source LLM" | Широкий контекст, много источников, большой top_k |
| Temporal | "Что изменилось за месяц?" | Вес по дате, фильтры date_from/date_to |
| Сравнительный | "GPT-4o vs Claude — что лучше?" | Multi-query, entity extraction |
| Multi-hop | "Как анонс NVIDIA повлиял на стартапы?" | Связь между документами, цепочка поисков |

Это то, что **ни один фреймворк не делает** из коробки.

### Ablation Study
Отключаем по одному компоненту, показываем дельту:
- Без reranker → faithfulness -X%
- Без refinement → recall -Y%
- Без hybrid (только dense) → precision -Z%
- Без adaptive strategy → сложные запросы -W%

---

## Что осталось (Phase 3) — приоритет по влиянию на портфолио

### Phase 3.0: Расширение коллекции [ЗАВЕРШЕНО]
- [x] Deep Research: 32 канала отобраны из 70+ кандидатов с evidence
- [x] Валидация: 37/37 каналов доступны (validate_channels.py)
- [x] Ingest: 36 каналов, 13124 точки, период 2025-07-01 → 2026-03-18
- [ ] Проверить quality: нет ли мусора, рекламы, форвардов-дублей
- [ ] Статистика: точек по каждому каналу, общий объём коллекции

### Phase 3.1: Evaluation Framework + Pipeline Optimization [ЗАВЕРШЕНО — recall 0.15→0.76]

**Eval pipeline (готово):**
- [x] Quick golden dataset v1: 10 вопросов (factual, temporal, channel, comparative, multi_hop, negative)
- [x] Quick golden dataset v2: 10 вопросов (entity, product, fact_check, cross_channel, recency, numeric, long_tail, negative)
- [x] Retrieval eval: 100 auto-generated queries, прямые Qdrant запросы без LLM (~5с/query)
- [x] Per-category fuzzy matching: ±5 factual, ±50 temporal/multi_hop
- [x] 22 прогона eval с разными конфигурациями (history в playbook)

**Pipeline optimization (recall@5: 0.15 → 0.76):**
- [x] Убрали dense re-score (recall 0.15 → 0.33)
- [x] Original query injection (recall 0.33 → 0.59)
- [x] Weighted RRF 3:1 + forced search + dynamic tools (recall 0.59 → 0.70)
- [x] Per-category matching (recall 0.70 → 0.76)
- [x] ColBERT reranking — jina-colbert-v2, recall@1 +97%, recall@5 +33% (retrieval eval)
- [x] Multi-query search fix — critical bug, v2 recall 0.46 → 0.61 (+33%)
- [x] Reranker upgrade — bge-reranker-v2-m3 (logit gap 8→18)
- [x] Channel dedup (max 2/channel)
- [x] MMR, PCA whitening 512/1024, DBSF — протестированы и отклонены с evidence

**Текущие метрики (2026-03-21):**
| Dataset | Recall@5 | Coverage | Тип |
|---------|----------|----------|-----|
| v1 (10 Qs) | **0.76** | 0.86 | Agent eval |
| v2 (10 Qs, сложные) | **0.685** | 0.81 | Agent eval (с LLM tool selection) |
| 100 Qs, RRF+ColBERT | **0.73** | — | Retrieval eval |

> v2 recall 0.61 → **0.685** (+12.3%) после LLM tool selection. LLM сам выбирает temporal_search/channel_search/broad search. Q8 (long_tail): 0.0→1.0.

**Что НЕ сработало (с evidence):**
| Техника | Результат | Почему |
|---------|-----------|--------|
| Cosine MMR | 0.70→0.11 | Re-promotes attractor documents |
| Dense re-score после RRF | 0.33→0.15 | Стирает BM25 вклад |
| PCA whitening 1024→512 | 0.70→0.56 | Слишком агрессивный cutoff |
| Whitening 1024→1024 | паритет | Dense не bottleneck при BM25 3:1 |
| DBSF fusion | 0.72 vs RRF 0.73 | RRF чуть лучше |

**Roadmap:** см. `docs/planning/retrieval_improvement_playbook.md`

### Phase 3.2: Adaptive Retrieval + Tool Router [КЛЮЧЕВОЕ — СЛЕДУЮЩИЙ ПРИОРИТЕТ]

> **Это ГЛАВНОЕ отличие от LlamaIndex/LangChain.** Все 4 исследования (R13-quick, R13-deep, R14-quick, R14-deep) единогласно рекомендуют.
> Подробный план: `docs/planning/adaptive_retrieval_plan.md`

**Корневая проблема**: pipeline линейный — каждый запрос идёт одним путём. Temporal, channel, entity запросы требуют разных стратегий. Это то, что фреймворки НЕ делают из коробки.

**Архитектура (консенсус 4 ресерчей):**

1. **Routing встроен в query_plan** — zero extra LLM calls
   - Расширение JSON schema: поле `strategy` (enum: broad/temporal/channel/entity) + `filters` (date_from, date_to, channels, entities)
   - Constrained decoding через llama-server grammar enforcement
   - Refs: R13-deep §1, R13-quick §1.2

2. **Rule-based pre-validator** (<1ms) — deterministic safety net
   - Regex: даты, имена каналов, entity patterns
   - Override LLM routing когда confidence > 0.8
   - Refs: R13-deep §1, R13-quick §1.4

3. **4 специализированных tools** — thin wrappers вокруг base_search
   - `broad_search(queries, k)` — текущий hybrid search (default/fallback)
   - `temporal_search(queries, date_from, date_to)` — Qdrant DatetimeRange filter
   - `channel_search(queries, channel)` — Qdrant MatchValue filter
   - `entity_search(entity, queries, date_from?, date_to?)` — BM25 keyword boost + optional filters
   - Все tools комбинируют фильтры: "gonzo_ml в январе 2026" = channel + temporal в одном запросе
   - Max 5 tools для Qwen3-30B (3B active) — "Less is More" paper показывает деградацию при >5-7
   - Refs: R13-quick §2, R13-deep §2, R14-quick §Приоритет 1

4. **Dynamic tool visibility** — сужение по query signals
   - Нет дат → скрыть temporal_search
   - Нет @каналов → скрыть channel_search
   - Refs: R13-deep §2, R13-quick §2.4

5. **3-tier fallback chain** — no query returns empty
   - Specialized → broadened (relax filters) → broad_search
   - ColBERT scores как quality gate: <0.3 → reformulate, 0.3-0.6 → expand, >0.6 → use
   - Refs: R13-deep §3, R14-deep (CRAG)

6. **AgentState расширение**
   - `strategy`, `applied_filters`, `routing_source` (llm/rules/fallback)
   - Refs: R13-quick §3.2

**Ожидаемый эффект:**
- +8-15% recall на temporal/channel/entity запросах (Adaptive-RAG paper, NAACL 2024)
- Решает: Q6 (NVIDIA/Vera Rubin — temporal+entity), Q1/Q3/Q7 (entity boost)
- Papers: Adaptive-RAG +5-31pp, CRAG +7-37%, RouteRAG 97.6-100% EM accuracy

**Задачи:**
- [x] Расширить query_plan JSON schema (strategy enum + filters)
- [x] Rule-based hint extraction (dates, channels, entities) — `src/services/query_signals.py`
- [x] Strategy dispatch с Qdrant filters (temporal: DatetimeRange, channel: MatchValue)
- [x] 3-tier fallback chain (specialized → broadened → broad)
- [x] AgentState: strategy, applied_filters, routing_source
- [x] LLM-visible specialized tools (temporal_search, channel_search) + dynamic visibility
- [x] LLM сам выбирает tool → v2 recall 0.61→0.685 (+12.3%)
- **Отложено**: entity inject в queries — ломает round-robin баланс
- [ ] CRAG-lite: ColBERT scores как quality gate
- [ ] Обновить eval dataset: альтернативные expected sources
- [ ] Routing accuracy eval (отдельный от recall)

**Результаты (2026-03-21):**
- LLM выбирает `temporal_search` для "NVIDIA в начале 2026" → date filter → Vera Rubin найден ✅
- LLM выбирает `channel_search` для "gonzo_ml про трансформеры" → channel filter ✅
- LLM выбирает `search` (broad) для "Сравни GPT-5 и Claude" → без фильтров ✅
- Dynamic visibility: 3-4 tools видны LLM вместо 7 → лучший accuracy tool selection

### Phase 3.3: Eval перестройка + Ablation Study [INTERVIEW KILLER]

> Strict document matching неадекватна для open-ended вопросов (strict recall=0.167, LLM judge=0.71 на тех же 6 ответах).
> "Чем больше градусников, тем лучше" — Андрей Соколов, Яндекс (R15).
> YaC AI Meetup (R15) подтверждает направление, но R15 — анализ конференции, не наш собственный ресерч. Треки из R15 требуют отдельных deep research (R16+) перед реализацией.

**Шаг 1 — Multi-criteria LLM judge (заменяет strict recall):**
- [ ] Прогнать eval v3 (30 Qs) на текущем коде (system prompt fix + hints injection)
- [ ] Claude как judge оценивает каждый ответ по 4 критериям (0-1): полезность, фактологичность, подтверждённость цитатами, полнота
- [ ] Каждый критерий оценивается отдельно (Яндекс: LLM judge fails на фактах, если оценивать одним числом)
- [ ] Strict recall@5 остаётся как вспомогательная метрика для entity/product вопросов
- **Конкретный результат**: таблица 30 вопросов × 4 критерия + overall score
- Ref: R15 — Соколов §10 (3 стадии eval), Вихров (LLM judge fails для фактов)

**Шаг 2 — Robustness-метрики:**
- [ ] **RSR (Retrieval Size Robustness)**: один и тот же вопрос с k=3, 5, 10, 15 — качество должно монотонно расти
- [ ] **Order robustness**: shuffle top-5 чанков перед генерацией — ответ не должен существенно меняться (lost-in-the-middle)
- [ ] **NDR (Non-Degradation Rate)**: ответ с контекстом не хуже ответа без контекста
- **Конкретный результат**: 3 новые метрики в eval pipeline, числа в playbook
- Ref: R15 — Соколов §6. Требует отдельный deep research (R16) для методологии измерения

**Шаг 3 — Ablation study:**
- [ ] LlamaIndex baseline: VectorStoreIndex + Qdrant + тот же корпус 13K docs
- [ ] Ablation table: отключаем по одному компоненту (ColBERT, RRF weighting, multi-query, tool selection) и меряем
- [ ] Per-component Δ: вклад каждого компонента в recall и latency
- **Конкретный результат**: таблица "с компонентом vs без" с числами

**Для собеса**: "Мой custom pipeline даёт recall 0.7X по 4 критериям, LlamaIndex out-of-box — 0.XX. Вот ablation каждого компонента. Плюс 3 robustness-метрики."

### Phase 3.4: Advanced Techniques [WEEK 2+]

> Треки из R15. Каждый требует отдельного ресерча (R16-R18) перед реализацией.

**RAG Necessity Classifier** — не всем запросам нужен поиск:
- [ ] Deep research (R16): изучить подходы — perplexity, tool call decision, обратный сигнал (Соколов), irrelevance subset при обучении (Цымбой)
- [ ] Простая эвристика как proof of concept: conversational queries → skip RAG
- [ ] Если эвристика работает → обучить быстрый классификатор на логах агента
- **Target**: -25% latency, +quality (Яндекс достиг этого)
- Ref: R15 — Соколов §7, Цымбой §irrelevance subset

**SFT/RLHF для RAG и Function Calling:**
- [ ] Deep research (R17): pipeline синтетических данных для FC (Цымбой: 1.5M сэмплов, Qwen 32B → Claude-level)
- [ ] Собрать 1K+ трейсов агента из eval прогонов
- [ ] Если достаточно данных → LoRA/QLoRA Qwen3-30B на V100
- **Target**: +19% по RAG (Соколов), Claude-level FC accuracy (Цымбой)
- Ref: R15 — Соколов §8, Цымбой §SFT+GRPO. Яндекс: fine-tune диспетчера НЕ помог (Вихров), но SFT самой модели помог

**NLI Citation Verification** — faithfulness metric:
- [ ] XLM-RoBERTa-large-xnli (Russian+English, ~400M params, CPU или 5060 Ti)
- [ ] Decompose ответ на claims → NLI entailment check
- [ ] Target faithfulness ≥ 0.92
- Ref: R14-deep §NLI

**NLI Citation Verification** — faithfulness metric:
- [ ] XLM-RoBERTa-large-xnli (Russian+English, ~400M params, CPU или 5060 Ti)
- [ ] Decompose ответ на claims → NLI entailment check
- [ ] Target faithfulness ≥ 0.92
- Ref: R14-deep §NLI

**Multi-Source Synthesis** — channel-aware analysis:
- [ ] Противоречия между каналами → явный флаг
- [ ] Channel attribution в ответах
- Ref: R14-deep (Astute RAG: +6.85%)

**Speculative RAG** — dual-GPU architecture:
- [ ] Drafter на 5060 Ti, Verifier на V100
- [ ] Несколько кандидатов ответов → выбор лучшего
- Ref: R14-deep (Google ICLR 2025: +12.97%, -50.83% latency)

**Context Compression:**
- [ ] Qwen2.5-1.5B-Instruct compressor на 5060 Ti
- [ ] 3.6% token compression, +3.2 EM (CORE 2025)

### Phase 3.5: README + Архитектурная диаграмма [КРИТИЧНО для первого впечатления]
- [ ] Mermaid диаграмма полного pipeline (включая adaptive routing)
- [ ] Quick start (docker compose up и готово)
- [ ] Скриншот web UI
- [ ] Ablation table + comparison table
- [ ] Секция "Design Decisions" — почему self-hosted, почему Qdrant, почему function calling
- [ ] Секция "What didn't work" — с цифрами (MMR, whitening, etc.)

### Phase 3.6: Observability [ВАЖНО для production-readiness]
- [ ] Latency tracking per tool step
- [ ] Token usage per request
- [ ] Routing decision logging (strategy, source, filters)
- [ ] Error rate monitoring
- [ ] Structured logging (уже частично есть)

### Phase 3.7: Web UI Polish [NICE TO HAVE]
- [ ] Markdown rendering в ответах
- [ ] История чатов (localStorage)
- [ ] Визуализация routing decision (какой tool выбран и почему)
- [ ] Фильтры по каналам в UI

---

## Что спрашивают на Applied LLM собесах

Типичные темы и как проект их покрывает:

| Тема | Что спросят | Покрытие в проекте |
|------|------------|-------------------|
| RAG | Chunking strategies, embedding selection, hybrid search | ✅ Полное |
| Evaluation | Как меришь качество RAG | ✅ Eval pipeline + golden dataset + ablation study |
| Prompt engineering | Grounding, system prompts | ✅ Citation-forced generation |
| Agent design | Tool calling, loops, when to stop | ✅ Function calling + coverage threshold |
| Scaling | Batching, caching, context management | ✅ Sub-batching, trim_messages |
| Trade-offs | RAG vs fine-tune, dense vs sparse | ✅ Есть research docs с анализом |
| Infra | Deployment, monitoring | ⚠️ Docker есть, observability нужна |
| LLM internals | Attention, tokenization, inference | ⚠️ Понимание есть, нужно уметь объяснить |

---

## Порядок работы

```
Phase 3.0: Расширение коллекции (36 каналов, 13124 точки)                    [DONE]
Phase 3.1: Evaluation + Pipeline Optimization (recall 0.15→0.76)             [DONE]
Phase 3.2: Adaptive Retrieval + Tool Router                                   [DONE — LLM tool selection работает]
Phase 3.3: Eval Revolution + Ablation Study                                   [NEXT — LLM judge + robustness]
Phase 3.4: Advanced Techniques (RAG classifier, SFT, NLI, Speculative RAG)  [Week 2+]
Phase 3.5: README + архитектурная диаграмма + comparison tables              [Параллельно с 3.3]
Phase 3.6: Observability                                                      [После 3.5]
Phase 3.7: UI polish                                                          [Lowest priority]

Исследовательская база: R01-R15 (15 отчётов, включая R15 — анализ Яндекс RAG конфы)
```

---

## Исследовательская база

### Завершённые ресерчи (ключевые для следующих фаз)

| # | Отчёт | Тема | Ключевой вывод |
|---|-------|------|----------------|
| R11 | `reports/R11-advanced-retrieval-strategies.md` | Advanced Retrieval | ColBERT, weighted RRF, whitening, entity extraction |
| R12 | `reports/R12-cluster-based-retrieval.md` | Кластеризация | Отложена — effort > impact при 13K docs |
| **R13-quick** | `reports/R13-quick-tool-router-architecture.md` | Tool Router (quick) | 4 tools, routing в query_plan, rule-based hints |
| **R13-deep** | `reports/R13-deep-tool-router-architecture.md` | Tool Router (deep) | Grammar enforcement, 3-tier fallback, ColBERT quality gate |
| **R14-quick** | `reports/R14-quick-beyond-frameworks-techniques.md` | Beyond Frameworks (quick) | A-RAG, CRAG, RouteRAG, interview strategy |
| **R14-deep** | `reports/R14-deep-beyond-frameworks-techniques.md` | Beyond Frameworks (deep) | Speculative RAG, NLI verification, temporal reasoning, 5-day plan |

### Ключевые papers (для собеседования)

| Paper | Год | Relevance | Ключевая цифра |
|-------|-----|-----------|-----------------|
| Adaptive-RAG (Jeong et al.) | NAACL 2024 | Routing architecture | +5-31pp accuracy |
| CRAG (Yan et al.) | 2024 | Self-correction | +7-37% accuracy |
| Self-RAG (Asai et al.) | ICLR 2024 | Adaptive retrieval | +31.4pp long-tail |
| A-RAG (Du et al.) | Feb 2026 | Hierarchical tools | 94.5% HotpotQA |
| "Less is More" (arXiv:2411.15399) | 2024 | Tool count limits | 35%→87% accuracy при subset tools |
| Speculative RAG (Wang et al.) | ICLR 2025 | Dual-GPU | +12.97%, -50.83% latency |
| RouteRAG (Bai et al.) | Sep 2025 | Rule-driven routing | 97.6-100% EM accuracy |
| FAIR-RAG | 2025 | Iterative refinement | F1 0.453 HotpotQA (+8.3pp) |
| Astute RAG (Wang et al., Google) | 2024 | Multi-source synthesis | +6.85% avg accuracy |
| ChronoQA | Nature 2025 | Temporal QA | 72% ошибок от temporal retrieval |

---

## Заметки

- Коллекция: `news_colbert` (dense 1024 + sparse BM25 + ColBERT 128-dim MaxSim)
- gpu_server.py: 3 модели (Qwen3-Embedding-0.6B + bge-reranker-v2-m3 + jina-colbert-v2), ~4-5 GB VRAM, ~11GB свободно на 5060 Ti
- RTX 5060 Ti недоступна в Docker Desktop (V100 TCC блокирует NVML) — DEC-0024
- Recall@5: 0.15 → 0.76 (v1), 0.61 (v2) через 22 eval прогона
- Retrieval recall@5: 0.73 на 100 запросах (ColBERT + RRF)
- Coverage: 0.86 (v1), 0.80 (v2)
- Latency: ~30-45с на полный pipeline (LLM inference ~12с × 2-3 calls — основное узкое место)
- Retrieval improvement playbook: `docs/planning/retrieval_improvement_playbook.md`
- Adaptive retrieval plan: `docs/planning/adaptive_retrieval_plan.md`
