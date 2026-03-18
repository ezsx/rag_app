# Project Scope: rag_app как портфолио-проект

> Живой документ. Обновляется по мере продвижения.
> Последнее обновление: 2026-03-18

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
- [x] Self-hosted reranker: BGE-M3 через TEI (временно, целевой — Qwen3-Reranker)
- [x] Telegram ingestion: 5 каналов, 1199 точек с smart chunking

### RAG Pipeline
- [x] Qdrant: dense (1024-dim cosine) + sparse (BM25 russian) named vectors
- [x] Hybrid retrieval: RRF fusion → dense re-scoring (MMR-like diversity)
- [x] TEI reranker integration с auto sub-batching
- [x] Two-tier chunking: posts <1500 chars целиком, >1500 recursive split
- [x] UUID5 deterministic point IDs

### Agent
- [x] Native function calling (не regex-парсинг ReAct)
- [x] 5 tools: query_plan → search → rerank → compose_context → final_answer
- [x] Coverage metric: 6-signal composite (cosine similarity based)
- [x] Auto-refinement: дополнительный поиск при coverage < 0.65
- [x] Grounding: citation-forced generation, source validation
- [x] Context overflow protection: _trim_messages

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

### Phase 3.0: Расширение коллекции [ПЕРВЫЙ ШАГ]
- [x] Deep Research: 32 канала отобраны из 70+ кандидатов с evidence
- [x] Валидация: 37/37 каналов доступны (validate_channels.py)
- [ ] Ingest: 32 новых канала, 2025-07-01 → 2026-03-18 (9 мес)
- [ ] Проверить quality: нет ли мусора, рекламы, форвардов-дублей
- [ ] Статистика: точек по каждому каналу, общий объём коллекции

### Phase 3.1: Evaluation Framework [КРИТИЧНО для собесов]
**Почему**: "Как ты меришь качество?" — первый вопрос на Applied LLM собесе.

- [ ] Evaluation dataset: 50+ вопросов, 5 типов (factual, analytical, temporal, comparative, multi-hop)
- [ ] Ground truth: для каждого вопроса — ожидаемые source документы + эталонный ответ
- [ ] Метрики: faithfulness, answer relevance, context recall, context precision
- [ ] Baseline: naive vector search → LLM (без reranker, без refinement)
- [ ] Benchmark: LlamaIndex с тем же Qdrant + тем же LLM
- [ ] Ablation study: отключаем компоненты по одному
- [ ] Автоматический прогон: `python scripts/evaluate_agent.py` → отчёт с таблицами
- [ ] A/B визуализация: графики, сравнительные таблицы

### Phase 3.2: Adaptive Retrieval [КЛЮЧЕВОЕ ОТЛИЧИЕ от фреймворков]
- [ ] Query Classifier: определение типа запроса
- [ ] Strategy Selector: разная стратегия retrieval для каждого типа
- [ ] Adaptive refinement: смена стратегии при низком coverage (не просто "искать ещё")
- [ ] Re-eval: показать дельту на сложных запросах

### Phase 3.3: README + Архитектурная диаграмма [КРИТИЧНО для первого впечатления]
**Почему**: первое что видит интервьюер на GitHub. 5 секунд чтобы зацепить.

- [ ] Mermaid диаграмма полного pipeline
- [ ] Quick start (docker compose up и готово)
- [ ] Скриншот web UI
- [ ] Ключевые метрики из evaluation (faithfulness, latency, coverage)
- [ ] Секция "Design Decisions" — почему self-hosted, почему Qdrant, почему function calling
- [ ] Сравнительная таблица vs LlamaIndex baseline

### Phase 3.4: Observability [ВАЖНО для production-readiness]
- [ ] Latency tracking per tool step
- [ ] Token usage per request
- [ ] Coverage distribution dashboard
- [ ] Error rate monitoring
- [ ] Structured logging (уже частично есть)

### Phase 3.5: Web UI Polish [NICE TO HAVE]
- [ ] Markdown rendering в ответах
- [ ] История чатов (localStorage)
- [ ] Фильтры по каналам в UI
- [ ] Настройки (temperature, max_steps) в UI

---

## Что спрашивают на Applied LLM собесах

Типичные темы и как проект их покрывает:

| Тема | Что спросят | Покрытие в проекте |
|------|------------|-------------------|
| RAG | Chunking strategies, embedding selection, hybrid search | ✅ Полное |
| Evaluation | Как меришь качество RAG | ⚠️ Нужен evaluation framework |
| Prompt engineering | Grounding, system prompts | ✅ Citation-forced generation |
| Agent design | Tool calling, loops, when to stop | ✅ Function calling + coverage threshold |
| Scaling | Batching, caching, context management | ✅ Sub-batching, trim_messages |
| Trade-offs | RAG vs fine-tune, dense vs sparse | ✅ Есть research docs с анализом |
| Infra | Deployment, monitoring | ⚠️ Docker есть, observability нужна |
| LLM internals | Attention, tokenization, inference | ⚠️ Понимание есть, нужно уметь объяснить |

---

## Порядок работы

```
Phase 3.0: Расширение коллекции (37 каналов, 9 мес) [IN PROGRESS]
Phase 3.1: Evaluation framework + benchmarks vs LlamaIndex
Phase 3.2: Adaptive retrieval (query classification + strategy selection)
Phase 3.3: README + архитектурная диаграмма + метрики
Phase 3.4: Observability
Phase 3.5: UI polish
```

---

## Заметки

- Qwen3-Reranker: ждём TEI PR #835, пока BGE-M3 как временная мера
- RTX 5060 Ti недоступна в Docker Desktop (V100 TCC блокирует NVML) — DEC-0024
- Coverage 0.71 после fix (было 0.29) — адекватный уровень
- Latency: несколько секунд на полный pipeline при прогретой модели
