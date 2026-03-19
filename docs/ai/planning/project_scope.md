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
- [x] Self-hosted reranker: BGE-M3 через gpu_server.py (временно, целевой — bge-reranker-v2-m3)
- [x] Telegram ingestion: 36 каналов, 13124 точки с smart chunking
- [x] gpu_server.py: embedding + reranker в одном HTTP-сервере (PyTorch cu128, RTX 5060 Ti)

### RAG Pipeline
- [x] Qdrant: dense (1024-dim cosine) + sparse (BM25 russian) named vectors
- [x] Weighted RRF fusion (BM25 weight=3, dense weight=1) — dense re-score убран (убивал BM25)
- [x] BGE-M3 reranker через gpu_server.py (AutoModelForSequenceClassification)
- [x] Two-tier chunking: posts <1500 chars целиком, >1500 recursive split
- [x] UUID5 deterministic point IDs
- [x] Оригинальный запрос пользователя всегда в subqueries (BM25 keyword match)

### Agent
- [x] Native function calling (не regex-парсинг ReAct)
- [x] 5 tools: query_plan → search → rerank → compose_context → final_answer
- [x] Dynamic tool availability: final_answer скрыт до выполнения search
- [x] Forced search: если LLM пропускает tools, принудительный search с оригинальным запросом
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

### Phase 3.0: Расширение коллекции [ЗАВЕРШЕНО]
- [x] Deep Research: 32 канала отобраны из 70+ кандидатов с evidence
- [x] Валидация: 37/37 каналов доступны (validate_channels.py)
- [x] Ingest: 36 каналов, 13124 точки, период 2025-07-01 → 2026-03-18
- [ ] Проверить quality: нет ли мусора, рекламы, форвардов-дублей
- [ ] Статистика: точек по каждому каналу, общий объём коллекции

### Phase 3.1: Evaluation Framework + Pipeline Optimization [В РАБОТЕ]
**Почему**: "Как ты меришь качество?" — первый вопрос на Applied LLM собесе.

**Eval pipeline (готово):**
- [x] Quick golden dataset: 10 вопросов, 5 типов (factual, temporal, channel-specific, comparative, multi-hop, negative)
- [x] Ground truth: channel:message_id + expected_answer для каждого вопроса
- [x] Автоматический прогон: `python scripts/evaluate_agent.py` → JSON + Markdown отчёты
- [x] Метрики: recall@5 (fuzzy ±5 msg_id), coverage, latency, answer rate
- [x] 11 прогонов eval с разными конфигурациями (history в results/raw/)

**Pipeline optimization (recall@5: 0.15 → 0.70):**
- [x] Убрали dense re-score после RRF (убивал BM25): 0.15 → 0.33
- [x] Оригинальный запрос в subqueries (LLM теряет entities): 0.33 → 0.59
- [x] Weighted RRF (BM25 3:1) + forced search + dynamic tools: 0.59 → 0.70
- [x] MMR протестирован и отключён (cosine-based MMR re-promotes attractors)
- [x] Ablation study: 4 конфигурации с измеренной дельтой

**Следующие шаги (recall@5 → 0.80+):**
- [ ] Global PCA whitening (1024→512 dim): ожидание +5-15%
- [ ] Замена реранкера bge-m3 → bge-reranker-v2-m3: ожидание +15-30%
- [ ] Расширение dataset до 50+ вопросов для stat. значимости
- [ ] Baseline: naive vector search (без reranker, без refinement)
- [ ] Benchmark: LlamaIndex с тем же Qdrant + тем же LLM
- [ ] Ablation study: отключаем компоненты по одному → таблица
- [ ] A/B визуализация: графики, сравнительные таблицы

**Roadmap к 0.80+ (из R11 + R12 research):** см. `docs/ai/planning/retrieval_improvement_playbook.md`

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
Phase 3.0: Расширение коллекции (36 каналов, 13124 точки) [DONE]
Phase 3.1: Evaluation + Pipeline Optimization [IN PROGRESS — recall@5 = 0.70]
Phase 3.2: Adaptive retrieval (query classification + strategy selection)
Phase 3.3: README + архитектурная диаграмма + метрики
Phase 3.4: Observability
Phase 3.5: UI polish
```

---

## Заметки

- Целевой реранкер: bge-reranker-v2-m3 (dedicated cross-encoder, +10 nDCG vs текущий bge-m3)
- RTX 5060 Ti недоступна в Docker Desktop (V100 TCC блокирует NVML) — DEC-0024
- gpu_server.py: embedding + reranker в одном HTTP-процессе (PyTorch cu128, WSL2 native)
- Recall@5: 0.15 → 0.70 через 11 eval прогонов (weighted RRF, forced search, original query injection)
- Coverage: 0.86 mean на текущем quick dataset
- Latency: ~25-30s на полный pipeline (LLM inference — основное узкое место)
- Retrieval improvement playbook: `docs/ai/planning/retrieval_improvement_playbook.md`
- Research reports: R11 (advanced retrieval), R12 (clustering) — ключевые для дальнейшего развития
