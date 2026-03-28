# Project Scope: rag_app как портфолио-проект

> Живой документ. Обновляется по мере продвижения.
> Последнее обновление: 2026-03-25

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
- [x] Self-hosted reranker: bge-reranker-v2-m3 (dedicated cross-encoder) через gpu_server.py
- [x] Telegram ingestion: 36 каналов, 13088 points в `news_colbert_v2` после payload enrichment + re-ingest
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
- [x] 13 tools: query_plan, search, temporal_search, channel_search, cross_channel_compare, summarize_channel, list_channels, rerank, compose_context, final_answer, related_posts, entity_tracker, arxiv_tracker
- [x] Dynamic tool availability: phase-based visibility (PRE-SEARCH / POST-SEARCH / NAV-COMPLETE / ANALYTICS-COMPLETE), max 5 tools visible одновременно
- [x] Forced search: если LLM пропускает tools, принудительный search с оригинальным запросом
- [x] Analytics short-circuit: entity_tracker/arxiv_tracker → analytics_done → можно отвечать без search; arxiv lookup может перейти в rerank/compose
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

**Итого: 36 каналов** в актуальной коллекции. Покрытие 10 тематических областей.
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

**Исторические метрики (legacy datasets, до golden_v1):**
| Dataset | Recall@5 | Coverage | Тип |
|---------|----------|----------|-----|
| v1 (10 Qs) | 0.76 | 0.86 | Agent eval |
| v2 (10 Qs, сложные) | 0.685 | 0.81 | Agent eval (с LLM tool selection) |
| 100 Qs, RRF+ColBERT | 0.73 | — | Retrieval eval |

> Legacy datasets заменены golden_v1 (25 Qs, 6 категорий). Актуальные метрики — в Phase 3.3.

**Что НЕ сработало (с evidence):**
| Техника | Результат | Почему |
|---------|-----------|--------|
| Cosine MMR | 0.70→0.11 | Re-promotes attractor documents |
| Dense re-score после RRF | 0.33→0.15 | Стирает BM25 вклад |
| PCA whitening 1024→512 | 0.70→0.56 | Слишком агрессивный cutoff |
| Whitening 1024→1024 | паритет | Dense не bottleneck при BM25 3:1 |
| DBSF fusion | 0.72 vs RRF 0.73 | RRF чуть лучше |

**Roadmap:** см. `docs/planning/retrieval_improvement_playbook.md`

### Phase 3.2: Adaptive Retrieval + Tool Router [ЗАВЕРШЕНО]

> Реализовано в SPEC-RAG-11 (adaptive retrieval) + SPEC-RAG-13 (tool expansion).

**Текущая архитектура:**
- **Базовый runtime из SPEC-RAG-13**: 11 LLM-visible tools (без analytics)
- **Текущий runtime после SPEC-RAG-15**: 13 LLM-visible tools, добавлены `entity_tracker` и `arxiv_tracker`
- **Phase-based dynamic visibility**: PRE-SEARCH → POST-SEARCH → NAV-COMPLETE → ANALYTICS-COMPLETE. Max 5 visible. Signal + keyword routing
- **Forced search**: если LLM пропускает tools, принудительный search с оригинальным запросом
- **Navigation short-circuit**: list_channels → navigation_answered → skip forced search, NAV-COMPLETE
- **Analytics short-circuit**: entity_tracker/arxiv_tracker → analytics_done → skip forced search, ANALYTICS-COMPLETE
- **Refusal policy**: prompt rules + temporal guard в _execute_action + forced search bypass

**Результаты:**
- Key Tool Accuracy: **0.955** (golden_v1, 25 Qs)
- Navigation: key_tool=1.0, latency=7s
- Specialized tools (cross_channel_compare, summarize_channel): routing correct, grounding improving

**Отложено:**
- [ ] CRAG-lite: ColBERT scores как quality gate
- [ ] Routing accuracy eval (отдельный от recall)

**Результаты (2026-03-21):**
- LLM выбирает `temporal_search` для "NVIDIA в начале 2026" → date filter → Vera Rubin найден ✅
- LLM выбирает `channel_search` для "gonzo_ml про трансформеры" → channel filter ✅
- LLM выбирает `search` (broad) для "Сравни GPT-5 и Claude" → без фильтров ✅
- Dynamic visibility: 3-4 tools видны LLM вместо 7 → лучший accuracy tool selection

### Phase 3.3: Evaluation Pipeline V2 [ЗАВЕРШЕНО — SPEC-RAG-14]

> R18 = целевой evaluation blueprint (release-grade).
> SPEC-RAG-14 = dev-phase subset — реализован + P0-P1.5 fixes.

**Реализовано (2026-03-24):**
- [x] Golden dataset v1: 25 hand-crafted вопросов, 6 категорий
- [x] SSE tool tracking: `step_started` с `visible_tools`, `tool_invoked` → tools_invoked list
- [x] Key tool accuracy (binary whitelist: key_tools ∪ alternatives, forbidden = hard 0)
- [x] Failure attribution: tool_hidden/tool_wrong/tool_failed/retrieval_empty/generation_wrong/refusal_wrong/judge_uncertain
- [x] LLM judge integration (Claude API): factual (0-1) + usefulness (0-2), `--judge claude|skip`
- [x] Unified JSON report (eval_metadata + aggregate + per_question)
- [x] Backward-compatible migration path для legacy eval datasets
- [x] Manual judge: консенсус Claude + Codex → factual=0.52, useful=1.14/2

**P0-P1.5 fixes (2026-03-24):**
- [x] Navigation short-circuit: list_channels → skip forced search → key_tool 0→1.0, latency -55%
- [x] Runtime error fix: context overflow (max_chars 36K→30K, -c 32768), Qwen3 prefill conflict
- [x] summarize_channel temporal boundary: relative to latest post, not now()
- [x] Eval error surfacing: runtime markers in answer → tool_execution_failed
- [x] Refusal hardening: prompt rules + temporal guard + forced search bypass
- [x] Summarize grounding: prompt hint → compose_context flow stabilized

**Текущие метрики (golden_v1, 2026-03-24):**
| Метрика | Значение |
|---------|----------|
| Key Tool Accuracy | **0.955** |
| Strict Recall@5 | ~0.43 (занижен: dataset strictness + alternative evidence) |
| Manual judge factual | **0.52** (консенсус) |
| Manual judge useful | **1.14/2** (консенсус) |
| Coverage | ~0.66 |
| Navigation | key_tool=1.0, latency=7s |
| Refusal | 1/3 fixed, 2/3 stochastic |

**Оставшиеся задачи (не blocker для Phase 3.4):**
- [ ] Audit zero-recall cases: true miss / dataset too strict / alternative valid evidence
- [ ] Soft metric для compare/summarize (channel-level matching)
- [ ] Stochastic refusal hardening (q19/q20)

**Checkpoint phase (после SPEC-RAG-15):**
- [ ] 100-150 вопросов (hand-crafted + synthetic)
- [ ] Citation grounding criterion
- [ ] RSR quick check, ablation quick screen

**Release phase (portfolio-grade, по R18):**
- [ ] 450-500 вопросов, synthetic pipeline + human verification
- [ ] Полный robustness suite: NDR + RSR + ROR
- [ ] Deep ablation study, Qwen local judge + Claude calibration

### Phase 3.4: Tool Expansion + Entity Analytics [В РАБОТЕ]

> R16 (generic tools) + R17 (domain-specific tools) завершены. SPEC-RAG-12/13 реализованы.

**Реализовано до SPEC-RAG-15 (foundation):**
- [x] Payload enrichment: entities[], arxiv_ids[], urls[], lang, year_week, year_month + 16 payload indexes
- [x] Entity dictionary: 95 AI/ML entities, 6 categories, case_sensitive, regex NER
- [x] Collection migration: news_colbert → news_colbert_v2 (enriched payload, 13088 points)
- [x] 4 new tools: list_channels, related_posts, cross_channel_compare, summarize_channel
- [x] Phase-based dynamic visibility (max 5 visible, signal + keyword routing)

**Реализовано в runtime + eval (SPEC-RAG-15, 2026-03-25):**
- [x] `entity_tracker` — Facet API на entities[] + year_week (top, timeline, compare, co_occurrence)
- [x] `arxiv_tracker` — Facet API на arxiv_ids[] (top, lookup, кто обсуждал)
- [x] Agent state: `analytics_done`, ANALYTICS-COMPLETE phase, forced search bypass
- [x] Golden dataset расширен до 30 вопросов (25 original + 5 analytics)
- [x] Manual judge: factual **1.79/2**, useful **1.72/2**, key tool accuracy **0.926**

**Осталось для закрытия фазы:**
- [ ] Unit tests для analytics tools и state machine
- [ ] Обновить stale docs: architecture flow, agent module, planning, decision log
- Enriched payload + indexes готовы, effort низкий

**Будущее (SPEC-RAG-16, опциональное):**
- [ ] `hot_topics` — pre-computed weekly digests (BERTopic cron + auxiliary collection)
- [ ] `channel_expertise` — pre-computed channel profiles, authority ranking

### Phase 3.5: Production-Grade Quality [СЛЕДУЮЩИЙ ПОСЛЕ hot_topics]

> 5 research tracks для перехода от demo к production. Каждый = deep research → spec → implementation.
> Порядок выбран по зависимостям: observability даёт числа для всех остальных.

**Track 1 — Observability + Latency Budget (2-3 дня):**
- Structured logging: OpenTelemetry-style spans per tool, per retrieval stage
- P50/P95/P99 latency distribution, token usage tracking per request
- Bottleneck analysis → SLO design (target <20s P95)
- **Делать первым** — даёт baseline числа для всех последующих треков
- Ref: production patterns из R18 §7

**Track 2 — NLI Citation Faithfulness (research 2-3 дня + impl 2-3 дня):**
- Decompose-then-verify: atomic claims → NLI check vs cited chunks
- XLM-RoBERTa-large-xnli (~1.3GB) влезает на 5060 Ti рядом с остальными моделями
- Target: faithfulness ≥ 0.92. **Killer feature** для interview
- Ref: R14-deep §NLI, Яндекс R15 предупреждение "LLM judge не работает для factual correctness"

**Track 3 — Retrieval Robustness NDR/RSR/ROR (3-5 дней):**
- NDR: добавление контекста улучшает или ухудшает ответ?
- RSR: k=3,5,10,15,20 — монотонное улучшение или lost-in-the-middle?
- ROR: порядок чанков в compose_context — влияет на Qwen3-30B-A3B (3B active)?
- Ref: R15 (Яндекс — ключевой трек конференции), R18 §robustness

**Track 4 — CRAG-lite / Quality-Gated Retrieval (2-3 дня):**
- ColBERT MaxSim scores уже доступны — zero overhead quality signal
- Calibration: ColBERT score × factual correctness → threshold
- 3 действия: Correct (answer) / Re-query (другая стратегия) / Refuse
- Зависит от данных Track 3. Ref: R13-deep §3, CRAG paper (+7% PopQA, +37% PubHealth)

**Track 5 — RAG Necessity / Adaptive Pipeline Depth (1-2 дня):**
- Query complexity classification: no-RAG / simple-RAG / full-pipeline
- Яндекс: ~25% queries не нужен RAG → -25% latency
- Ref: R15 — Соколов §7

**Отложено (не приоритет из-за constraints):**
- SFT/RLHF: нужно 200+ FC samples, 10K RAG samples. Описать в README как future work
- Speculative RAG: сложная dual-GPU orchestration, marginal gain
- Context Compression: отдельная модель (Qwen2.5-1.5B), marginal gain при текущем compose_context

### Phase 3.6: README + Архитектурная диаграмма [КРИТИЧНО для первого впечатления]
- [ ] Mermaid диаграмма полного pipeline (включая adaptive routing)
- [ ] Quick start (docker compose up и готово)
- [ ] Скриншот web UI
- [ ] Ablation table + comparison table
- [ ] Секция "Design Decisions" — почему self-hosted, почему Qdrant, почему function calling
- [ ] Секция "What didn't work" — с цифрами (MMR, whitening, etc.)

### Phase 3.7: Observability [ВАЖНО для production-readiness]
- [ ] Latency tracking per tool step
- [ ] Token usage per request
- [ ] Routing decision logging (strategy, source, filters)
- [ ] Error rate monitoring
- [ ] Structured logging (уже частично есть)

### Phase 3.8: Web UI Polish [NICE TO HAVE]
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
| Infra | Deployment, monitoring | ⚠️ Docker есть, observability запланирована (Track 1, Phase 3.5) |
| LLM internals | Attention, tokenization, inference | ⚠️ Понимание есть, нужно уметь объяснить |

---

## Порядок работы

```
Phase 3.0: Расширение коллекции (36 каналов, 13088 точек)                     [DONE]
Phase 3.1: Evaluation + Pipeline Optimization (recall 0.15→0.76)              [DONE]
Phase 3.2: Adaptive Retrieval + Tool Router (base runtime + dynamic visibility) [DONE]
Phase 3.3: Evaluation Pipeline V2 (SPEC-RAG-14 + P0-P1.5 fixes)              [DONE]
Phase 3.4: Tool Expansion + Entity Analytics (SPEC-RAG-15 runtime done)       [CLOSEOUT — unit tests + docs]
  → hot_topics + channel_expertise (SPEC-RAG-16)                              [NEXT]
Phase 3.5: Production-Grade Quality (5 research tracks)                        [AFTER hot_topics]
  → Track 1: Observability + Latency Budget
  → Track 2: NLI Citation Faithfulness
  → Track 3: Retrieval Robustness NDR/RSR/ROR
  → Track 4: CRAG-lite Quality Gate
  → Track 5: RAG Necessity Classifier
Phase 3.6: README + архитектурная диаграмма + ablation tables                 [Параллельно]
Phase 3.7: UI polish                                                           [Lowest priority]

Исследовательская база: R01-R18 (18 отчётов)
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
| **R15** | `reports/R15-yandex-rag-conference.md` | Яндекс RAG конференция | "Less is More" tools, FC bottleneck, SFT/GRPO, eval методология |
| **R16** | `reports/R16-deep-rag-agent-tools-expansion.md` | Generic RAG tools | 5 recommended, 2 rejected, phase-based visibility |
| **R17** | `reports/R17-deep-domain-specific-tools.md` | Domain-specific tools | entity_tracker, arxiv_tracker, hot_topics, channel_expertise |
| **R18** | `reports/R18-deep-evaluation-methodology-dataset.md` | Evaluation methodology | LLM judge, robustness (NDR/RSR/ROR), synthetic pipeline, 500Q dataset |

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

- Активная коллекция: `news_colbert_v2` (dense 1024 + sparse BM25 + ColBERT 128-dim MaxSim + enriched payload)
- gpu_server.py: 3 модели (Qwen3-Embedding-0.6B + bge-reranker-v2-m3 + jina-colbert-v2), ~4-5 GB VRAM, ~11GB свободно на 5060 Ti
- RTX 5060 Ti недоступна в Docker Desktop (V100 TCC блокирует NVML) — DEC-0024
- Recall@5: 0.15 → 0.76 (v1), 0.685 (v2) через iterative eval + adaptive tools
- Retrieval recall@5: 0.73 на 100 запросах (ColBERT + RRF)
- Coverage: 0.86 (v1), 0.80 (v2)
- Latency: ~30-45с на полный pipeline (LLM inference ~12с × 2-3 calls — основное узкое место)
- Retrieval improvement playbook: `docs/planning/retrieval_improvement_playbook.md`
- Adaptive retrieval plan: `docs/planning/adaptive_retrieval_plan.md`
- Evaluation strategy: `docs/research/reports/R18-deep-evaluation-methodology-dataset.md` (target-state) + `docs/specifications/active/SPEC-RAG-14-evaluation-pipeline.md` (dev phase)
