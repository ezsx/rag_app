# Project Scope: rag_app как портфолио-проект

> Живой документ. Обновляется по мере продвижения.
> Последнее обновление: 2026-03-30

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
- [x] 15 tools: query_plan, search, temporal_search, channel_search, cross_channel_compare, summarize_channel, list_channels, rerank, compose_context, final_answer, related_posts, entity_tracker, arxiv_tracker, hot_topics, channel_expertise
- [x] Dynamic tool availability: phase-based visibility (PRE-SEARCH / POST-SEARCH / NAV-COMPLETE / ANALYTICS-COMPLETE), max 5 tools visible одновременно
- [x] Forced search: если LLM пропускает tools, принудительный search с оригинальным запросом
- [x] Analytics short-circuit: entity_tracker/arxiv_tracker/hot_topics/channel_expertise → analytics_done → можно отвечать без search; analytics tools могут переходить сразу в final_answer
- [x] Coverage metric: 6-signal composite (cosine similarity based)
- [x] Auto-refinement: дополнительный поиск при coverage < 0.65
- [x] Grounding: citation-forced generation, source validation
- [x] Context overflow protection: _trim_messages
- [x] Channel dedup: max 2 docs/channel для diversity
- [x] RequestContext + ContextVar: per-request isolation
- [x] Cooperative deadline + visible-tool whitelist + auth hardening (SPEC-RAG-17)

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
- **Текущий runtime после SPEC-RAG-17**: 15 LLM-visible tools, добавлены `entity_tracker`, `arxiv_tracker`, `hot_topics`, `channel_expertise`
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

**Текущие метрики (golden_v2, 2026-04-01, SPEC-RAG-20d):**
| Метрика | Значение | Прошлый (03-30) |
|---------|----------|-----------------|
| Key Tool Accuracy | **1.000** (36 Qs) | 1.000 |
| Factual correctness | **0.875** (Claude judge, q27 fix) | ~0.80 |
| Usefulness | **1.917/2** | ~1.53 |
| Strict anchor recall | **0.637** | 0.461 |
| Mean latency | **23.6s** | 26.4s |
| Coverage mean | 0.346 (legacy cosine) | 0.421 |

**Предыдущие метрики (golden_v1, 2026-03-25, SPEC-RAG-15):**
| Метрика | Значение |
|---------|----------|
| Key Tool Accuracy | 0.970 (35/36) |
| Consensus factual | 0.833 (30/36) |
| Consensus useful | 1.611/2 |
| Mean latency | 26.4s |
| Eval run time | 15.8 min (36 Qs) |
| Model | Qwen3.5-35B-A3B Q4_K_M |
| Observability | Langfuse v3 (SPEC-RAG-19) |

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

### Phase 3.4: Tool Expansion + Entity Analytics [ЗАВЕРШЕНО]

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

**Реализовано после SPEC-RAG-15:**
- [x] `hot_topics` — BERTopic weekly digests + `weekly_digests` auxiliary collection
- [x] `channel_expertise` — monthly channel profiles + `channel_profiles` auxiliary collection
- [x] Weekly cron: full-corpus BERTopic fit + hot_score + LLM summary
- [x] Monthly cron: per-channel profile computation (authority/speed/breadth/volume) + LLM profile
- [x] SPEC-RAG-17: production hardening (request isolation, auth hardening, rate limiter fix, cooperative deadline, CORS, demo auth path)

**SPEC-RAG-18: Golden v2 + Offline Judge (2026-03-30):**
- [x] Golden v2 dataset: 36 Qs (18 retrieval_evidence, 13 analytics, 2 navigation, 3 refusal)
- [x] Eval modes: retrieval_evidence / analytics / navigation / refusal
- [x] Offline judge workflow: full artifact packet, batch export (30 Qs/batch)
- [x] Metric redesign: strict_anchor_recall → diagnostic only, primary = factual + useful + KTA
- [x] Judge consensus: Claude + Codex → **factual ~0.80, useful ~1.53/2, KTA 1.000**
- [x] hot_topics: 3 Qs, channel_expertise: 3 Qs — integrated and tested
- [x] Routing fixes: q24 dataset, q32 keywords, q33 tool description

**Осталось после закрытия фазы:**
- [ ] P1: q33 monthly hot_topics path fix + q36 channel_expertise routing fix
- [ ] P2: q21 deterministic out-of-range refusal
- [ ] Clean baseline rerun (36 Qs) после P1+P2
- [ ] Unit tests для analytics tools и state machine

### Phase 3.5: Pipeline Cleanup + Observability + Coverage Redesign [В ПРОЦЕССЕ]

> SPEC-RAG-20d. Codex audit (8 findings) + Claude obs audit (15 findings) + retrieval calibration (100 queries).

**SPEC-RAG-20d Pipeline Cleanup [ВЫПОЛНЕНО 2026-03-31]:**
- [x] serialize_tool_payload: tool-aware, compose prompt не обрезается (24K limit)
- [x] search hits stripped из history (id+snippet, не full text)
- [x] trim_messages: atomic blocks, pin compose_context
- [x] LLM 400 retry: сохраняет compose pair
- [x] Temporal guard для всех tools с date_from/date_to
- [x] k_per_query vs k_total разведены (cap до 30)
- [x] Lost-in-middle отключён (docs уже reranked)
- [x] QA fallback: agent context вместо legacy pipeline
- [ ] fetch_docs chunk stitching (отдельная спека, затрагивает ingest)

**Observability (Langfuse) [ВЫПОЛНЕНО 2026-03-31]:**
- [x] Double JSON encoding fix
- [x] Root trace: plan, tokens, coverage, strategy, citations_count
- [x] Tool spans: rich output + error marking
- [x] search_execution + compose_execution spans
- [x] LLM step names phase-aware, token aggregation
- [x] gpu_server: empty text guard

**Coverage + Reranker redesign [ВЫПОЛНЕНО 2026-04-01]:**
- [x] LANCER nugget coverage (services/agent/coverage.py) — DEC-0044
  - query_plan subqueries = nuggets
  - Implicit nuggets из search subqueries
  - Threshold 0.75, max_refinements 1
- [x] Targeted refinement по uncovered nuggets (SEAL-RAG style)
- [x] Cross-encoder → CRAG confidence filter (не reranking) — DEC-0045
  - ColBERT порядок сохраняется
  - CE отсекает docs с score < threshold
- [x] Retrieval calibration: 100 queries, recall@1-20, CE scores, pipeline v2 A/B

**Calibration results (100 queries, news_colbert_v2):**
- Recall: r@1=0.80, r@3=0.97, r@20=0.98 (monotonic)
- CE reranking degrades r@3: 0.97→0.94 → заменён на filter
- Pipeline v2 (RRF→CE→ColBERT): +0.02 r@2, не стоит усложнения
- CE score: relevant median=8.35, irrelevant median=-1.11 → threshold=0.0 (logit boundary)
- Latency improvement: 45-76s → 14-49s (−40-65%) за счёт устранения refinements

**Осталось:**
- [ ] CE filter_threshold=0.0 установить и smoke test
- [ ] Full eval 36 Qs — baseline после pipeline changes
- [ ] fetch_docs chunk stitching

### Phase 3.6: Production-Grade Quality [RESEARCH DONE]

> Research base: R19, R20, R21, R25.

**Данные (блокирует все последующие метрики):**
- [ ] Re-ingest свежих постов: 2026-03-18 → current date
- [ ] Weekly digests для всех ~37 недель
- [ ] Channel profiles re-compute после re-ingest

**Eval / proof layer:**
- [ ] P1 fixes: q33 monthly hot_topics + q36 channel_expertise routing
- [ ] Clean baseline 36 Qs after Phase 3.5 pipeline changes
- [ ] Eval expansion: 36 → 100+ вопросов
- [ ] Unit tests cleanup

**Track 2 — NLI Citation Faithfulness (R19 готов):**
- Hybrid C: Qwen3 decomposition + XLM-RoBERTa NLI
- Phase 1: eval-only metric, Phase 2: runtime integration

**Track 3 — Retrieval Robustness NDR/RSR/ROR (R20 готов):**
- NDR, RSR, ROR metrics
- Частично покрыто calibrate_coverage.py (recall curve, monotonicity)

**Track 4 — Fine-tune CE reranker:**
- 500 query-doc pairs из нашего corpus
- Reduce 3% degradation (DEC-0045 observation)

**Track 5 — RAG Necessity Classifier (R21, low priority):**
- Conservative pre-filter перед agent loop

**Production polish (non-blocking, но важные production patterns):**
- [ ] Observability: per-component latency, routing logs, error rate, token usage
- [ ] Prompt injection defense: delimiter boundaries между system instructions и retrieved content
- [ ] Health check endpoints: `/health`, `/ready`
- [ ] BERTopic labels cleanup / humanization

**Осознанно НЕ делаем сейчас:**
- Graph RAG
- Self-RAG
- Semantic caching
- Multi-provider fallback
- Custom NLI training
- SFT/RLHF
- Advanced multi-turn memory beyond simple sliding window

### Phase 3.6: README + Архитектурная диаграмма [КРИТИЧНО для первого впечатления]
- [ ] Mermaid диаграмма полного pipeline (включая adaptive routing)
- [ ] Quick start (docker compose up и готово)
- [ ] Скриншот web UI
- [ ] Ablation table + comparison table
- [ ] Секция "Design Decisions" — почему self-hosted, почему Qdrant, почему function calling
- [ ] Секция "What didn't work" — с цифрами (MMR, whitening, etc.)

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
| Infra | Deployment, monitoring | ✅ Docker + Langfuse v3 observability (SPEC-RAG-19) |
| LLM internals | Attention, tokenization, inference | ⚠️ Понимание есть, нужно уметь объяснить |

---

## Порядок работы

```
Phase 3.0: Расширение коллекции (36 каналов, 13088 точек)                     [DONE]
Phase 3.1: Evaluation + Pipeline Optimization (recall 0.15→0.76)              [DONE]
Phase 3.2: Adaptive Retrieval + Tool Router (base runtime + dynamic visibility) [DONE]
Phase 3.3: Evaluation Pipeline V2 (SPEC-RAG-14 + P0-P1.5 fixes)              [DONE]
Phase 3.4: Tool Expansion + Entity Analytics + hardening                       [DONE]
  → SPEC-RAG-15: entity_tracker + arxiv_tracker                               [DONE]
  → SPEC-RAG-16: hot_topics + channel_expertise                               [DONE]
  → SPEC-RAG-17: Production Hardening                                          [DONE]
  → SPEC-RAG-18: Golden v2 + Offline Judge                                     [DONE — baseline zafixed]
    Factual ~0.80/1 | Useful ~1.53/2 | KTA 1.000 | 36 Qs
    P1 fixes pending: q33 monthly path, q36 routing
Phase 3.5: Production-Grade Quality                                            [RESEARCH DONE, IMPL NEXT]
  → Data refresh: re-ingest + weekly digests + channel profiles
  → Eval: 100+ Qs + fresh baseline + tests + ablations + GPT-4o comparison
  → Substance: NLI + robustness + CRAG-lite + prompt injection defense
  → Low priority: RAG necessity classifier
Phase 3.6: README + архитектурная диаграмма + docs sync                        [Параллельно]
Phase 3.7: UI polish                                                           [Lowest priority]

Исследовательская база: R01-R26, plus quick/deep variants and artifacts
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
| **R19** | `reports/R19-deep-nli-citation-faithfulness.md` | NLI Faithfulness | Hybrid approach C (Qwen3 decomposition + XLM-RoBERTa NLI), 83.5% Russian accuracy, 1.12GB FP16 |
| **R20** | `reports/R20-deep-retrieval-robustness-ndr-rsr-ror.md` | Retrieval Robustness | Cao et al. adapted: 50Q/650 calls vs 1500Q/55K, two-stage sequential protocol, k=[3,5,10,20] |
| **R21** | `reports/R21-deep-rag-necessity-classifier.md` | RAG Necessity | Rule-based tiers (<1ms), LLM classifier неоправдан (12s/call), 10-20% non-RAG, conservative fallback |
| **R25** | `reports/R25-deep-production-gap-analysis.md` | Production Gap Analysis | Retrieval на production parity (Perplexity/Cohere level); gaps в proof layer, CRAG-lite, prompt injection, observability |
| **R26** | `reports/R26-golden-v2-eval-baseline.md` | Golden v2 Eval Baseline | Consensus judge: factual ~0.80, useful ~1.53/2, KTA 1.0 (36 Qs). P1: q33 monthly + q36 routing |

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
