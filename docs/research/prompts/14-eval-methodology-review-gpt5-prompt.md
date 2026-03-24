# Review prompt: R18 Evaluation Methodology для RAG-агента

## Роль

Ты — senior ML engineer и reviewer. Проведи критический анализ исследования по evaluation methodology для agentic RAG-системы. Ищи: ошибки в методологии, завышенные ожидания, неучтённые edge cases, over-engineering, и предложи конкретные упрощения где можно.

---

## Контекст проекта (кратко)

- **rag_app**: FastAPI + Qdrant (13088 points, 36 Telegram-каналов AI/ML, июль 2025 — март 2026)
- **LLM**: Qwen3-30B-A3B (3B active params, MoE) на V100 SXM2 32GB, ~40 сек/запрос агента
- **Retrieval**: BM25 + dense → weighted RRF 3:1 → ColBERT MaxSim → bge-reranker-v2-m3
- **Agent**: ReAct с native function calling, 11 LLM tools, dynamic visibility (max 5), phase-based
- **Текущий eval**: 3 датасета (10+10+30 Qs), только recall@5 + coverage, нет LLM judge
- **Цель**: портфолио для Applied LLM Engineer позиции

### Текущие 11 tools
Pre-search: query_plan, search, temporal_search, channel_search, cross_channel_compare, summarize_channel, list_channels
Post-search: rerank, compose_context, final_answer, related_posts

### Будущие tools (R17, ещё не реализованы)
- `entity_tracker` — entity timeline/comparison через Facet API
- `arxiv_tracker` — popular papers через Facet API
- `hot_topics` — pre-computed weekly digests (BERTopic cron)
- `channel_expertise` — channel authority profiles

---

## Исследование для review

Приложен полный отчёт R18 (Evaluation pipeline for agentic RAG with 11+ tools). Ключевые предложения:

### 1. Dataset: 450-500 вопросов, 17 категорий
- По категории на каждый tool (текущий + будущий) + negative + forbidden tool
- JSON schema с expected_tool_sequence (primary + alternatives + key_tools + forbidden_tools)
- Forward-looking questions с dual-sequence (baseline сейчас, future primary потом)
- Synthetic generation pipeline (7 stages)

### 2. LLM Judge: 5 независимых критериев
- Usefulness (0-2), Factual correctness (F1 через decompose-then-verify), Citation grounding (0-1), Completeness (0-2), Refusal accuracy (binary)
- Decompose-then-verify: разбить ответ на atomic claims → NLI проверка каждого claim vs expected_answer
- Tiered gating: FC < 0.3 = hard fail
- Qwen3 local (90%) + Claude API (10% calibration)

### 3. Tool selection: 5 метрик
- Key Tool Accuracy (0.40), Tool F1 (0.15), Exact Sequence Match (0.10), LCS-F1 (0.10), Negative Test Pass Rate (0.15), Visibility Accuracy (0.10)
- Разделение scaffold tools (query_plan, rerank, compose_context, final_answer) vs key tools (search variants)

### 4. Robustness: NDR, RSR, ROR
- NDR: k=0 vs k=3,5,10,15,20 — добавление контекста ≠ деградация
- RSR: k=3,5,10,15,20 — монотонный рост quality
- ROR: 6 shuffles compose_context — order robustness
- ~67 часов compute на полный suite

### 5. Ablation: phased
- Phase 1: 100 Qs × 11 configs = 1100 calls (~9 часов)
- Phase 2: 500 Qs × top 3 configs × 3 runs = 6000 calls (~67 часов)
- Paired bootstrap tests (10K iterations)

---

## ВАЖНОЕ ДОПОЛНЕНИЕ: итеративный подход к реализации

Автор проекта (и я) считаем что R18 описывает **целевое состояние**, но реализовывать нужно итеративно. Конкретно:

### Фаза "Dev" (первая реализация, сейчас)
- **20-30 golden questions** вручную (не 450-500)
- **LLM judge = Claude API или Codex** (не Qwen3 local) — экономим время V100 для inference агента
- **Метрики**: recall@5 + LLM judge (multi-criteria) + tool selection accuracy (key tool)
- **Без robustness** пока (NDR/RSR/ROR = checkpoint phase)
- **Без ablation** пока
- **Цель**: быстрый feedback loop (~20-30 мин на полный прогон), чтоб при добавлении новых tools сразу видеть что работает/сломалось

### Фаза "Checkpoint" (после добавления entity_tracker + arxiv_tracker)
- **100-150 questions** (hand-crafted + первый раунд synthetic)
- Добавить ablation quick screen (100 Qs × top configs)
- Добавить RSR quick check

### Фаза "Release" (финальная для портфолио)
- **450-500 questions** (synthetic pipeline + human verification)
- Полный robustness suite
- Qwen3 local judge + Claude calibration
- Ablation deep eval

---

## Вопросы для review

### Блокеры и ошибки
1. Есть ли **методологические ошибки** в decompose-then-verify подходе? Работает ли NLI на 3B-active Qwen для русского текста?
2. **Формула Factual_Precision** с 0.5 penalty за extrinsic claims — это обосновано? Не слишком ли жёстко/мягко?
3. **Key Tool Accuracy = 0.40 weight** — не слишком ли много? В каких случаях ToolF1 важнее?
4. **Robustness формулы**: RSR требует монотонного роста по ВСЕМ предыдущим k — это слишком строго? Один выброс = fail?

### Over-engineering
5. **17 категорий** — не слишком ли гранулярно? Можно ли объединить (например, все search variants в одну категорию)?
6. **expected_tool_sequence с alternatives и score_multiplier** — стоит ли это усложнение? Или достаточно key_tools + forbidden_tools?
7. **LCS-F1 и Exact Sequence Match** — добавляют ли они диагностической ценности сверх Key Tool Accuracy?
8. **7-stage synthetic pipeline** — можно ли упростить для dev phase? Какие стадии можно пропустить?

### Итеративный подход
9. **20-30 questions для dev phase** — достаточно ли для статистически значимых выводов? Или это ok как smoke test?
10. **Claude/Codex как judge вместо Qwen** — какие подводные камни? Смена judge между фазами не сломает сравнимость?
11. **Какой минимальный набор метрик** для dev phase? Recall@5 + LLM judge composite + key_tool_accuracy — хватит?
12. **Forward-looking questions**: имеет ли смысл включать 5-10 таких в dev phase (20-30 total), или они займут слишком большую долю?

### Что упущено
13. Есть ли **важные метрики или подходы** которые R18 не рассматривает?
14. **Latency budget**: R18 не учитывает что judge добавляет latency к eval. Для dev phase с 30 Qs: сколько времени займёт 5-criteria judge через Claude API?
15. **Dataset drift**: как обновлять eval dataset когда добавляются новые посты через ingest?

---

## Формат ответа

Для каждого вопроса: конкретный verdict (✅ норм / ⚠️ предупреждение / ❌ блокер) + обоснование + предложение если нужно. В конце — общий verdict по R18 и рекомендация по dev phase.
