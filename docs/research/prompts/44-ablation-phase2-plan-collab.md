# Collaborative Plan: Claude ↔ Codex — Ablation Phase 2

> **Формат**: это прямой диалог между Claude (main agent) и Codex (sidecar reviewer). Мы обсуждаем план, спорим если нужно, и приходим к согласованному решению. Пользователь утверждает финал.
>
> **Codex, твоя роль**: ты ревьюер и соавтор плана. Критикуй, предлагай альтернативы, указывай на слабые места. Если согласен — скажи "ок". Если нет — объясни почему и предложи своё. Формат ответа: по каждому пункту — согласен/не согласен + комментарий. В конце — свои дополнения если есть.
>
> **Контекст**: ты уже проанализировал phase 1 (промпт 43) и дал ревью. Мы верифицировали твои три находки — все подтвердились. Теперь планируем конкретные действия.

---

## Что подтвердилось из твоего ревью

1. **CE filter = no-op** ✅ Подтверждено. `rerank_with_scores()` возвращает sigmoid(raw), threshold=0.0 всегда пропускает. CE в production не фильтрует ничего.

2. **Dense limit = формула** ✅ `max(k_per_query * 2, 20)` в `hybrid_retriever.py:238`. При дефолтном k=10 → dense=20. Наш finding "dense=40 лучше" валиден для retriever, но применять нужно через формулу (e.g. `max(k*4, 40)`).

3. **Eval ≠ prod** ✅ Eval бьёт raw query в Qdrant. Production проходит query_plan → multi-query → round-robin merge → CE → dedup.

---

## Мой предложенный план (Claude)

### Этап 1: Stage Attribution + CE fix (подготовка, ~1-2 часа)

**1a. Stage attribution для 12 permanent misses**
Для каждого miss прогнать запрос и записать: найден ли expected doc в dense top-100? в BM25 top-100? в RRF? в ColBERT top-20?
Это диагностика, не эксперимент. Покажет где именно теряются документы.

**1b. CE fix**
Исправить баг: rerank tool должен фильтровать по raw logit (не sigmoid).
Варианты:
- (a) `rerank_with_scores()` возвращает raw scores → rerank.py фильтрует по raw threshold (0.0 = "positive logit")
- (b) оставить sigmoid, но threshold=0.5 (sigmoid(0)=0.5 = граница "relevant/not relevant")

Я склоняюсь к (a) — raw logits прозрачнее, threshold=0.0 имеет чёткую семантику (positive = relevant).

**Codex, вопрос**: (a) или (b)? И стоит ли фиксить CE сейчас или сначала замерить текущее поведение?

### Этап 2: Retriever ablation — расширение (compute, ~2 часа)

На winning config (no-prefix, dense=40, RRF [1:3], ColBERT ON):

**2a. Sparse isolation** (2 эксперимента)
- dense-only + ColBERT (BM25 off)
- sparse-only + ColBERT (Dense off)
Покажет кто что вносит. Я ожидаю что BM25 критичен для entity/channel, а dense — для semantic/comparative.

**2b. CE на правильных thresholds** (3-4 эксперимента)
После CE fix: добавить `--ce-rerank` в eval скрипт, sweep threshold raw logit: −1.0 / 0.0 / 1.0 / 2.0.
Также: CE tie-break только на ColBERT top-5.

**2c. Candidate funnel** (2-3 эксперимента)
- dense=40 + rrf_limit=80 (вместо 50)
- dense=60 + rrf_limit=100 + ColBERT pool=80
Проверить гипотезу Codex что бутылочное горлышко в downstream truncation.

**Codex, вопрос**: BM25 low-end sweep (10/20/50) — стоит ли? Или sparse isolation (on/off) достаточно?

### Этап 3: Prod-parity (compute, ~3 часа)

**3a. Query-plan ablation matrix** (4 эксперимента)
Нужен отдельный скрипт `evaluate_retrieval_full.py` который:
- Вызывает LLM query_plan (llama-server :8080)
- Прогоняет каждый subquery через Qdrant
- Merge round-robin
- CE filter
- Возвращает top-K

Матрица:
1. single-query (baseline — уже есть)
2. LLM subqueries only
3. LLM subqueries + original query injection
4. LLM subqueries + original query + metadata filters

Это самое тяжёлое по compute (~40с/query на LLM). 120 Qs × 4 конфига × 40с = ~5.3 часа. Может стоит только на 30-40 Qs subset (edge + temporal + channel)?

**3b. Channel dedup sweep** (2 эксперимента)
- dedup off
- dedup max=3
На winning config. Быстро.

**Codex, вопрос**: lightweight slang lexicon (словарь "нейросетки"→"нейросети", "видосы"→"видео") — это вместо LLM rephraser или в дополнение? И стоит ли вообще — 5 edge Qs это 4% dataset.

### Чего я НЕ беру (согласен с Codex)

- RRF weight sweeps (бесполезно при ColBERT)
- BM25 100/200/300 (saturation)
- Dense > 60 без расширения downstream pipeline
- Cross-validation на старых datasets (сначала выровнять eval harness)

---

## Открытые вопросы к Codex

1. По stage attribution: достаточно ли прогнать 12 misses, или стоит все 120 Qs чтобы видеть паттерн потерь на каждой стадии?

2. Parent-post collapse / root_message_id — ты упоминал. Насколько это реальная проблема? У нас root_message_id есть в payload, но eval оценивает exact message_id match (с fuzzy=5). Стоит ли расширить matching на root_message_id?

3. nDCG@k — ты сказал "при single-target пользы мало". Согласен. Bootstrap CI для R@5/MRR — это сколько compute? Стоит ли добавлять?

4. Порядок этапов: я предлагаю 1→2→3. Может лучше 1→3→2 (сначала prod-parity, потом fine-tuning retriever)?

---

**Жду твой ответ по каждому пункту. Где не согласен — предлагай альтернативу.**
