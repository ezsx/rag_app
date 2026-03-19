# Retrieval Improvement Playbook

> Живой документ. Ответ на вопрос "что дальше пробовать чтобы повысить recall?"
> Источник: R11-advanced-retrieval-strategies.md + собственные эксперименты.
> Последнее обновление: 2026-03-19

---

## Текущее состояние

**Recall@5 = 0.59** на quick dataset (10 вопросов).
**Target: 0.65-0.70** (реалистично), **0.75+** (stretch goal).

### История экспериментов

| Дата | Изменение | Recall@5 | Δ | Коммит |
|------|-----------|----------|---|--------|
| 2026-03-19 | Baseline: RRF → dense re-score | 0.15 | — | — |
| 2026-03-19 | Убрали dense re-score → чистый RRF | 0.33 | +0.18 | 72efb31 |
| 2026-03-19 | RRF → MMR (lambda=0.7) | 0.11 | -0.22 | reverted |
| 2026-03-19 | RRF → MMR (lambda=0.9) | 0.11 | -0.22 | reverted |
| 2026-03-19 | + Оригинальный запрос в subqueries | 0.59 | +0.26 | e0bd871 |

### Корневые проблемы (диагностированы)

1. **Embedding anisotropy** — все AI-тексты в cosine range [0.78-0.83]. "Attractor documents" попадают в top-10 любого запроса.
2. **Length-Induced Collapse** (paper arXiv:2410.24200) — тексты одинаковой длины кластерятся вместе вне зависимости от содержания. Наши посты все 300-1500 символов.
3. **Реранкер suboptimal** — bge-m3 (bi-encoder seq-cls mode) вместо bge-reranker-v2-m3 (dedicated cross-encoder, +10 nDCG).
4. **Cosine-based MMR не работает** — переиспользует сломанный cosine signal → re-promotes attractor documents.
5. **Query expansion теряет сущности** — LLM перефразирует и убирает ключевые слова. Частично решено (оригинальный запрос как subquery).

---

## Tier 1: Quick Wins (часы, не дни)

### 1.1 Embedding Whitening (mean-centering)
- **Суть**: вычесть среднее по всей коллекции из каждого вектора. Опционально PCA whitening.
- **Почему поможет**: раздвигает cosine range с [0.78-0.83] до [0.5-0.9]. Zero cost, без перетренировки.
- **Как**: один раз пройти по всем 13K точкам, посчитать mean vector, сохранить. При поиске вычитать из query embedding и из document embeddings (через Qdrant payload или переиндексацию).
- **Нюанс**: нужна переиндексация или on-the-fly вычитание. Можно начать с on-the-fly для query (вычитать mean из query embedding перед поиском) — это уже даст эффект.
- **Ожидание**: +3-8% recall
- **Статус**: [ ] не начато
- **Ссылки**: Liang et al. "Embedding anisotropy", Su et al. "Whitening Sentence Representations"

### 1.2 Weighted RRF (BM25 3:1 vs dense)
- **Суть**: Qdrant v1.17 поддерживает веса для prefetch веток в RRF. BM25 weight=3, dense weight=1.
- **Почему поможет**: BM25 правильно находит keyword-matched документы, но при equal weight dense "магниты" их перевешивают.
- **Как**: одна строка — `models.RrfQuery(rrf=models.Rrf(weights=[3.0, 1.0], k=2))`. Также асимметричный prefetch: BM25 limit=100, dense limit=20.
- **Ожидание**: +5-10% recall
- **Статус**: [ ] не начато

### 1.3 DBSF fusion (альтернатива RRF)
- **Суть**: Distribution-Based Score Fusion — нормализует скоры через mean ± 3σ перед слиянием.
- **Почему поможет**: когда dense scores сжаты в [0.78-0.83], нормализация растянет их и покажет реальные различия. RRF использует только ранги и теряет magnitude.
- **Как**: `query=models.FusionQuery(fusion=models.Fusion.DBSF)` вместо `Fusion.RRF`. Проверить: доступна ли DBSF в нашей версии Qdrant.
- **Ожидание**: неизвестно, нужен A/B тест с RRF
- **Статус**: [ ] не начато

### 1.4 Channel-based dedup (max 2 docs per channel)
- **Суть**: ограничить максимум 2 документа из одного канала в результатах.
- **Почему поможет**: prolific каналы (gonzo_ml, ai_machinelearning_big_data) монополизируют top-10. Dedup даёт шанс менее частым но релевантным каналам.
- **Как**: Qdrant `group_by` параметр, или post-processing в Python.
- **Ожидание**: +2-5% recall
- **Статус**: [ ] не начато

### 1.5 Замена реранкера: bge-m3 → bge-reranker-v2-m3
- **Суть**: текущий реранкер — bge-m3 загруженный как seq-cls. Dedicated cross-encoder bge-reranker-v2-m3 на 10+ пунктов nDCG выше.
- **Почему поможет**: специализированная модель лучше различает релевантные/нерелевантные пары. Скоры станут дифференцированными.
- **Как**: скачать модель на Windows → скопировать в WSL2 → поменять путь в gpu_server.py.
- **Нюанс**: нужен интернет для скачивания. VPN блокирует WSL2 — качать на Windows.
- **Ожидание**: +10-20% recall (самый impactful single fix)
- **Статус**: [ ] не начато
- **Альтернативы**: jina-reranker-v2-base-multilingual (278M, 15× throughput), bge-reranker-v2-gemma (2.5B, best quality)

---

## Tier 2: Архитектурные улучшения (1-3 дня каждое)

### 2.1 Query Classifier + Strategy Router
- **Суть**: определить тип запроса (factual/temporal/channel/comparative/multi-hop) → выбрать стратегию retrieval.
- **Почему поможет**: temporal запросы нуждаются в date filter, channel запросы — в channel filter, comparative — в parallel multi-query. Одна стратегия для всех = компромисс.
- **Как**: Qwen3-30B как classifier (structured JSON output). Map типа → стратегия (top_k, filters, diversity mode).
- **Ожидание**: +5-10% recall, -35% latency (enterprise benchmarks)
- **Статус**: [ ] не начато
- **Код из отчёта**:
```python
STRATEGIES = {
    "simple":      {"top_k": 3,  "diversity": "channel_dedup"},
    "temporal":    {"top_k": 10, "date_filter": True, "diversity": "temporal_buckets"},
    "comparative": {"top_k": 8,  "diversity": "bm25_mmr", "reranker": True},
    "multi_hop":   {"top_k": 5,  "diversity": "cluster_first", "iterative": True},
}
```

### 2.2 Entity Extraction (Natasha/Slovnet NER)
- **Суть**: извлекать именованные сущности (люди, компании, продукты) из запросов и документов.
- **Почему поможет**: "что писал Дженсен Хуанг" → entity="Дженсен Хуанг" → Qdrant payload filter. Точный recall на entity queries.
- **Как**: Natasha NER (27 МБ, CPU, F1~0.96, 25 docs/sec). Один раз прогнать по 13K постов (~10 мин), сохранить entities как Qdrant payload. При поиске — extract entities из query → filter.
- **Ожидание**: +5-15% recall на entity-specific запросах
- **Статус**: [ ] не начато

### 2.3 BM25-based Diversity (замена cosine MMR)
- **Суть**: MMR loop, но с BM25 pairwise similarity вместо cosine.
- **Почему поможет**: два поста про "трансформеры" vs "диффузию" — одинаковый cosine, но разный BM25 профиль. BM25 diversity работает когда cosine сломан.
- **Как**: модифицированный MMR loop в Python. `score = lambda * relevance - (1-lambda) * max(bm25_sim(doc, selected))`.
- **Ожидание**: +2-4% recall
- **Статус**: [ ] не начато

### 2.4 Genericity Score (штраф attractor documents)
- **Суть**: заранее посчитать для каждого документа — в скольких случайных запросах он попадает в top-10. Хранить как payload. Штрафовать при поиске.
- **Почему поможет**: напрямую атакует проблему attractor documents (gonzo_ml:4121, denissexy:10940).
- **Как**: прогнать 100 random queries → подсчитать frequency → сохранить как `genericity` payload. Qdrant formula query: `score = relevance * (1.0 - 0.3 * genericity)`.
- **Ожидание**: +3-5% recall
- **Статус**: [ ] не начато

### 2.5 Reranker-as-Fusion (без RRF)
- **Суть**: вместо RRF → rerank, сделать: BM25 top-50 + dense top-50 → deduplicate → cross-encoder reranks весь пул.
- **Почему поможет**: cross-encoder делает content-aware решения, не слепое rank fusion. Решает проблему "RRF весов" фундаментально.
- **Как**: два отдельных Qdrant search, merge, rerank 70-100 кандидатов. Latency ~150-300ms на RTX 5060 Ti.
- **Ожидание**: потенциально лучше RRF+rerank, нужен A/B тест
- **Статус**: [ ] не начато

---

## Tier 3: Глубокие улучшения (3-7 дней каждое)

### 3.1 ColBERT Reranking (jina-colbert-v2)
- **Суть**: per-token matching вместо single-vector cosine. Для каждого query token — MaxSim с document tokens.
- **Почему поможет**: **фундаментально** решает attractor problem. "Meta купила Manus" и "курс по трансформерам" — совершенно разные token profiles, даже если single-vector cosine одинаковый.
- **Как**: jina-colbert-v2 (560M, 89 языков, русский включён). Qdrant multi-vector config (MaxSim). Трёхэтапный: BM25+Dense → RRF → ColBERT rerank.
- **Storage**: ~500MB для 13K docs (100 tokens × 128 dim × float16).
- **Ожидание**: +6-10% nDCG
- **Статус**: [ ] не начато

### 3.2 Contextual Retrieval (Anthropic's technique)
- **Суть**: перед embedding'ом каждого чанка — LLM генерирует 2-3 предложения контекста ("Этот пост из канала X про тему Y"). Этот prefix disambiguates embedding.
- **Почему поможет**: посты "новая модель вышла" vs "бенчмарк модели" получают разные prefix'ы → разные embeddings. Anthropic измерили 35-67% reduction в retrieval failures.
- **Как**: прогнать Qwen3-30B по всем 13K постам (8-15 часов one-time на V100). Переиндексировать с prefix'ами.
- **Ожидание**: +10-20% recall (один из самых impactful, но трудоёмкий)
- **Статус**: [ ] не начато

### 3.3 Fine-tune Qwen3-Embedding-0.6B
- **Суть**: contrastive fine-tuning с hard negatives, добытыми из нашей же "сломанной" embedding space.
- **Почему поможет**: attractor documents = идеальные hard negatives. Модель учится различать именно те пары, которые сейчас путает.
- **Как**: 1) Qwen3-30B генерирует 3 query на пост → 39K положительных пар. 2) Для каждой пары top-50 "ложно похожих" = hard negatives. 3) sentence-transformers MultipleNegativesRankingLoss.
- **Ожидание**: +5-15% recall
- **Статус**: [ ] не начато
- **Ссылки**: Aurelio AI benchmarks, NV-Retriever (hard negative mining +2-5 nDCG)

### 3.4 CRAG (Corrective RAG)
- **Суть**: после search агент оценивает качество результатов. Если плохо — переформулирует запрос и ищет заново. Не просто "ещё один search", а анализ GAP'ов.
- **Почему поможет**: accuracy 58% → 83% в литературе. У нас уже есть refinement, но он только добавляет search, не переформулирует.
- **Как**: после compose_context, если coverage < threshold → анализ "чего не хватает" → новый query_plan с другими sub-queries → search.
- **Ожидание**: +10-15% на сложных запросах
- **Статус**: [ ] отложен (дорогой по latency, брать когда упрёмся)

---

## Tier 4: На будущее (держим в уме)

### 4.1 HyDE (Hypothetical Document Embedding)
- Генерировать "гипотетический ответ", embed его, искать похожие документы.
- +1-3с latency, не решает collapse в document space. **Complementary technique**.

### 4.2 Qwen3-Embedding-4B
- 8 GB VRAM, помещается рядом с реранкером на 16 GB. +5-10% vs 0.6B на benchmarks.
- Но "scaling alone won't fully solve domain-specific collapse" — не серебряная пуля.

### 4.3 Multi-Collection Architecture
- Разделить на коллекции по длине/типу/тематике. Разные embedding стратегии для каждой.
- Router выбирает в какие коллекции искать.

### 4.4 Link Expansion
- Многие посты = "ссылка + комментарий". Индексировать контент ссылок → +5-15% recall на запросах, ответ на которые в ссылке.

### 4.5 DPP Diversity (Determinantal Point Processes)
- Математически принципиальный diversity selection. YouTube recommendations использует.
- +5-15% diversity metrics vs MMR. Библиотека `dppy`.

### 4.6 Temporal Decay
- `fused_score = 0.7 × semantic + 0.3 × 0.5^(age_days/14)` — boost свежих документов.
- Тривиально через Qdrant payload.

### 4.7 Channel Authority Scoring
- Вручную расставить веса каналам (gonzo_ml=0.9, ml_product=0.5). Boost авторитетных.
- ReliabilityRAG (2025) — explicit source reliability signals.

### 4.8 Forward/Reply Chain Awareness
- Хранить `reply_to_id`, `forwarded_from` как payload. При нахождении поста — подтянуть всю цепочку.

---

## Research Track: Тематическая кластеризация коллекции

> **Статус**: требует отдельного исследования. Потенциально фундаментальное решение проблемы attractor documents и embedding collapse.

### Идея

Одна плоская коллекция на весь AI-корпус — наивно. Все документы "про AI" сливаются в embedding space.
Если кластеризовать по **темам** (M&A, релизы моделей, образование, research papers, инфраструктура...) — внутри каждого кластера cosine становится **осмысленным**: сравниваем яблоки с яблоками.

### Архитектура (предварительная)

```
Документы → TF-IDF → UMAP → HDBSCAN → 20-50 кластеров
                                         ↓
                              cluster_id как Qdrant payload
                                         ↓
Query → определить релевантные кластеры → filter по cluster_id → search
                                         ↓
                              Per-cluster whitening для dense vectors
```

**Одна коллекция**, кластеры через payload, не 50 отдельных коллекций (BM25 IDF нужен широкий корпус).

### Что это даёт

1. **Внутри кластера cosine range шире** — attractor documents не попадают в чужие кластеры
2. **Per-cluster whitening** — mean-centering от тематически однородной группы, эффективнее глобального
3. **Естественная diversity** — результаты из разных кластеров = разные темы
4. **Query routing** — classifier выбирает 3-5 релевантных кластеров → меньше шума

### Открытые вопросы (нужен ресёрч)

1. **Как выбирать кластеры для query?** Варианты: LLM classifier, embed query → nearest centroids, BM25 по описаниям кластеров, или искать во всех (whitening per-cluster всё равно помогает)
2. **Сколько кластеров оптимально?** При 13K документов: 20-50? Меньше = кластеры слишком широкие, больше = слишком мало документов в каждом
3. **Soft assignment**: документ на стыке 2-3 тем — дублировать или список cluster_ids?
4. **Кластеризация на TF-IDF vs на embeddings?** TF-IDF надёжнее (embeddings сломаны), но теряет семантику. Возможно BERTopic (TF-IDF + UMAP + HDBSCAN + LLM naming) — best of both worlds
5. **Как переиндексировать при добавлении новых документов?** Online clustering vs periodic rebuild

### Как начать исследование

1. Кластеризовать 13K документов (TF-IDF + HDBSCAN) — 30 мин
2. Посмотреть кластеры: осмысленные ли? Сколько получилось? Размеры?
3. Замерить cosine range внутри кластеров vs глобально
4. Если кластеры осмысленные — пилотировать на quick dataset

### Ожидание

Если сработает — может быть **самый impactful long-term fix**. Фундаментально решает embedding collapse на узком домене. Но требует исследования и экспериментов перед production-внедрением.

---

## Что НЕ работает / НЕ стоит пробовать

| Техника | Почему не работает |
|---------|-------------------|
| Cosine-based MMR | Переиспользует сломанный cosine signal → re-promotes attractors. Tested lambda 0.7 и 0.9, оба хуже baseline. |
| Dense re-score после RRF | Стирает BM25 вклад. Recall 0.33→0.15. |
| SPLADE для русского | Нет production-ready мультиязычных моделей. WSDM Cup 2026: "struggled to remain competitive". |
| HyDE как primary fix | +1-3с latency, не решает collapse в document space. Only complementary. |
| Scaling embedding alone | "Length-Induced Embedding Collapse" + domain-specific collapse = фундаментальные ограничения single-vector. 4B/8B помогут, но не решат. |

---

## Таблица реранкеров (для выбора)

| Модель | Params | VRAM | MIRACL Avg | Примечание |
|--------|--------|------|------------|------------|
| **bge-reranker-v2-m3** | 568M | ~1.2 GB | 69.32 | Best multilingual, наш target |
| jina-reranker-v2-base-multilingual | 278M | ~0.6 GB | Competitive | 15× throughput, `use_flash_attn=False` для sm_120 |
| jina-reranker-v3 | 0.6B | ~1.2 GB | 66.50 | BEIR SOTA (61.94) |
| bge-reranker-v2-gemma | 2.5B | ~5 GB | Higher | Best quality на 16GB |

---

## Бенчмарки для ориентации

- **Production RAG (structured docs)**: 0.70-0.85 recall@5
- **Short social media, non-English**: **0.55-0.70** recall@5 — наш реалистичный target
- **Stretch goal**: 0.75+
- **Текущий**: 0.59 (10 questions, stat. незначимо, CI [0.35-0.80])
- **Минимум для regression testing**: 50 вопросов
- **Минимум для значимости**: 200+ вопросов
