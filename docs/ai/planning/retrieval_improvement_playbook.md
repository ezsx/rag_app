# Retrieval Improvement Playbook

> Живой документ. Ответ на вопрос "что дальше пробовать чтобы повысить recall?"
> Источник: R11-advanced-retrieval-strategies.md + собственные эксперименты.
> Последнее обновление: 2026-03-19

---

## Текущее состояние

**Recall@5 = 0.70** на quick dataset (10 вопросов, 6 full + 1 partial из 9 answerable).
**Coverage = 0.86**, **Answer rate = 9/10**.
**Target: 0.80+** (реалистично с whitening + reranker), **0.85+** (stretch goal с BGE-M3).

### История экспериментов

| # | Дата | Изменение | Recall@5 | Δ | Файл результата | Коммит |
|---|------|-----------|----------|---|-----------------|--------|
| 1 | 2026-03-19 | Baseline: RRF → dense re-score (broken SSE) | 0.00 | — | eval_results_20260319-121112 | — |
| 2 | 2026-03-19 | SSE fix, baseline (dense re-score) | 0.00 | — | eval_results_20260319-122157 | — |
| 3 | 2026-03-19 | + SSE fix + fuzzy recall (limit 3) | 0.00 | — | eval_results_20260319-122430 | — |
| 4 | 2026-03-19 | Убрали dense re-score → чистый RRF | 0.15 | — | eval_results_20260319-123202 | — |
| 5 | 2026-03-19 | + Orig query в subqueries (best RRF) | 0.33 | +0.18 | eval_results_20260319-125400 | 72efb31 |
| 6 | 2026-03-19 | + MMR post-process (lambda=0.7) | 0.11 | -0.22 | eval_results_20260319-130723 | reverted |
| 7 | 2026-03-19 | + MMR post-process (lambda=0.9) | 0.11 | -0.22 | eval_results_20260319-131422 | reverted |
| 8 | 2026-03-19 | Pure RRF + orig query (best so far) | **0.59** | +0.26 | eval_results_20260319-133619 | e0bd871 |
| 9 | 2026-03-19 | Weighted RRF 3:1 (expired JWT → 0 agent) | 0.00 | broken | eval_results_20260319-174155 | — |
| 10 | 2026-03-19 | Weighted RRF 3:1 + new JWT | 0.48 | -0.11 | eval_results_20260319-174813 | testing |
| **11** | **2026-03-19** | **+ Forced search + dynamic tools** | **0.70** | **+0.11** | **eval_results_20260319-175853** | **036e54f** |

### Подробные результаты лучшего прогона (#11, recall@5=0.70)

| Q | Тип | Вопрос | Recall | Cov | Статус | Citations (top-4) |
|---|-----|--------|--------|-----|--------|-------------------|
| Q1 | factual | Financial Times ЧГ 2025 | **1.0** | 0.88 | ✅ | ai_ml_big_data:9245 ← exact match |
| Q2 | factual | GPT OSS параметры | **1.0** | 0.91 | ✅ | rybolos:1563 (±1 от :1562) |
| Q3 | factual | Meta + Manus AI | **1.0** | 0.83 | ✅ | ai_newz:4355 ← exact match, top-1 |
| Q4 | temporal | Декабрь 2025 Google/NVIDIA | 0.0 | 0.85 | ✅ ответ | Нашёл декабрьские, но не msg 9245/9226 |
| Q5 | temporal | Январь 2026 AI-каналы | 0.33 | 0.80 | ✅ partial | boris_again:3703 (±2 от :3701) |
| Q6 | channel | llm_under_hood reasoning GPT-5 | **1.0** | 0.92 | ✅ | llm_under_hood:648 ← exact match |
| Q7 | channel | boris_again Gemini 3 Flash | **1.0** | 0.91 | ✅ | boris_again нет в top-4, но в citations |
| Q8 | comparative | Deep Think vs o3-pro | **1.0** | 0.84 | ✅ | seeallochnaya:2711 ← exact match, top-1 |
| Q9 | multi_hop | LLM production 2 канала | 0.0 | 0.84 | ✅ ответ | llm_under_hood:641/723 (не :652/:769) |
| Q10 | negative | GPT-6 | N/A | 0.86 | ❌ корректный отказ | — |

**Анализ провалов:**
- Q4, Q9: retrieval находит **правильный канал и тему**, но другой msg_id (chunk boundary). Fuzzy ±5 не хватает — посты разбиты на чанки с разными msg_id.
- Q5: partial — нашёл 1 из 3 expected документов (boris_again:3703 ≈ :3701).

### Корневые проблемы (диагностированы)

1. **Embedding anisotropy** — все AI-тексты в cosine range [0.78-0.83]. "Attractor documents" попадают в top-10 любого запроса.
2. **Length-Induced Collapse** (paper arXiv:2410.24200) — тексты одинаковой длины кластерятся вместе вне зависимости от содержания. Наши посты все 300-1500 символов.
3. **Реранкер suboptimal** — bge-m3 (bi-encoder seq-cls mode) вместо bge-reranker-v2-m3 (dedicated cross-encoder, +10 nDCG).
4. **Cosine-based MMR не работает** — переиспользует сломанный cosine signal → re-promotes attractor documents.
5. **Query expansion теряет сущности** — LLM перефразирует и убирает ключевые слова. Частично решено (оригинальный запрос как subquery).

---

## Tier 1: Quick Wins (часы, не дни)

### 1.1 Embedding Whitening (PCA 1024→512)
- **Суть**: Global PCA whitening — mean-centering + scaling по eigenvalues + dimensionality reduction.
- **Почему поможет**: раздвигает cosine range. **Подтверждено экспериментально (2026-03-19)**:
  - ДО: pairwise cosine mean=0.7954, std=0.0933, range [0.33, 0.95]
  - ПОСЛЕ: pairwise cosine mean=0.0006, std=0.0484, range [-0.17, 0.61]
  - Variance explained: 96.2% (потеря 3.8% информации при 1024→512)
- **Whitening params сохранены**: `datasets/whitening_params.npz` (mean, components, explained_variance)
- **Нюанс**: query-only mean-centering **НЕ работает** (asymmetric — query в другом пространстве, docs в оригинальном). Нужна **полная переиндексация**: embed → whitening transform → upload whitened vectors в Qdrant.
- **Как реализовать**: скрипт переиндексации: загрузить все 13K vectors, применить whitening, создать новую коллекцию с dense_vector 512-dim, скопировать sparse + payload.
- **Ожидание**: +5-15% recall (по литературе: Su et al. +8-12 Spearman, WhitenRec +7-16% recall)
- **Результат эксперимента (2026-03-19)**: переиндексация выполнена в коллекцию `news_whitened`. **Recall упал с 0.70 до 0.56.** Whitening улучшил reranker score differentiation, но изменил RRF ranking непредсказуемо. Coverage metric требует рекалибровки (0.42 в whitened space). **Решение: откат на `news` (recall 0.70).** `news_whitened` сохранена для дальнейших экспериментов.
- **Статус**: [x] whitening подтверждён, params сохранены. [x] переиндексация выполнена. **Recall не улучшился — отложен.**
- **Ссылки**: Su et al. 2021 "BERT-whitening", WhitenRec 2024, WhiteningBERT (Huang et al. EMNLP 2021)

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

> **Статус**: исследование завершено (R12). **Вердикт: Phase 4, не Phase 1.** Есть более простые и impactful решения. Кластеризация становится оправданной при 50K+ документов или когда простые фиксы plateau'ят.
> **Источник**: [R12-cluster-based-retrieval.md](../../research/rag-stack/reports/R12-cluster-based-retrieval.md)

### Исходная идея

Одна плоская коллекция на весь AI-корпус — наивно. Все документы "про AI" сливаются в embedding space (cosine 0.78-0.83). Если кластеризовать по темам (M&A, релизы моделей, образование, research papers...) — внутри каждого кластера cosine станет осмысленным.

### Результаты исследования (R12)

**Comparison table (ключевой результат)**:

| Подход | Effort | Recall@5 Δ | Cumulative | Ops burden |
|--------|--------|------------|------------|------------|
| Weighted RRF tuning | 1-2 ч | +3-10% | 0.62-0.65 | None |
| **Global PCA whitening** (1024→512) | 2-4 ч | **+5-15%** | 0.67-0.73 | Near-zero |
| **bge-reranker-v2-m3** | 4-8 ч | **+15-30%** | 0.75-0.82 | ~200ms latency |
| BGE-M3 model swap (dense+sparse+ColBERT) | 8-12 ч | +25-40% | 0.80-0.88 | Medium |
| **Topic clustering + routing** | **20-40 ч** | +10-20% | 0.69-0.77 | **High (ongoing)** |

**Вывод**: кластеризация даёт **меньше recall** при **большем effort** чем whitening + reranker. Cross-encoder reranker "bypasses the cosine floor entirely" — обходит проблему embedding collapse фундаментально, без кластеров.

### Что мы узнали (ценные находки)

**Per-cluster whitening математически некорректен** при наших размерах:
- 200-600 docs/cluster, 1024 dims → ковариационная матрица rank-deficient (625 eigenvalues = 0)
- Деление на 0 при whitening = amplification шума
- **Безопасная операция**: только mean-centering per-cluster (без PCA)
- **Global whitening** корректен (N=13000, d=1024, ratio ~12.7)

**BERTopic — правильный инструмент** (не raw TF-IDF):
- TF-IDF плох для коротких мультиязычных текстов (40-120 tokens → шумные вектора)
- Ключевая идея: **отдельная embedding модель** для кластеризации (paraphrase-multilingual-mpnet-base-v2), не наша Qwen3-Embedding (которая сама "сломана")
- UMAP проецирует в 5 dims → re-scales compressed distance space → HDBSCAN работает
- 25-40 кластеров оптимально (min_cluster_size=100-200)
- Soft assignment через `approximate_distribution()` (top-3 clusters per doc, хранить как array payload в Qdrant)

**Routing: ensemble (centroid + BM25 frequency)**, не LLM:
- Embed query → nearest centroids (top-3) + BM25 full-corpus top-50 → count cluster frequency (top-3) → union
- Суммарная latency ~100-150ms на CPU
- LLM-classification **не рекомендуют**: "slow inference, high costs, poor accuracy on domain-specific topics"
- Fine-tuned BERT classifier (94% accuracy, ms latency) — если нужен ML-based router, но требует labeled data

**Incremental updates**: daily nearest-centroid assignment + weekly BERTopic `merge_models()`. Trigger re-clustering при drift > threshold или >500 orphan docs.

**"Embedding collapse on topically narrow corpus — expected behavior, не баг"**:
- Ethayarajh 2019: average BERT pairwise cosine = 0.99 (!)
- Zhou et al. ACL 2025 (Length-Induced Collapse): self-attention = low-pass filter, фундаментальное свойство архитектуры
- Domain homogeneity (все про AI) amplifies сужение

### Когда кластеризация НУЖНА

1. **50K+ документов** — reranker не справляется с шумным candidate pool, cluster filtering сужает HNSW search space на 90-95%
2. **Расширение за AI/ML** — crypto, biotech, finance → cross-domain filtering через кластеры
3. **Per-cluster whitening at scale** — при 500+ docs/cluster ковариация стабильна

### Архитектура (финальная, из R12)

```
Query → [Embed] + [BM25 full corpus]
           ↓              ↓
    [Global Whitening]  [Top-50 BM25]
    [PCA 1024→512]      [Cluster frequency]
           ↓              ↓
    [Centroid routing]  [top-3 clusters]
           ↓              ↓
         [Union: 3-5 clusters]
                  ↓
    [Qdrant Filtered Hybrid Search]
    filter: cluster_ids ∈ selected
    prefetch: dense(20) + BM25(20)
    fusion: weighted RRF
                  ↓
    [bge-reranker-v2-m3 top-20 → top-5]
                  ↓
         [Final Results]
Latency: ~240ms total
```

### Код для будущей реализации

```python
# BERTopic clustering
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

cluster_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
embeddings = cluster_model.encode(all_texts, batch_size=64)

topic_model = BERTopic(
    embedding_model=cluster_model,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
    hdbscan_model=HDBSCAN(min_cluster_size=150, min_samples=10, prediction_data=True),
    calculate_probabilities=True
)
topics, probs = topic_model.fit_transform(all_texts, embeddings)

# Soft assignment → Qdrant payload
topic_distr, _ = topic_model.approximate_distribution(all_texts, window=4, stride=1)
for i, point_id in enumerate(point_ids):
    top_ids = np.argsort(topic_distr[i])[::-1][:3]
    assigned = [int(t) for t in top_ids if topic_distr[i][t] > 0.05] or [int(top_ids[0])]
    client.set_payload("news", {"cluster_ids": assigned}, points=[point_id])
```

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
- **Short social media, non-English**: **0.55-0.70** recall@5
- **С whitening + cross-encoder reranker**: **0.75-0.82** (R12 estimate)
- **С BGE-M3 swap**: **0.80-0.88** (R12 estimate)
- **Текущий**: **0.70** (10 questions, quick dataset) — уже в production range!
- **Минимум для regression testing**: 50 вопросов
- **Минимум для значимости**: 200+ вопросов

## Рекомендуемый путь к 0.80+ (из R11 + R12)

```
Текущее состояние: recall@5 = 0.70
  ↓
Phase 1 (Day 1-2): Global PCA whitening (1024→512) + DBSF fusion test
  → Expected: 0.73-0.78
  ↓
Phase 2 (Day 3-5): bge-reranker-v2-m3 (dedicated cross-encoder)
  → Expected: 0.78-0.85
  ↓
Phase 3 (Week 2, если нужно): BGE-M3 model swap (dense+sparse+ColBERT в одной модели)
  → Expected: 0.83-0.88
  ↓
Phase 4 (Week 3+, при масштабировании): BERTopic кластеризация как metadata layer
  → Expected: дополнительные +5-10% при 50K+ docs
```
