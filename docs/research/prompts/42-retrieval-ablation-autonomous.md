# Prompt 42: Retrieval Ablation — Autonomous Research Session

## Контекст

rag_app — RAG платформа с ReAct агентом над 36 Telegram каналами (13777 points в Qdrant `news_colbert_v2`).
Текущий retrieval pipeline: `BM25(100) + Dense(20) → RRF [1.0, 3.0] → ColBERT → CE filter(0.0) → Channel dedup(2)`.

Baseline метрики (calibration 50 Qs): R@1=0.80, R@3=0.97, R@5=0.97, R@20=0.98.
Benchmark (auto-generated 100 Qs): R@1=0.94, MRR=0.944.

Разница auto vs hand-crafted: auto содержит exact text fragments (ColBERT token matching доминирует), hand-crafted = natural language (dense embedding уже ловит семантику). Нужен надёжный natural language dataset.

Многие техники отвергнуты по одной, но комбинации не проверены:
- DBSF: 0.72 vs RRF 0.73 (close)
- CE reranking: r@3 degrades 0.97→0.94 (но при другом threshold?)
- Instruction prefix: DEC-0042 says no prefix, но A/B тест на pplx-embed не проводился
- PCA whitening 1024→1024: parity (но с другими weights?)

## Задача — три фазы, выполнить последовательно до конца

### Фаза 1: Новый retrieval dataset (100-150 natural language Qs)

**Цель**: dataset имитирующий поведение реального пользователя. Не copy-paste из документов.

**Метод**:
1. Scroll Qdrant (`http://localhost:16333`, коллекция `news_colbert_v2`) — выбрать ~200 разнообразных документов (разные каналы, даты, темы). API: `POST /collections/news_colbert_v2/points/scroll`
2. По каждому документу сгенерировать 1-2 natural language вопроса как бы их задал пользователь. Вопросы на русском. Использовать самого себя (Claude) или Codex как sub-agent для генерации.
3. Категории вопросов (target distribution):
   - factual: "Кто/Что/Когда..." — 40%
   - temporal: "Что было в январе/на прошлой неделе..." — 15%
   - channel-specific: "Что писал gonzo_ml про..." — 15%
   - comparative: "Сравни мнения про X" — 10%
   - entity: "Что известно про NVIDIA/DeepSeek..." — 10%
   - edge cases: слэнг, неточные формулировки, длинные запросы — 10%
4. Для каждого вопроса: `{"id", "query", "expected_documents": ["channel:message_id"], "category", "source_text_preview"}`
5. Валидация: для каждого вопроса проверить что expected document находится в Qdrant через search

**Формат** (аналогичен `datasets/eval_retrieval_calibration.json`):
```json
[
  {
    "id": "ret_001",
    "query": "Какие компании разрабатывают человекоподобных роботов?",
    "expected_documents": ["ai_newz:4355", "techsparks:5510"],
    "category": "factual",
    "source_text_preview": "AgiBot отчитался о выпуске 10-тысячного..."
  }
]
```

**Сохранить**: `datasets/eval_retrieval_v3.json`

### Фаза 2: Параметризация evaluate_retrieval.py

**Файл**: `scripts/evaluate_retrieval.py`

Добавить CLI параметры (сейчас хардкожены в `src/adapters/search/hybrid_retriever.py`):
- `--rrf-weights` (default: "1.0,3.0") — Dense:BM25 weights
- `--bm25-limit` (default: 100) — BM25 prefetch limit
- `--dense-limit` (default: 20) — Dense prefetch limit
- `--ce-threshold` (default: None) — CE filter threshold (None = off)
- `--prefix` (flag) — instruction prefix для embedding
- `--no-dedup` (flag) — отключить channel dedup

Параметры пробрасываются через env vars или monkey-patch в HybridRetriever перед запуском.

**Важно**: не ломать существующий интерфейс. Без новых параметров = текущее поведение.

### Фаза 3: Ablation experiments

**Подход**: sequential sweep с carry-forward. Каждый шаг фиксирует лучший параметр.

**Перспективные эксперименты (не все подряд, а осмысленные)**:

| # | Experiment | Варианты | Почему перспективно |
|---|-----------|----------|-------------------|
| 1 | **Instruction prefix** | on / off | Никогда не A/B тестировали на pplx-embed. evaluate_retrieval.py уже использует prefix, production — нет |
| 2 | **RRF weight sweep** | 1:2, 1:3, 1:4, 1:5 | 3:1 эмпирический. 4:1 или 5:1 может усилить BM25 ещё |
| 3 | **DBSF fusion** | DBSF vs лучший RRF | Было close (0.72 vs 0.73). На новом dataset может выиграть |
| 4 | **BM25 top-K** | 50, 100, 200 | Больше candidates = лучше для ColBERT rerank? |
| 5 | **CE threshold** | -1, 0, 1, 2 | CE filter при 0.0 keeps 92% relevant. При 1.0 — строже, может убрать noise |
| 6 | **CE rerank (не filter)** | top-20 → top-5/10 CE rerank | Раньше degrades r@3, но при другом top_n? Комбо с DBSF? |
| 7 | **Dense top-K** | 10, 20, 40 | Больше dense candidates → RRF получает больше diversity? |

**Комбинации для проверки** (rejected по одной, но combo не тестировалось):
- Best RRF weight + instruction prefix
- DBSF + CE reranking (не filter)
- BM25 200 + CE threshold 1.0 (больше кандидатов + строже фильтр)

**Запуск каждого**: `python scripts/evaluate_retrieval.py --dataset datasets/eval_retrieval_v3.json --collection news_colbert_v2 [params]`

**Метрики**: Recall@1, Recall@3, Recall@5, Recall@20, MRR. Монотонность (recall не падает при увеличении k).

### Output

1. **Dataset**: `datasets/eval_retrieval_v3.json` (100-150 Qs)
2. **Результаты**: таблица experiment × metric, сохранить в `results/ablation/`
3. **Анализ**: какая конфигурация лучше baseline, на сколько, на каких категориях
4. **Рекомендация**: оптимальный pipeline config с evidence

## Критические файлы

- `scripts/evaluate_retrieval.py` — модифицировать для параметризации
- `src/adapters/search/hybrid_retriever.py` — конфигурационные точки:
  - строка 237: BM25 limit `max(k_per_query * 10, 100)`
  - строка 238: Dense limit `max(k_per_query * 2, 20)`
  - строка 266: RRF weights `[1.0, 3.0]`
  - строка 299: RRF weights (второй query path)
  - строка 310: Channel dedup `max_per_channel=2`
- `src/adapters/tei/embedding_client.py` — instruction prefix config
- `datasets/eval_retrieval_calibration.json` — reference формат (50 Qs)
- `datasets/eval_retrieval_100.json` — auto-generated reference (100 Qs)
- `benchmarks/config.py` — DENSE_TOP_K, SPARSE_TOP_K, FINAL_TOP_K

## Hardware (подтверждено работает)

- Qdrant: `http://localhost:16333`, коллекция `news_colbert_v2`, 13777 points
- gpu_server.py: `http://localhost:8082` — pplx-embed + Qwen3-Reranker + jina-colbert-v2
- llama-server: `http://localhost:8080` — Qwen3.5-35B (для LLM tasks если нужно)
- API: `http://localhost:8001` (Docker)

## ВАЖНО

- Не останавливаться на середине. Довести все три фазы до конца.
- Результаты собирать в структурированном виде для последующего анализа.
- При ошибках — debug и fix, не пропускать эксперимент.
- Если контекст приближается к лимиту — compact и продолжить по плану.
- Финальный deliverable: dataset + таблица результатов + рекомендация.
