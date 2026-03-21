# Deep Research: Techniques That Outperform Standard RAG Frameworks

## О проекте

**rag_app** — portfolio-grade RAG система для собеседований на позицию **Applied LLM Engineer**. Поисковик/агрегатор новостей из 36 AI/ML Telegram-каналов (~13K документов). Цель проекта — показать на собесе что custom pipeline **принципиально лучше** чем LlamaIndex/LangChain из коробки, с конкретными цифрами и архитектурными решениями.

**Автор**: backend-разработчик (Python), опыт ML (обнаружение аневризм мозга MRI, внедрён в больницах), VPN-сервис с нуля за 2.5 мес. Цель — Applied LLM Engineer позиции ($2-3k/мес).

---

## Железо и стек

- **LLM**: Qwen3-30B-A3B GGUF (MoE, 3B active) на V100 SXM2 32GB, llama-server.exe, Windows Host
- **GPU-сервер** (RTX 5060 Ti 16GB, WSL2 native, gpu_server.py):
  - Qwen3-Embedding-0.6B (dense 1024-dim)
  - bge-reranker-v2-m3 (cross-encoder, 568M)
  - jina-colbert-v2 (ColBERT per-token, 560M, 128-dim)
- **Qdrant v1.17**: dense + sparse(BM25) + ColBERT(multi-vector MaxSim)
- **Docker**: API + Qdrant (CPU only, GPU недоступен из Docker — V100 TCC блокирует NVML)
- **ReAct Agent**: native function calling, не regex-парсинг. 5 tools: query_plan → search → rerank → compose_context → final_answer

---

## Корпус

- 13124 документа из 36 Telegram-каналов
- AI/ML тематика, русский + английский, короткие посты (100-1500 символов)
- Каналы: от CTO Intento (papers) до директора Яндекса (индустрия), от PhD ex-Meta (новости) до AIRI FusionBrain (research)
- Период: 2025-07-01 → 2026-03-18 (9 месяцев)
- Payload: channel, date, message_id, links, media_types

---

## Что уже реализовано и работает

### Pipeline

1. **Weighted RRF** (BM25 3:1 over dense) — BM25 keyword match доминирует, решает attractor problem
2. **ColBERT reranking** (jina-colbert-v2) — per-token MaxSim, recall@1 +97%, recall@5 +33%
3. **Multi-query search** — LLM генерирует 5 subqueries, round-robin merge (раньше использовался только первый — critical bug, +33% recall после fix)
4. **Channel dedup** — max 2 docs/channel для diversity
5. **Dynamic tools** — final_answer скрыт до search, forced search если LLM не вызывает tools
6. **Original query injection** — оригинальный запрос всегда в BM25 subqueries

### Evaluation infrastructure

- **Agent eval**: 20 вопросов (v1 10 + v2 10), через LLM, ~40с/запрос
- **Retrieval eval**: 100 вопросов, прямые Qdrant queries, ~5с/запрос
- **22 прогона eval** с разными конфигурациями, полная история в playbook

### Метрики (2026-03-20)

| Dataset | Recall@5 | Coverage | Тип |
|---------|----------|----------|-----|
| v1 (10 Qs, factual/temporal/channel/comparative) | 0.76 | 0.86 | Agent eval |
| v2 (10 Qs, entity/product/cross_channel/recency) | 0.61 | 0.80 | Agent eval |
| 100 Qs, ColBERT + RRF | 0.73 | — | Retrieval eval |
| 100 Qs, RRF only (no ColBERT) | 0.55 | — | Retrieval eval |

### Что протестировано и не работает

| Техника | Результат | Почему |
|---------|-----------|--------|
| Cosine-based MMR | recall 0.70→0.11 | Re-promotes attractor documents через cosine |
| Dense re-score после RRF | recall 0.33→0.15 | Стирает BM25 вклад |
| PCA whitening 1024→512 | recall 0.70→0.56 | Слишком агрессивный cutoff |
| Whitening 1024→1024 | паритет | Dense не bottleneck при BM25 3:1 |
| DBSF fusion | recall 0.72 vs RRF 0.73 | RRF чуть лучше |

### Текущая корневая проблема

Pipeline **линейный** — каждый запрос идёт одним путём. LlamaIndex делает ровно то же самое. Нет адаптации:
- "Что было в январе 2026" → нужен date filter, а идёт generic search
- "Что писал gonzo_ml" → нужен channel filter, а идёт generic search
- "Vera Rubin NVIDIA" → LLM не знает entity, не генерирует правильный subquery
- "Сколько стоит Deep Think" → нужен точный fact lookup, а идёт broad search

---

## Что я ищу

### 1. Техники которые НЕ реализуемы стандартными фреймворками

Конкретные подходы которые:
- Требуют custom архитектуры (не настраиваются через config LlamaIndex)
- Дают **измеримый** прирост (цифры из papers/benchmarks)
- Реализуемы self-hosted на V100 + RTX 5060 Ti за 3-5 дней

Направления (не ограничиваться):
- **Adaptive retrieval** — разные стратегии для разных типов запросов
- **Self-reflection / CRAG** — агент оценивает качество поиска и корректирует
- **Agentic document discovery** — агент активно исследует коллекцию
- **Temporal reasoning** — "недавно", "в прошлом месяце", "после релиза X"
- **Multi-source synthesis** — собрать из 3-5 каналов, найти противоречия
- **Citation verification** — проверить что ответ следует из найденных документов
- **Query decomposition** — разбить сложный вопрос на подвопросы

### 2. Agentic RAG patterns (2025-2026)

Что нового за последний год? Конкретно:
- **CRAG** (Corrective RAG) — evaluator + query rewriting. Реальные цифры?
- **Self-RAG** — LLM решает когда искать. Применимо к Qwen3-30B?
- **Adaptive-RAG** — dynamic complexity assessment. Без fine-tuning?
- **Plan-and-Execute RAG** — отличие от нашего query_plan?
- **Reflexion for RAG** — self-reflection loop. Overhead vs improvement?
- **Что-то совсем новое** из 2026 что мы могли пропустить?

### 3. Что впечатляет на Applied LLM собеседованиях

- Какие архитектурные решения показывают "senior-level thinking"?
- Что отличает "собрал RAG из компонентов" от "спроектировал систему"?
- Ablation study: как правильно показать вклад каждого компонента?
- Какие метрики кроме recall ценятся: faithfulness, cost, scalability, latency?
- Примеры portfolio проектов RAG которые **впечатлили** hiring managers

### 4. Конкретный plan of attack на 3-5 дней

У нас recall@5 = 0.61-0.76 (agent), 0.73 (retrieval). Pipeline работает. Что даст максимальный impact:
- Реализовать tool router + specialized tools? (это наш план)
- Добавить CRAG/Self-RAG pattern?
- LlamaIndex baseline + сравнительная таблица?
- Глубокий ablation study?
- Всё вместе в каком порядке?

### 5. Open-source примеры

- GitHub репозитории с agentic RAG выходящим за рамки фреймворков
- Конкретные реализации tool routing для RAG
- Datasets и benchmarks для evaluation RAG agents

---

## Ограничения

- Qwen3-30B (3B active MoE) — не GPT-4, ненадёжные tool calls
- V100 32GB (LLM) + RTX 5060 Ti 16GB (3 модели ~4-5GB VRAM, ~11GB свободно)
- 13K документов, 36 каналов, русский + английский
- VPN блокирует сеть в WSL2
- 20 eval вопросов (стат. незначимо)
- Self-hosted only, без API calls

## Предыдущие исследования (для контекста)

- **R11**: Advanced Retrieval Strategies — ColBERT, weighted RRF, whitening, entity extraction, query classifier (+8% accuracy, -35% latency). Всё кроме entity extraction и query classifier реализовано.
- **R12**: Cluster-Based Retrieval — BERTopic кластеризация отложена (effort > impact при 13K docs). Cross-encoder reranker компенсирует embedding collapse.
- **R05**: RAG Evaluation — RAGAS metrics, golden dataset methodology, LLM-as-judge.
- **R07**: Pipeline Quality — embedding selection (Qwen3-Embedding-0.6B), reranker selection, chunking strategies.
