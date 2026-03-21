# Phase 2 Research Prompts

> **Дата:** 2026-03-17
> **Контекст:** Phase 1 завершён, E2E работает. Нужны улучшения quality.
> **Предыдущие исследования:** R01-R06 в `reports/`

---

## R-07: Retrieval & Agent Pipeline Quality

### Контекст проекта

Платформа `rag_app` — RAG-поисковик по русскоязычным Telegram-каналам (ML/AI/LLM тематика).

**Текущий стек retrieval:**
- Embedding: `intfloat/multilingual-e5-large` (1024-dim) через TEI HTTP
- Sparse: `fastembed SparseTextEmbedding` (Qdrant/bm25, language="russian")
- Vector store: Qdrant v1.17 (named vectors: dense_vector cosine + sparse_vector IDF)
- Search: Qdrant native `prefetch[dense, sparse] → FusionQuery(RRF) → MmrQuery`
- Reranker: `bge-reranker-v2-m3` через TEI HTTP (есть, но НЕ используется в основном agent pipeline — только в qa_service)
- Coverage: 5-signal composite metric (max_sim, mean_top_k, term_coverage, doc_count_adequacy, score_gap, above_threshold)

**Текущий agent pipeline (7+1 tools):**
```
router_select → query_plan → search → [rerank — пропускается] → compose_context → verify → final_answer
```
Агент использует текстовый ReAct формат (Thought/Action/Observation) через `/v1/completions` endpoint.
LLM парсинг: regex на Thought:/Action:/FinalAnswer: маркеры.

**Текущие проблемы:**
1. **Нет чанкования**: 1 сообщение Telegram = 1 вектор. Длинные дайджесты (2000-5000 символов с 5+ разными темами) получают размытый вектор, snижая precision.
2. **Reranker не в pipeline**: search → compose_context напрямую, без rerank между ними.
3. **Coverage ~0.42-0.55** на 885 points — формула может быть не откалибрована.
4. **LLM галлюцинирует**: вместо фактов из контекста придумывает generic ответы (GPT-4, медицина и т.д.)
5. **Текстовый ReAct парсинг хрупкий**: Qwen3 thinking mode ломает маркеры, пришлось стрипать preamble и форсить FinalAnswer программно.

**Данные:** 885 points из 5 русскоязычных ML/AI Telegram-каналов (Jan-Mar 2026).

**Железо для embedding/reranker:** RTX 5060 Ti 16GB (WSL2 native TEI).

### Вопросы для исследования

**Блок 1: Chunking**
1. Какая стратегия чанкования оптимальна для Telegram-контента? Типичный пост: 100-500 символов (короткая новость) или 2000-5000 (дайджест с 5+ темами). Варианты: fixed-size (300-800 chars), semantic (по абзацам/переносам строк), sentence-based, recursive.
2. Нужен ли overlap между чанками для коротких постов? Или это overhead?
3. Как чанкование влияет на dense_score при поиске? Ожидается рост precision — подтверждается ли эмпирически?
4. Стоит ли чанковать ВСЕ сообщения или только длинные (>N символов)? Какой порог?

**Блок 2: Reranker в pipeline**
5. Какой прирост quality даёт reranker (bge-reranker-v2-m3) при k=10 после hybrid search? При k=20? Бенчмарки на русскоязычном контенте.
6. Есть ли более современные reranker модели (2025-2026) лучше bge-reranker-v2-m3? Особенно для русского языка. Рассмотреть: Qwen reranker, Cohere rerank v3, Jina reranker v2.
7. Latency impact: TEI rerank на RTX 5060 Ti при batch 10-20 документов — приемлемо ли для real-time agent?

**Блок 3: Embedding model**
8. `multilingual-e5-large` (1024-dim, 2023) vs `Qwen3-Embedding-0.6B` (2025) — quality comparison на MTEB Multilingual, особенно русский сегмент.
9. Qwen3-Embedding требует instruction prefix (`query:` / `passage:`) — совместим ли с текущим TEI HTTP setup? Нужны ли изменения в TEIEmbeddingClient?
10. Стоит ли рассматривать другие embedding 2025-2026: `BGE-M3`, `Jina-embeddings-v3`, `GTE-Qwen2`?
11. Dimension vs quality tradeoff: 1024-dim e5-large vs 1024-dim Qwen3 vs Matryoshka (variable dim)?

**Блок 4: Agent architecture**
12. Текстовый ReAct (Thought/Action/Observation через regex) vs native function calling (`/v1/chat/completions` с tools schema) — что надёжнее для Qwen3? Меньше ли галлюцинаций при function calling?
13. Наш pipeline: 7+1 tools. Какие инструменты лишние? Чего не хватает? Сравнить с state-of-art RAG agent architectures 2025-2026.
14. Grounding problem: как заставить LLM использовать ТОЛЬКО контекст и не галлюцинировать? Лучшие практики 2025-2026: prompt techniques, constrained generation, citation-forced prompts.
15. Стоит ли разделить agent на два этапа: retrieval agent (search+rerank+compose) → generation agent (answer based on context)? Или single-agent лучше?

### Формат ответа

Для каждого блока:
1. Текущее состояние art (что делают в 2025-2026)
2. Конкретные рекомендации для нашего стека
3. Приоритет (must-have / nice-to-have / future)
4. Оценка effort (часы/дни) и expected impact

---

## R-08: LLM Selection для V100 SXM2 32GB

### Контекст железа

**Текущая конфигурация:**
- GPU: NVIDIA Tesla V100 SXM2 32GB (compute 7.0, PCIe на Windows Host)
- Режим: TCC (Tesla Compute Cluster) — означает что GPU работает как чисто вычислительное устройство, без дисплея
- Runtime: `llama-server.exe` (llama.cpp build 8354) на Windows Host, порт 8080
- Текущие параметры: `-c 16384 --parallel 2 --flash-attn on -ngl 99 --main-gpu 0`
- VRAM usage: ~13GB из 32GB (модель 8.5GB + KV cache 2×16K)

**Ограничения V100:**
- Compute capability 7.0 — НЕ поддерживает BF16, только FP16
- Нет Tensor Core для INT8/INT4 matmul (это Ampere+)
- Flash Attention: поддерживается в llama.cpp (SM 7.0+), но без hardware-optimized FlashAttn2
- Bandwidth: 900 GB/s (HBM2) — хороша для inference, хуже чем A100 (2TB/s)
- TCC mode на Windows: Docker GPU недоступен (NVML blocked), только native процессы

**Текущая модель:** Qwen3-8B Q8_0 (8.5GB)
- Качество ответов: умеренное. Галлюцинирует при RAG, thinking mode расходует tokens.
- Скорость: query_plan ~10-18s, final answer ~5-10s. Приемлемо для single-user.

**Вторая GPU:** RTX 5060 Ti 16GB (compute 12.0, Blackwell) — занята TEI embedding+reranker.

**Сценарий использования:** 1 пользователь, не параллельный. Real-time agent с SSE streaming. Можно позволить 2-5 секунд на шаг.

**Runtime ограничение:** Только llama-server.exe на Windows. vLLM (Linux-only) — невозможен сейчас. Возможен в будущем при переходе на Proxmox/bare-metal Linux, но это отдельный upgrade.

### Вопросы для исследования

**Блок 1: Размер модели**
1. На V100 32GB с llama.cpp: какой максимальный размер модели вместе с KV cache для `-c 16384 --parallel 1`?
   - Q8_0: ~Xgb модель + ~Ygb KV = max model size?
   - Q4_K_M: аналогично
   - Q5_K_M: аналогично
2. Qwen3-14B Q5_K_M (~10GB) — влезет ли с 16K контекстом? Какой прирост quality?
3. Qwen3-32B Q4_K_M (~18GB) — влезет ли с 8K контекстом? Стоит ли жертвовать контекстом ради quality?
4. Есть ли MoE модели (Mixture of Experts) которые дают quality 30B+ при VRAM footprint 12-15GB? DeepSeek V3 lite?

**Блок 2: Выбор модели (2025-2026 landscape)**
5. Для RAG agent с function calling/ReAct на русском языке — какие модели показывают лучшие результаты?
   - Qwen3 family (8B, 14B, 32B)
   - Mistral/Mixtral (Nemo 12B, Medium)
   - Llama 4 family (Maverick, Scout)
   - DeepSeek V3/V4
   - Gemma 3 (27B)
   - Phi-4 (14B)
6. Какие модели лучше всего справляются с grounding (использование только контекста, без галлюцинаций)?
7. Какие модели лучше поддерживают русский язык в RAG-сценарии?

**Блок 3: Thinking mode**
8. Qwen3 thinking mode: количественный impact на quality для RAG-задач. Стоит ли overhead ~200-500 tokens на шаг?
9. Hybrid thinking: `/no_think` для tool-calling шагов + thinking для final answer. Возможно ли технически с llama-server?
10. Альтернативы thinking: DeepSeek reasoning, Llama reasoning — есть ли модели с более эффективным CoT?

**Блок 4: llama.cpp оптимизация для V100**
11. Flash Attention на V100 (SM 7.0): работает ли корректно? Benchmark прирост?
12. KV cache quantization (Q8_0 или Q4_0 KV) — доступно ли для V100? Это позволит увеличить контекст без роста VRAM.
13. Speculative decoding с маленькой draft-моделью (Qwen3-0.6B) — возможно ли на V100? Прирост tokens/sec?
14. Context shifting vs YaRN для расширения контекста — что поддерживает llama.cpp для Qwen3?

### Формат ответа

1. **Top-3 рекомендации моделей** с конкретными файлами (HuggingFace repo, quantization, size)
2. **VRAM budget таблица**: model_size + KV_cache = total для каждого варианта
3. **Quality comparison**: если есть бенчмарки на RAG/function-calling/Russian
4. **Конкретная команда llama-server** для каждого рекомендованного варианта
5. **Migration effort**: что нужно поменять в коде при смене модели
