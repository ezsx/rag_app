# RAG Stack Research — Synthesis

**Дата**: 2026-03-16
**Статус**: Финальный
**Источники**: R01–R06 (все 6 research-треков завершены), R07 ⬜ (Proxmox — отдельный трек, не блокирует архитектуру)
**Следующий шаг**: architecture design → specifications

---

## Executive Summary

Текущий стек (ChromaDB + кастомный BM25 + Qwen2.5-7B + llama-server + requests.Session) **функционален, но несёт ~1400 строк кода, которые заменяются инфраструктурой**. Ключевые выводы:

1. **Storage** — Qdrant заменяет ChromaDB + BM25IndexManager целиком. Нативные sparse vectors (`Qdrant/bm25` + `language="russian"`), нативный RRF, нативный MMR. ~400 строк кода уходит.
2. **LLM сейчас** — Qwen3-8B GGUF запускается на V100 через существующий llama-server **прямо сейчас**, без Proxmox. Заменяет оба LLM (7B agent + 3B planner CPU).
3. **LLM будущее** — vLLM v0.15.1 после Proxmox (V100 → Linux VM). Даёт xgrammar (100% valid JSON), prefix caching, нативный Hermes tool calling. Требует AgentService rewrite.
4. **Coverage metric** — текущий `citation_coverage` (document count) бесполезен. Заменяется composite из 5 cosine-сигналов. Порог снижается с 0.80 до 0.65–0.70.
5. **Evaluation** — custom LLM-judge на Qwen3-8B + DeepEval как CI/CD runner. RAGAS — только как reference. Работает уже на llama-server.
6. **Async** — httpx.AsyncClient как промежуточный фикс OPEN-02. AsyncOpenAI + vLLM как финальное состояние после Proxmox.

---

## 1. Validated Assumptions

### Подтверждены ✅

| # | Гипотеза | Источник | Статус |
|---|----------|----------|--------|
| H1 | Qdrant поддерживает нативный sparse (BM25) + dense в одной коллекции | R01 | **✅ Подтверждено** (`named vectors`, prefetch+FusionQuery) |
| H2 | `Qdrant/bm25` с `language="russian"` корректно токенизирует русский | R01 | **✅ Подтверждено** (Snowball stemmer, не BM42) |
| H3 | Нативный RRF в Qdrant эквивалентен кастомному Python RRF | R01 | **✅ Подтверждено** (prefetch + FusionQuery, тот же алгоритм) |
| H4 | MMR реализован нативно в Qdrant | R01 | **✅ Подтверждено** (с v1.15.0, `rescore: mmr`) |
| H5 | vLLM даёт лучший structured output чем llama-server (GBNF) | R02 | **✅ Подтверждено** (xgrammar vs GBNF: надёжнее, быстрее) |
| H6 | V100 (SM7.0) поддерживается vLLM | R02, R03 | **✅ С оговорками** (только vLLM ≤ v0.15.1 + xformers; V1 engine SM≥8.0 → fallback на V0) |
| H7 | Qwen3-8B FP16 помещается в V100 32GB | R03 | **✅ Подтверждено** (~16.4 GB + KV-кэш 8192 токенов → ~19 GB) |
| H8 | Qwen3-8B заменяет оба LLM (7B agent + 3B planner) | R03 | **✅ Подтверждено** (Qwen3-8B ≥ Qwen2.5-14B по бенчмаркам) |
| H9 | RRF-скоры непригодны для coverage estimation | R04 | **✅ Подтверждено** (max RRF ≈ 0.0328, не cross-query сравнимы) |
| H10 | Пропущенный retrieval вреднее лишнего retrieval | R04 | **✅ Подтверждено** (Google ICLR 2025: 66.1% галлюцинаций с insufficient context vs 10.2% без контекста) |
| H11 | Qwen3-8B пригодна как LLM-судья для binary/3-point оценок | R05 | **✅ Подтверждено** (JudgeBoard 2025: judging > problem-solving для 8B) |
| H12 | RAGAS стабильна для production | R05 | **❌ Опровергнута** — 2 breaking changes за год, NaN-скоры на vLLM |
| H13 | Один тип вопросов достаточен для eval датасета | R05 | **❌ Опровергнута** — без принудительного распределения 95% factual → завышение метрик |
| H14 | Qwen3-8B GGUF запускается на V100 через llama-server | R03 + сессия | **✅ Подтверждено** (llama.cpp не имеет SM7.0 ограничений, GGUF F16/Q8_0/Q4_K_M) |

### Опровергнуты ❌

| # | Гипотеза | Источник | Реальность |
|---|----------|----------|-----------|
| H15 | `Qdrant/bm42` подходит для русского | R01 | **❌ English-only**. BM42 обучен на английском, для русского = случайные токены. |
| H16 | AWQ/GPTQ-Marlin/FP8 работают на V100 | R03 | **❌ SM7.0**. AWQ требует SM≥7.5 (V100 = 7.0), GPTQ-Marlin SM≥8.0, FP8 SM≥8.9. FP16 — единственный вариант. |
| H17 | vLLM v0.17.0+ работает на V100 | R02 | **❌ Сломан**. v0.17.0 убрал xformers-зависимость, V100 без xformers не работает. Пинить v0.15.1. |
| H18 | Документо-счётный `citation_coverage` измеряет достаточность контекста | R04 | **❌ Не измеряет**. Высокий count ≠ релевантность; один точный документ = coverage 1.0, но count ratio = 0.2. |
| H19 | Порог coverage 0.80 откалиброван | R04 | **❌ Слишком агрессивен**. Composite metric естественно сжимает оценки, 0.80 вызывает лишние поиски. |

### Частично подтверждены ⚠️

| # | Гипотеза | Нюанс |
|---|----------|-------|
| H20 | Один клиент AsyncOpenAI закрывает OPEN-02 | ⚠️ Закрывает, но только после Proxmox + vLLM. Промежуточный фикс: httpx.AsyncClient. |
| H21 | vLLM производительнее llama-server на V100 | ⚠️ Наоборот по throughput: llama-server ~80 tok/s vs vLLM ~40 tok/s. vLLM выигрывает по async, structured output, prefix caching. |
| H22 | `PoLL` (несколько малых моделей) лучше одного GPT-4 судьи | ⚠️ Да, но требует несколько разных моделей. У нас одна Qwen3-8B → MAJ (несколько инстансов) как компромисс. |

---

## 2. Key Decisions

| # | Решение | Варианты | Рекомендация | Обоснование | Статус |
|---|---------|----------|-------------|-------------|--------|
| **D1** | Vector store | A) ChromaDB + кастомный BM25 B) Qdrant | **B — Qdrant** | Нативный sparse, RRF, MMR. ~400 строк кода уходит. MCP уже поднят. | **DECIDED** |
| **D2** | Sparse model для русского | A) BM42 B) `Qdrant/bm25` + `language="russian"` | **B — `Qdrant/bm25` + Snowball** | BM42 English-only. Snowball stemmer корректно обрабатывает русские флексии. | **DECIDED** |
| **D3** | LLM сейчас (до Proxmox) | A) Остаться на Qwen2.5-7B B) Перейти на Qwen3-8B GGUF через llama-server | **B — Qwen3-8B GGUF** | Доступно прямо сейчас. Качество Qwen2.5-14B при 8B весе. Заменяет плanner. | **DECIDED** |
| **D4** | LLM будущее (после Proxmox) | A) Остаться на llama-server B) vLLM v0.15.1 C) Ollama | **B — vLLM v0.15.1** | xgrammar, prefix caching, нативный Hermes tool calling, AsyncOpenAI. Требует Linux → Proxmox. | **DECIDED** |
| **D5** | GGUF квантизация (до vLLM) | Q4_K_M / Q8_0 / F16 | **Q8_0 или F16** | V100 32GB с запасом. Q8_0 (~9 GB) — баланс. F16 (~16.4 GB) — максимум качества, 13 GB свободно. | **DECIDED** |
| **D6** | Planner LLM | A) CPU Ollama 3B (отдельный) B) Тот же V100 endpoint | **B — тот же endpoint** | CPU плanner = 5–15 tok/s узкое место. Qwen3-8B: 40–60 tok/s. Одна точка доступа (`LLM_BASE_URL`). | **DECIDED** |
| **D7** | Thinking mode Qwen3 | A) Включён B) Отключён | **B — Отключён** | `<think>...</think>` блоки ломают текущий ReAct-парсер. Экономит 250–1250 токенов. Промпт: `/no_think` (llama-server) / `extra_body={"enable_thinking": False}` (vLLM). | **DECIDED** |
| **D8** | Язык system prompt | A) Русский B) Английский + "respond in Russian" | **B — Английский** | 30–40% меньше токенов. Лучше instruction following для структурных задач (JSON, tool calling). Отдельная инструкция на выходной язык. | **DECIDED** |
| **D9** | Coverage metric | A) Document count ratio B) Composite 5-сигналов (cosine-based) | **B — Composite** | max_sim (0.25) + mean_top_k (0.20) + term_coverage (0.20) + doc_count_adequacy (0.15) + score_gap (0.15). Microseconds, no LLM calls. | **DECIDED** |
| **D10** | Coverage threshold | A) 0.80 (текущий) B) 0.65–0.70 | **B — 0.65–0.70** | Composite metric сжимает оценки. 0.80 → лишние поиски. Bias toward retrieval (false-negative хуже). | **DECIDED** |
| **D11** | Max refinements | A) 1 (текущий) B) 2 | **B — 2** | F1: 0.398 (1 iter) → 0.447 (3 iter). Одна дополнительная итерация даёт прирост. Plateau после 3. | **DECIDED** |
| **D12** | Eval framework | A) RAGAS B) DeepEval C) Custom judge | **C + B** — Custom judge (Qwen3-8B) + DeepEval как CI/CD runner | RAGAS нестабилен (breaking changes, NaN). DeepEval — pytest-интеграция, стабильный API. Custom judge — полный контроль промптов на русском. | **DECIDED** |
| **D13** | Async фикс OPEN-02 | A) asyncio.run_in_executor B) httpx.AsyncClient (промежуточный) C) AsyncOpenAI + vLLM (финальный) | **B сейчас → C после Proxmox** | httpx.AsyncClient — минимальный фикс без смены инфраструктуры. AsyncOpenAI = финальное состояние. | **DECIDED** |

---

## 3. Contradictions & Conflicts

### 3.1 vLLM vs llama-server: когда и зачем

**R02** рекомендует переход на vLLM как первый приоритет.
**R03** описывает Qwen3-8B в контексте vLLM launch команды.
**Реальность**: vLLM не работает на Windows, требует Proxmox + VFIO (R07). llama-server стабилен и достаточен.

**Резолюция**: llama-server остаётся основным до Proxmox. vLLM — опциональное улучшение после. Выгоды реальны (xgrammar 92→98%, prefix cache, Hermes tool calling), но умеренны для single user. Агентский рефакторинг (ReAct текст → Hermes) — отдельная большая задача, делается вместе с vLLM, не раньше.

### 3.2 Qwen3-8B: нужен ли Proxmox?

**R03** описывает Qwen3-8B только в контексте vLLM.
**Реальность**: llama.cpp поддерживает Qwen3-8B через GGUF. SM7.0 ограничения (FP8, AWQ) — только для vLLM/transformers.

**Резолюция**: Qwen3-8B GGUF **работает прямо сейчас** через llama-server. Нет зависимости от Proxmox для модели. Proxmox нужен только для vLLM-специфичных фич.

### 3.3 Async: httpx vs AsyncOpenAI

**R06** описывает httpx.AsyncClient как целевое решение.
**R02** указывает AsyncOpenAI как правильный async клиент для vLLM.

**Резолюция**: httpx.AsyncClient — промежуточный фикс, работает с llama-server сейчас. AsyncOpenAI — финальное состояние после vLLM. Оба правильны на своём этапе. Архитектура `LlamaServerClient` уже абстрагирует детали — смена клиента не затрагивает AgentService.

### 3.4 Max refinements: 1 vs 2–3

**Текущий код**: `max_refinements = 1` (агрессивный лимит).
**R04**: F1 растёт до 3 итераций, но plateau наступает после первой дополнительной.

**Резолюция**: увеличить до 2. Третья итерация даёт минимальный прирост при удвоении latency. Coverage threshold снижение (0.80 → 0.65) само по себе уменьшит число ненужных refinements.

### 3.5 Eval: нужен ли vLLM для evaluate_agent.py?

**R05** описывает eval pipeline с `VLLM_URL` как эндпоинтом для LLM-judge.
**Реальность**: llama-server — OpenAI-compatible API. Eval скрипт не привязан к vLLM.

**Резолюция**: eval работает на llama-server **сейчас**. Достаточно сменить `VLLM_URL` на llama-server URL и модель на имя GGUF файла. Единственная потеря — guided decoding (100% valid JSON), но `temperature=0` + try/except достаточно.

---

## 4. Remaining Unknowns

| # | Вопрос | Влияние | Как снять | Приоритет |
|---|--------|---------|-----------|-----------|
| U1 | vLLM v0.15.1 + Qwen3 совместимость | Qwen3 вышел апрель 2025, vLLM v0.15.1 — раньше. Может не поддерживаться. | Тест при настройке vLLM после Proxmox. Запасной вариант: ждать vLLM v0.16+ с xformers backport. | P1 (после Proxmox) |
| U2 | Hermes tool calling vs текущий ReAct regex | Полный переход требует AgentService rewrite. Объём и риски неизвестны. | Оценить scope при начале vLLM трека. Возможно поэтапно: сначала vLLM с text ReAct, потом Hermes. | P2 |
| U3 | Реальный прирост качества Qwen3-8B vs Qwen2.5-7B на нашем домене | Бенчмарки общие. Качество на русских Telegram-новостях — неизвестно. | Eval с LLM-judge после датасета из Qdrant. Baseline vs Qwen3-8B. | P1 |
| U4 | Calibration coverage threshold: 0.65 достаточно? | R04 рекомендует 0.65–0.70 как стартовую точку, не финальную. Требует калибровки на 30–50 размеченных примерах. | Ручная разметка примеров после eval датасета. Найти threshold с минимальной взвешенной ошибкой (false-negative ×3). | P1 |
| U5 | Производительность Qdrant vs ChromaDB на нашем корпусе | Теоретически быстрее. Практически — на Windows Docker с HDD могут быть сюрпризы. | Замерить latency после миграции. Named volumes — mandatory. | P2 |
| U6 | Cohen's κ для Qwen3-8B judge на наших данных | R05 требует κ ≥ 0.6 против эксперта. Без проверки судья ненадёжен. | Разметить 30–50 примеров вручную, вычислить κ после eval системы. | P1 |

---

## 5. Architecture Decisions (ADR)

### ADR-001: Storage Migration (ChromaDB → Qdrant)

```
Текущее: ChromaDB (dense) + custom BM25IndexManager (disk-based pickle)
Целевое: Qdrant single collection с named vectors

Схема коллекции:
  dense_vector:   multilingual-e5-large (1024-dim, cosine)
  sparse_vector:  Qdrant/bm25 (language="russian", Snowball)

Hybrid search:
  prefetch=[
    Prefetch(using="dense_vector", limit=20),
    Prefetch(using="sparse_vector", limit=20)
  ]
  query=FusionQuery(fusion=Fusion.RRF)
  limit=10

MMR (при включении):
  rescore=QueryRescore(rescore=RescoringQuery(mmr=MmrQuery(lambda_mult=0.5)), limit=5)

Payload (per point):
  text: str                # полный текст чанка
  source: str              # имя Telegram-канала
  channel_id: int          # id канала
  message_id: int          # id сообщения
  date: str                # ISO 8601
  author: str | null       # автор если есть

Windows Docker: ТОЛЬКО named volumes (bind mounts → silent data corruption)
```

**Что уходит**: `src/adapters/chroma.py` (~200 строк), `src/adapters/bm25.py` (~200 строк), `bm25-index/` директория.
**Что остаётся**: `HybridRetriever` — упрощается до 1 вызова `client.query_points()`. Payload-фильтры сохраняются.

### ADR-002: LLM Stack (двухэтапный)

```
Этап 1 — Сейчас (llama-server, Windows):
  Модель:     Qwen3-8B-Instruct (GGUF, Q8_0 или F16)
  Сервер:     llama-server на хосте (V100 SXM2 32GB, CUDA 12.4)
  Клиент:     LlamaServerClient (httpx.AsyncClient → заменяет requests.Session)
  Thinking:   ОТКЛЮЧЁН (/no_think в system prompt)
  Planner:    тот же endpoint (не отдельный CPU процесс)
  Performance: ~60–80 tok/s (F16, V100, batch=1)

Этап 2 — После Proxmox (vLLM, Linux VM):
  Модель:     Qwen/Qwen3-8B (safetensors, FP16)
  Сервер:     vLLM v0.15.1 (pinned; v0.17.0+ сломал xformers → V100 не работает)
  Запуск:
    vllm serve Qwen/Qwen3-8B \
      --dtype half \
      --enforce-eager \           # обязательно: V100 нет FlashAttention2 (SM<8.0)
      --max-model-len 8192 \
      --gpu-memory-utilization 0.92 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes
  Клиент:     AsyncOpenAI(base_url=LLM_BASE_URL, api_key="EMPTY")
  Thinking:   ОТКЛЮЧЁН (extra_body={"enable_thinking": False})
  Агент:      text ReAct regex → нативный Hermes tool calling (требует AgentService rewrite)
  Performance: ~40–60 tok/s (FP16, enforce-eager, V100)

V100 SM7.0 ограничения (vLLM):
  ❌ AWQ (требует SM≥7.5)
  ❌ GPTQ-Marlin (SM≥8.0)
  ❌ FP8 (SM≥8.9)
  ❌ FlashAttention2 (SM≥8.0) → --enforce-eager обязателен
  ❌ vLLM V1 engine (SM≥8.0) → V0 legacy
  ✅ FP16 + xformers + vLLM V0 (только ≤ v0.15.1)
```

### ADR-003: Coverage Metric

```python
def calculate_coverage(
    query: str,
    retrieved_docs: list[dict],  # каждый doc содержит 'cosine_sim' и 'text'
    relevance_threshold: float = 0.55,
    target_k: int = 5
) -> float:
    """
    Composite из 5 сигналов. Вычисляется в compose_context.
    Не использует RRF-скоры — они для ранжирования, не для coverage.
    Cosine similarity запрашивается отдельно: with_vectors=True в Qdrant запросе.
    """
    if not retrieved_docs:
        return 0.0
    scores = sorted([d['cosine_sim'] for d in retrieved_docs], reverse=True)
    top_k = scores[:target_k]

    max_sim = scores[0]
    mean_top_k = sum(top_k) / len(top_k)
    relevant_count = sum(1 for s in scores if s >= relevance_threshold)
    count_adequacy = min(1.0, relevant_count / target_k)
    gap = 1.0 - (scores[0] - scores[min(target_k-1, len(scores)-1)]) / scores[0] if scores[0] > 0 else 0
    term_cov = _query_term_overlap(query, retrieved_docs)  # доля ключевых слов из query в тексте

    return min(1.0,
        0.25 * max_sim +
        0.20 * mean_top_k +
        0.20 * term_cov +
        0.15 * count_adequacy +
        0.15 * gap +
        0.05 * (relevant_count / len(scores))
    )

# Параметры агента (agent_service.py):
COVERAGE_THRESHOLD = 0.65   # было 0.80
MAX_REFINEMENTS = 2          # было 1

# Cosine thresholds (ориентир):
# 0.85+ near-paraphrase, 0.75-0.85 strong, 0.60-0.75 moderate,
# 0.45-0.60 tangential, <0.45 irrelevant
# <0.30 → abort и вернуть "insufficient information" вместо галлюцинации
```

**Что меняется в коде**: `compose_context()` получает `query` как параметр; Qdrant запрос добавляет `with_vectors=True`; cosine считается как `dot(query_vec, doc_vec)` (L2-нормированные). Алгоритм coverage полностью заменяется.

### ADR-004: Eval Framework

```
Архитектура:
  Custom LLM-judge (Qwen3-8B) → обёртка в DeepEval BaseMetric → pytest CI/CD

Метрики судьи (промпты на русском):
  faithfulness:      binary QAG (разбить ответ на claims, каждый проверить по контексту)
                     threshold: score >= 0.80, verdict: pass/fail
  relevance:         шкала 1-5 (отвечает ли на вопрос)
                     threshold: score >= 3, verdict: pass/fail
  completeness:      шкала 1-5 (покрыты ли все аспекты)
                     threshold: score >= 3, verdict: pass/fail
  citation_accuracy: binary per-citation (источник существует × claim подтверждён)
                     threshold: score >= 0.90, verdict: pass/fail

Retrieval метрики (без LLM):
  recall@5, precision@5, MRR, NDCG@5 (по expected_document_ids из датасета)

System метрики:
  latency P50/P95/P99, answer_rate, negative_rejection_rate

CI/CD:
  PR smoke: 50 примеров, ~15 мин. Quality gate: faithfulness ≥ 0.80, recall@5 ≥ 0.70
  Nightly: 200 примеров, ~60 мин. Alert при регрессии > 5%

Судья — параметры:
  temperature=0, seed=42, max_tokens=1024
  thinking mode ОТКЛЮЧЁН (evaluation = структурная задача, не требует reasoning)

Валидация судьи:
  Разметить вручную 30–50 примеров → Cohen's κ ≥ 0.6 (substantial agreement)

Eval датасет (generate_eval_dataset.py из Qdrant):
  Целевой размер: 200 примеров (margin of error ±5.5% при 95% CI)
  Распределение типов (ОБЯЗАТЕЛЬНО):
    factual 35%, temporal 20%, aggregation 20%, negative 15%, comparative 10%
  Critique-фильтр: groundedness + relevance + standalone ≥ 3/5 → ~50% отсеивается
  Генерировать с запасом: 400 точек → ~200 итоговых
```

### ADR-005: Async Architecture

```
Этап 1 — Сейчас:
  LlamaServerClient: requests.Session → httpx.AsyncClient
  Изменения: минимальные, только клиент внутри адаптера
  Результат: закрывает OPEN-02 без изменений AgentService

Этап 2 — После vLLM:
  Клиент: AsyncOpenAI(base_url=..., api_key="EMPTY")
  State isolation: contextvars.ContextVar для per-request state
  ToolRunner: async-нативные инструменты (async def tools)
  SSE: обработка client disconnect через asyncio.CancelledError
  Init: FastAPI lifespan для инициализации Qdrant/reranker при старте

Проблема сейчас: AgentService._current_step — атрибут класса (не per-request)
Фикс: перевести в contextvars.ContextVar ИЛИ в AgentState (R06 design)
```

---

## 6. Implementation Phases

```
Phase 0 — Сейчас (нет блокеров):
  - Qwen3-8B GGUF на llama-server (просто смена модели и /no_think в prompt)
  - httpx.AsyncClient в LlamaServerClient (закрывает OPEN-02)
  - max_refinements = 2, coverage_threshold = 0.65 (одна строка каждое)

Phase 1 — После Qdrant setup (зависит от docker-compose + named volumes):
  - Qdrant коллекция: dense + sparse (Qdrant/bm25, language="russian")
  - Миграция данных: dense vectors → напрямую, sparse → регенерировать
  - HybridRetriever: заменить на client.query_points() с prefetch+FusionQuery
  - coverage: добавить with_vectors=True, composite metric

Phase 2 — После Phase 1 (зависит от Qdrant):
  - generate_eval_dataset.py: 400 точек → ~200 отфильтрованных примеров
  - evaluate_agent.py расширенный: retrieval metrics + LLM-judge
  - DeepEval BaseMetric обёртки для CI/CD
  - Baseline measurement: текущая система vs Qwen3-8B

Phase 3 — После Proxmox + VFIO (R07):
  - V100 → Linux VM (VFIO passthrough)
  - vLLM v0.15.1: проверить совместимость с Qwen3 (риск U1)
  - AsyncOpenAI клиент
  - Если совместимость ОК: AgentService rewrite → Hermes tool calling
  - Если нет: остаться на llama-server GGUF, ждать vLLM fix
```

---

## 7. Risk Matrix

| # | Риск | Вероятность | Влияние | Митигация |
|---|------|-------------|---------|-----------|
| R1 | vLLM v0.15.1 не поддерживает Qwen3 | Средняя (Qwen3 вышел после v0.15.1) | Среднее — vLLM трек откладывается | llama-server + Qwen3 GGUF как бессрочный fallback. Мониторить xformers backport в vLLM v0.16+. |
| R2 | Hermes tool calling реwrite сложнее ожидаемого | Средняя | Высокое — AgentService рефакторинг | Поэтапно: сначала vLLM с text ReAct, Hermes — отдельным шагом. Не блокирует vLLM миграцию. |
| R3 | Cohen's κ < 0.6 для Qwen3-8B judge | Низкая–Средняя | Среднее — eval метрики ненадёжны | MAJ (multi-agent judging с несколькими инстансами), более детальные рубрики, упрощение шкал до binary. |
| R4 | Qdrant на Windows Docker: производительность | Низкая (named volumes решают corruption) | Низкое–Среднее | Named volumes обязательны. Мониторить latency после миграции. |
| R5 | Coverage threshold 0.65 слишком низкий для нашего корпуса | Средняя — нужна калибровка | Среднее — лишние retrieval итерации | Калибровать на 30–50 размеченных примерах. Threshold — отправная точка, не константа. |
| R6 | Qwen3 thinking mode прорывается через /no_think | Низкая | Высокое — ломает ReAct парсер | Добавить явную фильтрацию `<think>...</think>` в LlamaServerClient как safeguard. |
| R7 | V100 driver конфликт (RTX 5060 Ti требует 590+, V100 max 581) | Высокая (уже есть) | Среднее — блокирует vLLM | Proxmox VFIO изолирует драйверы. До Proxmox — llama-server на хосте. Текущее состояние стабильно. |

---

## 8. Reference Parameters

### Qdrant hybrid search

| Параметр | Значение | Источник |
|----------|----------|---------|
| Dense model | multilingual-e5-large | текущий стек |
| Dense dim | 1024 | E5-large spec |
| Distance metric | Cosine | R01 |
| Sparse model | Qdrant/bm25 | R01 |
| BM25 language | russian (Snowball) | R01 |
| RRF k constant | 60 (default) | R01 |
| Hybrid prefetch limit | 20 per retriever | R01 |
| Final limit | 10 | текущий стек |
| MMR lambda | 0.5 (diversity/relevance баланс) | R01 |
| Docker volumes | Named (НЕ bind mounts) | R01 |

### LLM (Этап 1 — llama-server)

| Параметр | Значение | Источник |
|----------|----------|---------|
| Модель | Qwen3-8B-Instruct GGUF | R03, сессия |
| Квантизация | Q8_0 (баланс) или F16 (max) | R03 |
| VRAM (Q8_0) | ~9 GB | R03 |
| VRAM (F16) | ~16.4 GB | R03 |
| Context length | 8192 токенов | R02 |
| Thinking mode | ОТКЛЮЧЁН (/no_think) | R03 |
| Tool calling | text ReAct (текущий, без изменений) | R03 |
| Performance | ~60–80 tok/s (V100, batch=1) | R03 |

### LLM (Этап 2 — vLLM, после Proxmox)

| Параметр | Значение | Источник |
|----------|----------|---------|
| Модель | Qwen/Qwen3-8B (HF safetensors) | R03 |
| Precision | FP16 (--dtype half) | R03 |
| Engine | V0 legacy (SM7.0 < 8.0) | R02, R03 |
| enforce-eager | Обязателен (нет FlashAttention2) | R03 |
| max-model-len | 8192 | R02 |
| gpu-memory-utilization | 0.92 | R03 |
| tool-call-parser | hermes | R03 |
| Performance | ~40–60 tok/s | R03 |

### Coverage & Retrieval

| Параметр | Значение | Источник |
|----------|----------|---------|
| Coverage threshold | 0.65–0.70 | R04 |
| Max refinements | 2 | R04 |
| Relevance threshold (cosine) | 0.55 | R04 |
| Abort threshold (cosine) | < 0.30 | R04 |
| Term overlap weight | 0.20 | R04 |
| Max sim weight | 0.25 | R04 |
| Mean top-k weight | 0.20 | R04 |

---

## 9. Open Tracks

| # | Трек | Приоритет | Зависимость |
|---|------|-----------|-------------|
| T1 | Qdrant migration + данные | P0 | docker-compose + named volumes |
| T2 | Qwen3-8B GGUF на llama-server | P0 | скачать GGUF, обновить prompt |
| T3 | httpx.AsyncClient в LlamaServerClient | P0 | нет зависимостей |
| T4 | coverage metric composite + threshold 0.65 | P1 | T1 (нужен Qdrant for with_vectors) |
| T5 | generate_eval_dataset.py + 200 примеров | P1 | T1 (нужен Qdrant) |
| T6 | evaluate_agent.py + LLM-judge | P1 | T5 |
| T7 | Cohen's κ калибровка судьи | P1 | T6 |
| T8 | Coverage threshold калибровка | P2 | T5, T4 |
| T9 | DeepEval CI/CD интеграция | P2 | T6 |
| R07 | Proxmox + VFIO research | P1 | когда будет время |
| vLLM | vLLM v0.15.1 + Qwen3 migration | P2 | R07, проверка U1 |

---

## 10. Итого: готовность к разработке

**Go** — с чёткими этапами:

1. ✅ Storage решение ясно: Qdrant нативный hybrid (R01). ~400 строк кода уходит.
2. ✅ LLM путь ясен: Qwen3-8B GGUF сейчас, vLLM FP16 после Proxmox (R02, R03).
3. ✅ Coverage metric ясна: composite 5-сигналов, threshold 0.65, max 2 refinements (R04).
4. ✅ Eval стратегия ясна: custom judge + DeepEval, 200 примеров из Qdrant (R05).
5. ✅ Async путь ясен: httpx сейчас, AsyncOpenAI после vLLM (R06).
6. ⚠️ vLLM v0.15.1 + Qwen3 совместимость — **не проверена** (риск R1). llama-server — надёжный fallback.
7. ⚠️ Hermes tool calling rewrite — масштаб неизвестен (риск R2). Не блокирует остальное.
8. ⚠️ Coverage threshold 0.65 — стартовая точка, требует калибровки (риск R5).

**Следующий шаг**: architecture design → модульная структура компонентов → спецификации на изменения.
