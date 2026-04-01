# SPEC-RAG-21: NLI Citation Faithfulness

> Статус: DRAFT v2 (post-Codex review)
> Автор: Claude + Human
> Дата: 2026-04-01
> Research base: R19 (NLI citation faithfulness)
> Зависимости: evaluate_agent.py, gpu_server.py, Langfuse API
> Judge prompt version: v1

---

## Цель

Добавить метрику **faithfulness** (grounding) — проверку что ответ агента опирается на найденные документы, а не на parametric knowledge LLM. Eval-only, не runtime.

**Зачем:**
- Factual correctness (текущая метрика) отвечает на "правильный ли ответ?"
- Faithfulness отвечает на "опирается ли ответ на документы?"
- Это ортогональные метрики. High factual + low faithfulness = LLM галлюцинирует правильный ответ "из головы"

**Целевые метрики:** baseline faithfulness ≥ 0.85 на retrieval_evidence вопросах (17 из 36 Qs).

---

## Архитектура

```
┌──────────────────────────────────────────────────────────────┐
│                      EVAL PIPELINE                            │
│                                                              │
│  1. evaluate_agent.py  (без изменений)                       │
│     └─ 36 Qs → eval_results.json (уже содержит enriched     │
│        citations с полными текстами из Qdrant)               │
│                                                              │
│  2. evaluate_agent.py --export-offline-judge  (расширить)     │
│     └─ eval_results.json → judge_artifact.md                 │
│        (расширенный: + tool outputs для analytics)           │
│                                                              │
│  3a. Claude DECOMPOSITION (новый чат, промпт v1)             │
│      └─ Читает артефакт → per-question claims[]              │
│      └─ Сохраняет → claims_YYYYMMDD.json                    │
│                                                              │
│  3b. Claude JUDGE (тот же или отдельный чат, промпт v1)      │
│      └─ Читает артефакт → per-question:                      │
│         • factual (0 / 0.5 / 1.0)                           │
│         • useful (0 / 1 / 2)                                 │
│         • reasoning                                          │
│      └─ Сохраняет → judge_verdicts_YYYYMMDD.json            │
│                                                              │
│  4. scripts/run_nli.py                          [NEW]        │
│     └─ claims (JSON) × documents (JSON) → XLM-RoBERTa NLI   │
│        • faithfulness score per question                      │
│     └─ Сохраняет → nli_scores_YYYYMMDD.json                 │
│                                                              │
│  5. scripts/merge_eval_report.py                [NEW]        │
│     └─ eval_results + judge_verdicts + claims + nli_scores   │
│        → final_eval_report.json + final_eval_report.md       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Ключевые решения (post-review)

1. **Нет отдельного build_judge_artifact.py** — evaluate_agent.py уже enriches citations из Qdrant (offline_judge_packet). Расширяем существующий `--export-offline-judge`.
2. **Decomposition и Judge — два раздельных прохода** — ошибка в judging не влияет на качество claims, и наоборот.
3. **JSON as source of truth** — run_nli.py читает JSON (eval_results + claims), не парсит markdown. Markdown артефакт — для человека.
4. **Faithfulness считается только на retrieval_evidence** — analytics (13 Qs) и navigation/refusal (5 Qs) = N/A.

---

## Шаг 1: Eval прогон

Без изменений. `evaluate_agent.py` прогоняет 36 Qs, сохраняет в `results/raw/`. Langfuse записывает traces.

**Важно**: evaluate_agent.py уже обогащает citations полными текстами из Qdrant (строки 704-732, offline_judge_packet). Это source of truth для documents.

---

## Шаг 2: Расширенный артефакт

Расширяем `build_offline_judge_markdown()` в evaluate_agent.py. Добавляем:
- Полные тексты tool output для analytics вопросов
- Metadata per question для машинной обработки

**Markdown артефакт** — для Claude judge (человекочитаемый):

```markdown
## Q04: Что нового у DeepSeek?

**eval_mode**: retrieval_evidence
**category**: factual
**expected_answer**: DeepSeek выпустил V3.2, Speciale, V3.1...
**required_claims**:
- DeepSeek выпустил несколько версий V3

### Agent Response
**answer**: DeepSeek в последние месяцы активно развивает...
**tools_invoked**: query_plan → search → rerank → compose_context → final_answer
**latency**: 28.3s

### Cited Documents
[1] id: abc123 | channel: @ai_newz | date: 2026-02-15
> DeepSeek выпустил V3.2 с улучшенной производительностью.
> Модель показывает результаты на уровне GPT-4o...

[2] id: def456 | channel: @gonzo_ml | date: 2026-01-20
> Линейка DeepSeek V3: базовая V3, затем V3.1 с OCR...
```

**JSON (eval_results)** — для run_nli.py (машинночитаемый). Уже содержит `offline_judge_packet.cited_documents[]` с полными текстами.

### Для analytics вопросов (eval_mode=analytics)

Секция "Tool Output" вместо "Cited Documents":

```markdown
### Tool Output (entity_tracker, co_occurrence)
> summary: Чаще всего с NVIDIA упоминаются: OpenAI (125), Google (101)...
> data: [{entity: OpenAI, count: 125}, ...]
```

Analytics вопросы получают faithfulness=N/A (нет documents для NLI).

### Для refusal вопросов (expected_refusal=true)

Claims=[], faithfulness=N/A.

---

## Шаг 3a: Claude Decomposition (отдельный проход)

### Процесс

1. Открыть **новый чат** с Claude (чистый контекст)
2. Вставить **промпт decomposition** (ниже)
3. Вставить артефакт (батчами по 10-12 вопросов, каждый батч с полным промптом)
4. Claude возвращает JSON с claims per question
5. Сохранить в `results/raw/claims_YYYYMMDD.json`

### Промпт decomposition (v1)

Хранится в `datasets/prompts/decomposition_v1.md`.

```
Ты — эксперт по разбиению текста на атомарные утверждения.
Твоя задача — разбить каждый ответ RAG-агента на независимые claims.

ПРАВИЛА:

1. Каждый claim — ровно один проверяемый факт.
2. Каждый claim — полное утвердительное предложение на русском языке,
   понятное без контекста. Раскрой все местоимения.
3. Классификация:
   - "verifiable" — утверждение требует подтверждения документом
   - "common_knowledge" — общеизвестный факт, не требует подтверждения
   - "meta" — структурное/связующее высказывание ("Согласно данным...",
     "Далее перечислены...")

4. НЕ добавляй claims которых нет в ответе.
5. Если ответ пустой — верни пустой список claims.
6. Если ответ = refusal ("не найдено в базе") — верни пустой список claims.
7. Числовые claims: "125 упоминаний" — это один claim, не разбивай на
   "есть упоминания" и "их количество 125".

ФОРМАТ ОТВЕТА (JSON):

{
  "questions": [
    {
      "id": "golden_q04",
      "claims": [
        {"text": "DeepSeek выпустил модель V3.2", "type": "verifiable"},
        {"text": "V3.2 показывает результаты на уровне GPT-4o", "type": "verifiable"},
        {"text": "DeepSeek — китайская компания", "type": "common_knowledge"}
      ]
    },
    ...
  ]
}

Далее тебе будет предоставлен артефакт с вопросами. Разбей каждый ответ на claims.
```

---

## Шаг 3b: Claude Judge (отдельный проход)

### Процесс

1. Открыть **новый чат** (или продолжить, но с **отдельным промптом**)
2. Вставить **промпт judge** (ниже)
3. Вставить артефакт (батчами по 10-12 вопросов, каждый батч с полным промптом)
4. Claude возвращает JSON с оценками
5. Сохранить в `results/raw/judge_verdicts_YYYYMMDD.json`

### Промпт judge (v1)

Хранится в `datasets/prompts/judge_v1.md`.

```
Ты — независимый эксперт-оценщик RAG-системы. Твоя задача — объективно
оценить качество ответов поисковой системы по AI/ML новостям.

ПРАВИЛА ОЦЕНКИ:

## Factual correctness (0 / 0.5 / 1.0)
Сравни ответ агента с expected_answer и required_claims.

- 1.0 — ответ содержит ВСЕ ключевые факты из expected. Допускаются
  дополнительные верные факты сверх expected.
- 0.5 — ответ содержит ЧАСТЬ ключевых фактов (больше половины), но
  пропускает значимые. ИЛИ содержит все факты но с неточностями.
- 0.0 — ответ пустой, отказ при answerable=true, или содержит менее
  половины ключевых фактов, или фактически неверен.

## Useful (0 / 1 / 2)
Был бы ответ полезен реальному пользователю?

- 2 — ответ полезен, информативен, хорошо структурирован.
- 1 — ответ частично полезен. Есть верная информация но неполная.
- 0 — ответ бесполезен. Пустой, нерелевантный, или полностью неверный.

ВАЖНЫЕ ПРИНЦИПЫ:

1. НЕ ПЕРЕСУЖИВАЙ. Если ответ содержит 5 из 6 фактов expected —
   это 0.5 factual, НЕ 0.0. Partial credit обязателен.

2. НЕ ПРЕУКРАШИВАЙ. Если ответ пустой или явно мимо — это 0.0,
   без "но попытка была хорошая".

3. ДОПОЛНИТЕЛЬНЫЕ ФАКТЫ — НЕ ошибка. Если ответ содержит всё из
   expected + ещё верные факты сверху — это 1.0, не снижай.

4. REFUSAL при answerable=true — это 0.0 factual, 0 useful.
   REFUSAL при expected_refusal=true — это 1.0 factual, 2 useful.

5. Analytics вопросы (eval_mode=analytics): ответ должен опираться
   на tool output, не на обычный search. Если agent правильно
   использовал entity_tracker/hot_topics/channel_expertise и
   ответ корректен — это 1.0.

6. Navigation вопросы (eval_mode=navigation): ответ должен содержать
   список каналов. Если list_channels вызван и ответ корректен — 1.0.

7. Цифры: допускай разумную погрешность. "125 упоминаний" vs
   "около 120" — не снижай. "125" vs "50" — снижай.

8. Оценивай ОТВЕТ, не pipeline. Если agent дошёл до ответа странным
   путём но ответ верный — оценивай ответ.

ФОРМАТ ОТВЕТА (JSON):

{
  "verdicts": [
    {
      "id": "golden_q01",
      "factual": 1.0,
      "useful": 2,
      "reasoning": "Краткое обоснование в 1-2 предложения"
    },
    ...
  ]
}

Далее тебе будет предоставлен артефакт с вопросами. Оцени каждый.
```

### Калибровочные примеры (вставляются после промпта)

```
Вот примеры правильной оценки для калибровки:

ПРИМЕР 1 — полный ответ, retrieval_evidence (1.0):
Query: "Что упоминается вместе с NVIDIA?"
Expected: "OpenAI, Google, Anthropic, Gemini"
Answer: "OpenAI (125), Google (101), Microsoft (50), Anthropic (47), DeepSeek (41)"
→ factual: 1.0 (все expected entities + дополнительные верные)
→ useful: 2

ПРИМЕР 2 — частичный ответ, retrieval_evidence (0.5):
Query: "Какие open-source модели для генерации изображений?"
Expected: "Kandinsky 5.0, FLUX.2, HunyuanImage"
Answer: "Kandinsky 5.0 и HunyuanImage — ведущие модели"
→ factual: 0.5 (FLUX.2 пропущен, но 2 из 3 есть)
→ useful: 2

ПРИМЕР 3 — пустой ответ (0.0):
Query: "Что обычно упоминается вместе с NVIDIA?"
Answer: ""
→ factual: 0.0 (пустой ответ)
→ useful: 0

ПРИМЕР 4 — analytics вопрос (1.0):
Query: "Какие горячие темы были на неделе 2026-W11?"
eval_mode: analytics
Tool output: hot_topics вернул 5 тем с hot_score
Answer: "На неделе 2026-W11 обсуждались: HRM/KV/MLP (21 пост), Eval (18)..."
→ factual: 1.0 (корректно использовал hot_topics, данные совпадают)
→ useful: 2

ПРИМЕР 5 — правильный refusal (1.0):
Query: "Что писали каналы про квантовые компьютеры в 2024?"
expected_refusal: true
Answer: "В базе данных отсутствуют посты за 2024 год..."
→ factual: 1.0 (правильный отказ)
→ useful: 2

ПРИМЕР 6 — navigation (1.0):
Query: "Какие каналы есть в базе?"
eval_mode: navigation
Answer: "В базе 36 каналов: @ai_newz, @gonzo_ml, @seeallochnaya..."
→ factual: 1.0 (list_channels вызван, список корректен)
→ useful: 2

ПРИМЕР 7 — ответ с числовыми данными (0.5):
Query: "Какие самые обсуждаемые AI-компании?"
Expected: "NVIDIA (450 упоминаний), OpenAI (380), Google (320)"
Answer: "NVIDIA лидирует с 450 упоминаниями, OpenAI на втором месте с 380"
→ factual: 0.5 (Google пропущен, но NVIDIA и OpenAI верно)
→ useful: 2
```

---

## Шаг 4: XLM-RoBERTa NLI

### Модель

**Primary**: `joeddav/xlm-roberta-large-xnli` (560M params, 1.12 GB FP16)
- 83.5% accuracy на XNLI Russian
- 3-way classification: entailment / neutral / contradiction
- Max input: 512 tokens (premise + hypothesis)

**Fallback** (если VRAM тесно): `cointegrated/rubert-base-cased-nli-threeway` (180M, 0.36 GB)

### Интеграция в gpu_server.py

Новый endpoint `/nli`. **Lazy loading** — модель загружается при первом запросе, не при старте. Флаг `--with-nli` для явной предзагрузки.

```python
POST /nli
{
  "pairs": [
    {"premise": "Текст документа...", "hypothesis": "OpenAI упоминается с NVIDIA в 125 случаях"},
    ...
  ]
}

Response:
{
  "results": [
    {"label": "entailment", "scores": {"entailment": 0.92, "neutral": 0.06, "contradiction": 0.02}},
    ...
  ]
}

Errors:
  503 — модель не загружена (lazy load failed / OOM)
  422 — невалидный input
  504 — timeout (30s)
```

### Document chunking (512-token limit)

XLM-RoBERTa лимит 512 токенов на пару (premise + hypothesis). Русские Telegram-посты могут быть длиннее. Стратегия из R19 (Risk 4):

1. Если document + claim ≤ 512 tokens → подать как есть
2. Если document > ~400 tokens → разбить document на чанки по 400 tokens с overlap 50
3. Проверить каждый чанк отдельно
4. Взять **max entailment score** across all chunks

Tokenization через `AutoTokenizer` из той же модели (не приблизительный count).

### Скрипт `scripts/run_nli.py`

**Вход**: 
- `results/raw/eval_results_YYYYMMDD.json` — содержит cited documents с текстами (JSON)
- `results/raw/claims_YYYYMMDD.json` — claims от Claude decomposition (JSON)

**Выход**: `results/raw/nli_scores_YYYYMMDD.json`

**Важно**: run_nli.py читает ТОЛЬКО JSON, никогда markdown.

Логика:
1. Для каждого вопроса:
   - Если eval_mode != retrieval_evidence → faithfulness = null, skip
   - Берёт claims с type="verifiable" из claims JSON
   - Если 0 verifiable claims → faithfulness = null
   - Если 0 cited documents (agent ответил без citations) → faithfulness = 0.0 (all claims unsupported)
   - Берё�� cited documents из eval_results JSON (offline_judge_packet.citations — массив объектов с полями id, text, channel, date)
   - Chunking документов если > 400 tokens
   - Формирует pairs: каждый claim × каждый document chunk
   - Отправляет batch в gpu_server.py /nli
2. Per claim: best entailment score across all document chunks
   - entailment > 0.5 → supported (1.0)
   - contradiction > 0.5 → contradicted (0.0)
   - else → neutral (0.5 lenient / 0.0 strict)
3. Per question: `faithfulness = mean(claim_scores)` по verifiable claims only

### Метрики per question:

```json
{
  "query_id": "golden_q04",
  "eval_mode": "retrieval_evidence",
  "faithfulness": 0.875,
  "faithfulness_strict": 0.75,
  "claims_total": 8,
  "claims_verifiable": 7,
  "claims_supported": 6,
  "claims_contradicted": 0,
  "claims_neutral": 1,
  "claims_common_knowledge": 1,
  "per_claim": [
    {
      "text": "DeepSeek выпустил модель V3.2",
      "type": "verifiable",
      "nli_label": "entailment",
      "nli_score": 0.94,
      "best_document_id": "abc123",
      "best_chunk_idx": 0
    }
  ],
  "contradictions": []
}
```

### NLI threshold calibration

Начальный порог: entailment > 0.5, contradiction > 0.5.

**После первого прогона**: вручную разметить 30-50 пар (claim, document, expected_label) из реальных результатов. Подобрать оптимальный порог по F1. Зафиксировать в decision-log.

### Citation Precision (ALCE-style)

Привязка к конкретным цитатам [1], [2] в ответе агента:

```
Citation Precision = 1 - (irrelevant_citations / total_citations)
```

Citation считается irrelevant если **ни один verifiable claim** не получает entailment от этого конкретного документа. Метрика зависит от полноты decomposition — при citation_precision < 0.5 проверить качество claims вручную.

**Примечание**: Citation Recall из R19 (ALCE-style) в текущей версии тавтологична с faithfulness. Убираем как отдельную метрику. Faithfulness = главная grounding метрика.

---

## Шаг 5: Merge финального отчёта

**Скрипт**: `scripts/merge_eval_report.py`

Объединяет:
- `results/raw/eval_results_YYYYMMDD.json` — agent performance
- `results/raw/judge_verdicts_YYYYMMDD.json` — Claude judge
- `results/raw/claims_YYYYMMDD.json` — Claude decomposition
- `results/raw/nli_scores_YYYYMMDD.json` — XLM-RoBERTa faithfulness

**Выход**:
- `results/reports/final_eval_YYYYMMDD.json` — машинночитаемый
- `results/reports/final_eval_YYYYMMDD.md` — человекочитаемый

### Aggregate метрики:

| Метрика | Источник | Scope | Описание |
|---------|----------|-------|----------|
| **Factual** | Claude judge | All 36 Qs | Корректность vs expected answer |
| **Useful** | Claude judge | All 36 Qs | Полезность для пользователя |
| **KTA** | evaluate_agent.py | All 36 Qs | Key Tool Accuracy |
| **Faithfulness** | XLM-RoBERTa NLI | retrieval_evidence only (~17 Qs) | Grounding: claims supported by docs |
| **Faithfulness (strict)** | XLM-RoBERTa NLI | retrieval_evidence only | Без partial credit для neutral |
| **Citation Precision** | XLM-RoBERTa NLI | retrieval_evidence only | Доля полезных citations (ALCE) |
| **Latency** | evaluate_agent.py | All 36 Qs | Mean per question |

### Отдельная секция Contradictions

Если NLI находит contradiction (claim противоречит документу) — это сигнал hallucination. В финальном отчёте отдельная секция со всеми contradictions:

```markdown
## Contradictions (hallucination signals)

| Question | Claim | Document | Contradiction score |
|----------|-------|----------|-------------------|
| q12 | "GPT-5.4 вышел в январе 2026" | doc:xyz "GPT-5.3 анонсирован в ноябре 2025" | 0.87 |
```

---

## Файловая структура

```
src/services/eval/           — eval-only пакет (НЕ runtime)
  __init__.py
  nli.py                     — NLIVerifier class (HTTP client to gpu_server /nli)

scripts/
  run_nli.py                 — claims × documents → faithfulness scores
  merge_eval_report.py       — judge + NLI → final report

datasets/prompts/            — версионированные промпты
  decomposition_v1.md        — промпт для Claude decomposition
  judge_v1.md                — промпт для Claude judge

results/
  reports/
    judge_artifact_YYYYMMDD.md      — input для Claude (человекочитаемый)
    final_eval_YYYYMMDD.md          — финальный отчёт с всеми метриками
    final_eval_YYYYMMDD.json        — машинночитаемый
  raw/
    eval_results_YYYYMMDD.json      — agent eval output (source of truth)
    claims_YYYYMMDD.json            — Claude decomposition
    judge_verdicts_YYYYMMDD.json    — Claude judge
    nli_scores_YYYYMMDD.json        — XLM-RoBERTa NLI
```

---

## VRAM Budget

| Модель | Текущий | После NLI |
|--------|---------|-----------|
| pplx-embed-v1-0.6B | ~1.5 GB | ~1.5 GB |
| Qwen3-Reranker-0.6B | ~1.5 GB | ~1.5 GB |
| jina-colbert-v2 | ~1.5 GB | ~1.5 GB |
| XLM-RoBERTa-large-xnli | — | ~1.7 GB |
| PyTorch runtime overhead | ~0.5 GB | ~1.5 GB |
| **Total** | **~5.5 GB** | **~8.2 GB** |
| **RTX 5060 Ti free** | **~10.5 GB** | **~7.8 GB** |

Lazy loading: NLI модель загружается только при первом запросе к `/nli`. В runtime (без eval) overhead = 0.

---

## Acceptance Criteria

- [ ] XLM-RoBERTa-large-xnli загружается в gpu_server.py (lazy), endpoint `/nli` работает
- [ ] `/nli` обрабатывает 512-token truncation корректно (chunking с overlap)
- [ ] `/nli` возвращает 503 при OOM, 504 при timeout
- [ ] Промпты decomposition и judge сохранены в `datasets/prompts/` с версией
- [ ] `evaluate_agent.py --export-offline-judge` генерирует расширенный артефакт
- [ ] Claude decomposition → claims JSON с type classification
- [ ] Claude judge → verdicts JSON с factual/useful/reasoning
- [ ] `run_nli.py` обрабатывает 17 retrieval_evidence Qs за < 5 минут
- [ ] `run_nli.py` читает только JSON, не парсит markdown
- [ ] Document chunking при > 400 tokens, max entailment across chunks
- [ ] `merge_eval_report.py` собирает финальный отчёт с 7 aggregate метриками
- [ ] Contradictions секция в финальном отчёте
- [ ] Baseline faithfulness ≥ 0.85 на retrieval_evidence вопросах
- [ ] NLI threshold calibration: 30-50 вручную размеченных пар после первого прогона

---

## Порядок реализации

1. **datasets/prompts/**: сохранить промпты decomposition_v1.md и judge_v1.md
2. **gpu_server.py**: добавить XLM-RoBERTa модель + `/nli` endpoint (lazy loading)
3. **src/services/eval/nli.py**: NLI HTTP client class
4. **evaluate_agent.py**: расширить `--export-offline-judge` (tool outputs для analytics)
5. **scripts/run_nli.py**: NLI прогон (JSON input, chunking, batch)
6. **scripts/merge_eval_report.py**: merge всех источников → финальный отчёт
7. Прогон eval → артефакт → Claude decomposition → Claude judge → NLI → merge
8. Calibrate NLI threshold на 30-50 парах
9. Зафиксировать baseline faithfulness

---

## Не делаем

- Runtime NLI (Phase 2 из R19) — только eval
- Fine-tune XLM-RoBERTa на нашем домене — baseline first
- Sentence-level splitting вместо Claude decomposition — Claude даёт атомарные claims лучше
- NLI для analytics/navigation/refusal вопросов — faithfulness = N/A
- Citation Recall как отдельная метрика — тавтологична с faithfulness в текущем дизайне
- Claude API автоматизация — нет доступа к API, ручной чат workflow
