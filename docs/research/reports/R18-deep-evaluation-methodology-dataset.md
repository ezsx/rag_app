# Evaluation pipeline for agentic RAG with 11+ tools

**The single biggest unlock for your eval system is replacing monolithic LLM scoring with decompose-then-verify factual checking against expected answers.** This sidesteps the Yandex warning entirely — you never ask the LLM to judge correctness holistically, only to classify claim-level entailment (a task where even 3B-active-param models achieve >90% agreement with humans). Combined with a 450–500 question golden dataset, tool selection tracking via SSE events, and phased ablation studies, this pipeline transforms your eval from 30 questions with strict recall into a statistically powered, multi-criteria system running entirely self-hosted on your V100.

The specification below covers all 7 sections with concrete JSON schemas, prompt templates in Russian, formulas, and protocols. Every component runs on Qwen3-30B-A3B locally, with Claude API reserved for ≤10% of evaluations (subjective criteria and calibration only).

---

## Section 1: Dataset design — schema, categories, and sample sizes

### Complete JSON schema for evaluation questions

Every evaluation question carries fields for multi-criteria scoring, tool sequence validation, negative constraints, and forward-compatibility with future tools:

```json
{
  "id": "eval_q_001",
  "version": "1.0",
  "question": "Какие основные отличия между LoRA и QLoRA обсуждались в каналах AI_News и ML_Practice в январе 2026?",
  "expected_answer": "В канале AI_News обсуждалось, что LoRA требует больше VRAM...",
  "expected_answer_type": "long_form",
  "category": "cross_channel_compare",
  "subcategory": "technology_comparison",
  "difficulty": "hard",
  "multi_hop": true,
  "hop_count": 2,
  "expected_tool_sequence": {
    "primary": ["query_plan", "cross_channel_compare", "search", "search", "rerank", "compose_context", "final_answer"],
    "alternatives": [
      {
        "sequence": ["query_plan", "channel_search", "channel_search", "rerank", "compose_context", "final_answer"],
        "score_multiplier": 0.7,
        "reason": "achieves comparison via multiple channel_search calls"
      }
    ],
    "key_tools": ["cross_channel_compare"],
    "scaffold_tools": ["query_plan", "rerank", "compose_context", "final_answer"],
    "forbidden_tools": ["summarize_channel", "list_channels"]
  },
  "negative_constraints": {
    "is_unanswerable": false,
    "out_of_scope": false,
    "out_of_timerange": false,
    "expected_refusal": false,
    "refusal_reason": null
  },
  "temporal_constraints": {
    "requires_temporal": true,
    "time_range": {"start": "2026-01-01", "end": "2026-01-31"},
    "temporal_type": "range"
  },
  "source_post_ids": ["post_12345", "post_67890"],
  "source_channels": ["AI_News", "ML_Practice"],
  "future_tool_flag": false,
  "baseline_tool_sequence": null,
  "metadata": {
    "created_at": "2026-03-24",
    "created_by": "synthetic_pipeline_v1",
    "verified_by_human": false,
    "generation_method": "multi_hop_cross_channel",
    "tags": ["lora", "qlora", "fine-tuning"]
  }
}
```

The **critical design choice** here is separating `key_tools` from `scaffold_tools`. Scaffold tools (query_plan, rerank, compose_context, final_answer) appear in nearly every sequence and carry minimal discriminative signal. Key tools — the search variant selected — carry almost all diagnostic value and should be weighted accordingly in scoring.

### Question categories covering all 15 tools

The dataset must span 17 question categories mapping to 15 tools (11 current + 4 future), plus 2 negative-test categories:

| Category | Tool tested | Questions | Difficulty mix |
|---|---|---|---|
| Simple factual | search | 40–50 | 50% easy, 30% med, 20% hard |
| Temporal queries | temporal_search | 30–40 | 20% easy, 50% med, 30% hard |
| Channel-specific | channel_search | 30–40 | 40% easy, 40% med, 20% hard |
| Cross-channel comparison | cross_channel_compare | 25–30 | 10% easy, 40% med, 50% hard |
| Channel summaries | summarize_channel | 15–20 | 50% easy, 50% med |
| Metadata queries | list_channels | 15–20 | 80% easy, 20% med |
| Complex decomposition | query_plan (multi-hop) | 25–35 | 100% med–hard |
| Follow-up exploration | related_posts | 20–25 | 30% easy, 50% med, 20% hard |
| Fact verification | verify | 15–20 | 30% med, 70% hard |
| Entity tracking (future) | entity_tracker | 15–20 | 100% hard |
| Paper tracking (future) | arxiv_tracker | 15–20 | 100% hard |
| Trending topics (future) | hot_topics | 15–20 | 50% med, 50% hard |
| Channel expertise (future) | channel_expertise | 15–20 | 50% med, 50% hard |
| Unanswerable | (none — refusal) | 30–40 | N/A |
| Forbidden tool | (negative selection) | 20–30 | N/A |

**Total: ~450–500 questions.** This aligns with the Berkeley Function Calling Leaderboard (BFCL) which uses 100–400 per category and with the 500-pair recommendation from the Yandex talk.

### Statistical justification for per-category sample size

For binary pass/fail evaluation, **30 samples per category** detects a 20-percentage-point difference (e.g., 60% → 80% accuracy) at α=0.05 with ~80% power. This is the Central Limit Theorem threshold where sample means approximate normality. The BFCL uses 200–400 per major category; with 15 categories the pragmatic minimum is **30 per core category, 15–20 for low-variance categories** (list_channels, summarize_channel). The "Know Your RAG" paper (arXiv 2411.19710) confirms that diverse label distribution matters more than raw quantity — 500 well-distributed questions outperform 2000 skewed ones.

### Negative case design

Six unanswerable question types (drawn from the ACL 2025 paper "Unanswerability Evaluation for RAG"):

- **Out-of-database**: Topic-adjacent but absent from corpus ("Что обсуждалось о квантовых вычислениях?" when no channel covers it)
- **Out-of-timerange**: References dates outside July 2025–March 2026 window
- **False presupposition**: Assumes incorrect facts ("Почему Meta уменьшила Llama 3 до 30B параметров?")
- **Underspecified**: Missing crucial context ("Какая модель лучше?" — which model?)
- **Modality-limited**: Asks about images/video in text-only corpus
- **Safety-concerned**: Should be refused on policy grounds

Target **15–20% negative questions** in the dataset (SQuAD 2.0 uses 33%, but for production RAG 15–20% is realistic). Each negative question specifies `expected_refusal: true` and `refusal_reason`.

### Forward-looking questions for future tools

Questions for entity_tracker, arxiv_tracker, hot_topics, and channel_expertise use a dual-sequence format — answerable now via search+compose_context as baseline, but designed to be answered *better* by the future tool:

```json
{
  "id": "future_entity_001",
  "question": "Как менялось обсуждение модели Mistral в каналах за последние 3 месяца?",
  "category": "entity_tracker",
  "future_tool_flag": true,
  "expected_tool_sequence": {
    "primary": ["query_plan", "temporal_search", "search", "rerank", "compose_context", "final_answer"],
    "future_primary": ["query_plan", "entity_tracker", "compose_context", "final_answer"],
    "key_tools": ["temporal_search"],
    "future_key_tools": ["entity_tracker"]
  },
  "evaluation_mode": "baseline",
  "upgrade_criteria": "Re-evaluate with future_primary when entity_tracker ships; expect higher completeness scores"
}
```

This lets you track improvement when each SPEC-RAG-15/16 tool ships — baseline scores on these questions become the "before" measurement automatically.

---

## Section 2: Synthetic question generation pipeline

### Seven-stage pipeline

**Stage 1 — Corpus preparation.** Load all 13,088 posts from Qdrant. Extract metadata (channel_name, timestamp, text, post_id). Cluster posts by topic using Qwen3-Embedding-0.6B embeddings. Build an entity co-occurrence graph across channels — entities appearing in different channels become multi-hop bridge candidates.

**Stage 2 — Stratified sampling.** Sample posts proportionally across channels and months. Enforce minimums: ≥3 posts per channel per month represented. For multi-hop questions, select post pairs sharing entities across different channels.

**Stage 3 — Category-specific generation.** For each sampled post (or post-pair), call Qwen3-30B-A3B with category-specific prompts (templates below). Generate question, expected_answer, key_facts, and difficulty estimate.

**Stage 4 — Quality filtering.** Score each QA pair via LLM on groundedness, standalone clarity, and relevance (each 1–5). Filter threshold: all scores ≥ 4 (per Hugging Face RAG Evaluation cookbook). Remove near-duplicates via embedding cosine similarity > 0.85.

**Stage 5 — Tool sequence annotation.** Separate LLM call classifies each question into expected_tool_sequence with required/optional/forbidden tools.

**Stage 6 — Diversity enforcement.** Check distribution against target. If any category is underrepresented, generate more questions targeting that category. Enforce: each of 36 channels in ≥5 questions, each month (July 2025–March 2026) in ≥20 questions, difficulty split of 30% easy / 40% medium / 30% hard, ≥20% multi-hop.

**Stage 7 — Human verification.** Random-sample **20–30%** for human review (promotes to "gold" status). 100% human review of all negative/edge cases. For 450 questions, this means ~90–135 manually reviewed. Use Gwet's AC1 (more stable than Cohen's κ on skewed distributions) to measure inter-annotator agreement; target AC1 ≥ 0.7. Use Dawid-Skene EM aggregation when 3+ annotators disagree.

### Prompt templates for Qwen3-30B-A3B

**Template A — Single-hop factual (search/channel_search):**

```
<system>
Ты — эксперт по созданию вопросов для оценки RAG-системы. Генерируй вопросы на русском языке.
</system>
<user>
Ниже приведён пост из Telegram-канала "{channel_name}", дата: {date}.

--- ПОСТ ---
{post_text}
--- КОНЕЦ ПОСТА ---

Задача: Создай один фактологический вопрос, ответ на который содержится ТОЛЬКО в данном посте.

Требования:
1. Вопрос должен быть понятен без контекста поста
2. Вопрос должен быть конкретным (не общим)
3. Ответ должен быть обоснован текстом поста
4. Не упоминай "пост", "канал", "Telegram" в вопросе

Формат ответа (строго JSON):
{
  "question": "...",
  "expected_answer": "...",
  "difficulty": "easy|medium|hard",
  "key_facts": ["факт1", "факт2"]
}
</user>
```

**Template B — Temporal search:**

```
<system>
Ты — эксперт по созданию вопросов с временной привязкой для оценки RAG-системы.
</system>
<user>
Ниже приведены два поста из разных периодов:

--- ПОСТ 1 ({date1}, канал: {channel1}) ---
{post_text_1}
--- ПОСТ 2 ({date2}, канал: {channel2}) ---
{post_text_2}

Задача: Создай вопрос, который требует поиска информации за определённый временной период.

Требования:
1. Вопрос должен указывать на конкретный временной диапазон
2. Ответ должен синтезировать информацию из обоих постов
3. Вопрос должен начинаться с "Что обсуждалось...", "Какие...", "Как изменилось..."

Формат ответа (строго JSON):
{
  "question": "...",
  "expected_answer": "...",
  "time_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "difficulty": "medium|hard"
}
</user>
```

**Template C — Cross-channel comparison:**

```
<system>
Ты — эксперт по созданию сравнительных вопросов между каналами.
</system>
<user>
Ниже приведены посты из двух разных каналов на схожую тему:

--- КАНАЛ "{channel1}" ({date1}) ---
{post_text_1}
--- КАНАЛ "{channel2}" ({date2}) ---
{post_text_2}

Задача: Создай вопрос, требующий сравнения позиций/информации из двух каналов.

Требования:
1. Вопрос должен требовать информацию из ОБОИХ каналов
2. Ответ невозможно дать, зная только один из постов
3. Сравнение должно быть содержательным

Формат ответа (строго JSON):
{
  "question": "...",
  "expected_answer": "...",
  "channels_required": ["{channel1}", "{channel2}"],
  "comparison_aspect": "...",
  "difficulty": "hard"
}
</user>
```

**Template D — Entity tracker (future tool baseline):**

```
<system>
Ты — эксперт по созданию вопросов отслеживания сущностей во времени.
</system>
<user>
Ниже приведены 3 поста, упоминающие сущность "{entity}" в разные даты:

--- ПОСТ 1 ({date1}) ---
{post_text_1}
--- ПОСТ 2 ({date2}) ---
{post_text_2}
--- ПОСТ 3 ({date3}) ---
{post_text_3}

Задача: Создай вопрос о том, как менялось обсуждение/отношение к "{entity}" со временем.

Формат ответа (строго JSON):
{
  "question": "...",
  "expected_answer": "...",
  "entity": "{entity}",
  "timeline": ["{date1}", "{date2}", "{date3}"],
  "difficulty": "hard"
}
</user>
```

**Template E — Negative/unanswerable question:**

```
<system>
Ты — эксперт по созданию вопросов, на которые НЕТ ответа в корпусе.
</system>
<user>
Ниже приведён пост из канала "{channel_name}":

--- ПОСТ ---
{post_text}
--- КОНЕЦ ПОСТА ---

Задача: Создай вопрос, который СВЯЗАН с темой поста, но ответа на него НЕТ в тексте.

Стратегии:
- Спроси о деталях, которые не упомянуты (цена, дата выхода, автор)
- Спроси о сравнении с чем-то, что не упоминается
- Спроси о будущих планах, если пост их не содержит

Формат ответа (строго JSON):
{
  "question": "...",
  "unanswerable_reason": "detail_not_mentioned|topic_not_covered|future_prediction|false_presupposition",
  "expected_system_response": "honest_uncertainty",
  "related_topic": "..."
}
</user>
```

### Multi-hop question generation

The multi-hop pipeline uses entity co-occurrence graphs. Extract named entities (models, companies, technologies) from all 13,088 posts. Find entity pairs appearing in different channels or time periods. For each bridge entity E appearing in Post A (Channel X) and Post B (Channel Y): generate single-hop sub-questions for each post, then compose a multi-hop question via LLM requiring both answers. Finally, verify multi-hop necessity — if the question can be answered from a single post, discard it (anti-shortcut check per CRUMQs methodology, arXiv 2510.11956).

### Quality verification rates

**100% automated filtering** (LLM scoring on groundedness/relevance/standalone ≥ 4). **20–30% human verification** of questions passing automated filters. **100% human review** of all negative and edge cases. This yields ~90–135 manually reviewed questions from a 450-question dataset. The "Know Your RAG" paper found that 95% of simple prompt-generated data falls into a single category ("fact_single"), making explicit diversity control and human verification essential.

---

## Section 3: LLM judge — multi-criteria with factual correctness solution

### Five independent criteria

Each criterion gets its own score, its own prompt, and its own scale. Never combine into a single aggregate prompt — this is the key reliability insight from Databricks and Eugene Yan's survey.

| Criterion | Scale | Method | LLM calls |
|---|---|---|---|
| Usefulness | 0–2 (3-point) | Direct G-Eval with CoT | 1 |
| Factual correctness | 0.0–1.0 (continuous F1) | Decompose-then-verify | 3–4 |
| Citation grounding | 0.0–1.0 (ratio) | Per-claim binary check | 1–2 |
| Completeness | 0–2 (3-point) | Reference-based scoring | 1 |
| Refusal accuracy | Binary 0/1 | Classification | 1 |

The 3-point scale (0–2) for subjective criteria is optimal because Databricks found >80% human-LLM agreement on 0–3 scales, while finer scales produce arbitrary scores from smaller LLMs. **Continuous scores (factual correctness, citation grounding) are computed programmatically from binary per-claim judgments**, avoiding LLM scale arbitrariness entirely.

### Factual correctness — the decompose-then-verify solution

This is the critical innovation that addresses the Yandex warning. The key insight: **your system HAS expected_answer.** This transforms the problem from "verify against world knowledge" (where LLM judges fail) to "verify against reference text" (where LLM judges as NLI classifiers work reliably).

**Step 1 — Decompose answer into atomic claims** (LLM call #1):

```
<system>
Разложите данный ответ на атомарные фактические утверждения. Каждое утверждение должно содержать ровно один проверяемый факт.
</system>
<user>
Ответ: {answer}

Извлеките все фактические утверждения. Каждое утверждение должно быть:
- Самодостаточным (понятным без контекста)
- Атомарным (один факт)
- Проверяемым

Формат (JSON):
{"claims": [{"id": 1, "claim": "..."}, {"id": 2, "claim": "..."}], "total_claims": N}
</user>
```

**Step 2 — Decompose expected_answer into atomic claims** (LLM call #2): Same prompt applied to expected_answer.

**Step 3 — Verify each answer claim against expected_answer** (LLM call #3):

```
<system>
Вы — строгий верификатор фактов. Верифицируйте ТОЛЬКО на основе эталонного ответа. НЕ используйте собственные знания.
</system>
<user>
Утверждение: {claim}
Эталонный ответ: {expected_answer}

Определите категорию:
- "supported" — утверждение прямо подтверждается эталонным ответом
- "contradicted" — утверждение противоречит эталонному ответу
- "not_mentioned" — не может быть ни подтверждено, ни опровергнуто

Формат (JSON):
{"claim": "...", "verdict": "supported|contradicted|not_mentioned", "evidence": "...", "reasoning": "..."}
</user>
```

**Step 4 — Check coverage** (LLM call #4): For each expected_answer claim, check if it's covered in the answer.

**Step 5 — Compute score programmatically:**

```
TP = |answer claims that are "supported"|
FP = |answer claims that are "contradicted"|
Extrinsic = |answer claims that are "not_mentioned"|
FN = |expected_answer claims missing from answer|

Factual_Precision = TP / (TP + FP + 0.5 × Extrinsic)
Factual_Recall = TP / (TP + FN)
Factual_F1 = 2 × Precision × Recall / (Precision + Recall)
```

The **0.5 penalty** for extrinsic claims is configurable — strict mode penalizes additions not in expected_answer, lenient mode (0.0) only penalizes contradictions. This adapts FActScore (Min et al., EMNLP 2023) to a reference-based setting where you have ground truth.

**Why this works despite the Yandex warning**: You never ask the LLM to judge factual correctness holistically. You ask it to perform NLI classification (supported/contradicted/not_mentioned) — a binary task where even small models achieve >90% agreement with humans. Each claim is checked against expected_answer, not the LLM's own knowledge. The final score is computed mathematically, with no LLM bias in aggregation. Every step is auditable.

**Supplementary non-LLM cross-reference** (fast sanity check):

```python
# Hybrid final score
Final_FC = 0.7 * Factual_F1 + 0.2 * entity_overlap(answer, expected) + 0.1 * semantic_similarity(answer, expected)
```

Entity overlap catches numeric/name mismatches that NLI might miss. Semantic similarity (via Qwen3-Embedding-0.6B) provides a coarse backup signal.

### Prompt templates for other criteria

**Usefulness (0–2):**

```
<system>
Вы — эксперт по оценке качества ответов AI-ассистента по AI/ML Telegram-каналам.
</system>
<user>
Вопрос: {question}
Ответ системы: {answer}

Рубрика:
0 — Бесполезный: не содержит релевантной информации
1 — Частично полезный: некоторая полезная информация, но неполный
2 — Полезный: полностью отвечает, хорошо структурирован, конкретен

Шаг 1: Определите информационную потребность.
Шаг 2: Оцените покрытие.
Шаг 3: Учтите структуру и конкретность.

Формат (JSON): {"reasoning": "...", "score": 0|1|2, "criterion": "usefulness"}
</user>
```

**Citation grounding (0.0–1.0):**

```
<system>
Проверьте, подкреплено ли каждое утверждение в ответе ссылкой на источник.
</system>
<user>
Ответ: {answer}
Контекст (retrieved documents): {contexts}

Для каждого утверждения: выделите его, проверьте наличие ссылки, проверьте корректность ссылки по контексту.

Формат (JSON):
{"claims": [{"claim": "...", "has_citation": true|false, "citation_correct": true|false|null}], "grounding_score": 0.0-1.0}
</user>
```

Score = count(has_citation AND citation_correct) / total_claims.

**Completeness (0–2):**

```
<system>
Оцените полноту ответа относительно эталонного ответа.
</system>
<user>
Вопрос: {question}
Ответ системы: {answer}
Эталонный ответ: {expected_answer}

Рубрика:
0 — Пропущено >50% ключевых аспектов
1 — Покрыто 50-80% ключевых аспектов
2 — Покрыто >80% ключевых аспектов

Формат (JSON):
{"key_aspects": ["..."], "covered_aspects": ["..."], "missing_aspects": ["..."], "coverage_ratio": 0.0-1.0, "score": 0|1|2}
</user>
```

**Refusal accuracy (binary):**

```
<system>
Для negative questions оцените, правильно ли система отказалась отвечать.
</system>
<user>
Вопрос: {question}
Ответ: {answer}
Ожидаемое поведение: {expected_behavior}

score=1 если поведение совпадает, score=0 если нет.

Формат (JSON): {"expected_behavior": "refuse|answer", "actual_behavior": "refused|answered", "score": 0|1}
</user>
```

### Aggregation — tiered gating plus weighted average

```python
def compute_overall(scores):
    # Gate 1: Hard failures
    if scores['refusal_accuracy'] == 0:  # Wrong refusal behavior
        return 0.0
    if scores['factual_correctness'] < 0.3:  # Severe hallucination
        return scores['factual_correctness'] * 0.5  # Capped low
    
    # Gate 2: Weighted combination (all normalized to 0-1)
    return (
        0.35 * scores['factual_correctness'] +
        0.25 * (scores['completeness'] / 2.0) +
        0.20 * scores['citation_grounding'] +
        0.20 * (scores['usefulness'] / 2.0)
    )
```

Factual correctness has both a **gate** (hard floor at 0.3) and the **highest weight** (0.35) — double protection against hallucination-tolerant scores. This tiered approach ensures no hallucinating answer passes while allowing nuanced ranking above the floor.

### Qwen3-30B-A3B vs Claude API — tiered deployment

**Tier 1 — Qwen local (90% of evaluations):** All binary claim verification (supported/contradicted/not_mentioned), citation grounding checks, refusal classification, claim decomposition. These are NLI-class tasks where 3B active params suffice.

**Tier 2 — Claude API (10% of evaluations):** Usefulness scoring (subjective, benefits from stronger reasoning), completeness scoring (nuanced comparison), disagreement resolution when Qwen gives borderline results.

**Tier 3 — Monthly calibration:** 50–100 samples evaluated by both. Measure agreement drift. If Qwen-Claude agreement drops below 85%, retune Qwen prompts.

**Cost:** At 500 questions × 5 criteria × ~500 tokens/eval, Qwen handles ~2.5M tokens locally at $0. Claude handles ~250K tokens at **~$1–4/day** — negligible.

Critical Qwen reliability practices: always use **3–5 few-shot examples** per criterion (single biggest quality booster for smaller models), require **JSON output** with explicit schema, use **CoT reasoning** ("Шаг 1... Шаг 2...") before verdict, set **temperature=0** for reproducibility.

---

## Section 4: Robustness metrics — NDR, RSR, and retrieval order robustness

These three metrics originate from Cao et al. (arXiv 2505.21870, "Evaluating the Retrieval Robustness of Large Language Models", Bloomberg/UMich, May 2025) and were subsequently discussed at the Yandex meetup. Each measures a different failure mode of RAG context sensitivity.

### NDR (Non-Degradation Rate)

NDR measures how often adding retrieval context **doesn't make things worse** compared to no retrieval at all.

**Formula:**

```
NDR = (1/Z) × Σ_{q∈Q} Σ_{k∈K} 𝟙[f(q, k) ≥ f(q, 0)]

where Z = |Q| × |K|
```

**f(q, k)** is the LLM judge correctness score for query q with k retrieved documents. **f(q, 0)** is the score with no retrieval (LLM answers from parametric knowledge). **Any drop counts as degradation** — no threshold, strictly binary per sample.

**Protocol:**
1. Run each of 500 questions with k=0 (retrieval disabled, LLM answers from own knowledge)
2. Run each question with k=3, 5, 10, 15, 20 (natural retrieval results, not synthetic noise)
3. For each (q, k) pair, check if f(q,k) ≥ f(q,0)
4. NDR = proportion of non-degraded pairs

**Key finding from the paper:** Larger LLMs have *lower* NDR because richer parametric knowledge sets a higher baseline, making degradation more likely. For Qwen3-30B-A3B with 3B active params, expect relatively high NDR since the no-retrieval baseline will be weaker.

### RSR (Retrieval Size Robustness)

RSR measures whether increasing k **never hurts** answer quality. A robust system should maintain or improve as it sees more context.

**Formula:**

```
RSR(q, k_i) = 𝟙[∀j<i: f(q, k_i) ≥ f(q, k_j)]
RSR = (1 / |Q| × (|K|-1)) × Σ_{q∈Q} Σ_{k_i∈K, i>1} RSR(q, k_i)
```

For k = [3, 5, 10, 15, 20] with scores [0.8, 0.9, 0.7, 0.9, 0.9]: k=5 passes (0.9 ≥ 0.8), k=10 fails (0.7 < 0.8), k=15 passes (0.9 ≥ all previous), k=20 passes. RSR for this query = 3/4 = 0.75.

**What should grow monotonically:** Answer quality (LLM judge correctness), not recall@k. This is an **end-to-end generation quality metric** — it catches the "lost in the middle" problem where more context actually degrades generation even if retrieval recall improves.

### ROR (Retrieval Order Robustness)

ROR measures sensitivity to the **ordering of chunks** passed to compose_context.

**Formula:**

```
ROR = (1 / |Q| × |K|) × Σ_{q∈Q} Σ_{k∈K} (1 - 2 × σ_{o∈O}[f(q, k, o)])
```

**σ** is the standard deviation of correctness scores across orderings. The factor of 2 scales σ (bounded [0, 0.5] for binary metrics) to [0, 1]. Higher ROR = more consistent.

**Shuffling protocol:** Use 3 orderings minimum (original reranker order, reversed, random shuffle), ideally **6 orderings** (original + reversed + 4 random shuffles) for stable standard deviation estimates.

**Measuring "answer didn't change":** The original paper uses an LLM judge for binary correctness. For your system, recommended approaches ranked by quality:

- **LLM judge (pairwise)** — most accurate, handles paraphrasing. Primary method.
- **BERTScore F1 ≥ 0.85** using ai-forever/ruBert-large — fast, semantic, good for Russian. Secondary/cheap method.
- **Embedding cosine similarity ≥ 0.90** via Qwen3-Embedding-0.6B — very fast, coarse. Screening only.

### Integration protocol and time budget

```
Phase 1: NDR+RSR sweep (shared runs)
  Baseline (k=0): 500 runs → 5.6 hours
  k=3,5,10,15,20: 2,500 runs → 27.8 hours
  Total: 3,000 runs → 33.3 hours → computes both NDR and RSR

Phase 2: ROR shuffle test (single k=10)
  6 orderings × 500 questions = 3,000 runs → 33.3 hours
  
Phase 3: LLM judge scoring
  ~6,000 runs × 5 criteria ≈ done concurrently with generation

Pragmatic total: ~6,000 runs → ~67 hours → ~2.8 days continuous
```

**Prioritization:** RSR first (directly actionable — tells you optimal k and whether your reranker pipeline is robust). NDR second (shares runs with RSR, safety check). ROR third (most expensive, but critical for your compose_context step given the "lost in the middle" effect from Liu et al., arXiv 2307.03172).

**Robustness test config format:**

```json
{
  "robustness_eval": {
    "tests": [
      {
        "name": "NDR_RSR_sweep",
        "k_values": [0, 3, 5, 10, 15, 20],
        "orders": ["reranker_score"],
        "metrics_computed": ["NDR", "RSR"],
        "scoring": {"method": "llm_judge", "model": "qwen3-30b-a3b", "output": "binary"}
      },
      {
        "name": "ROR_shuffle",
        "k_values": [10],
        "orders": ["original", "reversed", "shuffle_1", "shuffle_2", "shuffle_3", "shuffle_4"],
        "shuffle_seed_base": 42,
        "metrics_computed": ["ROR"]
      }
    ]
  }
}
```

---

## Section 5: Tool selection metrics — key tool accuracy and negative testing

### Expected tool sequence format

The core insight from T-Eval, BFCL, and the Advancing Agentic Systems paper: decompose the tool sequence into **scaffold tools** (always called, low diagnostic value) and **key tools** (the discriminative choice). For your system, scaffold = {query_plan, rerank, compose_context, final_answer}, key = {search, temporal_search, channel_search, cross_channel_compare, summarize_channel, list_channels, related_posts, verify}.

```json
{
  "expected_tool_sequence": {
    "primary": ["query_plan", "temporal_search", "rerank", "compose_context", "final_answer"],
    "alternatives": [
      {
        "sequence": ["query_plan", "search", "rerank", "compose_context", "final_answer"],
        "score_multiplier": 0.6,
        "reason": "search works but loses temporal filtering"
      }
    ],
    "key_tools": ["temporal_search"],
    "scaffold_tools": ["query_plan", "rerank", "compose_context", "final_answer"],
    "forbidden_tools": ["cross_channel_compare", "list_channels", "summarize_channel"]
  }
}
```

The `score_multiplier` on alternatives handles ambiguity — agent used a valid but suboptimal path, receiving partial credit.

### Five accuracy metrics with formulas

**1. Key Tool Selection Accuracy** (most important, weight 0.40):

```
KeyToolAcc = |predicted_key_tools ∩ expected_key_tools| / |expected_key_tools|
```

Did the agent pick temporal_search over search? This single metric carries the most diagnostic signal. Borrowed from BFCL's "Multiple Function" category.

**2. Tool F1** (weight 0.15):

```
ToolPrecision = |tools_called ∩ expected_tools| / |tools_called|
ToolRecall = |tools_called ∩ expected_tools| / |expected_tools|  
ToolF1 = 2 × ToolPrecision × ToolRecall / (ToolPrecision + ToolRecall)
```

From the Advancing Agentic Systems paper (Gabriel et al., 2024) — treats tool selection as set membership.

**3. Exact Sequence Match** (weight 0.10):

```
ESM = 𝟙[predicted_sequence == expected_sequence]
```

Strict but useful as a ceiling metric. From T-Eval and ToolBench Plan.EM.

**4. LCS-F1** (weight 0.10, for partial credit):

```
LCS_P = len(LCS(predicted, expected)) / len(predicted)
LCS_R = len(LCS(predicted, expected)) / len(expected)
LCS_F1 = 2 × LCS_P × LCS_R / (LCS_P + LCS_R)
```

Captures mostly-correct sequences with minor insertions/deletions.

**5. Negative Test Pass Rate** (weight 0.15):

```
NegativePass = |cases passing all forbidden constraints| / |total negative test cases|
FalsePositiveToolRate = |forbidden_tools_called| / |total_tools_called|  (across negative tests)
```

### Composite tool selection score

```
ToolScore = 0.40 × KeyToolAcc + 0.15 × ToolF1 + 0.10 × ESM + 0.10 × LCS_F1 + 0.15 × NegativePass + 0.10 × VisibilityAcc
```

**Minimum threshold:** KeyToolAcc ≥ 0.85 is a hard gate. Below this, the agent is fundamentally selecting wrong tools.

### Negative tool selection test format

```json
{
  "id": "neg_tool_001",
  "question": "Какие каналы у вас есть?",
  "category": "negative_tool_test",
  "expected_tool_sequence": {
    "primary": ["query_plan", "list_channels", "compose_context", "final_answer"],
    "key_tools": ["list_channels"],
    "forbidden_tools": ["search", "temporal_search", "cross_channel_compare", "channel_search", "rerank"],
    "negative_scoring": {
      "critical_forbidden": ["search", "temporal_search"],
      "penalty_per_violation": 0.5
    }
  }
}
```

Negative score = max(0, 1.0 − Σ penalty per forbidden tool called). Critical forbidden tools (search, temporal_search) get full 0.5 penalty; minor violations get 0.25. This mirrors BFCL V4's hallucination scoring (10% of overall score).

### Dynamic visibility evaluation

The signal router (max 5 visible tools per step) needs its own metrics:

```
VisibilityPrecision = |shown_tools ∩ relevant_tools| / |shown_tools|
KeyToolVisibilityRate = 𝟙[optimal_key_tool ∈ visible_set]
```

**KeyToolVisibilityRate is the gating metric.** If the router hides temporal_search for a temporal query, the agent *cannot succeed* regardless of LLM quality. Decompose failures into:

```
EndToEndAccuracy = P(correct_tool_visible) × P(correct_tool_selected | correct_tool_visible)
```

This isolates routing failures from LLM selection failures — crucial for debugging.

---

## Section 6: Ablation study — phased protocol with paired bootstrap tests

### Components to ablate (priority order)

**Priority 1 — ColBERT reranking on/off.** Last quality gate before context composition. Research shows cross-encoders can outperform ColBERT but ColBERT is faster. Tests whether two-stage reranking (ColBERT MaxSim then bge-reranker) adds value over bge-reranker alone.

**Priority 2 — bge-reranker-v2-m3 on/off.** The Dell RAG Fusion paper (arXiv 2603.02153, March 2026) found reranking neutralizes fusion recall gains entirely. This ablation reveals whether your cross-encoder is the primary quality driver.

**Priority 3 — Multi-query search on/off.** The same Dell paper found multi-query fusion increased raw recall but Hit@10 actually *decreased* from 0.51 to 0.48 after reranking and truncation. Most potentially surprising ablation.

**Priority 4 — RRF weights (3:1 vs 1:1 vs 1:3).** Hybrid RAG research (aimultiple.com, 2026) found improperly tuned fusion weights can make hybrid search worse than single-retriever.

**Priority 5 — Forced search vs agent decides.** Prolego's LLM RAG study found agents didn't improve over forced retrieval.

**Priority 6 — Original query injection into multi-query.** Does injecting the original alongside sub-queries help or add noise?

**Priority 7 — Dynamic tool visibility (max 5 vs all tools).** Validates the Yandex finding that hiding irrelevant tools improves accuracy.

### Protocol — phased approach

**Phase 1 — Quick screen (≈9 hours):** Baseline + 7 ablations + 2 RRF variants + 1 negative control = 11 configs × 100 stratified questions × 1 run = **1,100 inference calls**. Identifies top 3 impactful components.

**Phase 2 — Deep eval (≈67 hours):** Baseline + top 3 ablations × 500 questions × 3 runs = **6,000 inference calls**. Produces statistically significant results for the components that matter most.

**Phase 3 — Interaction effects (optional, ≈50 hours):** 2–3 suspected interactions (e.g., ColBERT + multi-query) on 500 × 3 = 4,500 calls.

**Total: ~5–7 days** instead of 7.6 days for full factorial, with more focused data.

### Statistical significance tests

**Paired bootstrap resampling** (gold standard for NLP, Berg-Kirkpatrick et al., 2012): Resample query-level score pairs with replacement, B=10,000 iterations. Always use **paired** tests on same queries — unpaired tests readily produce false significance (arXiv 2511.19794).

```python
def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000):
    n = len(scores_a)
    delta_orig = np.mean(scores_a) - np.mean(scores_b)
    count = sum(
        1 for _ in range(n_bootstrap)
        if np.mean(scores_a[idx := np.random.randint(0, n, n)]) 
         - np.mean(scores_b[idx]) >= 2 * delta_orig
    )
    return count / n_bootstrap
```

For 500 queries, **2+ percentage-point differences** are very likely significant at p<0.01. Report 95% confidence intervals alongside point estimates.

### Visualization

Present ablation results as a **heatmap** (components × metrics, color-coded by % change from baseline) for the primary view, supplemented by a **latency vs. quality scatter plot** showing the Pareto frontier across configs. For presentations, use a spider/radar chart with each axis as a metric and each polygon as an ablation config.

### Ablation config format

```yaml
configs:
  - id: "baseline"
    name: "Full Pipeline"
    overrides: {}
  - id: "no_colbert"
    name: "ColBERT Disabled"
    overrides: {colbert_rerank_enabled: false}
  - id: "no_bge_reranker"
    name: "BGE Reranker Disabled"
    overrides: {bge_reranker_enabled: false}
  - id: "no_reranking"
    name: "All Reranking Disabled (Negative Control)"
    overrides: {colbert_rerank_enabled: false, bge_reranker_enabled: false}
  - id: "rrf_1_1"
    name: "RRF Equal Weights"
    overrides: {rrf_weights: {bm25: 1, dense: 1}}
  - id: "rrf_1_3"
    name: "RRF Dense-Heavy"
    overrides: {rrf_weights: {bm25: 1, dense: 3}}
  - id: "forced_search"
    name: "Forced Search"
    overrides: {forced_search: true}
  - id: "all_tools"
    name: "All Tools Visible"
    overrides: {tool_visibility: "all"}
  - id: "no_query_injection"
    name: "No Original Query Injection"
    overrides: {original_query_injection: false}
  - id: "single_query"
    name: "Multi-Query Disabled"
    overrides: {multi_query_enabled: false}
  - id: "minimal"
    name: "Minimal Pipeline (Negative Control)"
    overrides: {colbert_rerank_enabled: false, bge_reranker_enabled: false, multi_query_enabled: false, original_query_injection: false}
```

---

## Section 7: Eval pipeline integration — architecture, execution, and reporting

### Architecture

```
EVAL ORCHESTRATOR (eval_runner.py)
├── Config Loader (reads ablation_configs.yaml)
├── Question Loader (reads questions.jsonl)
├── Execution Engine
│   ├── SSE Client (httpx async streaming)
│   ├── Event Collector
│   │   ├── tool_invoked events → tool sequence tracking
│   │   ├── search_results events → retrieval metrics
│   │   ├── context_composed events → context analysis
│   │   ├── answer_chunk events → answer assembly
│   │   └── citations events → citation tracking
│   └── Robustness Runner (N runs × M configs per question)
├── Metrics Calculator
│   ├── Retrieval metrics (recall@K, MRR, NDCG)
│   ├── LLM Judge (local Qwen3 multi-criteria)
│   ├── Tool selection accuracy (all 5 metrics)
│   ├── Robustness (NDR, RSR, ROR)
│   └── Latency breakdown
├── Statistical Analyzer
│   ├── Paired bootstrap tests
│   ├── Confidence intervals
│   └── Before/after comparison
└── Report Writer (JSON + markdown + visualizations)
```

**SSE event tracking** captures tool_invoked events to reconstruct the actual tool sequence without modifying the agent code:

```python
async def evaluate_query(question, config):
    tools_invoked, answer_chunks = [], []
    start = time.time()
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{config['base_url']}/query",
            json={"query": question, **config.get("overrides", {})}) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event["type"] == "tool_invoked":
                        tools_invoked.append(event["tool_name"])
                    elif event["type"] == "answer_chunk":
                        answer_chunks.append(event["text"])
    return {
        "answer": "".join(answer_chunks),
        "tools_invoked": tools_invoked,
        "latency": time.time() - start
    }
```

### Output JSON report format

```json
{
  "eval_report": {
    "metadata": {
      "eval_id": "eval_20260324_143022_ablation_colbert_off",
      "timestamp": "2026-03-24T14:30:22Z",
      "duration_seconds": 21500,
      "rag_app_version": "1.4.2",
      "model": "Qwen3-30B-A3B",
      "judge_model": "Qwen3-30B-A3B",
      "hardware": "V100-SXM2-32GB",
      "git_commit": "a3f8c2d",
      "config_hash": "sha256:abc123"
    },
    "ablation_config": {
      "id": "no_colbert",
      "overrides": {"colbert_rerank_enabled": false},
      "baseline_ref": "eval_20260324_100000_baseline"
    },
    "dataset": {
      "total_questions": 500,
      "runs_per_question": 3,
      "total_inference_calls": 1500,
      "categories": {"factual": 180, "temporal": 95, "comparative": 80, "negative": 65, "future_baseline": 80}
    },
    "aggregate_metrics": {
      "retrieval": {
        "recall_at_5": {"mean": 0.782, "ci_95": [0.741, 0.823]},
        "recall_at_10": {"mean": 0.856, "ci_95": [0.821, 0.891]},
        "mrr": {"mean": 0.694, "ci_95": [0.645, 0.743]}
      },
      "generation": {
        "factual_correctness": {"mean": 0.73, "ci_95": [0.69, 0.77]},
        "usefulness": {"mean": 1.62, "ci_95": [1.55, 1.69]},
        "citation_grounding": {"mean": 0.81, "ci_95": [0.77, 0.85]},
        "completeness": {"mean": 1.44, "ci_95": [1.38, 1.50]},
        "refusal_accuracy": {"mean": 0.92, "ci_95": [0.87, 0.97]},
        "composite_score": {"mean": 0.71, "ci_95": [0.67, 0.75]}
      },
      "tool_selection": {
        "key_tool_accuracy": 0.89,
        "tool_f1": 0.91,
        "exact_sequence_match": 0.72,
        "negative_test_pass_rate": 0.85,
        "visibility_accuracy": 0.94
      },
      "robustness": {
        "NDR": 0.87,
        "RSR": 0.82,
        "ROR": 0.94
      },
      "system": {
        "latency_p50_s": 35.2,
        "latency_p95_s": 52.1,
        "error_rate": 0.004
      }
    },
    "comparison_to_baseline": {
      "baseline_eval_id": "eval_20260324_100000_baseline",
      "deltas": {
        "recall_at_5": {"delta": -0.038, "p_value": 0.003, "significant": true},
        "factual_correctness": {"delta": -0.05, "p_value": 0.001, "significant": true},
        "latency_p50_s": {"delta": -4.9, "note": "improvement (faster)"}
      },
      "statistical_test": "paired_bootstrap",
      "n_bootstrap": 10000,
      "regressions": [{"question_id": "q_142", "baseline_score": 0.9, "experiment_score": 0.3}]
    },
    "per_question_results": "see per_question.jsonl"
  }
}
```

### Five execution modes

```python
@click.command()
@click.option("--mode", type=click.Choice(["full", "quick", "ablation", "robustness", "compare"]))
@click.option("--config", default="ablation_configs.yaml")
@click.option("--subset", default=None, type=int)
@click.option("--ablation-id", default=None)
@click.option("--compare-a", default=None)
@click.option("--compare-b", default=None)
def run_eval(mode, config, subset, ablation_id, compare_a, compare_b): ...
```

**full** — All 500 questions × 3 runs × baseline config. ~16.7 hours. Run after major changes.

**quick** — Stratified 100-question subset × 1 run. ~67 minutes. Daily smoke test during development.

**ablation** — All configs (or specific --ablation-id) × full dataset. Phase 1 quick-screen: ~9 hours. Phase 2 deep: ~67 hours.

**robustness** — NDR+RSR+ROR protocol from Section 4. ~67 hours for full suite, ~13 hours for RSR-only quick check.

**compare** — Loads two existing report JSONs, computes paired bootstrap deltas with p-values, identifies per-question regressions, generates comparison markdown.

### Storage and versioning

```
eval_results/
├── reports/
│   ├── 20260324_100000_baseline/
│   │   ├── report.json
│   │   ├── summary.md
│   │   ├── per_question.jsonl
│   │   ├── config_snapshot.yaml
│   │   └── visualizations/
│   ├── 20260324_143022_no_colbert/
│   └── comparisons/
│       └── baseline_vs_no_colbert.json
├── datasets/
│   ├── questions_v3.jsonl
│   └── questions_v3_quick100.jsonl
├── configs/
│   └── ablation_configs.yaml
└── index.json  # Registry of all eval runs with key metrics
```

Git-track configs, dataset files, and index.json. Store large per_question.jsonl locally (gitignored). Each report directory includes a frozen config snapshot for reproducibility. The index.json serves as a lightweight registry enabling `--compare` mode to find runs by ID.

---

## Conclusion: the three highest-leverage actions

The entire pipeline rests on three architectural decisions that compound in value. **First**, decompose-then-verify factual scoring transforms the Yandex "LLM judge doesn't work" problem into a solved NLI classification problem — this alone should move your effective recall from the current 0.167 (strict match) to ~0.70+ (claim-level F1 against expected answers), matching your ad-hoc LLM judge observations but with auditability and reproducibility. **Second**, separating key tool accuracy from scaffold tool accuracy reveals the true diagnostic signal — your system's quality hinges on whether temporal_search vs search vs channel_search is selected correctly, not on whether query_plan was called. **Third**, the phased ablation design (100-question screen → 500-question deep eval on top 3 components) delivers 80% of the insight in 20% of the compute time, critical when each query takes 40 seconds on a single V100.

The dataset should be grown iteratively: start with 100 hand-crafted golden questions (20 per core category), expand to 450–500 via the synthetic pipeline, verify 20–30% manually, and use Dawid-Skene aggregation for disagreements. Robustness testing adds ~67 hours of compute but surfaces failure modes invisible to standard eval — particularly RSR, which may reveal that your ColBERT+bge-reranker double-reranking is the primary reason your system tolerates larger k values without "lost in the middle" degradation.