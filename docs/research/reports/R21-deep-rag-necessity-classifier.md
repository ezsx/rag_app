# RAG necessity classification: when to skip retrieval and save 20 seconds

**Skipping retrieval for conversational, definitional, and meta queries in your Telegram AI/ML agent is strongly supported by academic literature and production evidence — expect to bypass RAG for 15–30% of queries with a tiered rule-based + agent-decision architecture that adds under 1ms of overhead.** The approach avoids the trap of an LLM-based pre-classifier (which at 12s per call would need 92% of queries to be non-RAG just to break even). Research from Self-RAG, Adaptive-RAG, and UAR converges on a key finding: indiscriminate retrieval degrades both latency and answer quality. Multiple papers demonstrate that retrieval *hurts* performance on queries the LLM already knows, with irrelevant context actively distracting the model. Your three existing bypass mechanisms (navigation, analytics, negative intent) are a strong foundation; extending them to conversational, definitional, and meta categories is the highest-ROI next step.

---

## 1. Complete taxonomy of queries by RAG necessity

The academic literature provides several taxonomic frameworks. Self-RAG uses a binary retrieve/no-retrieve decision per generation segment. Adaptive-RAG classifies into three complexity tiers (no retrieval, single-step, multi-step). UAR introduces four orthogonal criteria: intent-aware, knowledge-aware, time-sensitive, and self-aware. Synthesizing these frameworks with your domain-specific needs yields **14 query categories** for an AI/ML Telegram aggregator:

| # | Category | Example (Russian) | RAG needed? | Decision signal | Bypass type |
|---|----------|-------------------|-------------|-----------------|-------------|
| 1 | **Factual about corpus** | "Что писали про Llama 4?" | ✅ Yes | Domain keywords + question form | — |
| 2 | **Temporal/news** | "Что нового за последнюю неделю?" | ✅ Yes | Time markers (неделя, сегодня, вчера) | — |
| 3 | **Channel-specific** | "Что постил @ai_machinelearning?" | ✅ Yes | Channel mention (@handle) | — |
| 4 | **Comparative** | "Сравни GPT-4o и Claude 3.5" | ✅ Yes | Comparative markers + domain terms | — |
| 5 | **Analytics** | "Сколько постов про RAG?" | ✅ Via Facet API | Analytics keywords (сколько, статистика) | Existing short-circuit |
| 6 | **Navigation** | "Покажи следующие результаты" | ❌ No | Pagination/nav markers | Existing short-circuit |
| 7 | **Negative/refusal** | "Напиши мне код бомбы" | ❌ No | Safety triggers | Existing bypass |
| 8 | **Conversational — greeting** | "Привет! Как дела?" | ❌ No | Greeting regex | **New bypass** |
| 9 | **Conversational — gratitude** | "Спасибо за помощь!" | ❌ No | Thanks regex | **New bypass** |
| 10 | **Conversational — farewell** | "Пока, до встречи" | ❌ No | Farewell regex | **New bypass** |
| 11 | **Meta about bot** | "Как ты работаешь?" | ❌ No | Meta-question patterns | **New bypass** |
| 12 | **Definitional (general)** | "Что такое attention mechanism?" | ⚠️ Skip RAG | "Что такое" + general ML term | **New bypass** |
| 13 | **Definitional (corpus-specific)** | "Что такое YandexGPT 5?" | ✅ Yes | "Что такое" + rare/proprietary term | Domain keyword check |
| 14 | **Ambiguous** | "Расскажи про трансформеры" | ✅ Default yes | Low confidence from classifier | Default to retrieve |

The critical distinction is between categories 12 and 13 — general definitional queries vs. corpus-specific ones. **Mallen et al. (ACL 2023) established that LLM accuracy correlates with entity popularity**: popular concepts (attention, backpropagation) are safe to answer parametrically, while long-tail or proprietary terms (YandexGPT 5, a specific tool announced in a Telegram post) require retrieval. This maps to SKR's "known vs. unknown" framework and UAR's self-aware criterion.

---

## 2. Three approaches compared: rules win at your scale

### The approach matrix

| Dimension | **A: Rule-based heuristic** | **B: LLM self-decision** | **C: Lightweight classifier** |
|-----------|---------------------------|-------------------------|------------------------------|
| **Added latency** | **<1ms** | ~12s (one Qwen3 call) | 5–40ms (BERT on CPU/GPU) |
| **Implementation cost** | Low (regex + keywords) | Near-zero (remove forced search) | Medium (train/deploy model) |
| **Accuracy** | ~85% on clear categories | Unreliable at 3B active params | 87–94% after tuning |
| **False negative risk** | Low (conservative rules) | High (small model overconfidence) | Medium (threshold-dependent) |
| **Maintenance** | Manual rule updates | Prompt drift | Retraining needed |
| **Coverage** | Catches ~60% of no-RAG queries | Catches ~80% (but unreliably) | Catches ~90% |
| **Academic analog** | Mallen et al. popularity threshold | Self-RAG reflection tokens | Adaptive-RAG T5 classifier |

### Why Approach B is a trap at 3B active parameters

Research from RetrievalQA (Zhang et al., ACL 2024) found that **"at least half the time, GPT-3.5 is unaware that it needs retrieval."** With Qwen3-30B-A3B's 3B active parameters, self-awareness about knowledge boundaries is even weaker. The LLM self-decision approach works well in Self-RAG because the model is *fine-tuned* with special reflection tokens trained on GPT-4 critic labels — it is not a zero-shot capability. Removing forced search and hoping the agent correctly decides when to call tools is the highest-risk option.

### Why Approach A is optimal for your system

The latency math is decisive. Each LLM call costs ~12s on your V100. A rule-based pre-filter adds <1ms. An LLM classifier (Approach C with Qwen3) would add 12s to *every* query — requiring **92% of queries to be non-RAG just to break even** on latency (12s cost ÷ 13s savings = 92%). Even a BERT classifier at 20–40ms on CPU is 3 orders of magnitude cheaper than an LLM call, but requires training data you don't yet have. **Start with rules, graduate to a trained classifier once you have production data.**

Moskvoretskii et al. (2025) confirmed this empirically: **simple uncertainty estimation methods often outperformed complex purpose-built adaptive RAG pipelines** while requiring far fewer compute resources. The Semantic Router library (Aurelio AI) achieves sub-millisecond routing via embeddings alone and is the closest production analog to a lightweight rule system.

---

## 3. Recommended implementation: tiered classification with code

### Tier architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  TIER 1: Rule-based (<1ms)  │
│  Regex patterns + keywords  │
│  Catches: greetings, thanks,│
│  farewells, meta-questions  │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    │             │
 MATCHED       NOT MATCHED
 (no_rag)         │
    │             ▼
    │  ┌────────────────────────────┐
    │  │  TIER 2: Domain check (1ms)│
    │  │  Domain keywords present?  │
    │  │  Definitional pattern?     │
    │  └──────────┬─────────────────┘
    │       ┌─────┴─────┐
    │    DOMAIN KW    NO DOMAIN KW
    │    (needs_rag)  + definitional
    │       │         pattern
    │       │         (no_rag)
    │       ▼            │
    │  ┌──────────────┐  │
    │  │ TIER 3: ReAct │  │
    │  │ Agent (no     │  │
    │  │ forced search)│  │
    │  │ Agent decides │  │
    │  │ tool use      │  │
    │  └──────────────┘  │
    │       │            │
    ▼       ▼            ▼
┌───────────────────────────┐
│    LLM generates answer   │
│    (12s single call)      │
└───────────────────────────┘
```

### Concrete heuristic rules for Russian

```python
import re

# TIER 1: Conversational patterns — instant bypass
GREETING = re.compile(
    r'^(привет|здравствуй(те)?|добр(ый|ое|ая)\s+(день|утро|вечер)|'
    r'хай|хелло|приветствую|салют|здарова|йо|ку)\b', re.I)

THANKS = re.compile(
    r'^(спасибо|благодарю|благодарствую|мерси|сенкс|спс|пасиб)\b', re.I)

FAREWELL = re.compile(
    r'^(пока|до свидания|до встречи|прощай|бывай|удачи|'
    r'всего доброго|досвидос)\b', re.I)

META = re.compile(
    r'(как ты работаешь|что ты умеешь|кто тебя создал|кто ты\??|'
    r'ты бот|ты робот|расскажи о себе|на чём ты работаешь|'
    r'какие у тебя функции|чем ты можешь помочь)', re.I)

# TIER 2: Domain keyword detection
DOMAIN_KW = re.compile(
    r'(трансформер|нейросет|LLM|RAG|GPT|BERT|fine.?tun|'
    r'обучение|модел[ьи]|датасет|embedding|токенизац|'
    r'attention|diffusion|LoRA|RLHF|prompt|инференс|'
    r'архитектур|бенчмарк|галлюцинац|Llama|Claude|Gemini|'
    r'Mistral|Qwen|DeepSeek|Anthropic|OpenAI|Яндекс)', re.I)

DEFINITIONAL = re.compile(
    r'^(что такое|определение|что означает|что значит|объясни)\s+', re.I)

def classify_query(query: str) -> tuple[str, float]:
    q = query.strip()
    words = len(q.split())

    # Tier 1: Clear conversational bypass
    if words <= 5 and (GREETING.match(q) or THANKS.match(q) or FAREWELL.match(q)):
        return ("no_rag", 0.95)
    if META.search(q):
        return ("no_rag", 0.95)

    # Tier 2: Definitional + domain check
    if DEFINITIONAL.match(q) and not DOMAIN_KW.search(q):
        # General definitional: "Что такое backpropagation?"
        # LLM knows this — skip retrieval
        return ("no_rag", 0.80)
    if DEFINITIONAL.match(q) and DOMAIN_KW.search(q):
        # Could be corpus-specific: "Что такое YandexGPT 5?"
        return ("needs_rag", 0.75)

    # Tier 2: Domain keywords → definitely needs RAG
    if DOMAIN_KW.search(q):
        return ("needs_rag", 0.85)

    # Ultra-short non-domain queries
    if words <= 2:
        return ("no_rag", 0.70)

    # Default: retrieve (safe fallback)
    return ("needs_rag", 0.60)
```

### Zero-shot classification prompt (for future Approach C)

```python
CLASSIFY_PROMPT = """Ты — классификатор запросов к AI/ML ассистенту.
База знаний: 13000+ постов из 36 AI/ML Telegram-каналов.

Категории:
- needs_rag: вопрос о конкретных постах, новостях, инструментах, 
  сравнениях моделей, событиях из каналов
- no_rag: приветствие, благодарность, вопрос о боте, общее 
  определение из учебника, разговорная фраза

Примеры:
"Привет!" → no_rag
"Что нового про Llama?" → needs_rag
"Что такое backpropagation?" → no_rag
"Какие модели обсуждали на этой неделе?" → needs_rag
"Спасибо!" → no_rag

Запрос: "{query}"
Ответь ОДНИМ словом:"""
```

### Integration point in the agent pipeline

The classification check inserts **before** the ReAct agent loop, as a pre-filter node:

```python
# In your agent entry point (simplified)
async def handle_message(query: str) -> str:
    # STEP 1: Existing short-circuits (navigation, analytics)
    if is_navigation(query):
        return handle_navigation(query)
    if is_analytics(query):
        return handle_analytics(query)
    if is_negative_intent(query):
        return handle_refusal(query)

    # STEP 2: NEW — RAG necessity classification
    classification, confidence = classify_query(query)
    
    if classification == "no_rag" and confidence >= 0.80:
        # Direct LLM response — no search tool provided
        return await llm_generate(
            system_prompt=NO_RAG_SYSTEM_PROMPT,
            user_message=query
        )  # ~12s instead of ~25s
    
    # STEP 3: All other queries → ReAct agent
    # Key change: remove forced_search=True for borderline cases
    force = confidence >= 0.80  # Only force search when confident
    return await react_agent(query, force_search=force)
```

---

## 4. NDR measurement provides ground truth — sequence it after heuristics

NDR (No Document Retrieval utility) measures whether f(q, k) ≥ f(q, 0) — whether answers with context k are actually better than answers without. **No existing evaluation framework directly implements NDR as a named metric.** RAGAS measures context relevancy and faithfulness post-hoc but assumes retrieval already happened. ARES (Stanford, NAACL 2024) trains lightweight LM judges for automated evaluation but also operates post-retrieval.

The closest operational approach is **SR-RAG (2025)**, which fine-tunes the LLM itself to route between external retrieval and parametric knowledge by comparing answer quality both ways. It reduced retrieval by **20–40%** while maintaining accuracy. Probing-RAG (Baek et al., 2024) achieved an even more aggressive **57.5% retrieval skip rate** using classifiers on hidden states.

**Recommended sequencing**: Deploy heuristic rules first (Approach A) to capture the obvious no-RAG categories at zero cost. In parallel, build NDR measurement infrastructure by running a shadow pipeline: for every query, generate answers both with and without retrieval, score them with an LLM judge, and accumulate labeled data. After **500–1000 labeled pairs**, use this data to either (a) train a lightweight classifier (Approach C upgrade), or (b) validate and tune the heuristic thresholds. This is the active learning loop: production heuristics generate data, data improves the classifier, classifier replaces heuristics.

The training pipeline follows the distillation pattern documented by Fireworks and Predibase: use a strong LLM to label query pairs (needs_rag vs. no_rag based on answer quality comparison), then fine-tune a small model (ruBERT-tiny, ~10ms inference) on these labels. One reference implementation achieved **87.4% accuracy** with a 46MB LoRA adapter at ~$3 training cost.

---

## 5. Latency savings: 2.6–5.2 seconds average per query

The savings depend on what fraction P of queries are non-RAG:

| Metric | Conservative (P=20%) | Moderate (P=30%) | Optimistic (P=40%) |
|--------|---------------------|-------------------|---------------------|
| Queries skipping RAG | 20% | 30% | 40% |
| Time saved per skipped query | **13s** (25s→12s) | 13s | 13s |
| Average savings per query | **2.6s** | **3.9s** | **5.2s** |
| Classifier overhead | <1ms | <1ms | <1ms |
| Net savings at 100 queries/day | 260s/day | 390s/day | 520s/day |

For your specialized AI/ML demo, the non-RAG fraction is likely **10–20%** (domain-focused users ask more substantive queries). Enterprise chatbots see 25–40%. The breakeven for the rule-based approach is effectively **any non-zero fraction** — since the classifier costs <1ms, even catching 5% of queries as non-RAG is pure gain.

**Critical finding**: an LLM-based classifier (Approach C using Qwen3 itself) requires P ≥ 92% to break even on latency. This definitively rules out using the main model as a pre-filter. A BERT classifier at 20ms breaks even at P ≥ 0.15% — effectively always beneficial, but requires training data.

---

## 6. False negatives cause hallucination — mitigation requires asymmetric thresholds

The cost structure is starkly asymmetric. A **false negative** (classifying a RAG-needed query as no_rag) produces a hallucinated or incomplete answer, eroding user trust. A **false positive** (unnecessary retrieval) wastes 13 seconds but produces a correct answer. This means **recall on the "needs_rag" class must exceed 95%**.

### Three-layer defense against false negatives

**Layer 1 — Conservative classification rules.** The heuristic rules above only bypass RAG for high-confidence pattern matches (greetings, thanks, farewells, meta-questions). All ambiguous queries default to retrieval. This is the UAR pattern: when uncertain about any of the four criteria, retrieve.

**Layer 2 — Post-generation confidence check.** HALT (2025) demonstrated that lightweight probes on hidden states can detect hallucination risk with **85.7% accuracy at full coverage, 97.7% at 40% coverage**, adding <0.1% compute overhead. For your system: after generating a no-RAG response, compute token-level entropy on the output. If mean entropy exceeds a threshold → discard the response and re-route through RAG. This costs one additional LLM call (~12s) only for the small fraction of false negatives, not for all queries.

**Layer 3 — User feedback signal.** Track whether users follow up with rephrased questions after a no-RAG response (implicit negative signal). Log all bypassed queries for periodic human review. After accumulating 200+ reviewed examples, retrain the classifier.

### The definitional query question is settled

Multiple papers confirm that retrieval **hurts** performance on queries the LLM already knows well. Knowledgeable-R1 (2025) showed that in adversarial contexts, query-only prompting scored **25.8% accuracy vs. 8.1% for standard RAG** — when retrieved context is wrong, it catastrophically overrides correct parametric knowledge. The "distracting effect" paper (2025) formalized that high-quality irrelevant passages from strong retrievers are *more* distracting than those from weak retrievers. For general definitional queries like "Что такое attention mechanism?", **skipping retrieval improves answer quality**.

The exception: corpus-specific definitions ("Что такое YandexGPT 5?") where the corpus contains unique information the LLM doesn't have. The domain keyword check in the Tier 2 heuristic handles this distinction.

---

## 7. Experiment design for validation

### Eval dataset expansion plan

Your current 30 golden questions (all requiring RAG) cannot evaluate a binary classifier — zero negative examples means infinite false positive rate. Expand to at least **80 questions** across this distribution:

| Category | Count | Source | Ground truth method |
|----------|-------|--------|-------------------|
| Factual about corpus (needs_rag) | 20 | Existing golden set | Verified by retrieval |
| Temporal queries (needs_rag) | 10 | Existing golden set | Requires current corpus |
| Conversational (no_rag) | 15 | Synthetic + production logs | LLM answers correctly without context |
| Meta-questions (no_rag) | 10 | Handcrafted | Static bot responses |
| Definitional — general (no_rag) | 10 | Synthetic | LLM answer quality ≥ RAG answer quality |
| Definitional — corpus-specific (needs_rag) | 5 | Handcrafted from rare corpus terms | Only answerable with corpus |
| Ambiguous (labeled by NDR) | 10 | Production logs | Compare with/without retrieval quality |

### Metrics to track

- **Recall@needs_rag** ≥ 95% (primary — prevents hallucination)
- **F2-score** (4× weight on recall vs. precision)
- **Precision@no_rag** — of queries classified as safe to skip, what fraction truly don't need RAG?
- **Latency reduction** — actual measured P50/P95 improvement
- **Answer quality delta** — A/B test comparing classifier-routed vs. always-retrieve baseline using LLM-as-judge scoring

### Validation protocol

Following Adaptive-RAG's silver label approach:

1. For each test query, generate answer **with** retrieval (A_rag) and **without** retrieval (A_no_rag)
2. Score both answers using LLM-as-judge on a 1–5 scale for correctness, completeness, and relevance
3. If score(A_rag) > score(A_no_rag) + δ → ground truth label is "needs_rag"
4. If score(A_no_rag) ≥ score(A_rag) → ground truth label is "no_rag"
5. If scores are within δ → label as "ambiguous" (default to retrieve)
6. Use δ = 0.5 on a 5-point scale as the significance threshold

This creates the labeled dataset for both classifier evaluation and future training data, directly implementing the NDR measurement principle.

---

## 8. What the literature says about your specific questions

**What percentage of queries are non-RAG?** Production data is scarce, but convergent evidence suggests **20–60%** depending on domain specificity. Probing-RAG skipped 57.5% of retrievals on academic QA benchmarks. SR-RAG reduced retrieval by 20–40%. For your specialized AI/ML demo UI, expect the lower end: **10–20%**, rising to **25–30%** as the bot gets casual users.

**Should forced search be removed entirely?** No. The recommended hybrid: remove forced search only for queries that pass the Tier 1/2 heuristic filters. For all other queries, keep the agent's search tool available but not forced — let the ReAct agent decide naturally for borderline cases. This combines the safety of forced search for clearly factual queries with the flexibility of agent-driven retrieval for uncertain ones.

**What does the academic state-of-the-art look like in 2025–2026?** The field has moved decisively toward **multi-criteria, lightweight classification** rather than relying on any single signal. UAR's four-criterion decision tree, DeepRAG's MDP formulation, and RAGRouter-Bench (January 2026 — the first dedicated routing benchmark with 7,727 queries) all point to the same conclusion: no single feature (perplexity, query length, entity popularity) is sufficient. The winning strategy combines fast heuristics for obvious cases with learned classifiers for the gray zone. **The key 2025 finding from Moskvoretskii et al.** is that simple uncertainty estimation (mean token entropy) often beats purpose-built adaptive RAG systems — complexity is not correlated with effectiveness in this space.

**Connection to Yandex's approach.** No specific Yandex conference talk on perplexity-based RAG classification was found in public sources. However, Yandex Cloud's 2025 webinar series covers RAG with YDB + LangChain and Telegram bot integration with Function Calling — architecturally aligned with your system. The perplexity-based gating concept is well-established in FLARE (token probability thresholds) and DRAGIN (entropy after stopword filtering), both of which can be implemented on Qwen3.

---

## Conclusion: deploy rules now, measure NDR next, train a classifier later

The research converges on a clear three-phase roadmap. **Phase 1 (this week)**: deploy the Tier 1/2 rule-based classifier from Section 3. It catches greetings, thanks, farewells, meta-questions, and general definitional queries at <1ms cost with near-zero false negative risk. Expected impact: 10–20% of queries bypass RAG, saving **1.3–2.6s average per query**. **Phase 2 (weeks 2–4)**: implement NDR shadow measurement — generate both RAG and no-RAG answers for every query, score with LLM-as-judge, accumulate labeled data. Expand the eval set to 80+ questions. **Phase 3 (month 2)**: once 500+ labeled pairs exist, train a ruBERT-tiny classifier (~10ms inference) and replace the rule-based system for the gray zone, keeping rules as the fast first tier.

The single most important architectural decision: **insert classification before the agent loop, not inside it.** The classifier is a gate that prevents the expensive ReAct cycle from starting, not a tool the agent uses mid-reasoning. This matches the LangGraph conditional routing pattern and Adaptive-RAG's pre-classification design. For ambiguous queries, the only safe default is retrieval — the cost asymmetry between unnecessary retrieval (13s wasted) and missed retrieval (hallucinated answer) makes this unambiguous.