# SPEC-RAG-23: NDR/RSR/ROR Robustness — Bypass Pipeline

> Статус: DRAFT v2 (детализированная)
> Автор: Claude + Human
> Дата: 2026-04-01
> Research base: R15 (Yandex RAG conf — Соколов), R20 (Cao et al. 2025), R29 (comprehensive metrics)
> Зависимости: calibrate_coverage.py (retrieval), llama-server (LLM), gpu_server.py (embed/colbert), eval_golden_v2.json

---

## Цель

Измерить три структурных свойства retrieval pipeline:

1. **NDR** — retrieval помогает или мешает?
2. **RSR** — больше документов = лучше? Монотонность.
3. **ROR** — порядок документов влияет на ответ?

**Подход**: bypass script — прямые вызовы retriever + LLM, без agent loop.

---

## Реализация: scripts/evaluate_ndr_rsr_ror.py

### Структура скрипта (4 фазы)

```python
def main():
    # Phase 1: Load dataset + init sparse model
    # Phase 2: Retrieve (once per question, cache)
    # Phase 3: Generate (vary k / ordering per question)
    # Phase 4: Score + compute metrics + report
```

---

### Phase 1: Load dataset

```python
import json
from pathlib import Path

def load_golden(path: Path) -> list[dict]:
    """Загрузить golden_v2, вернуть список questions."""
    data = json.load(path.open("r", encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("questions", data)
```

Фильтрация по тестам:
- **NDR**: все 36 Qs (все eval_modes — parametric baseline интересен для всех)
- **RSR/ROR**: только `eval_mode == "retrieval_evidence"` (остальные не используют varying k)

```python
RETRIEVAL_MODES = {"retrieval_evidence"}

def filter_for_test(questions: list[dict], test: str) -> list[dict]:
    if test == "ndr":
        return questions  # все
    return [q for q in questions if q.get("eval_mode") in RETRIEVAL_MODES]
```

### Phase 2: Retrieve (reuse из calibrate_coverage.py)

**Прямой import** трёх функций из `calibrate_coverage.py`:

```python
# Добавить scripts/ в path
sys.path.insert(0, str(Path(__file__).parent))
from calibrate_coverage import embed_query, colbert_encode, search_qdrant
```

Также нужен sparse model (fastembed BM25):

```python
from fastembed import SparseTextEmbedding
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")
```

**Retrieve + cache**:

```python
def retrieve_and_cache(
    questions: list[dict],
    embedding_url: str,
    qdrant_url: str,
    collection: str,
    sparse_model,
    cache: dict,       # checkpoint["retrieval_cache"]
) -> dict:
    """Retrieve top-20 для каждого question. Skip already cached."""
    for q in questions:
        qid = q["id"]
        if qid in cache:
            continue
        query = q["query"]
        points = search_qdrant(
            query, embedding_url, qdrant_url, collection, sparse_model,
            use_colbert=True, top_k=20,
        )
        # Извлечь текст и metadata из points
        docs = []
        for p in points:
            payload = p.get("payload", {})
            docs.append({
                "id": str(p.get("id", "")),
                "text": payload.get("text", ""),
                "channel": payload.get("channel", ""),
                "score": p.get("score", 0),
            })
        cache[qid] = docs
    return cache
```

Результат: `cache[qid]` = list of 20 docs, отсортированных ColBERT score (production order).

### Phase 3: Generate (LLM calls)

**LLM client** — прямой HTTP к llama-server (OpenAI-compatible):

```python
import urllib.request

def call_llm(
    messages: list[dict],
    llm_url: str = "http://localhost:8080",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 42,
) -> str:
    """Прямой вызов llama-server /v1/chat/completions."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return resp["choices"][0]["message"]["content"]
```

**Prompt builder** — два варианта (с docs и без):

```python
SYSTEM_WITH_DOCS = (
    "Ты — помощник для поиска информации по AI/ML новостям из Telegram-каналов.\n"
    "Отвечай на русском языке. Опирайся ТОЛЬКО на предоставленные документы.\n"
    "Если в документах нет ответа — скажи \"информация не найдена в предоставленных документах\"."
)

SYSTEM_NO_DOCS = (
    "Ты — помощник для поиска информации по AI/ML новостям.\n"
    "Отвечай на русском языке на основе своих знаний.\n"
    "Если не знаешь ответа — скажи честно."
)

MAX_DOC_CHARS = 800  # truncate каждый doc до 800 символов (как compose_context)

def build_messages(query: str, docs: list[dict]) -> list[dict]:
    """Построить messages для LLM. docs=[] → parametric mode."""
    if not docs:
        return [
            {"role": "system", "content": SYSTEM_NO_DOCS},
            {"role": "user", "content": query},
        ]

    doc_parts = []
    for i, doc in enumerate(docs, 1):
        text = doc["text"][:MAX_DOC_CHARS]
        channel = doc.get("channel", "unknown")
        doc_parts.append(f"[{i}] ({channel}) {text}")
    docs_text = "\n\n".join(doc_parts)

    user_content = f"Документы:\n{docs_text}\n\nВопрос: {query}"
    return [
        {"role": "system", "content": SYSTEM_WITH_DOCS},
        {"role": "user", "content": user_content},
    ]
```

**Условия генерации**:

```python
import random

def get_conditions(test: str, docs: list[dict]) -> list[tuple[str, list[dict]]]:
    """Вернуть list of (condition_name, docs_for_condition)."""
    if test == "ndr":
        return [
            ("k=0", []),
            ("k=20", docs),
        ]
    elif test == "rsr":
        return [
            ("k=3", docs[:3]),
            ("k=5", docs[:5]),
            ("k=10", docs[:10]),
            ("k=20", docs[:20]),
        ]
    elif test == "ror":
        reversed_docs = list(reversed(docs))
        shuffled = docs.copy()
        random.Random(42).shuffle(shuffled)
        return [
            ("original", docs),
            ("reversed", reversed_docs),
            ("shuffled", shuffled),
        ]
    return []
```

**Generation loop** (с checkpoint):

```python
def run_generation(
    questions: list[dict],
    retrieval_cache: dict,
    test: str,
    llm_url: str,
    checkpoint: dict,
    checkpoint_path: Path,
) -> None:
    """Generate answers для всех (question, condition) пар. Saves to checkpoint."""
    filtered = filter_for_test(questions, test)

    for q in filtered:
        qid = q["id"]
        docs = retrieval_cache.get(qid, [])
        conditions = get_conditions(test, docs)

        for cond_name, cond_docs in conditions:
            key = f"{qid}:{test}:{cond_name}"
            if key in checkpoint.get("generations", {}):
                continue  # already done

            messages = build_messages(q["query"], cond_docs)
            t0 = time.time()
            answer = call_llm(messages, llm_url)
            latency = time.time() - t0

            checkpoint.setdefault("generations", {})[key] = {
                "answer": answer,
                "latency_sec": round(latency, 1),
                "condition": cond_name,
                "n_docs": len(cond_docs),
            }
            save_checkpoint(checkpoint_path, checkpoint)
            logger.info("  %s | %s | %s | %.1fs", qid, test, cond_name, latency)
```

### Phase 4: Score + Compute Metrics

**BERTScore scoring**:

```python
def score_all(
    questions: list[dict],
    generations: dict,    # checkpoint["generations"]
    test: str,
) -> dict[str, dict[str, float]]:
    """BERTScore F1 для всех (question, condition)."""
    # Lazy init BERTScorer (same as evaluate_agent.py)
    from bert_score import BERTScorer
    scorer = BERTScorer(
        model_type="ai-forever/ruBert-large",
        num_layers=18, idf=False, lang="ru",
        rescale_with_baseline=False,
    )
    scorer._tokenizer.model_max_length = 512  # fix OverflowError

    scores = {}  # key → float
    filtered = filter_for_test(questions, test)

    for q in filtered:
        qid = q["id"]
        expected = q.get("expected_answer", "")
        if not expected:
            continue
        conditions = get_conditions(test, [])  # just need names
        # Reconstruct condition names
        for cond_name, _ in get_conditions(test, [{}] * 20):
            key = f"{qid}:{test}:{cond_name}"
            gen = generations.get(key, {})
            answer = gen.get("answer", "")
            if not answer:
                continue
            _, _, F1 = scorer.score([answer], [expected])
            scores[key] = round(float(F1[0]), 4)

    return scores
```

**NDR computation**:

```python
def compute_ndr(questions: list, scores: dict) -> dict:
    """NDR = fraction where RAG ≥ no-RAG."""
    hits = 0
    total = 0
    per_question = []

    for q in questions:
        qid = q["id"]
        s_rag = scores.get(f"{qid}:ndr:k=20")
        s_no_rag = scores.get(f"{qid}:ndr:k=0")
        if s_rag is None or s_no_rag is None:
            continue
        total += 1
        hit = s_rag >= s_no_rag
        if hit:
            hits += 1
        per_question.append({
            "id": qid,
            "score_rag": s_rag,
            "score_no_rag": s_no_rag,
            "delta": round(s_rag - s_no_rag, 4),
            "ndr_hit": hit,
        })

    return {
        "ndr": round(hits / total, 4) if total else None,
        "hits": hits,
        "total": total,
        "per_question": sorted(per_question, key=lambda x: x["delta"]),
    }
```

**RSR computation**:

```python
def compute_rsr(questions: list, scores: dict, epsilon: float = 0.02) -> dict:
    """RSR = fraction where scores are monotonically non-decreasing across k."""
    K_VALUES = [3, 5, 10, 20]
    hits = 0
    total = 0
    violations = []

    filtered = filter_for_test(questions, "rsr")
    for q in filtered:
        qid = q["id"]
        k_scores = []
        for k in K_VALUES:
            s = scores.get(f"{qid}:rsr:k={k}")
            k_scores.append(s)

        if any(s is None for s in k_scores):
            continue
        total += 1

        monotonic = True
        for i in range(1, len(k_scores)):
            if k_scores[i] < k_scores[i-1] - epsilon:
                monotonic = False
                violations.append({
                    "id": qid,
                    "k_from": K_VALUES[i-1],
                    "k_to": K_VALUES[i],
                    "score_from": k_scores[i-1],
                    "score_to": k_scores[i],
                    "drop": round(k_scores[i-1] - k_scores[i], 4),
                })
        if monotonic:
            hits += 1

    return {
        "rsr": round(hits / total, 4) if total else None,
        "hits": hits,
        "total": total,
        "violations": violations,
    }
```

**ROR computation**:

```python
import statistics

def compute_ror(questions: list, scores: dict) -> dict:
    """ROR = mean(1 - 2σ) across questions."""
    ORDERINGS = ["original", "reversed", "shuffled"]
    ror_values = []

    filtered = filter_for_test(questions, "ror")
    for q in filtered:
        qid = q["id"]
        order_scores = []
        for o in ORDERINGS:
            s = scores.get(f"{qid}:ror:{o}")
            if s is not None:
                order_scores.append(s)

        if len(order_scores) < 2:
            continue
        sigma = statistics.stdev(order_scores) if len(order_scores) > 1 else 0.0
        ror_q = max(0.0, 1.0 - 2 * sigma)
        ror_values.append({"id": qid, "scores": order_scores, "sigma": round(sigma, 4), "ror": round(ror_q, 4)})

    mean_ror = sum(r["ror"] for r in ror_values) / len(ror_values) if ror_values else None
    return {
        "ror": round(mean_ror, 4) if mean_ror else None,
        "per_question": sorted(ror_values, key=lambda x: x["ror"]),
    }
```

---

### Checkpoint schema

```json
{
  "started_at": "2026-04-01T12:00:00",
  "config": {
    "dataset": "datasets/eval_golden_v2.json",
    "collection": "news_colbert_v2",
    "tests": ["ndr", "rsr", "ror"]
  },
  "retrieval_cache": {
    "golden_q01": [
      {"id": "uuid", "text": "...", "channel": "ch", "score": 0.95},
      ...
    ]
  },
  "generations": {
    "golden_q01:ndr:k=0": {
      "answer": "...",
      "latency_sec": 12.3,
      "condition": "k=0",
      "n_docs": 0
    },
    "golden_q01:ndr:k=20": { ... },
    "golden_q01:rsr:k=3": { ... },
    ...
  }
}
```

Key = `{question_id}:{test}:{condition}`. При resume — skip existing keys.

---

### Report schema (output JSON)

```json
{
  "metadata": {
    "timestamp": "2026-04-01T14:00:00",
    "dataset": "eval_golden_v2.json",
    "total_questions": 36,
    "retrieval_questions": 17,
    "total_llm_calls": 160,
    "total_time_sec": 5400,
    "scoring": "bertscore_f1_rubert_large_l18"
  },
  "ndr": {
    "rate": 0.86,
    "hits": 31,
    "total": 36,
    "per_question": [ ... ]
  },
  "rsr": {
    "rate": 0.76,
    "hits": 13,
    "total": 17,
    "violations": [ {"id": "q06", "k_from": 10, "k_to": 20, "drop": 0.05} ],
  },
  "ror": {
    "rate": 0.82,
    "per_question": [ ... ]
  },
  "composite": 0.81
}
```

### Report (markdown)

```markdown
# NDR/RSR/ROR Robustness Report

**Date**: 2026-04-01
**Dataset**: eval_golden_v2.json (36 Qs, 17 retrieval)
**Scoring**: BERTScore F1 (ruBert-large, layer 18)

## Summary
| Metric | Value | Interpretation |
|--------|-------|----------------|
| NDR | 0.86 | Retrieval helps in 86% cases |
| RSR | 0.76 | Monotonic in 76% cases |
| ROR | 0.82 | Order-robust in 82% cases |
| Composite | 0.81 | |

## NDR Details (36 Qs)
| Question | score(RAG) | score(no-RAG) | Δ | Hit |
| ...

## RSR Monotonicity Violations
| Question | k_from | k_to | score_from | score_to | Drop |
| ...

## ROR Per-Question (17 Qs)
| Question | original | reversed | shuffled | σ | ROR |
| ...
```

---

### Judge artifact (для Claude manual scoring)

Export **20-30 worst-delta pairs** (where NDR fails or RSR violates monotonicity):

```markdown
# Robustness Judge Artifact

## Question: golden_q06
**Query**: Что обсуждалось в AI-каналах в январе 2026?
**Expected**: ...

### Condition A: k=0 (no retrieval)
**Answer A**: [LLM answer without docs]

### Condition B: k=20 (with retrieval)
**Answer B**: [LLM answer with docs]

### Your assessment:
- Factual A (0/0.5/1.0): ___
- Factual B (0/0.5/1.0): ___
```

---

### CLI (argparse)

```python
parser = argparse.ArgumentParser(description="NDR/RSR/ROR robustness eval (bypass)")
parser.add_argument("--dataset", type=Path, default=Path("datasets/eval_golden_v2.json"))
parser.add_argument("--collection", default="news_colbert_v2")
parser.add_argument("--qdrant-url", default="http://localhost:16333")
parser.add_argument("--embedding-url", default="http://localhost:8082")
parser.add_argument("--llm-url", default="http://localhost:8080")
parser.add_argument("--tests", nargs="*", default=["ndr", "rsr", "ror"],
                    choices=["ndr", "rsr", "ror"])
parser.add_argument("--output", type=Path, default=Path("results/robustness"))
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--max-doc-chars", type=int, default=800)
parser.add_argument("--llm-max-tokens", type=int, default=1024)
parser.add_argument("--llm-temperature", type=float, default=0.0)
parser.add_argument("--llm-seed", type=int, default=42)
parser.add_argument("--rsr-epsilon", type=float, default=0.02,
                    help="Tolerance для monotonicity check")
```

---

## Compute Budget (уточнённый)

| Test | Questions | Conditions | LLM Calls | Reuse | Unique Calls |
|------|-----------|------------|-----------|-------|-------------|
| NDR | 36 | k=0, k=20 | 72 | k=20 reused in RSR | 36 (only k=0 new) |
| RSR | 17 | k=3, k=5, k=10, k=20 | 68 | k=20 reused from NDR | 51 (k=3,5,10 new) |
| ROR | 17 | orig, reversed, shuffled | 51 | orig = RSR k=20 | 34 (rev+shuffle new) |
| **Total** | | | **191** | | **~121 unique** |

При 100 calls/hour → **~1.2 часа** compute. Retrieval: 36 calls × ~2s = ~1 min.

---

## Acceptance Criteria

- [ ] `evaluate_ndr_rsr_ror.py` runs end-to-end: `python scripts/evaluate_ndr_rsr_ror.py --dataset datasets/eval_golden_v2.json`
- [ ] Retrieval: import `search_qdrant` из `calibrate_coverage.py`, top-20 cached per question
- [ ] LLM: direct `urllib.request` к `llama-server /v1/chat/completions`, `enable_thinking: False`, `seed: 42`
- [ ] NDR: k=0 (system prompt без docs) vs k=20, BERTScore comparison, rate computed
- [ ] RSR: k=[3,5,10,20], monotonicity check with ε=0.02, violations listed
- [ ] ROR: original/reversed/shuffled(seed=42), σ per-question, mean ROR
- [ ] Checkpoint: JSON at `{output}/checkpoint.json`, key=`{qid}:{test}:{condition}`, resume skips existing
- [ ] BERTScore: `ai-forever/ruBert-large` layer 18, `model_max_length=512` patch
- [ ] Report JSON: matches schema выше (ndr/rsr/ror/composite/per_question)
- [ ] Report markdown: summary table + per-question details + violations
- [ ] Judge artifact: top 20-30 worst-delta pairs in markdown for Claude manual review

---

## Что НЕ входит

- Agent API integration — bypass only
- Query perturbation — SPEC-RAG-22 (отдельная задача)
- BERTScore/SummaC в evaluate_agent.py — уже реализовано в SPEC-RAG-22
- Automated Claude judge — manual chat only
- Multiple retrievers comparison — single pipeline (production config)
- k > 20 — ColBERT возвращает top-20, это наш ceiling

---

## Ссылки

- R15: Yandex RAG conf — NDR/RSR/ROR importance
- R20: Cao et al. (2025) arXiv:2505.21870 — protocol
- R29: comprehensive RAG eval metrics
- DEC-0045: CE confidence filter
- `scripts/calibrate_coverage.py` lines 85-169: `embed_query()`, `colbert_encode()`, `search_qdrant()`
- `src/adapters/llm/llama_server_client.py` lines 129-162: LLM call format reference
