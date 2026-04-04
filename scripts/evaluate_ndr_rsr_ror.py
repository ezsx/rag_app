#!/usr/bin/env python3
"""NDR/RSR/ROR Robustness Evaluation — Bypass Pipeline (SPEC-RAG-23).

Прямые вызовы retriever + LLM, без agent loop. Полный контроль над k и ordering.

Измеряет:
- NDR: retrieval помогает или мешает (k=0 vs k=20)
- RSR: монотонность по k (k=3,5,10,20)
- ROR: устойчивость к порядку документов (original, reversed, shuffled)

Использование:
    python scripts/evaluate_ndr_rsr_ror.py \\
        --dataset datasets/eval_golden_v2.json \\
        --output results/robustness/

    python scripts/evaluate_ndr_rsr_ror.py --tests ndr --output results/robustness/
    python scripts/evaluate_ndr_rsr_ror.py --resume results/robustness/checkpoint.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Reuse retrieval functions из calibrate_coverage.py
sys.path.insert(0, str(Path(__file__).parent))
from calibrate_coverage import search_qdrant

# ─── Config ───────────────────────────────────────────────────────

RETRIEVAL_EVAL_MODES = {"retrieval_evidence"}
K_VALUES = [3, 5, 10, 20]
ORDERINGS = ["original", "reversed", "shuffled"]
RSR_EPSILON = 0.02  # tolerance для monotonicity

SYSTEM_WITH_DOCS = (
    "Ты — помощник для поиска информации по AI/ML новостям из Telegram-каналов.\n"
    "Отвечай на русском языке. Опирайся ТОЛЬКО на предоставленные документы.\n"
    'Если в документах нет ответа — скажи "информация не найдена в предоставленных документах".'
)

SYSTEM_NO_DOCS = (
    "Ты — помощник для поиска информации по AI/ML новостям.\n"
    "Отвечай на русском языке на основе своих знаний.\n"
    "Если не знаешь ответа — скажи честно."
)

MAX_DOC_CHARS = 800


# ─── Dataset ──────────────────────────────────────────────────────


def load_dataset(path: Path) -> list[dict]:
    """Загрузить golden_v2 dataset."""
    data = json.load(path.open("r", encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("questions", data)


def filter_for_test(questions: list[dict], test: str) -> list[dict]:
    """NDR = все вопросы, RSR/ROR = только retrieval_evidence."""
    if test == "ndr":
        return questions
    return [q for q in questions if q.get("eval_mode") in RETRIEVAL_EVAL_MODES]


# ─── Checkpoint ───────────────────────────────────────────────────


def load_checkpoint(path: Path) -> dict:
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"retrieval_cache": {}, "generations": {}}


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ─── LLM Generation ──────────────────────────────────────────────


def build_messages(query: str, docs: list[dict]) -> list[dict]:
    """Промпт для LLM: с документами или без (parametric)."""
    if not docs:
        return [
            {"role": "system", "content": SYSTEM_NO_DOCS},
            {"role": "user", "content": query},
        ]

    doc_parts = []
    for i, doc in enumerate(docs, 1):
        text = doc["text"][:MAX_DOC_CHARS]
        channel = doc.get("channel", "")
        doc_parts.append(f"[{i}] ({channel}) {text}")
    docs_text = "\n\n".join(doc_parts)

    return [
        {"role": "system", "content": SYSTEM_WITH_DOCS},
        {"role": "user", "content": f"Документы:\n{docs_text}\n\nВопрос: {query}"},
    ]


def call_llm(
    messages: list[dict],
    llm_url: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 42,
) -> tuple[str, float]:
    """Прямой вызов llama-server. Возвращает (answer, latency_sec)."""
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
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        answer = resp["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        answer = f"[ERROR] {exc}"
    latency = time.time() - t0
    return answer, round(latency, 1)


# ─── Retrieval ────────────────────────────────────────────────────


def retrieve_and_cache(
    questions: list[dict],
    args,
    sparse_model,
    cache: dict,
) -> dict:
    """Retrieve top-20 для каждого question. Cache в checkpoint."""
    for q in questions:
        qid = q["id"]
        if qid in cache:
            continue
        query = q["query"]
        logger.info("Retrieving: %s", qid)
        points = search_qdrant(
            query, args.embedding_url, args.qdrant_url,
            args.collection, sparse_model,
            use_colbert=True, top_k=20,
        )
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


# ─── Conditions ───────────────────────────────────────────────────


def get_conditions(test: str, docs: list[dict]) -> list[tuple[str, list[dict]]]:
    """Вернуть (condition_name, docs) для теста."""
    if test == "ndr":
        return [
            ("k=0", []),
            ("k=20", docs[:20]),
        ]
    elif test == "rsr":
        return [(f"k={k}", docs[:k]) for k in K_VALUES]
    elif test == "ror":
        reversed_docs = list(reversed(docs[:20]))
        shuffled = docs[:20].copy()
        random.Random(42).shuffle(shuffled)
        return [
            ("original", docs[:20]),
            ("reversed", reversed_docs),
            ("shuffled", shuffled),
        ]
    return []


def canonical_key(qid: str, test: str, cond: str) -> str:
    """Ключ для checkpoint. Reuse: k=20/original — одна генерация."""
    # k=20 original shared между NDR, RSR, ROR
    if cond in ("k=20", "original"):
        return f"{qid}:shared:k=20_original"
    return f"{qid}:{test}:{cond}"


# ─── Generation Loop ─────────────────────────────────────────────


def run_test(
    questions: list[dict],
    test: str,
    retrieval_cache: dict,
    generations: dict,
    llm_url: str,
    checkpoint: dict,
    checkpoint_path: Path,
) -> None:
    """Генерация ответов для всех (question, condition)."""
    filtered = filter_for_test(questions, test)
    logger.info("=== %s: %d questions ===", test.upper(), len(filtered))

    for q in filtered:
        qid = q["id"]
        docs = retrieval_cache.get(qid, [])
        conditions = get_conditions(test, docs)

        for cond_name, cond_docs in conditions:
            key = canonical_key(qid, test, cond_name)
            if key in generations:
                continue

            messages = build_messages(q["query"], cond_docs)
            answer, latency = call_llm(messages, llm_url)

            generations[key] = {
                "answer": answer,
                "latency_sec": latency,
                "condition": cond_name,
                "n_docs": len(cond_docs),
                "question_id": qid,
            }
            save_checkpoint(checkpoint_path, checkpoint)
            logger.info("  %s | %s | %s | %.1fs | %d chars",
                        qid, test, cond_name, latency, len(answer))


# ─── BERTScore Scoring ────────────────────────────────────────────


_scorer = None


def bertscore_f1(candidate: str, reference: str) -> float | None:
    """BERTScore F1 (lazy init ruBert-large)."""
    global _scorer
    if not candidate or not reference:
        return None
    if _scorer is None:
        try:
            from bert_score import BERTScorer
            _scorer = BERTScorer(
                model_type="ai-forever/ruBert-large",
                num_layers=18, idf=False, lang="ru",
                rescale_with_baseline=False,
            )
            _scorer._tokenizer.model_max_length = 512
            logger.info("BERTScore loaded: ruBert-large layer 18")
        except Exception as exc:
            logger.warning("BERTScore unavailable: %s", exc)
            return None
    _, _, F1 = _scorer.score([candidate], [reference])
    return round(float(F1[0]), 4)


def score_generations(
    questions: list[dict],
    generations: dict,
    tests: list[str],
) -> dict[str, float]:
    """BERTScore для всех generations. Возвращает {canonical_key: score}."""
    scores = {}
    q_map = {q["id"]: q for q in questions}

    for key, gen in generations.items():
        qid = gen["question_id"]
        q = q_map.get(qid)
        if not q or not q.get("expected_answer"):
            continue
        answer = gen["answer"]
        if answer.startswith("[ERROR]"):
            continue
        score = bertscore_f1(answer, q["expected_answer"])
        if score is not None:
            scores[key] = score

    return scores


# ─── Metric Computation ──────────────────────────────────────────


def compute_ndr(questions: list[dict], generations: dict, scores: dict) -> dict:
    """NDR = fraction where RAG (k=20) ≥ no-RAG (k=0)."""
    results = []
    for q in questions:
        qid = q["id"]
        key_rag = canonical_key(qid, "ndr", "k=20")
        key_no = canonical_key(qid, "ndr", "k=0")
        s_rag = scores.get(key_rag)
        s_no = scores.get(key_no)
        if s_rag is None or s_no is None:
            continue
        hit = s_rag >= s_no
        results.append({
            "id": qid,
            "query": q["query"][:80],
            "score_rag": s_rag,
            "score_no_rag": s_no,
            "delta": round(s_rag - s_no, 4),
            "ndr_hit": hit,
        })

    hits = sum(1 for r in results if r["ndr_hit"])
    total = len(results)
    return {
        "rate": round(hits / total, 4) if total else 0.0,
        "hits": hits,
        "total": total,
        "per_question": sorted(results, key=lambda x: x["delta"]),
    }


def compute_rsr(questions: list[dict], generations: dict, scores: dict) -> dict:
    """RSR = fraction where scores are monotonically non-decreasing across k."""
    filtered = filter_for_test(questions, "rsr")
    results = []
    violations = []

    for q in filtered:
        qid = q["id"]
        k_scores = []
        for k in K_VALUES:
            cond = f"k={k}"
            key = canonical_key(qid, "rsr", cond)
            k_scores.append(scores.get(key))

        if any(s is None for s in k_scores):
            continue

        monotonic = True
        for i in range(1, len(k_scores)):
            if k_scores[i] < k_scores[i - 1] - RSR_EPSILON:
                monotonic = False
                violations.append({
                    "id": qid,
                    "query": q["query"][:60],
                    "k_from": K_VALUES[i - 1],
                    "k_to": K_VALUES[i],
                    "score_from": k_scores[i - 1],
                    "score_to": k_scores[i],
                    "drop": round(k_scores[i - 1] - k_scores[i], 4),
                })
        results.append({"id": qid, "k_scores": k_scores, "monotonic": monotonic})

    hits = sum(1 for r in results if r["monotonic"])
    total = len(results)
    return {
        "rate": round(hits / total, 4) if total else 0.0,
        "hits": hits,
        "total": total,
        "violations": violations,
        "per_question": results,
    }


def compute_ror(questions: list[dict], generations: dict, scores: dict) -> dict:
    """ROR = mean(1 - 2σ) across questions."""
    import statistics
    filtered = filter_for_test(questions, "ror")
    results = []

    for q in filtered:
        qid = q["id"]
        order_scores = []
        for o in ORDERINGS:
            key = canonical_key(qid, "ror", o)
            s = scores.get(key)
            if s is not None:
                order_scores.append(s)

        if len(order_scores) < 2:
            continue
        sigma = statistics.stdev(order_scores)
        ror_q = max(0.0, 1.0 - 2 * sigma)
        results.append({
            "id": qid,
            "query": q["query"][:60],
            "scores": {o: scores.get(canonical_key(qid, "ror", o)) for o in ORDERINGS},
            "sigma": round(sigma, 4),
            "ror": round(ror_q, 4),
        })

    mean_ror = sum(r["ror"] for r in results) / len(results) if results else 0.0
    return {
        "rate": round(mean_ror, 4),
        "per_question": sorted(results, key=lambda x: x["ror"]),
    }


# ─── Report ───────────────────────────────────────────────────────


def build_report(ndr: dict, rsr: dict, ror: dict, metadata: dict) -> str:
    """Markdown report."""
    composite = 0.0
    rates = [ndr.get("rate", 0), rsr.get("rate", 0), ror.get("rate", 0)]
    non_zero = [r for r in rates if r > 0]
    if non_zero:
        product = 1.0
        for r in non_zero:
            product *= r
        composite = round(product ** (1.0 / len(non_zero)), 4)

    lines = [
        "# NDR/RSR/ROR Robustness Report",
        "",
        f"**Date**: {metadata['timestamp']}",
        f"**Dataset**: {metadata['dataset']} ({metadata['total_questions']} Qs)",
        "**Scoring**: BERTScore F1 (ruBert-large, layer 18)",
        "**Pipeline**: BM25(100)+Dense(20) → RRF 3:1 → ColBERT → top-20",
        "",
        "## Summary",
        "",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| **NDR** | **{ndr.get('rate', 'N/A')}** | Retrieval helps in {ndr.get('hits', 0)}/{ndr.get('total', 0)} cases |",
        f"| **RSR** | **{rsr.get('rate', 'N/A')}** | Monotonic in {rsr.get('hits', 0)}/{rsr.get('total', 0)} cases |",
        f"| **ROR** | **{ror.get('rate', 'N/A')}** | Mean order robustness |",
        f"| **Composite** | **{composite}** | Geometric mean |",
        "",
        "> Note: simplified protocol (see docs/progress/experiment_log.md for comparison with Cao et al.)",
        "",
    ]

    # NDR details
    if ndr.get("per_question"):
        lines.extend([
            "## NDR Details",
            "",
            "| Question | score(RAG) | score(no-RAG) | Δ | Hit |",
            "|----------|-----------|---------------|-----|-----|",
        ])
        for r in ndr["per_question"]:
            hit = "✓" if r["ndr_hit"] else "✗"
            lines.append(f"| {r['id']} | {r['score_rag']} | {r['score_no_rag']} | {r['delta']:+.4f} | {hit} |")
        lines.append("")

    # RSR violations
    if rsr.get("violations"):
        lines.extend([
            "## RSR Monotonicity Violations",
            "",
            "| Question | k_from | k_to | score_from | score_to | Drop |",
            "|----------|--------|------|-----------|----------|------|",
        ])
        for v in rsr["violations"]:
            lines.append(f"| {v['id']} | {v['k_from']} | {v['k_to']} | {v['score_from']} | {v['score_to']} | {v['drop']} |")
        lines.append("")

    # ROR details
    if ror.get("per_question"):
        lines.extend([
            "## ROR Per-Question",
            "",
            "| Question | original | reversed | shuffled | σ | ROR |",
            "|----------|----------|----------|----------|---|-----|",
        ])
        for r in ror["per_question"]:
            s = r["scores"]
            lines.append(
                f"| {r['id']} | {s.get('original', 'N/A')} | {s.get('reversed', 'N/A')} | "
                f"{s.get('shuffled', 'N/A')} | {r['sigma']} | {r['ror']} |"
            )
        lines.append("")

    return "\n".join(lines)


def build_judge_artifact(
    questions: list[dict],
    ndr: dict,
    generations: dict,
    top_n: int = 20,
) -> str:
    """Export worst-delta NDR pairs для Claude manual judge."""
    q_map = {q["id"]: q for q in questions}
    lines = [
        "# Robustness Judge Artifact",
        "",
        f"Top {top_n} worst NDR delta pairs for manual Claude judge verification.",
        "Score each answer: factual_correctness 0 / 0.5 / 1.0",
        "",
    ]

    worst = [r for r in ndr.get("per_question", []) if r["delta"] < 0.05][:top_n]

    for r in worst:
        qid = r["id"]
        q = q_map.get(qid, {})
        key_rag = canonical_key(qid, "ndr", "k=20")
        key_no = canonical_key(qid, "ndr", "k=0")
        gen_rag = generations.get(key_rag, {})
        gen_no = generations.get(key_no, {})

        lines.extend([
            f"## {qid}",
            f"**Query**: {q.get('query', '')}",
            f"**Expected**: {q.get('expected_answer', '')[:300]}",
            "",
            "### A: No retrieval (k=0)",
            gen_no.get("answer", "N/A")[:500],
            "",
            "### B: With retrieval (k=20)",
            gen_rag.get("answer", "N/A")[:500],
            "",
            "| | factual (0/0.5/1) |",
            "|---|---|",
            "| A (no retrieval) | ___ |",
            "| B (with retrieval) | ___ |",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NDR/RSR/ROR robustness eval (SPEC-RAG-23)")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/eval_golden_v2.json"))
    parser.add_argument("--collection", default="news_colbert_v2")
    parser.add_argument("--qdrant-url", default="http://localhost:16333")
    parser.add_argument("--embedding-url", default="http://localhost:8082")
    parser.add_argument("--llm-url", default="http://localhost:8080")
    parser.add_argument("--tests", nargs="*", default=["ndr", "rsr", "ror"],
                        choices=["ndr", "rsr", "ror"])
    parser.add_argument("--output", type=Path, default=Path("results/robustness"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    # Load
    questions = load_dataset(args.dataset)
    logger.info("Loaded %d questions from %s", len(questions), args.dataset)

    checkpoint_path = args.output / "checkpoint.json"
    checkpoint = load_checkpoint(args.resume or checkpoint_path)
    checkpoint.setdefault("retrieval_cache", {})
    checkpoint.setdefault("generations", {})

    # Sparse model для BM25
    logger.info("Loading fastembed BM25...")
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")

    # Phase 1: Retrieve
    logger.info("Phase 1: Retrieval (cached)")
    retrieve_and_cache(questions, args, sparse_model, checkpoint["retrieval_cache"])
    save_checkpoint(checkpoint_path, checkpoint)

    # Phase 2: Generate
    logger.info("Phase 2: Generation")
    for test in args.tests:
        run_test(
            questions, test,
            checkpoint["retrieval_cache"],
            checkpoint["generations"],
            args.llm_url,
            checkpoint,
            checkpoint_path,
        )

    # Phase 3: Score
    logger.info("Phase 3: BERTScore scoring")
    scores = score_generations(questions, checkpoint["generations"], args.tests)
    logger.info("Scored %d generations", len(scores))

    # Phase 4: Compute metrics
    logger.info("Phase 4: Computing metrics")
    ndr_result = compute_ndr(questions, checkpoint["generations"], scores) if "ndr" in args.tests else {}
    rsr_result = compute_rsr(questions, checkpoint["generations"], scores) if "rsr" in args.tests else {}
    ror_result = compute_ror(questions, checkpoint["generations"], scores) if "ror" in args.tests else {}

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(args.dataset),
        "total_questions": len(questions),
        "tests": args.tests,
        "scoring": "bertscore_f1_rubert_large_l18",
    }

    # Raw JSON
    raw = {
        "metadata": metadata,
        "ndr": ndr_result,
        "rsr": rsr_result,
        "ror": ror_result,
        "scores": scores,
    }

    # Composite
    rates = [ndr_result.get("rate", 0), rsr_result.get("rate", 0), ror_result.get("rate", 0)]
    non_zero = [r for r in rates if r > 0]
    product = 1.0
    for r in non_zero:
        product *= r
    raw["composite"] = round(product ** (1.0 / len(non_zero)), 4) if non_zero else 0.0

    raw_path = args.output / f"ndr_rsr_ror_raw_{ts}.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    # Markdown report
    report = build_report(ndr_result, rsr_result, ror_result, metadata)
    report_path = args.output / f"ndr_rsr_ror_report_{ts}.md"
    report_path.write_text(report, encoding="utf-8")

    # Judge artifact
    if ndr_result:
        artifact = build_judge_artifact(questions, ndr_result, checkpoint["generations"])
        artifact_path = args.output / f"judge_artifact_{ts}.md"
        artifact_path.write_text(artifact, encoding="utf-8")

    logger.info("Results: %s", raw_path)
    logger.info("Report: %s", report_path)

    # Print summary
    print(f"\n{'='*50}")
    print(f"NDR: {ndr_result.get('rate', 'N/A')} ({ndr_result.get('hits', 0)}/{ndr_result.get('total', 0)})")
    print(f"RSR: {rsr_result.get('rate', 'N/A')} ({rsr_result.get('hits', 0)}/{rsr_result.get('total', 0)})")
    print(f"ROR: {ror_result.get('rate', 'N/A')}")
    print(f"Composite: {raw['composite']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
