#!/usr/bin/env python3
"""NLI faithfulness scoring (SPEC-RAG-21).

Вход:
  --eval-results  results/raw/eval_results_YYYYMMDD.json
  --claims        results/raw/claims_YYYYMMDD.json  (Claude decomposition)
Выход:
  results/raw/nli_scores_YYYYMMDD.json

Читает ТОЛЬКО JSON, никогда markdown.
Faithfulness считается только для retrieval_evidence вопросов.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Добавляем корень проекта в path для импорта src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.eval.nli import NLIVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_nli")


def load_eval_results(path: Path) -> dict:
    """Загружает eval results JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Поддержка обоих форматов: {per_question: [...]} или [{...}, ...]
    if isinstance(data, dict) and "per_question" in data:
        return data
    raise ValueError(f"Unexpected eval results format: {type(data)}")


def load_claims(path: Path) -> dict:
    """Загружает claims JSON от Claude decomposition."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Формат: {"questions": [{"id": "golden_q01", "claims": [...]}, ...]}
    if isinstance(data, dict) and "questions" in data:
        return {q["id"]: q.get("claims", []) for q in data["questions"]}
    # Альтернативный формат: список напрямую
    if isinstance(data, list):
        return {q["id"]: q.get("claims", []) for q in data}
    raise ValueError(f"Unexpected claims format: {type(data)}")


def extract_documents(question_result: dict) -> list:
    """Извлекает cited documents с текстами из eval result.

    Ищет в offline_judge_packet.citations (основной путь)
    или в agent.citation_hits (fallback).
    """
    # Основной путь: offline_judge_packet.citations (enriched из Qdrant)
    packet = question_result.get("offline_judge_packet", {})
    citations = packet.get("citations", [])
    docs = []
    for cit in citations:
        text = cit.get("text", "")
        if text:
            docs.append({
                "id": str(cit.get("id", "")),
                "text": text,
                "channel": cit.get("channel"),
                "date": cit.get("date"),
            })
    if docs:
        return docs

    # Fallback: agent.citation_hits
    agent = question_result.get("agent", {})
    for hit in agent.get("citation_hits", []):
        text = hit.get("text", hit.get("snippet", ""))
        if text:
            docs.append({
                "id": str(hit.get("id", "")),
                "text": text,
            })
    return docs


def run_nli(
    eval_results_path: Path,
    claims_path: Path,
    output_path: Path,
    gpu_server_url: str = "http://localhost:8082",
    entailment_threshold: float = 0.5,
    contradiction_threshold: float = 0.5,
) -> dict:
    """Основной pipeline: claims × documents → NLI faithfulness scores."""

    logger.info("Загрузка eval results: %s", eval_results_path)
    eval_data = load_eval_results(eval_results_path)
    questions = eval_data["per_question"]

    logger.info("Загрузка claims: %s", claims_path)
    claims_map = load_claims(claims_path)

    verifier = NLIVerifier(
        gpu_server_url=gpu_server_url,
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )

    results = []
    total_pairs = 0
    t0 = time.time()

    for q in questions:
        qid = q.get("query_id", q.get("id", ""))
        eval_mode = q.get("eval_mode", "")

        # Получаем claims для этого вопроса
        q_claims = claims_map.get(qid, [])

        # Извлекаем documents
        docs = extract_documents(q)

        logger.info(
            "%s [%s]: %d claims, %d docs",
            qid, eval_mode, len(q_claims), len(docs),
        )

        # Верификация
        faith = verifier.verify_question(
            query_id=qid,
            eval_mode=eval_mode,
            claims=q_claims,
            documents=docs,
        )

        total_pairs += faith.nli_pairs_count
        results.append(asdict(faith))

        if faith.faithfulness is not None:
            logger.info(
                "  → faithfulness=%.3f (strict=%.3f), supported=%d/%d, contradicted=%d",
                faith.faithfulness, faith.faithfulness_strict,
                faith.claims_supported, faith.claims_verifiable,
                faith.claims_contradicted,
            )

    elapsed = time.time() - t0

    # Aggregate
    retrieval_results = [r for r in results if r["faithfulness"] is not None]
    avg_faith = (
        sum(r["faithfulness"] for r in retrieval_results) / len(retrieval_results)
        if retrieval_results else None
    )
    avg_faith_strict = (
        sum(r["faithfulness_strict"] for r in retrieval_results) / len(retrieval_results)
        if retrieval_results else None
    )
    total_supported = sum(r["claims_supported"] for r in retrieval_results)
    total_verifiable = sum(r["claims_verifiable"] for r in retrieval_results)
    total_contradicted = sum(r["claims_contradicted"] for r in retrieval_results)

    all_contradictions = []
    for r in results:
        for c in r.get("contradictions", []):
            c["query_id"] = r["query_id"]
            all_contradictions.append(c)

    # Citation precision (ALCE-style): per retrieval question
    # Для каждого вопроса: citation irrelevant если ни один claim не entailed от этого doc
    citation_precisions = []
    for r in retrieval_results:
        per_claim = r.get("per_claim", [])
        if not per_claim:
            continue
        # Собираем doc IDs которые поддержали хотя бы один claim
        supporting_docs = {
            c["best_document_id"] for c in per_claim
            if c.get("nli_label") == "entailment" and c.get("best_document_id")
        }
        # Все doc IDs из этого вопроса
        q_data = next((q for q in questions if q.get("query_id") == r["query_id"]), {})
        all_doc_ids = {str(d["id"]) for d in extract_documents(q_data) if d.get("id")}
        if all_doc_ids:
            irrelevant = len(all_doc_ids - supporting_docs)
            precision = 1.0 - (irrelevant / len(all_doc_ids))
            citation_precisions.append(precision)

    avg_citation_precision = (
        sum(citation_precisions) / len(citation_precisions)
        if citation_precisions else None
    )

    output = {
        "metadata": {
            "eval_results_path": str(eval_results_path),
            "claims_path": str(claims_path),
            "gpu_server_url": gpu_server_url,
            "entailment_threshold": entailment_threshold,
            "contradiction_threshold": contradiction_threshold,
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed, 1),
            "total_nli_pairs": total_pairs,
            "judge_prompt_version": "v1",
        },
        "aggregate": {
            "faithfulness": round(avg_faith, 4) if avg_faith is not None else None,
            "faithfulness_strict": round(avg_faith_strict, 4) if avg_faith_strict is not None else None,
            "citation_precision": round(avg_citation_precision, 4) if avg_citation_precision is not None else None,
            "retrieval_questions": len(retrieval_results),
            "total_questions": len(results),
            "total_claims_verifiable": total_verifiable,
            "total_claims_supported": total_supported,
            "total_claims_contradicted": total_contradicted,
        },
        "contradictions": all_contradictions,
        "per_question": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("=== NLI Results ===")
    logger.info("Faithfulness (lenient): %s", output["aggregate"]["faithfulness"])
    logger.info("Faithfulness (strict):  %s", output["aggregate"]["faithfulness_strict"])
    logger.info("Citation Precision:     %s", output["aggregate"]["citation_precision"])
    logger.info("Retrieval Qs: %d/%d", len(retrieval_results), len(results))
    logger.info("Claims: %d supported / %d verifiable / %d contradicted",
                total_supported, total_verifiable, total_contradicted)
    logger.info("Contradictions: %d", len(all_contradictions))
    logger.info("Elapsed: %.1fs, NLI pairs: %d", elapsed, total_pairs)
    logger.info("Output: %s", output_path)

    return output


def main():
    parser = argparse.ArgumentParser(description="NLI faithfulness scoring (SPEC-RAG-21)")
    parser.add_argument("--eval-results", required=True, help="Path to eval_results JSON")
    parser.add_argument("--claims", required=True, help="Path to claims JSON (Claude decomposition)")
    parser.add_argument("--output", default=None, help="Output path (default: results/raw/nli_scores_YYYYMMDD.json)")
    parser.add_argument("--gpu-server", default="http://localhost:8082", help="GPU server URL")
    parser.add_argument("--entailment-threshold", type=float, default=0.5)
    parser.add_argument("--contradiction-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d")
        args.output = f"results/raw/nli_scores_{ts}.json"

    run_nli(
        eval_results_path=Path(args.eval_results),
        claims_path=Path(args.claims),
        output_path=Path(args.output),
        gpu_server_url=args.gpu_server,
        entailment_threshold=args.entailment_threshold,
        contradiction_threshold=args.contradiction_threshold,
    )


if __name__ == "__main__":
    main()
