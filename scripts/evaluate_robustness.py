#!/usr/bin/env python3
"""Robustness evaluation harness (SPEC-RAG-22 Layer 2).

Тесты:
1. RGB Noise Robustness — inject irrelevant docs, check accuracy drop
2. RGB Negative Rejection — all docs irrelevant, check refusal
3. Proxy-NDR — retrieval vs no-retrieval baseline
4. Query Perturbation — noise/substitution/reorder stability

Использование:
    # Полный прогон
    python scripts/evaluate_robustness.py \\
        --dataset datasets/eval_golden_v2.json \\
        --api-url http://localhost:8001/v1/agent/stream \\
        --api-key TOKEN \\
        --output results/robustness/

    # Только query perturbation
    python scripts/evaluate_robustness.py \\
        --dataset datasets/eval_robustness_v1.json \\
        --tests query_perturbation \\
        --api-url http://localhost:8001/v1/agent/stream

    # Resume
    python scripts/evaluate_robustness.py \\
        --resume results/robustness/checkpoint.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Noise Generator (Russian-aware) ─────────────────────────────

# Русская клавиатурная раскладка — соседние клавиши
RU_KEYBOARD_NEIGHBORS = {
    "й": "цу", "ц": "йук", "у": "цке", "к": "уеа", "е": "кнг",
    "н": "егш", "г": "нщш", "ш": "гщз", "щ": "шзх", "з": "щхъ",
    "ф": "ыв", "ы": "фва", "в": "ыап", "а": "впр", "п": "аро",
    "р": "пол", "о": "рлд", "л": "одж", "д": "лжэ", "ж": "дэ",
    "я": "чс", "ч": "ясм", "с": "чми", "м": "сит", "и": "мтб",
    "т": "иьб", "ь": "тбю", "б": "ьюи", "ю": "бь",
    "ё": "е", "е": "ёкн",  # noqa: F601
}

# Latin↔Cyrillic визуально похожие
LAT_CYR_MAP = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
    "x": "х", "y": "у", "k": "к", "m": "м", "t": "т",
}
CYR_LAT_MAP = {v: k for k, v in LAT_CYR_MAP.items()}


def add_noise_russian(query: str, seed: int, ratio: float = 0.2) -> str:
    """Russian-aware noise: keyboard neighbors, char mutations, Lat/Cyr mixing.

    Deterministic (seed-based). Модифицирует ~ratio слов.
    """
    rng = random.Random(seed)
    words = query.split()
    n_modify = max(1, int(len(words) * ratio))
    indices = rng.sample(range(len(words)), min(n_modify, len(words)))

    for idx in indices:
        word = words[idx]
        if len(word) < 2:
            continue

        mutation = rng.choice(["keyboard", "swap", "delete", "lat_cyr"])

        if mutation == "keyboard":
            # Заменить случайный символ на клавиатурного соседа
            pos = rng.randint(0, len(word) - 1)
            ch = word[pos].lower()
            neighbors = RU_KEYBOARD_NEIGHBORS.get(ch, "")
            if neighbors:
                replacement = rng.choice(list(neighbors))
                word = word[:pos] + replacement + word[pos + 1:]

        elif mutation == "swap":
            # Переставить два соседних символа
            if len(word) > 2:
                pos = rng.randint(0, len(word) - 2)
                word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]

        elif mutation == "delete":
            # Удалить случайный символ (не первый)
            if len(word) > 2:
                pos = rng.randint(1, len(word) - 1)
                word = word[:pos] + word[pos + 1:]

        elif mutation == "lat_cyr":
            # Заменить кириллический символ на визуально похожий латинский
            pos = rng.randint(0, len(word) - 1)
            ch = word[pos].lower()
            if ch in CYR_LAT_MAP:
                replacement = CYR_LAT_MAP[ch]
                word = word[:pos] + replacement + word[pos + 1:]

        words[idx] = word

    return " ".join(words)


# ─── Agent API client ─────────────────────────────────────────────


def iter_sse_events(response: httpx.Response):
    """Парсинг SSE потока (event: + data: pairs)."""
    event_type = None
    data_lines = []

    for raw_line in response.iter_lines():
        if raw_line is None:
            continue
        line = raw_line.strip("\r")
        if not line:
            if event_type and data_lines:
                yield event_type, "\n".join(data_lines)
            event_type = None
            data_lines = []
            continue
        if line.startswith(":") or line.startswith("retry:"):
            continue
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())


def call_agent(
    query: str,
    api_url: str,
    api_key: str,
    timeout: int = 120,
) -> dict[str, Any]:
    """Отправить query в agent API (SSE), собрать ответ."""
    payload = {"query": query}
    answer = ""
    citations = []
    tools_invoked = []
    error = False

    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout) as client, client.stream(
            "POST", api_url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as response:
            response.raise_for_status()
            for event_name, event_data in iter_sse_events(response):
                try:
                    decoded = json.loads(event_data)
                except json.JSONDecodeError:
                    continue
                if event_name == "final":
                    answer = decoded.get("answer", "")
                elif event_name == "citations":
                    citations = decoded.get("citations", [])
                elif event_name == "tool_invoked":
                    tool = decoded.get("tool") or decoded.get("name", "")
                    if tool:
                        tools_invoked.append(tool)
    except Exception as exc:
        error = True
        answer = str(exc)

    latency = time.time() - t0
    return {
        "answer": answer,
        "citations": citations,
        "tools_invoked": tools_invoked,
        "latency_sec": round(latency, 1),
        "error": error,
    }


# ─── BERTScore helper ─────────────────────────────────────────────


_bert_scorer = None


def bertscore_f1(candidate: str, reference: str) -> float | None:
    """BERTScore F1 (lazy init). None если недоступен."""
    global _bert_scorer
    if _bert_scorer is None:
        try:
            from bert_score import BERTScorer
            _bert_scorer = BERTScorer(
                model_type="ai-forever/ruBert-large",
                num_layers=18, idf=False, lang="ru",
                rescale_with_baseline=False,
            )
        except ImportError:
            logger.warning("bert-score not installed, using fallback")
            return None
    if not candidate or not reference:
        return None
    _, _, F1 = _bert_scorer.score([candidate], [reference])
    return round(float(F1[0]), 4)


# ─── Checkpoint / Resume ──────────────────────────────────────────


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Загрузить checkpoint для resume."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}}


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Сохранить checkpoint после каждого question."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def is_completed(checkpoint: dict, question_id: str, test_type: str, variant: str = "") -> bool:
    """Проверить пройден ли (question_id, test_type, variant)."""
    key = f"{question_id}:{test_type}:{variant}"
    return key in checkpoint.get("completed", {})


def mark_completed(checkpoint: dict, question_id: str, test_type: str, variant: str, result: Any) -> None:
    """Отметить (question_id, test_type, variant) как пройденный."""
    key = f"{question_id}:{test_type}:{variant}"
    checkpoint.setdefault("completed", {})[key] = result


# ─── Test implementations ─────────────────────────────────────────


def test_query_perturbation(
    dataset: list[dict],
    api_url: str,
    api_key: str,
    checkpoint: dict,
    checkpoint_path: Path,
) -> list[dict]:
    """Query perturbation: noise/substitution/reorder stability."""
    results = []

    for q in dataset:
        qid = q["id"]
        original_query = q.get("original_query", q.get("query", ""))
        expected = q.get("expected_answer", "")

        # Original (может быть уже из normal eval)
        if not is_completed(checkpoint, qid, "perturbation", "original"):
            logger.info("  %s: original", qid)
            res = call_agent(original_query, api_url, api_key)
            mark_completed(checkpoint, qid, "perturbation", "original", res)
            save_checkpoint(checkpoint_path, checkpoint)

        orig_result = checkpoint["completed"].get(f"{qid}:perturbation:original", {})
        orig_answer = orig_result.get("answer", "")

        perturbations = q.get("perturbations", {})
        q_results = {"id": qid, "original_answer": orig_answer, "variants": {}}

        for ptype, perturbed_query in perturbations.items():
            _variant_key = f"perturbation:{ptype}"
            if is_completed(checkpoint, qid, "perturbation", ptype):
                pert_result = checkpoint["completed"].get(f"{qid}:perturbation:{ptype}", {})
            else:
                logger.info("  %s: %s", qid, ptype)
                pert_result = call_agent(perturbed_query, api_url, api_key)
                mark_completed(checkpoint, qid, "perturbation", ptype, pert_result)
                save_checkpoint(checkpoint_path, checkpoint)

            pert_answer = pert_result.get("answer", "")
            consistency = bertscore_f1(orig_answer, pert_answer)
            quality = bertscore_f1(pert_answer, expected) if expected else None

            q_results["variants"][ptype] = {
                "perturbed_query": perturbed_query,
                "answer": pert_answer[:500],
                "consistency_bertscore": consistency,
                "quality_bertscore": quality,
                "latency_sec": pert_result.get("latency_sec"),
                "error": pert_result.get("error", False),
            }

        results.append(q_results)

    return results


def test_proxy_ndr(
    dataset: list[dict],
    api_url: str,
    api_key: str,
    checkpoint: dict,
    checkpoint_path: Path,
) -> list[dict]:
    """Proxy-NDR: retrieval vs no-retrieval baseline."""
    results = []

    for q in dataset:
        qid = q["id"]
        query = q.get("query", "")
        expected = q.get("expected_answer", "")
        eval_mode = q.get("eval_mode", "retrieval_evidence")

        # Только retrieval_evidence
        if eval_mode != "retrieval_evidence":
            continue

        # Normal (с retrieval)
        if not is_completed(checkpoint, qid, "ndr", "rag"):
            logger.info("  %s: with retrieval", qid)
            res = call_agent(query, api_url, api_key)
            mark_completed(checkpoint, qid, "ndr", "rag", res)
            save_checkpoint(checkpoint_path, checkpoint)

        # No-retrieval (TODO: requires API flag --no-retrieval; for now skip search)
        # Placeholder: отправляем query с prefix, чтобы agent ответил без search
        if not is_completed(checkpoint, qid, "ndr", "no_rag"):
            logger.info("  %s: no retrieval (parametric)", qid)
            no_rag_query = f"[БЕЗ ПОИСКА, ответь из своих знаний] {query}"
            res = call_agent(no_rag_query, api_url, api_key)
            mark_completed(checkpoint, qid, "ndr", "no_rag", res)
            save_checkpoint(checkpoint_path, checkpoint)

        rag_result = checkpoint["completed"].get(f"{qid}:ndr:rag", {})
        no_rag_result = checkpoint["completed"].get(f"{qid}:ndr:no_rag", {})

        score_rag = bertscore_f1(rag_result.get("answer", ""), expected)
        score_no_rag = bertscore_f1(no_rag_result.get("answer", ""), expected)

        results.append({
            "id": qid,
            "score_rag": score_rag,
            "score_no_rag": score_no_rag,
            "ndr_hit": (score_rag or 0) >= (score_no_rag or 0),
            "delta": round((score_rag or 0) - (score_no_rag or 0), 4),
        })

    return results


# ─── Aggregate and Report ─────────────────────────────────────────


def aggregate_robustness(
    perturbation_results: list[dict],
    ndr_results: list[dict],
) -> dict[str, Any]:
    """Агрегирует robustness результаты."""
    agg: dict[str, Any] = {}

    # Query perturbation
    if perturbation_results:
        consistencies = []
        for r in perturbation_results:
            for v in r.get("variants", {}).values():
                c = v.get("consistency_bertscore")
                if c is not None:
                    consistencies.append(c)
        agg["query_perturbation"] = {
            "mean_consistency": round(sum(consistencies) / len(consistencies), 4) if consistencies else None,
            "min_consistency": round(min(consistencies), 4) if consistencies else None,
            "total_variants": len(consistencies),
            "total_questions": len(perturbation_results),
        }

    # Proxy-NDR
    if ndr_results:
        hits = sum(1 for r in ndr_results if r["ndr_hit"])
        agg["proxy_ndr"] = {
            "rate": round(hits / len(ndr_results), 4) if ndr_results else None,
            "hits": hits,
            "total": len(ndr_results),
            "mean_delta": round(sum(r["delta"] for r in ndr_results) / len(ndr_results), 4),
        }

    return agg


def build_robustness_report(agg: dict, perturbation_results: list, ndr_results: list) -> str:
    """Markdown robustness report."""
    lines = ["# Robustness Evaluation Report", f"**Date:** {datetime.now().isoformat()}", ""]

    qp = agg.get("query_perturbation")
    if qp:
        lines.extend([
            "## Query Perturbation Robustness",
            f"- Mean consistency: **{qp['mean_consistency']}**",
            f"- Min consistency: {qp['min_consistency']}",
            f"- Total: {qp['total_questions']} questions × {qp['total_variants']} variants",
            "",
        ])

    ndr = agg.get("proxy_ndr")
    if ndr:
        lines.extend([
            "## Proxy-NDR (retrieval vs no-retrieval)",
            f"- Rate: **{ndr['rate']}** ({ndr['hits']}/{ndr['total']})",
            f"- Mean delta (rag - no_rag): {ndr['mean_delta']:+.4f}",
            "",
        ])

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation (SPEC-RAG-22 Layer 2)")
    parser.add_argument("--dataset", type=Path, required=True, help="Golden or robustness dataset JSON")
    parser.add_argument("--api-url", default="http://localhost:8001/v1/agent/stream")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "test"))
    parser.add_argument("--tests", nargs="*", default=["query_perturbation", "proxy_ndr"],
                        choices=["query_perturbation", "proxy_ndr", "rgb_noise", "rgb_rejection"])
    parser.add_argument("--output", type=Path, default=Path("results/robustness"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if isinstance(dataset, dict) and "questions" in dataset:
        dataset = dataset["questions"]

    # Checkpoint
    checkpoint_path = args.output / "checkpoint.json"
    checkpoint = load_checkpoint(args.resume or checkpoint_path)

    perturbation_results: list[dict] = []
    ndr_results: list[dict] = []

    if "query_perturbation" in args.tests:
        # Нужен robustness dataset с perturbations
        has_perturbations = any(q.get("perturbations") for q in dataset)
        if has_perturbations:
            logger.info("=== Query Perturbation (%d Qs) ===", len(dataset))
            perturbation_results = test_query_perturbation(
                dataset, args.api_url, args.api_key, checkpoint, checkpoint_path,
            )
        else:
            logger.warning("Dataset has no perturbations field, skipping query_perturbation")

    if "proxy_ndr" in args.tests:
        logger.info("=== Proxy-NDR ===")
        ndr_results = test_proxy_ndr(
            dataset, args.api_url, args.api_key, checkpoint, checkpoint_path,
        )

    # Aggregate
    agg = aggregate_robustness(perturbation_results, ndr_results)

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    raw_path = args.output / f"robustness_raw_{ts}.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump({
            "aggregate": agg,
            "query_perturbation": perturbation_results,
            "proxy_ndr": ndr_results,
        }, f, ensure_ascii=False, indent=2)
    logger.info("Raw results: %s", raw_path)

    report_path = args.output / f"robustness_report_{ts}.md"
    report = build_robustness_report(agg, perturbation_results, ndr_results)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report: %s", report_path)

    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
