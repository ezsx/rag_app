"""SummaC-ZS faithfulness — sentence-level NLI без claim decomposition (SPEC-RAG-22 §1.3).

Полностью автоматическая faithfulness метрика:
answer sentences × document sentences → NLI entailment matrix → aggregate.

Отличие от claim-level NLI (nli.py / SPEC-RAG-21):
- summac_faithfulness: sentence-level, автоматический, каждый прогон
- claim_faithfulness: claim-level, требует Claude decomposition, deep diagnostic

Reference: Laban et al. "SummaC: Re-Visiting NLI-based Models for
Inconsistency Detection in Summarization" (TACL 2022).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# razdel лучше spaCy для русского Telegram-стиля
try:
    from razdel import sentenize as _razdel_sentenize
    _HAS_RAZDEL = True
except ImportError:
    _HAS_RAZDEL = False
    logger.warning("razdel не установлен, SummaC fallback на naive split. pip install razdel")


def sentenize(text: str) -> list[str]:
    """Разбить текст на предложения. razdel если доступен, иначе naive split."""
    if not text or not text.strip():
        return []
    if _HAS_RAZDEL:
        return [s.text.strip() for s in _razdel_sentenize(text) if s.text.strip()]
    # Naive fallback: split по ". ", "! ", "? ", "\n"
    import re
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]


@dataclass
class SummaCResult:
    """Результат SummaC-ZS для одного вопроса."""
    query_id: str
    summac_faithfulness: float | None = None
    answer_sentences: int = 0
    supported_sentences: int = 0  # max_entailment ≥ threshold
    unsupported_sentences: int = 0  # max_entailment < threshold
    nli_pairs_count: int = 0
    per_sentence: list[dict[str, Any]] = field(default_factory=list)


class SummaCVerifier:
    """SummaC-ZS: sentence-level NLI faithfulness через ruBERT.

    Для каждого предложения ответа находит лучший supporting evidence
    среди всех предложений всех cited documents. Итоговый score =
    mean(max_entailment per answer sentence).
    """

    def __init__(
        self,
        gpu_server_url: str = "http://localhost:8082",
        entailment_threshold: float = 0.4,
        timeout: int = 60,
        batch_size: int = 64,
    ) -> None:
        self.gpu_server_url = gpu_server_url.rstrip("/")
        self.entailment_threshold = entailment_threshold
        self.timeout = timeout
        self.batch_size = batch_size

    def verify_question(
        self,
        query_id: str,
        answer: str,
        documents: list[dict[str, Any]],
    ) -> SummaCResult:
        """Верифицирует answer против cited documents на уровне предложений.

        documents: [{"id": "...", "text": "..."}, ...]
        """
        result = SummaCResult(query_id=query_id)

        answer_sents = sentenize(answer)
        if not answer_sents:
            return result
        result.answer_sentences = len(answer_sents)

        # Собираем все предложения из документов
        doc_sents: list[str] = []
        for doc in documents:
            text = doc.get("text", "")
            if text:
                doc_sents.extend(sentenize(text))

        if not doc_sents:
            # Нет документов — все unsupported
            result.unsupported_sentences = len(answer_sents)
            result.summac_faithfulness = 0.0
            return result

        # Строим NLI pair matrix: каждый answer_sent × каждый doc_sent
        all_pairs: list[dict[str, str]] = []
        pair_map: list[int] = []  # answer_sent index для каждой пары

        for ai, a_sent in enumerate(answer_sents):
            for d_sent in doc_sents:
                all_pairs.append({"premise": d_sent, "hypothesis": a_sent})
                pair_map.append(ai)

        result.nli_pairs_count = len(all_pairs)

        # Batch NLI calls
        nli_results = self._call_nli_batched(all_pairs)
        if not nli_results:
            return result

        # Для каждого answer sentence — max entailment across all doc sentences
        max_entailments: dict[int, float] = {i: 0.0 for i in range(len(answer_sents))}
        best_evidence: dict[int, str] = {}

        for pair_idx, nli_res in enumerate(nli_results):
            ai = pair_map[pair_idx]
            ent_score = nli_res.get("entailment", 0.0)
            if ent_score > max_entailments[ai]:
                max_entailments[ai] = ent_score
                best_evidence[ai] = all_pairs[pair_idx]["premise"][:100]

        # Aggregate
        scores = []
        for ai, a_sent in enumerate(answer_sents):
            max_ent = max_entailments[ai]
            supported = max_ent >= self.entailment_threshold
            if supported:
                result.supported_sentences += 1
            else:
                result.unsupported_sentences += 1
            scores.append(max_ent)
            result.per_sentence.append({
                "sentence": a_sent[:200],
                "max_entailment": round(max_ent, 4),
                "supported": supported,
                "best_evidence": best_evidence.get(ai, ""),
            })

        result.summac_faithfulness = round(sum(scores) / len(scores), 4) if scores else None
        return result

    def _call_nli_batched(self, pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Вызов /nli endpoint батчами."""
        if not pairs:
            return []

        all_results: list[dict[str, Any]] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            try:
                resp = httpx.post(
                    f"{self.gpu_server_url}/nli",
                    json={"pairs": batch},
                    timeout=self.timeout,
                )
                if resp.status_code == 503:
                    raise RuntimeError("NLI model not loaded (503)")
                resp.raise_for_status()
                data = resp.json()
                # gpu_server.py возвращает {"results": [{"label": ..., "scores": {"entailment": ...}}]}
                raw = data if isinstance(data, list) else data.get("results", [])
                # Нормализуем: извлекаем scores из вложенной структуры
                normalized = []
                for item in raw:
                    if "scores" in item:
                        normalized.append(item["scores"])
                    else:
                        normalized.append(item)
                all_results.extend(normalized)
            except Exception as exc:
                logger.error("SummaC NLI batch %d failed: %s", i // self.batch_size, exc)
                # Pad с neutral результатами
                all_results.extend([{"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}] * len(batch))

        return all_results
