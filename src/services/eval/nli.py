"""NLI Verifier — HTTP client для XLM-RoBERTa NLI endpoint (SPEC-RAG-21).

Eval-only модуль, НЕ runtime. Проверяет grounding ответов агента
через Natural Language Inference: claim × document → entailment/neutral/contradiction.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Порог для NLI классификации.
# Калибровка на smoke test (21 claims, rubert-base-cased-nli-threeway):
# median entailment = 0.968, min supported = 0.430 ($5 трлн парафраз).
# При 0.45: 21/21 supported. При 0.50: 20/21 (теряем верный claim).
DEFAULT_ENTAILMENT_THRESHOLD = 0.45
DEFAULT_CONTRADICTION_THRESHOLD = 0.55


@dataclass
class ClaimResult:
    """Результат NLI верификации одного claim."""
    text: str
    claim_type: str  # verifiable / common_knowledge / meta
    nli_label: str | None = None  # entailment / neutral / contradiction
    nli_score: float = 0.0
    best_document_id: str | None = None
    best_chunk_idx: int = 0


@dataclass
class QuestionFaithfulness:
    """Faithfulness результат для одного вопроса."""
    query_id: str
    eval_mode: str
    faithfulness: float | None = None  # None для analytics/navigation/refusal
    faithfulness_strict: float | None = None
    claims_total: int = 0
    claims_verifiable: int = 0
    claims_supported: int = 0
    claims_contradicted: int = 0
    claims_neutral: int = 0
    claims_common_knowledge: int = 0
    nli_pairs_count: int = 0  # фактическое число NLI пар (с учётом chunking)
    per_claim: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)


class NLIVerifier:
    """HTTP client для gpu_server.py /nli endpoint."""

    def __init__(
        self,
        gpu_server_url: str = "http://localhost:8082",
        entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        max_doc_tokens: int = 400,
        chunk_overlap: int = 50,
        timeout: float = 30.0,
    ):
        self.gpu_server_url = gpu_server_url.rstrip("/")
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.max_doc_tokens = max_doc_tokens
        self.chunk_overlap = chunk_overlap
        self.timeout = timeout

    @staticmethod
    def _clean_premise(text: str) -> str:
        """Очистка текста документа от шума перед NLI.

        Убирает эмодзи, URLs, markdown — они не несут NLI-семантику
        и могут сбивать XLM-RoBERTa на informal русском тексте.
        """
        # Эмодзи
        text = re.sub(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
            r'\U0001F900-\U0001F9FF\U00002702-\U000027B0'
            r'\U00002600-\U000026FF\U0000FE00-\U0000FE0F'
            r'\u200d\u2B50\u26A1\u2764\u2705\u274C]+',
            ' ', text,
        )
        # URLs
        text = re.sub(r'https?://\S+', '', text)
        # Markdown bold/italic/code
        text = re.sub(r'[*_~`]{1,3}', '', text)
        # Hashtags: #AI → AI
        text = re.sub(r'#(\w+)', r'\1', text)
        # Множественные пробелы/переносы
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _chunk_document(self, text: str) -> list[str]:
        """Разбивает длинный документ на чанки по словам (~max_doc_tokens).

        Приблизительный подсчёт: 1 русский токен ≈ 3-4 символа.
        Для точности нужен tokenizer, но для chunking достаточно приближения.
        """
        # Приближение: ~400 tokens ≈ 1200-1600 chars для русского текста.
        # Точный tokenizer не нужен — XLM-RoBERTa truncation на 512 tokens
        # на стороне сервера обеспечивает safety net.
        max_chars = self.max_doc_tokens * 4
        overlap_chars = self.chunk_overlap * 4

        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            # Ищем ближайший конец предложения или пробел
            if end < len(text):
                # Пытаемся разрезать по концу предложения
                for sep in [". ", ".\n", "! ", "? "]:
                    cut = text.rfind(sep, start + max_chars // 2, end)
                    if cut > 0:
                        end = cut + 1
                        break
                else:
                    # Режем по пробелу
                    cut = text.rfind(" ", start + max_chars // 2, end)
                    if cut > 0:
                        end = cut
            chunks.append(text[start:end].strip())
            start = end - overlap_chars
        return [c for c in chunks if c]

    def _call_nli(self, pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Вызов /nli endpoint на gpu_server.py."""
        if not pairs:
            return []
        try:
            resp = httpx.post(
                f"{self.gpu_server_url}/nli",
                json={"pairs": pairs},
                timeout=self.timeout,
            )
            if resp.status_code == 503:
                raise RuntimeError("NLI model not loaded (503)")
            resp.raise_for_status()
            return resp.json().get("results", [])
        except httpx.TimeoutException:
            logger.error("NLI request timeout (%.1fs)", self.timeout)
            raise
        except Exception as exc:
            logger.error("NLI request failed: %s", exc)
            raise

    def verify_question(
        self,
        query_id: str,
        eval_mode: str,
        claims: list[dict[str, str]],
        documents: list[dict[str, Any]],
    ) -> QuestionFaithfulness:
        """Верифицирует claims одного вопроса против cited documents.

        claims: [{"text": "...", "type": "verifiable"}, ...]
        documents: [{"id": "...", "text": "..."}, ...]
        """
        result = QuestionFaithfulness(query_id=query_id, eval_mode=eval_mode)
        result.claims_total = len(claims)

        # Analytics/navigation/refusal → faithfulness=N/A
        if eval_mode != "retrieval_evidence":
            return result

        verifiable = [c for c in claims if c.get("type") == "verifiable"]
        result.claims_verifiable = len(verifiable)
        result.claims_common_knowledge = sum(
            1 for c in claims if c.get("type") == "common_knowledge"
        )

        if not verifiable:
            return result

        # Нет documents → все verifiable claims unsupported
        doc_texts_exist = any(d.get("text") for d in documents) if documents else False
        if not documents or not doc_texts_exist:
            result.faithfulness = 0.0
            result.faithfulness_strict = 0.0
            result.claims_neutral = len(verifiable)
            result.per_claim = [
                {"text": c["text"], "type": c["type"],
                 "nli_label": "neutral", "nli_score": 0.0,
                 "best_document_id": None, "best_chunk_idx": 0}
                for c in verifiable
            ]
            return result

        # Готовим пары claim × document chunks
        # Структура: (claim_idx, doc_id, chunk_idx, premise, hypothesis)
        all_pairs: list[dict[str, str]] = []
        pair_meta: list[dict[str, Any]] = []  # параллельный массив метаданных
        for ci, claim in enumerate(verifiable):
            for doc in documents:
                doc_text = doc.get("text", "")
                doc_id = doc.get("id", "unknown")
                if not doc_text:
                    continue
                cleaned = self._clean_premise(doc_text)
                chunks = self._chunk_document(cleaned)
                for chunk_idx, chunk in enumerate(chunks):
                    all_pairs.append({
                        "premise": chunk,
                        "hypothesis": claim["text"],
                    })
                    pair_meta.append({
                        "claim_idx": ci,
                        "doc_id": str(doc_id),
                        "chunk_idx": chunk_idx,
                    })

        if not all_pairs:
            result.faithfulness = 0.0
            result.faithfulness_strict = 0.0
            return result

        # Batch NLI
        result.nli_pairs_count = len(all_pairs)
        nli_results = self._call_nli(all_pairs)

        # Агрегация: per claim — best entailment + max contradiction across all doc chunks
        claim_best: dict[int, ClaimResult] = {}
        claim_scores_map: dict[int, dict[str, float]] = {}  # best entailment scores
        claim_max_contradiction: dict[int, tuple] = {}  # (score, doc_id)
        for ci, claim in enumerate(verifiable):
            claim_best[ci] = ClaimResult(
                text=claim["text"],
                claim_type=claim.get("type", "verifiable"),
            )
            claim_scores_map[ci] = {}
            claim_max_contradiction[ci] = (0.0, None)

        if len(nli_results) != len(all_pairs):
            logger.warning(
                "NLI results count mismatch: got %d, expected %d",
                len(nli_results), len(all_pairs),
            )

        for idx, nli_res in enumerate(nli_results):
            if idx >= len(pair_meta):
                break
            meta = pair_meta[idx]
            ci = int(meta["claim_idx"])
            scores = nli_res.get("scores", {})
            ent_score = scores.get("entailment", 0.0)
            con_score = scores.get("contradiction", 0.0)
            # Трекаем best entailment
            if ent_score > claim_best[ci].nli_score:
                claim_best[ci].nli_score = ent_score
                claim_best[ci].nli_label = str(nli_res.get("label", "neutral"))
                claim_best[ci].best_document_id = str(meta["doc_id"])
                claim_best[ci].best_chunk_idx = int(meta["chunk_idx"])
                claim_scores_map[ci] = scores
            # Трекаем max contradiction только от best-entailment документа.
            # Нерелевантные документы дают false positive contradiction
            # (MNLI bias: "документ о другом" → contradiction вместо neutral).
            if meta["doc_id"] == claim_best[ci].best_document_id:
                if con_score > claim_max_contradiction[ci][0]:
                    claim_max_contradiction[ci] = (con_score, meta["doc_id"])

        # Классификация и scoring
        lenient_scores = []
        strict_scores = []
        for ci, cr in sorted(claim_best.items()):
            scores = claim_scores_map.get(ci, {})
            ent = scores.get("entailment", cr.nli_score)
            # Contradiction только от best-entailment doc — избегаем false positives
            con = claim_max_contradiction[ci][0]

            if ent > self.entailment_threshold:
                cr.nli_label = "entailment"
                lenient_scores.append(1.0)
                strict_scores.append(1.0)
                result.claims_supported += 1
            elif con > self.contradiction_threshold:
                cr.nli_label = "contradiction"
                lenient_scores.append(0.0)
                strict_scores.append(0.0)
                result.claims_contradicted += 1
                result.contradictions.append({
                    "claim": cr.text,
                    "document_id": claim_max_contradiction[ci][1],
                    "contradiction_score": round(con, 4),
                })
            else:
                cr.nli_label = "neutral"
                lenient_scores.append(0.5)
                strict_scores.append(0.0)
                result.claims_neutral += 1

            result.per_claim.append({
                "text": cr.text,
                "type": cr.claim_type,
                "nli_label": cr.nli_label,
                "nli_score": round(cr.nli_score, 4),
                "best_document_id": cr.best_document_id,
                "best_chunk_idx": cr.best_chunk_idx,
            })

        result.faithfulness = round(sum(lenient_scores) / len(lenient_scores), 4)
        result.faithfulness_strict = round(sum(strict_scores) / len(strict_scores), 4)
        return result
