from __future__ import annotations

import logging
from typing import List, Dict, Any

from schemas.search import SearchPlan, Candidate
from utils.ranking import rrf_merge

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, settings):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.settings = settings

    def search_with_plan(self, query_text: str, plan: SearchPlan) -> List[Candidate]:
        # BM25 ветка
        bm25_candidates = self.bm25.search(
            query_text, plan, k=self.settings.hybrid_top_bm25
        )

        # Dense ветка — собираем между под‑запросами и сливаем RRF
        dense_lists_items: List[List[Dict[str, Any]]] = []
        dense_filters: Dict[str, Any] = (
            plan.metadata_filters.dict(exclude_none=True)
            if plan.metadata_filters
            else {}
        )
        for q in plan.normalized_queries or [query_text]:
            try:
                items = self.dense.search(
                    q, k=self.settings.hybrid_top_dense, filters=dense_filters
                )
            except Exception as exc:
                logger.error("Dense search failed for query '%s': %s", q, exc)
                logger.debug("Dense filters: %s", dense_filters)
                items = []
            dense_lists_items.append(items)

        # Сливаем dense по под‑запросам через RRF
        dense_fused: List[Candidate] = []
        if any(dense_lists_items):
            # подготовим формат для rrf_merge: (doc, distance, meta)
            rrf_input = []
            for lst in dense_lists_items:
                triples = [
                    (
                        it.get("text", ""),
                        float(it.get("distance", 0.0)),
                        it.get("metadata", {}),
                    )
                    for it in lst
                ]
                rrf_input.append(triples)
            fused = rrf_merge(rrf_input, k=self.settings.k_fusion)
            # обратно в Candidate, восстановив id по метаданным/хешу
            for doc, _dist, meta in fused:
                doc_id = None
                if isinstance(meta, dict):
                    cid = meta.get("channel_id")
                    mid = meta.get("msg_id") or meta.get("message_id")
                    if cid is not None and mid is not None:
                        doc_id = f"{cid}:{mid}"
                if not doc_id:
                    doc_id = f"hash:{hash(doc)}"
                dense_fused.append(
                    Candidate(
                        id=doc_id,
                        text=doc,
                        metadata=meta,
                        bm25_score=None,
                        dense_score=None,  # уже RRF
                        source="dense",
                    )
                )

        # Объединяем BM25 и Dense, снова RRF
        rrf_all_input = []
        # BM25 → (doc, distance, meta); RRF учитывает только ранги, потому проставим rank как distance
        rrf_all_input.append(
            [
                (c.text, float(rank), c.metadata)
                for rank, c in enumerate(bm25_candidates)
            ]
        )
        if dense_fused:
            rrf_all_input.append(
                [
                    (c.text, float(rank), c.metadata)
                    for rank, c in enumerate(dense_fused)
                ]
            )

        if len(rrf_all_input) == 1:
            # RRF из одной ветки — это исходные BM25 кандидаты
            fused_all = [
                (c.text, float(rank), c.metadata)
                for rank, c in enumerate(bm25_candidates)
            ]
        else:
            fused_all = rrf_merge(rrf_all_input, k=self.settings.k_fusion)

        # Дедуп по id
        seen = set()
        out: List[Candidate] = []
        for doc, _dist, meta in fused_all:
            doc_id = None
            if isinstance(meta, dict):
                cid = meta.get("channel_id")
                mid = meta.get("msg_id") or meta.get("message_id")
                if cid is not None and mid is not None:
                    doc_id = f"{cid}:{mid}"
            if not doc_id:
                doc_id = f"hash:{hash(doc)}"
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(
                Candidate(
                    id=doc_id,
                    text=doc,
                    metadata=meta,
                    bm25_score=None,
                    dense_score=None,
                    source="hybrid",
                )
            )
        return out
