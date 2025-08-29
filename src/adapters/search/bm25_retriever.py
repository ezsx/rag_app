from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any

from adapters.search.bm25_index import BM25IndexManager, BM25Query
from schemas.search import SearchPlan, Candidate

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, index_manager: BM25IndexManager, settings):
        self.index_manager = index_manager
        self.settings = settings

    def _date_to_days(self, iso_str: Optional[str]) -> Optional[int]:
        if not iso_str:
            return None
        try:
            import datetime as dt

            if len(iso_str) >= 10:
                y, m, d = map(int, iso_str[:10].split("-"))
                return dt.date(y, m, d).toordinal()
        except Exception:
            return None
        return None

    def search(
        self, query_text: str, plan: Optional[SearchPlan], k: int
    ) -> List[Candidate]:
        collection = self.settings.current_collection

        must_terms: List[str] = []
        should_terms: List[str] = []
        filters: Dict[str, Any] = {}

        if plan:
            must_terms = [t for t in plan.must_phrases if t]
            should_terms = [t for t in plan.should_phrases if t]
            # Добавим normalized_queries в should для расширения
            should_terms.extend([q for q in plan.normalized_queries if q])

            mf = (
                plan.metadata_filters.dict(exclude_none=True)
                if plan.metadata_filters
                else {}
            )
            if mf.get("channel_usernames"):
                filters["channel_usernames"] = mf["channel_usernames"]
            if mf.get("channel_ids"):
                filters["channel_ids"] = mf["channel_ids"]
            if mf.get("reply_to") is not None:
                filters["reply_to"] = mf["reply_to"]
            if mf.get("min_views") is not None:
                filters["min_views"] = mf["min_views"]
            if mf.get("date_from"):
                filters["date_from_days"] = self._date_to_days(mf["date_from"])  # type: ignore
            if mf.get("date_to"):
                filters["date_to_days"] = self._date_to_days(mf["date_to"])  # type: ignore

        q = BM25Query(must_terms=must_terms, should_terms=should_terms, filters=filters)
        hits = self.index_manager.search(
            collection, q, top_k=k or self.settings.bm25_default_top_k
        )

        result: List[Candidate] = []
        for h in hits:
            result.append(
                Candidate(
                    id=h.doc_id,
                    text=h.text,
                    metadata=h.metadata,
                    bm25_score=h.bm25_score,
                    dense_score=None,
                    source="bm25",
                )
            )
        return result
