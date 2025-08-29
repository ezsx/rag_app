from types import SimpleNamespace

from adapters.search.hybrid_retriever import HybridRetriever
from adapters.search.bm25_index import BM25IndexManager, BM25Doc
from adapters.search.bm25_retriever import BM25Retriever


class _DenseStub:
    def __init__(self, items):
        self._items = items

    def search(self, q, k, filters=None):
        return self._items


class _Settings(SimpleNamespace):
    hybrid_top_bm25: int = 10
    hybrid_top_dense: int = 10
    k_fusion: int = 60


def test_hybrid_basic(tmp_path):
    mgr = BM25IndexManager(index_root=str(tmp_path))
    coll = "test"
    docs = [
        BM25Doc(
            doc_id="1:1",
            text="bm25 документ",
            channel_id=1,
            channel_username="@c",
            date_days=750000,
            date_iso="2024-01-01",
            views=1,
            reply_to=None,
            msg_id=1,
        )
    ]
    mgr.add_documents(coll, docs, commit_every=1)
    bm25_ret = BM25Retriever(
        mgr, SimpleNamespace(current_collection=coll, bm25_default_top_k=10)
    )
    dense_stub = _DenseStub(
        [
            {
                "text": "dense документ",
                "distance": 0.1,
                "metadata": {"channel_id": 1, "msg_id": 2},
            },
        ]
    )
    settings = _Settings()
    hybrid = HybridRetriever(bm25_ret, dense_stub, SimpleNamespace(**settings.__dict__))

    from schemas.search import SearchPlan

    plan = SearchPlan(
        normalized_queries=["документ"],
        must_phrases=[],
        should_phrases=[],
        metadata_filters=None,
        k_per_query=5,
        fusion="rrf",
    )

    res = hybrid.search_with_plan("документ", plan)
    assert len(res) >= 1
    ids = {c.id for c in res}
    assert any(i.startswith("1:") for i in ids)
