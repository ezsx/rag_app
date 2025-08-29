from types import SimpleNamespace
from adapters.search.bm25_index import BM25IndexManager, BM25Doc, BM25Query
from adapters.search.bm25_retriever import BM25Retriever


class _Settings(SimpleNamespace):
    current_collection: str = "test"
    bm25_default_top_k: int = 10


def test_bm25_retriever(tmp_path):
    mgr = BM25IndexManager(index_root=str(tmp_path))
    coll = "test"
    docs = [
        BM25Doc(
            doc_id="1:1",
            text="документ про науку",
            channel_id=1,
            channel_username="@c",
            date_days=750000,
            date_iso="2024-01-01",
            views=1,
            reply_to=None,
            msg_id=1,
        ),
        BM25Doc(
            doc_id="1:2",
            text="наука и техника",
            channel_id=1,
            channel_username="@c",
            date_days=750001,
            date_iso="2024-01-02",
            views=2,
            reply_to=None,
            msg_id=2,
        ),
    ]
    mgr.add_documents(coll, docs, commit_every=1)
    settings = _Settings()
    ret = BM25Retriever(mgr, settings)
    res = ret.search("наука", plan=None, k=5)
    assert len(res) >= 1
    assert res[0].source == "bm25"
