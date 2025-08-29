from adapters.search.bm25_index import BM25IndexManager, BM25Doc, BM25Query


def test_bm25_index_roundtrip(tmp_path):
    mgr = BM25IndexManager(index_root=str(tmp_path))
    coll = "test"
    docs = [
        BM25Doc(
            doc_id="1:1",
            text="тестовый документ по экономике",
            channel_id=1,
            channel_username="@ch1",
            date_days=750000,
            date_iso="2024-01-01",
            views=10,
            reply_to=None,
            msg_id=1,
        ),
        BM25Doc(
            doc_id="1:2",
            text="новости спорта и экономики",
            channel_id=1,
            channel_username="@ch1",
            date_days=750001,
            date_iso="2024-01-02",
            views=20,
            reply_to=None,
            msg_id=2,
        ),
    ]
    mgr.add_documents(coll, docs, commit_every=1)
    q = BM25Query(must_terms=["экономике"], should_terms=[], filters={})
    hits = mgr.search(coll, q, top_k=5)
    assert len(hits) >= 1
    assert any(h.doc_id == "1:1" for h in hits)
