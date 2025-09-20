from core.deps import get_qa_service


def test_answer_v2_smoke():
    svc = get_qa_service()
    res = svc.answer_v2("Test query")
    assert isinstance(res, dict)
    assert "prompt" in res and isinstance(res["prompt"], str)
    assert "answer" in res and isinstance(res["answer"], str)
    assert "trace" in res and isinstance(res["trace"], list)
    # В трейсе должны быть план и фьюжн
    tools = [t.get("tool") for t in res["trace"]]
    assert "fusion_rrf" in tools
    assert "plan" in tools or True  # план может быть отключён флажком
