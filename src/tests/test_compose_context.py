from services.tools.compose_context import compose_context


def test_compose_context_basic_limit():
    docs = [
        {"id": "1", "text": "a" * 2000, "metadata": {}},
        {"id": "2", "text": "b" * 2000, "metadata": {}},
    ]
    out = compose_context(docs, max_tokens_ctx=100)  # ~400 chars
    prompt = out["prompt"]
    citations = out["citations"]
    contexts = out["contexts"]
    assert len(prompt) <= 400 + 20  # запас на префиксы [i]
    assert len(citations) == len(contexts)
    assert citations[0]["id"] == "1" and citations[0]["index"] == 1
    assert citations[-1]["index"] == len(citations)
