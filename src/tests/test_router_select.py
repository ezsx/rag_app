from services.tools.router_select import router_select


def test_router_select_short_numeric():
    assert router_select("usd 2024-09")["route"] == "bm25"


def test_router_select_entity():
    assert router_select("@channel updates")["route"] == "bm25"


def test_router_select_conversational():
    r = router_select("почему вырос курс нефти в 2023?")
    assert r["route"] in ("dense", "hybrid")


def test_router_select_filters_mixed():
    r = router_select("channel:news after:2024-01 what happened?")
    assert r["route"] == "hybrid"


def test_router_select_long_text():
    q = """This is a very long conversational query about macroeconomic indicators and their correlation
    with crude oil prices over the last five years including multiple narratives and explanatory notes"""
    assert router_select(q)["route"] == "dense"
