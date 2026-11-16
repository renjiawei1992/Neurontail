from a5_neurotail.data.fertility import text_stats


def test_text_stats_basic():
    s = text_stats(["hello world", "this is a test"])
    assert s["lines"] == 2
    assert s["words"] == 6
    assert s["chars"] > 0
    assert s["bytes"] >= s["chars"]
