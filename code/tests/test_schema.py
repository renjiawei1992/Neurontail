import json
import tempfile
from pathlib import Path

from a5_neurotail.data.schema import MCExample, read_jsonl, write_jsonl


def test_mc_roundtrip():
    row = MCExample(
        id="x",
        language="ton_Latn",
        passage="p",
        question="q",
        options=["a", "b", "c", "d"],
        answer_idx=2,
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        assert write_jsonl(p, [row]) == 1
        got = read_jsonl(p)
    assert got[0]["answer_idx"] == 2
    assert got[0]["options"][3] == "d"
    json.dumps(got[0])
