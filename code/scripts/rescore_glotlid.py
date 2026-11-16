#!/usr/bin/env python
"""Re-score intervention outputs with GlotLID (covers fij/mri/zul — replacing
the lexicon fallback) and compare against the previous scoring."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fasttext  # noqa: E402

GLOTLID = "/home/ubuntu/.cache/huggingface/hub/models--cis-lmu--glotlid/snapshots/e9592e0a9bc3b6f2a946de94faaa188d1dd6f522/model.bin"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="comma-separated jsonl paths")
    ap.add_argument("--out", default=str(ROOT.parent / "outputs/eval/glotlid_rescore.csv"))
    args = ap.parse_args()

    m = fasttext.load_model(GLOTLID)
    import csv
    rows = []
    for f in [x.strip() for x in args.files.split(",") if x.strip()]:
        p = Path(f)
        if not p.exists():
            print("missing:", f)
            continue
        data = [json.loads(l) for l in open(p, encoding="utf-8")]
        lang = data[0].get("language", p.stem.split("_")[0] + "_Latn")
        hits = 0
        preds = {}
        for r in data:
            pred = m.predict(r["output"].replace("\n", " "))[0][0].replace("__label__", "")
            preds[pred] = preds.get(pred, 0) + 1
            hits += int(pred == lang)
        lss = hits / max(len(data), 1)
        rows.append({"file": p.stem, "lang": lang, "n": len(data), "glotlid_lss": round(lss, 3),
                     "top_preds": str(sorted(preds.items(), key=lambda x: -x[1])[:3])})
        print(f"{p.stem:34s} {lang:12s} LSS={lss:.3f} top={rows[-1]['top_preds']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "lang", "n", "glotlid_lss", "top_preds"])
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
