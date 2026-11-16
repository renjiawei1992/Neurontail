#!/usr/bin/env python
"""M2 CSG scoring (CPU): LSS on all m2 outputs + CSG table.

CSG = on_target_drop - max(off_target_drop, random_drop)
where drop = baseline_LSS - intervened_LSS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.eval.lss import LanguageIdentifier, compute_lss  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def lss_of(path: Path, lang: str, lid) -> float:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return compute_lss([r["output"] for r in rows], lang, lid)["lss"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--langs", required=True)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--lid", default=str(ROOT.parent / "data/raw/lid.176.ftz"))
    ap.add_argument("--out-root", default=str(ROOT.parent / "outputs"))
    args = ap.parse_args()

    lid = LanguageIdentifier("fasttext", args.lid)
    base = Path(args.out_root) / "interventions" / "m2" / args.model
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    rows = []
    for lang in langs:
        lss_base = lss_of(base / f"{lang}_base.jsonl", lang, lid)
        lss_lape = lss_of(base / f"{lang}_ko_lape.jsonl", lang, lid)
        lss_steer = lss_of(base / f"{lang}_steer_lape.jsonl", lang, lid)
        off_file = base / f"{lang}_ko_off_eng_Latn.jsonl"
        if not off_file.exists():
            off_file = base / f"{lang}_ko_off_bod_Tibt.jsonl"
        lss_off = lss_of(off_file, lang, lid)
        rand_lss = [lss_of(base / f"{lang}_ko_rand{s:02d}.jsonl", lang, lid)
                    for s in range(args.n_random)]
        rand_mean = float(np.mean(rand_lss))
        rand_max = float(np.max(rand_lss))
        on_drop = lss_base - lss_lape
        off_drop = lss_base - lss_off
        rand_drop = lss_base - rand_mean
        csg_mean = on_drop - max(off_drop, rand_drop)
        csg_max = on_drop - max(off_drop, lss_base - rand_max)
        rows.append({
            "language": lang,
            "lss_base": lss_base, "lss_ko_lape": lss_lape, "lss_steer": lss_steer,
            "lss_off": lss_off, "lss_rand_mean": rand_mean, "lss_rand_std": float(np.std(rand_lss)),
            "on_drop": on_drop, "off_drop": off_drop, "rand_drop": rand_drop,
            "csg_mean": csg_mean, "csg_max": csg_max,
        })
        print(f"{lang:12s} base={lss_base:.2f} ko={lss_lape:.2f} steer={lss_steer:.2f} "
              f"off={lss_off:.2f} rand={rand_mean:.2f}±{np.std(rand_lss):.2f} "
              f"CSG_mean={csg_mean:+.3f} CSG_max={csg_max:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.out_root) / "eval" / f"m2_csg_{args.model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
