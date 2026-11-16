#!/usr/bin/env python
"""Build compact paper tables from a results CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.analysis.tables import save_table, summarize_csg, summarize_lss  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", choices=["csg", "lss"], default="csg")
    ap.add_argument("--by", default="language,model")
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    by = [x.strip() for x in args.by.split(",") if x.strip()]
    table = summarize_csg(df, by) if args.metric == "csg" else summarize_lss(df, by)
    save_table(table, args.out)
    print(table.to_string(index=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
