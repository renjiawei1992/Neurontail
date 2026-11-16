#!/usr/bin/env python
"""Aggregate intervention results into CSG tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.eval.csg import causal_specificity_gap  # noqa: E402


def load_rows(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    data = json.load(open(p, encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(load_rows(path))
    if not rows:
        raise SystemExit("no rows")
    df = pd.DataFrame(rows)
    if {"on_target_drop", "off_target_drop", "random_drop"} <= set(df.columns):
        df["csg"] = [
            causal_specificity_gap(r["on_target_drop"], r["off_target_drop"], r["random_drop"])
            for _, r in df.iterrows()
        ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    if "csg" in df.columns:
        cols = [c for c in ("language", "model", "method") if c in df.columns]
        if cols:
            print(df.groupby(cols)["csg"].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
