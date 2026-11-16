#!/usr/bin/env python
"""Prepare unified JSONL data from FLORES, Belebele, and native CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.config import languages_for_run, load_config  # noqa: E402
from a5_neurotail.data.loaders import (  # noqa: E402
    load_belebele,
    load_flores_parallel,
    load_native_csv,
)
from a5_neurotail.data.prompts import language_forcing_prompts  # noqa: E402
from a5_neurotail.data.schema import write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--languages", default=str(ROOT / "configs/languages.yaml"))
    ap.add_argument("--models", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--paths", default=None)
    ap.add_argument("--flores-dir", default=None)
    ap.add_argument("--belebele-dir", default=None)
    ap.add_argument("--native-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--n-lss", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config, args.languages, args.models, args.paths)
    out = Path(args.out_dir or cfg["paths"]["processed_dir"])
    flores = Path(args.flores_dir or (Path(cfg["paths"]["raw_dir"]) / "flores"))
    belebele = Path(args.belebele_dir or (Path(cfg["paths"]["raw_dir"]) / "belebele/extracted/Belebele"))
    native = Path(args.native_dir or (Path(cfg["paths"]["processed_dir"]) / "native"))
    for sub in ("parallel", "mcqa", "prompts"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    langs = languages_for_run(cfg, args.main_only)
    summary = {"parallel": {}, "mcqa": {}, "prompts": {}}

    for lang in langs:
        code = lang["code"]
        try:
            rows = load_flores_parallel(flores, code)
            n = write_jsonl(out / "parallel" / f"{code}.jsonl", rows)
            summary["parallel"][code] = n
        except FileNotFoundError:
            summary["parallel"][code] = 0
        p = belebele / f"{code}.jsonl"
        if p.exists():
            rows = load_belebele(p, code)
            n = write_jsonl(out / "mcqa" / f"belebele_{code}.jsonl", rows)
            summary["mcqa"][f"belebele_{code}"] = n
        p = native / f"{code}.csv"
        if p.exists():
            rows = load_native_csv(p, code)
            n = write_jsonl(out / "mcqa" / f"native_{code}.jsonl", rows)
            summary["mcqa"][f"native_{code}"] = n
        prompts = language_forcing_prompts(lang["name"], args.n_lss)
        n = write_jsonl(out / "prompts" / f"lss_{code}.jsonl", prompts)
        summary["prompts"][code] = n
        print(f"{code}: parallel={summary['parallel'][code]} lss={n}")

    print("\nsummary:", summary)


if __name__ == "__main__":
    main()
