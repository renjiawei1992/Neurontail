#!/usr/bin/env python
"""Evaluate MCQA accuracy or LSS on generated outputs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.config import load_config  # noqa: E402
from a5_neurotail.data.schema import read_jsonl  # noqa: E402
from a5_neurotail.eval.lss import LanguageIdentifier, compute_lss  # noqa: E402
from a5_neurotail.eval.mcqa import evaluate_mcqa  # noqa: E402
from a5_neurotail.models.loader import load_model, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--languages", default=str(ROOT / "configs/languages.yaml"))
    ap.add_argument("--models", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--task", choices=["mcqa", "lss"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lid-backend", default=None)
    ap.add_argument("--lid-model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.languages, args.models)
    rows = read_jsonl(args.data)
    if args.task == "mcqa":
        if not args.model:
            raise SystemExit("--model required for mcqa")
        set_seed(int(cfg.get("seed", 0)))
        model, tokenizer = load_model(cfg["models"][args.model])
        res = evaluate_mcqa(
            model, tokenizer, rows,
            max_length=int(cfg["eval"].get("mcqa_max_length", 512)),
            limit=args.limit,
        )
        out = {"task": "mcqa", "model": args.model, "accuracy": res["accuracy"],
               "n": res["n"], "rows": res["rows"]}
    else:
        backend = args.lid_backend or cfg["eval"].get("lid_backend", "fasttext")
        lid_path = args.lid_model or cfg["eval"].get("lid_model")
        lid = LanguageIdentifier(backend=backend, model_path=lid_path)
        by_lang = defaultdict(list)
        for r in rows:
            by_lang[r["language"]].append(r["output"])
        out = {"task": "lss", "backend": backend, "languages": {}}
        for lang, outs in by_lang.items():
            res = compute_lss(outs, lang, lid)
            out["languages"][lang] = {"n": res["n"], "lss": res["lss"]}
            print(lang, res["lss"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
