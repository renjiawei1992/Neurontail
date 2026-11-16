#!/usr/bin/env python
"""M2 / B2 causal decomposition driver (ARIS experiment-bridge).

Per language: baseline, LAPE knockout (L12, 1%), 20 random-control knockouts,
LAPE steering (alpha=3), off-target knockout. Writes JSONL to
outputs/interventions/m2/. LSS/CSG scoring is a separate step (CPU).

Usage:
  m2_causal_block.py --model qwen25_7b --langs eng_Latn,bod_Tibt,... \
      [--layer 12] [--n-random 20] [--limit 100] [--max-new-tokens 48]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.config import load_config  # noqa: E402
from a5_neurotail.data.schema import read_jsonl  # noqa: E402
from a5_neurotail.intervene.knockout import knockout_generate, random_indices  # noqa: E402
from a5_neurotail.intervene.steering import steer_generate  # noqa: E402
from a5_neurotail.models.loader import load_model, set_seed  # noqa: E402
from a5_neurotail.models.modules import find_mlp_refs  # noqa: E402


def load_selection(path: Path) -> list[int]:
    return [int(x) for x in json.load(open(path, encoding="utf-8"))["selected"]]


def write_jsonl(path: Path, prompts, outs, meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, o in zip(prompts, outs):
            f.write(json.dumps({"prompt": p, "output": o, **meta}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--langs", required=True, help="comma-separated FLORES codes")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--out-root", default=str(ROOT.parent / "outputs"))
    args = ap.parse_args()

    cfg = load_config(
        ROOT / "configs/experiments.yaml",
        ROOT / "configs/languages.yaml",
        ROOT / "configs/models.yaml",
        ROOT / "configs/paths.yaml",
    )
    out_root = Path(args.out_root)
    sel_root = out_root / "neurons" / args.model / "lape"
    key = f"L{args.layer}"
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    set_seed(int(cfg.get("seed", 0)))
    model, tokenizer = load_model(cfg["models"][args.model])
    refs = find_mlp_refs(model)
    ref = next(r for r in refs if r.key == key)
    n_neurons = ref.intermediate_size

    base_dir = out_root / "interventions" / "m2" / args.model
    for lang in langs:
        p = Path(cfg["paths"]["processed_dir"]) / "prompts" / f"lss_{lang}.jsonl"
        prompts = [r["prompt"] for r in read_jsonl(p)][: args.limit]
        sel = load_selection(sel_root / lang / f"{key}.json")
        meta = {"language": lang, "model": args.model, "key": key, "n_neurons": len(sel)}
        print(f"[{lang}] {len(prompts)} prompts, {len(sel)} selected units", flush=True)

        # baseline (none)
        from a5_neurotail.eval.lss import generate_plain
        outs = generate_plain(model, tokenizer, prompts, max_new_tokens=args.max_new_tokens)
        write_jsonl(base_dir / f"{lang}_base.jsonl", prompts, outs, {**meta, "method": "none"})
        print(f"[{lang}] baseline done", flush=True)

        # LAPE knockout (necessity)
        outs = knockout_generate(model, tokenizer, prompts, refs, key, sel,
                                 max_new_tokens=args.max_new_tokens)
        write_jsonl(base_dir / f"{lang}_ko_lape.jsonl", prompts, outs, {**meta, "method": "knockout", "variant": "lape"})
        print(f"[{lang}] lape knockout done", flush=True)

        # steering (sufficiency)
        outs = steer_generate(model, tokenizer, prompts, refs, key, sel, args.alpha,
                              max_new_tokens=args.max_new_tokens)
        write_jsonl(base_dir / f"{lang}_steer_lape.jsonl", prompts, outs,
                    {**meta, "method": "steering", "variant": "lape", "alpha": args.alpha})
        print(f"[{lang}] lape steering done", flush=True)

        # 20 random-control knockouts
        for s in range(args.n_random):
            rand = random_indices(n_neurons, len(sel), seed=1000 + s)
            outs = knockout_generate(model, tokenizer, prompts, refs, key, rand,
                                     max_new_tokens=args.max_new_tokens)
            write_jsonl(base_dir / f"{lang}_ko_rand{s:02d}.jsonl", prompts, outs,
                        {**meta, "method": "knockout", "variant": "random", "seed": s})
        print(f"[{lang}] {args.n_random} random knockouts done", flush=True)

        # off-target knockout (use English units for non-English; Tibetan for English)
        off_lang = "eng_Latn" if lang != "eng_Latn" else "bod_Tibt"
        off_sel = load_selection(sel_root / off_lang / f"{key}.json")
        outs = knockout_generate(model, tokenizer, prompts, refs, key, off_sel,
                                 max_new_tokens=args.max_new_tokens)
        write_jsonl(base_dir / f"{lang}_ko_off_{off_lang}.jsonl", prompts, outs,
                    {**meta, "method": "knockout", "variant": "off_target", "off_lang": off_lang})
        print(f"[{lang}] off-target knockout ({off_lang}) done", flush=True)

    print("M2 generation block complete")


if __name__ == "__main__":
    main()
