#!/usr/bin/env python
"""M2 control: random-set steering (20 sets) vs LAPE steering for long-tail langs."""
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
from a5_neurotail.intervene.steering import steer_generate  # noqa: E402
from a5_neurotail.models.loader import load_model, set_seed  # noqa: E402
from a5_neurotail.models.modules import find_mlp_refs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--langs", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    cfg = load_config(
        ROOT / "configs/experiments.yaml",
        ROOT / "configs/languages.yaml",
        ROOT / "configs/models.yaml",
        ROOT / "configs/paths.yaml",
    )
    out_root = ROOT.parent / "outputs"
    key = f"L{args.layer}"
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    set_seed(int(cfg.get("seed", 0)))
    model, tokenizer = load_model(cfg["models"][args.model])
    refs = find_mlp_refs(model)
    ref = next(r for r in refs if r.key == key)
    n_neurons = ref.intermediate_size

    base_dir = out_root / "interventions" / "m2" / args.model
    sel = json.load(open(out_root / "neurons" / args.model / "lape" / f"{langs[0]}" / f"{key}.json"))
    k = len(sel["selected"])
    for lang in langs:
        p = Path(cfg["paths"]["processed_dir"]) / "prompts" / f"lss_{lang}.jsonl"
        prompts = [r["prompt"] for r in read_jsonl(p)][: args.limit]
        for s in range(args.n_random):
            rng = np.random.default_rng(2000 + s)
            rand = np.sort(rng.choice(n_neurons, size=k, replace=False))
            outs = steer_generate(model, tokenizer, prompts, refs, key, rand, args.alpha,
                                  max_new_tokens=args.max_new_tokens)
            out = base_dir / f"{lang}_steer_rand{s:02d}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for pr, o in zip(prompts, outs):
                    f.write(json.dumps({"prompt": pr, "output": o, "language": lang,
                                        "model": args.model, "key": key, "method": "steering",
                                        "variant": "random", "alpha": args.alpha, "seed": s},
                                       ensure_ascii=False) + "\n")
        print(f"[{lang}] {args.n_random} random steering sets done", flush=True)
    print("random steering block complete")


if __name__ == "__main__":
    main()
