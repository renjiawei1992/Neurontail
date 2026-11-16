#!/usr/bin/env python
"""M2 task-side necessity: MCQA accuracy under LAPE steering vs random steerings.

For each language with Belebele: base acc, LAPE-steering acc, 3 random-steering accs.
Task CSG = acc_gain(lape) - max(acc_drop(rand)).
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
from a5_neurotail.eval.mcqa import evaluate_mcqa  # noqa: E402
from a5_neurotail.models.hooks import AddHook, intervene  # noqa: E402
from a5_neurotail.models.loader import load_model, set_seed  # noqa: E402
from a5_neurotail.models.modules import find_mlp_refs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--langs", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--limit", type=int, default=200)
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
    results = {}

    for lang in langs:
        p = Path(cfg["paths"]["processed_dir"]) / "mcqa" / f"belebele_{lang}.jsonl"
        if not p.exists():
            print(f"[{lang}] no belebele, skip", flush=True)
            continue
        items = read_jsonl(p)[: args.limit]
        sel = json.load(open(out_root / "neurons" / args.model / "lape" / lang / f"{key}.json"))["selected"]
        base = evaluate_mcqa(model, tokenizer, items)["accuracy"]
        ctx = lambda idx: intervene([ref], {key: AddHook(idx, 3.0)})
        lape = evaluate_mcqa(model, tokenizer, items, ctx=ctx(sel))["accuracy"]
        rands = []
        for s in range(args.n_random):
            rng = np.random.default_rng(4000 + s)
            rand = np.sort(rng.choice(n_neurons, size=len(sel), replace=False))
            rands.append(evaluate_mcqa(model, tokenizer, items, ctx=ctx(rand))["accuracy"])
        rand_mean = float(np.mean(rands))
        # specific gain = LAPE gain - strongest control gain (gain = acc - base)
        task_gain = (lape - base) - max(rand_mean - base, 0.0)
        results[lang] = {"base": base, "steer_lape": lape, "steer_rand": rands,
                         "task_specific_gain": task_gain}
        print(f"{lang:12s} base={base:.3f} steer_lape={lape:.3f} steer_rand={[round(r,3) for r in rands]} "
              f"task_specific_gain={task_gain:+.3f}", flush=True)

    out = out_root / "eval" / f"m2_task_steer_{args.model}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
