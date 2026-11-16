#!/usr/bin/env python
"""B3 donor-patching matrix: patch target-language LAPE units with donor means.

For each (target, donor) pair: generate on target-language prompts with target
LAPE neurons replaced by donor-language mean activations; score LSS (target)
and LSS (donor) to compute transfer gain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.config import load_config  # noqa: E402
from a5_neurotail.data.schema import read_jsonl  # noqa: E402
from a5_neurotail.models.hooks import PatchHook, intervene  # noqa: E402
from a5_neurotail.models.loader import load_model, set_seed  # noqa: E402
from a5_neurotail.models.modules import find_mlp_refs  # noqa: E402
from a5_neurotail.models.stats_io import load_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True, help="comma-separated target:donor codes")
    ap.add_argument("--layers", default="20,24,28", help="comma-separated layer indices to patch")
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
    keys = [f"L{x.strip()}" for x in args.layers.split(",") if x.strip()]
    act_dir = out_root / "activations" / args.model
    sel_dir = out_root / "neurons" / args.model / "lape"

    set_seed(int(cfg.get("seed", 0)))
    model, tokenizer = load_model(cfg["models"][args.model])
    refs = find_mlp_refs(model)

    donor_stats = {}
    base_dir = out_root / "interventions" / "b3" / args.model
    for pair in [x.strip() for x in args.pairs.split(",") if x.strip()]:
        target, donor = pair.split(":")
        if donor not in donor_stats:
            donor_stats[donor] = load_stats(act_dir / f"{donor}.npz")
        p = Path(cfg["paths"]["processed_dir"]) / "prompts" / f"lss_{target}.jsonl"
        prompts = [r["prompt"] for r in read_jsonl(p)][: args.limit]
        # multi-key patch: patch target LAPE units with donor means at every key
        hooks = {}
        for key in keys:
            sel = json.load(open(sel_dir / target / f"{key}.json"))["selected"]
            values = donor_stats[donor][key]["mean"][np.asarray(sel, dtype=int)]
            hooks[key] = PatchHook(sel, values.tolist())
        device = next(model.parameters()).device
        outs = []
        with torch.inference_mode():
            with intervene(refs, hooks):
                for start in range(0, len(prompts), 8):
                    batch = prompts[start : start + 8]
                    enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, pad_token_id=tokenizer.pad_token_id)
                    new = gen[:, enc["input_ids"].shape[1] :]
                    outs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
        out = base_dir / f"{target}_patch_{donor}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for pr, o in zip(prompts, outs):
                f.write(json.dumps({"prompt": pr, "output": o, "language": target,
                                    "model": args.model, "keys": keys, "method": "donor_patch_multikey",
                                    "donor": donor}, ensure_ascii=False) + "\n")
        print(f"{target} <- {donor}: wrote {len(outs)} outputs (keys={keys})", flush=True)
    print("B3 donor matrix complete")


if __name__ == "__main__":
    main()
