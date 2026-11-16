#!/usr/bin/env python
"""B2 layer-sweep: LAPE knockout per layer with random control (necessity-by-layer)."""
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
from a5_neurotail.models.hooks import KnockoutHook, intervene  # noqa: E402
from a5_neurotail.models.loader import load_model, set_seed  # noqa: E402
from a5_neurotail.models.modules import find_mlp_refs  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--langs", required=True)
    ap.add_argument("--layers", default="8,16,20,24")
    ap.add_argument("--n-random", type=int, default=1)
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
    sel_dir = out_root / "neurons" / args.model / "lape"
    keys = [f"L{x.strip()}" for x in args.layers.split(",") if x.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    set_seed(int(cfg.get("seed", 0)))
    model, tokenizer = load_model(cfg["models"][args.model])
    refs = find_mlp_refs(model)
    ref_map = {r.key: r for r in refs}
    device = next(model.parameters()).device

    base_dir = out_root / "interventions" / "b2sweep" / args.model
    for lang in langs:
        p = Path(cfg["paths"]["processed_dir"]) / "prompts" / f"lss_{lang}.jsonl"
        prompts = [r["prompt"] for r in read_jsonl(p)][: args.limit]
        for key in keys:
            sel = json.load(open(sel_dir / lang / f"{key}.json"))["selected"]
            # LAPE knockout
            with torch.inference_mode():
                with intervene([ref_map[key]], {key: KnockoutHook(sel)}):
                    outs = []
                    for s in range(0, len(prompts), 8):
                        batch = prompts[s : s + 8]
                        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                        enc = {k: v.to(device) for k, v in enc.items()}
                        g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=tokenizer.pad_token_id)
                        new = g[:, enc["input_ids"].shape[1] :]
                        outs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
            out = base_dir / f"{lang}_ko_{key}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for pr, o in zip(prompts, outs):
                    f.write(json.dumps({"prompt": pr, "output": o, "language": lang,
                                        "model": args.model, "key": key, "method": "knockout",
                                        "variant": "lape"}, ensure_ascii=False) + "\n")
            # random control
            for rs in range(args.n_random):
                rng = np.random.default_rng(9000 + rs)
                rand = np.sort(rng.choice(ref_map[key].intermediate_size, size=len(sel), replace=False))
                with torch.inference_mode():
                    with intervene([ref_map[key]], {key: KnockoutHook(rand)}):
                        outs = []
                        for s in range(0, len(prompts), 8):
                            batch = prompts[s : s + 8]
                            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                            enc = {k: v.to(device) for k, v in enc.items()}
                            g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                               do_sample=False, pad_token_id=tokenizer.pad_token_id)
                            new = g[:, enc["input_ids"].shape[1] :]
                            outs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
                out = base_dir / f"{lang}_ko_rand_{key}_s{rs}.jsonl"
                with open(out, "w", encoding="utf-8") as f:
                    for pr, o in zip(prompts, outs):
                        f.write(json.dumps({"prompt": pr, "output": o, "language": lang,
                                            "model": args.model, "key": key, "method": "knockout",
                                            "variant": "random", "seed": rs}, ensure_ascii=False) + "\n")
            print(f"[{lang}] {key} lape + {args.n_random} rand done", flush=True)
    print("layer sweep complete")


if __name__ == "__main__":
    main()
