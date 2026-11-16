#!/usr/bin/env python
"""Run knockout or steering on language-forcing prompts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.config import load_config  # noqa: E402
from a5_neurotail.data.prompts import language_forcing_prompts  # noqa: E402
from a5_neurotail.data.schema import read_jsonl, write_jsonl  # noqa: E402
from a5_neurotail.intervene.knockout import knockout_generate  # noqa: E402
from a5_neurotail.intervene.steering import steer_generate  # noqa: E402
from a5_neurotail.eval.lss import generate_plain  # noqa: E402
from a5_neurotail.models.loader import load_model, set_seed  # noqa: E402
from a5_neurotail.models.modules import find_mlp_refs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--languages", default=str(ROOT / "configs/languages.yaml"))
    ap.add_argument("--models", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--paths", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--selection", default=None, help="JSON from identify_neurons.py")
    ap.add_argument("--key", default=None, help="MLP key, e.g. L12")
    ap.add_argument("--language", required=True)
    ap.add_argument("--method", choices=["knockout", "steering", "none"], required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k-frac", type=float, default=None)
    ap.add_argument("--indices", default=None, help="comma-separated; overrides k-frac")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.languages, args.models, args.paths)
    set_seed(int(cfg.get("seed", 0)))
    indices: list[int] = []
    if args.method != "none":
        if not args.selection or not args.key:
            raise SystemExit("--selection and --key required for knockout/steering")
        with open(args.selection, encoding="utf-8") as f:
            sel = json.load(f)
        if args.indices:
            indices = [int(x) for x in args.indices.split(",")]
        else:
            indices = list(sel.get("selected", []))
            if args.k_frac:
                indices = indices[: max(1, int(round(len(indices) * args.k_frac / 0.01)))]
        if not indices:
            raise SystemExit("empty neuron selection")

    p = Path(args.prompts) if args.prompts else (
        Path(cfg["paths"]["processed_dir"]) / "prompts" / f"lss_{args.language}.jsonl"
    )
    if p.exists():
        rows = read_jsonl(p)
        prompts = [r["prompt"] for r in rows]
    else:
        # fallback: build prompts with the language NAME, not the FLORES code
        lang_name = cfg["language_map"].get(args.language, {}).get("name", args.language)
        prompts = [r.prompt for r in language_forcing_prompts(lang_name)]
    if args.limit:
        prompts = prompts[: args.limit]

    model, tokenizer = load_model(cfg["models"][args.model])
    refs = find_mlp_refs(model)
    if args.method == "none":
        outs = generate_plain(model, tokenizer, prompts, max_new_tokens=args.max_new_tokens)
    elif args.method == "knockout":
        outs = knockout_generate(
            model, tokenizer, prompts, refs, args.key, indices,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        outs = steer_generate(
            model, tokenizer, prompts, refs, args.key, indices, args.alpha,
            max_new_tokens=args.max_new_tokens,
        )
    rows = [
        {"prompt": p, "output": o, "language": args.language, "model": args.model,
         "method": args.method, "key": args.key, "n_neurons": len(indices),
         "alpha": args.alpha if args.method == "steering" else None}
        for p, o in zip(prompts, outs)
    ]
    n = write_jsonl(args.out, rows)
    print(f"wrote {n} outputs to {args.out}")


if __name__ == "__main__":
    main()
