#!/usr/bin/env python
"""Train full/random/neuron-targeted LoRA adapters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.config import load_config  # noqa: E402
from a5_neurotail.data.schema import read_jsonl  # noqa: E402
from a5_neurotail.models.loader import load_model, set_seed  # noqa: E402
from a5_neurotail.models.modules import find_mlp_refs  # noqa: E402
from a5_neurotail.train.lora import (  # noqa: E402
    neuron_map_from_selection,
    random_neuron_map,
    train_lora,
)


def read_texts(path: str) -> list[str]:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = read_jsonl(p)
        return [r.get("text") or r.get("tgt_text") or r.get("content", "") for r in rows]
    return [line.strip() for line in open(p, encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--languages", default=str(ROOT / "configs/languages.yaml"))
    ap.add_argument("--models", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--selection", default=None, help="JSON from identify_neurons.py")
    ap.add_argument("--key", default=None)
    ap.add_argument("--mode", choices=["full_lora", "random_lora", "neuron_lora"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.languages, args.models)
    set_seed(int(cfg["lora"].get("seeds", [0])[0]))
    texts = read_texts(args.data)
    if args.limit:
        texts = texts[: args.limit]
    model, tokenizer = load_model(cfg["models"][args.model])
    refs = find_mlp_refs(model)

    neuron_map = None
    if args.mode in ("neuron_lora", "random_lora"):
        if not args.selection or not args.key:
            raise SystemExit("--selection and --key required for neuron/random LoRA")
        sel = json.load(open(args.selection, encoding="utf-8"))
        selection = {args.key: sel["selected"]}
        if args.mode == "neuron_lora":
            neuron_map = neuron_map_from_selection(selection, refs)
        else:
            neuron_map = random_neuron_map(refs, selection, seed=int(cfg.get("seed", 0)))

    out = train_lora(
        model, tokenizer, texts, args.out,
        mode=args.mode, neuron_map=neuron_map, cfg=cfg["lora"],
    )
    print("saved", out)


if __name__ == "__main__":
    main()
