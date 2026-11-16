#!/usr/bin/env python
"""Extract per-neuron activation statistics for each language."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.config import languages_for_run, load_config  # noqa: E402
from a5_neurotail.data.loaders import load_flores_texts  # noqa: E402
from a5_neurotail.data.schema import read_jsonl  # noqa: E402
from a5_neurotail.identify.langfir import random_token_texts  # noqa: E402
from a5_neurotail.models.activations import extract_stats, save_stats  # noqa: E402
from a5_neurotail.models.loader import describe_environment, load_model, set_seed  # noqa: E402
from a5_neurotail.models.modules import find_mlp_refs, select_layers  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--languages", default=str(ROOT / "configs/languages.yaml"))
    ap.add_argument("--models", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--flores-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--max-texts", type=int, default=1000)
    ap.add_argument("--random-tokens", type=int, default=512)
    ap.add_argument("--random-length", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config, args.languages, args.models, args.paths)
    if args.model not in cfg["models"]:
        raise SystemExit(f"unknown model {args.model}; choices: {list(cfg['models'])}")
    set_seed(int(cfg.get("seed", 0)))
    print("environment:", describe_environment())
    model, tokenizer = load_model(cfg["models"][args.model])
    refs = find_mlp_refs(model)
    act_cfg = cfg["activation"]
    refs = select_layers(refs, act_cfg.get("layers"))
    print(f"MLP refs: {len(refs)}; keys={[r.key for r in refs][:5]} ...")

    data_dir = Path(args.data_dir or cfg["paths"]["processed_dir"])
    flores_dir = Path(args.flores_dir or (Path(cfg["paths"]["raw_dir"]) / "flores"))
    out = Path(args.out_dir or cfg["paths"]["activations_dir"]) / args.model
    out.mkdir(parents=True, exist_ok=True)

    for lang in languages_for_run(cfg, args.main_only):
        code = lang["code"]
        p = data_dir / "parallel" / f"{code}.jsonl"
        if p.exists():
            texts = [r["tgt_text"] for r in read_jsonl(p)]
        if not texts:
            # empty jsonl (e.g. pivot language) -> fall back to raw FLORES
            try:
                texts = load_flores_texts(flores_dir, code)
            except FileNotFoundError:
                print(f"skip {code}: no parallel data")
                continue
        texts = texts[: args.max_texts]
        if not texts:
            print(f"skip {code}: empty")
            continue
        stats = extract_stats(
            model,
            tokenizer,
            texts,
            refs,
            batch_size=int(act_cfg.get("batch_size", 4)),
            max_length=int(act_cfg.get("max_length", 256)),
            hist_bins=act_cfg.get("hist_bins"),
            token_position=str(act_cfg.get("token_position", "all")),
        )
        meta = {
            "model": args.model,
            "hf_id": cfg["models"][args.model]["hf_id"],
            "language": code,
            "n_texts": len(texts),
            "token_position": act_cfg.get("token_position", "all"),
        }
        save_stats(out / f"{code}.npz", stats, meta)
        print(f"saved {code}: {len(texts)} texts, {len(stats)} refs")

    if args.random_tokens > 0:
        texts = random_token_texts(tokenizer, args.random_tokens, args.random_length)
        stats = extract_stats(
            model,
            tokenizer,
            texts,
            refs,
            batch_size=int(act_cfg.get("batch_size", 4)),
            max_length=int(act_cfg.get("max_length", 256)),
            hist_bins=act_cfg.get("hist_bins"),
            token_position=str(act_cfg.get("token_position", "all")),
        )
        save_stats(
            out / "_random.npz",
            stats,
            {"model": args.model, "kind": "random_tokens", "n_texts": len(texts)},
        )
        print("saved random-token control")


if __name__ == "__main__":
    main()
