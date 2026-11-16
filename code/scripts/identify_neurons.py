#!/usr/bin/env python
"""Run LAPE / DLN / LangFIR neuron identification from cached statistics."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a5_neurotail.config import languages_for_run, load_config  # noqa: E402
from a5_neurotail.identify.dln import run_dln  # noqa: E402
from a5_neurotail.identify.langfir import run_langfir  # noqa: E402
from a5_neurotail.identify.lape import run_lape  # noqa: E402
from a5_neurotail.models.stats_io import load_stats  # noqa: E402

_LAYER_RE = re.compile(r"L(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--languages", default=str(ROOT / "configs/languages.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--methods", default="lape,dln,langfir")
    ap.add_argument("--layers", default=None, help="comma-separated layer indices")
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--min-prob", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.config, args.languages, str(ROOT / "configs/models.yaml"))
    stats_dir = Path(args.stats_dir or (Path(cfg["paths"]["activations_dir"]) / args.model))
    out_dir = Path(args.out_dir or (Path(cfg["paths"]["neurons_dir"]) / args.model))
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    langs = languages_for_run(cfg)
    stats_by_lang = {}
    for lang in langs:
        p = stats_dir / f"{lang['code']}.npz"
        if p.exists():
            stats_by_lang[lang["code"]] = load_stats(p)
    if not stats_by_lang:
        raise SystemExit(f"no stats in {stats_dir}")
    print(f"loaded {len(stats_by_lang)} languages")

    keys = sorted(next(iter(stats_by_lang.values())).keys())
    if layers:
        keys = [k for k in keys if _LAYER_RE.search(k) and int(_LAYER_RE.search(k).group(1)) in layers]
    print(f"keys: {len(keys)}")

    random_stats = None
    rp = stats_dir / "_random.npz"
    if rp.exists():
        random_stats = load_stats(rp)

    for target in stats_by_lang:
        for key in keys:
            for method in methods:
                try:
                    if method == "lape":
                        res = run_lape(
                            stats_by_lang, key, target,
                            top_frac=args.top_frac, min_prob=args.min_prob,
                        )
                    elif method == "dln":
                        res = run_dln(stats_by_lang, key, target, top_frac=args.top_frac)
                    elif method == "langfir":
                        if random_stats is None:
                            continue
                        res = run_langfir(
                            stats_by_lang[target], random_stats, key,
                            top_frac=args.top_frac, min_prob=args.min_prob,
                        )
                    else:
                        continue
                except Exception as exc:
                    print(f"skip {method} {target} {key}: {exc}")
                    continue
                out = out_dir / method / target
                out.mkdir(parents=True, exist_ok=True)
                with open(out / f"{key}.json", "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False)
        print(f"done {target}")

    print("wrote", out_dir)


if __name__ == "__main__":
    main()
