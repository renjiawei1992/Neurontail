#!/usr/bin/env python
"""Print environment and optional model config (no weights downloaded)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model key from configs/models.yaml")
    ap.add_argument("--models", default=str(ROOT / "configs/models.yaml"))
    args = ap.parse_args()
    import torch

    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"gpu[{i}]: {p.name} {p.total_memory/1e9:.1f} GB")
    try:
        import transformers

        print("transformers:", transformers.__version__)
    except Exception as exc:
        print("transformers: missing", exc)
    try:
        import peft

        print("peft:", peft.__version__)
    except Exception as exc:
        print("peft: missing", exc)
    total, used, free = shutil.disk_usage(ROOT)
    print(f"disk {ROOT}: total={total/1e9:.1f} GB free={free/1e9:.1f} GB")

    if args.model:
        from a5_neurotail.config import load_yaml

        models = load_yaml(args.models)["models"]
        if args.model not in models:
            raise SystemExit(f"unknown model {args.model}")
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(models[args.model]["hf_id"])
        print("model config:", {
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "intermediate_size": getattr(cfg, "intermediate_size", None),
            "num_experts": getattr(cfg, "num_experts", None),
        })


if __name__ == "__main__":
    main()
