"""Activation extraction and on-disk statistics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hooks import ActivationAccumulator, capture
from .modules import MLPRef, select_layers


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def tokenize_texts(tokenizer, texts: list[str], max_length: int):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return enc


def _token_mask(attention_mask: torch.Tensor, last_only: bool) -> torch.Tensor:
    mask = attention_mask.bool()
    if not last_only:
        return mask
    out = torch.zeros_like(mask)
    lengths = mask.sum(dim=1)
    for i, n in enumerate(lengths.tolist()):
        if n > 0:
            out[i, n - 1] = True
    return out


def extract_stats(
    model,
    tokenizer,
    texts: list[str],
    refs: list[MLPRef],
    batch_size: int = 4,
    max_length: int = 256,
    hist_bins: int | None = 32,
    token_position: str = "all",
    layers: list[int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    refs = select_layers(refs, layers)
    if not refs:
        raise ValueError("no MLP refs selected")
    accs = {
        r.key: ActivationAccumulator(
            r.intermediate_size, hist_bins=hist_bins
        )
        for r in refs
    }
    device = _device(model)
    last_only = token_position == "last"
    with torch.inference_mode():
        with capture(refs, accs) as hooks:
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                enc = tokenize_texts(tokenizer, batch_texts, max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                mask = _token_mask(enc["attention_mask"], last_only)
                for h in hooks.values():
                    h.set_mask(mask)
                model(**enc)
    return {k: acc.summary() for k, acc in accs.items()}


from .stats_io import (  # noqa: F401
    activation_probability,
    load_stats,
    mean_activation,
    save_stats,
)

def extract_to_file(
    model,
    tokenizer,
    texts: list[str],
    refs: list[MLPRef],
    out_path: str | Path,
    meta: dict[str, Any],
    **kwargs,
) -> None:
    stats = extract_stats(model, tokenizer, texts, refs, **kwargs)
    save_stats(out_path, stats, meta)
