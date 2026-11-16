"""Activation addition / steering experiments."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..models.hooks import AddHook, VectorAddHook
from ..models.modules import MLPRef
from .common import base_nll, generate_with_hook, nll_with_hook


def diffmean_direction(
    src_stats: dict[str, dict[str, np.ndarray]],
    tgt_stats: dict[str, dict[str, np.ndarray]],
    key: str,
) -> np.ndarray:
    """Per-neuron DiffMean direction: mean(tgt) - mean(src) from cached stats."""
    if key not in src_stats or key not in tgt_stats:
        raise KeyError(f"missing stats key {key}")
    return tgt_stats[key]["mean"].astype(np.float64) - src_stats[key]["mean"].astype(np.float64)


def diffmean_steer_generate(
    model,
    tokenizer,
    prompts: list[str],
    refs: list[MLPRef],
    key: str,
    direction: np.ndarray,
    beta: float,
    indices: list[int] | np.ndarray | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 48,
) -> list[str]:
    """Steer along a DiffMean direction: add beta * direction[neuron] at `key`.

    If `indices` is None, the direction applies to ALL neurons of the MLP.
    """
    n = direction.shape[0]
    idx = np.arange(n) if indices is None else np.asarray(indices, dtype=int)
    values = beta * direction[idx]
    hook = VectorAddHook(idx.tolist(), values.tolist())
    return generate_with_hook(
        model, tokenizer, prompts, refs, key, hook, batch_size, max_new_tokens
    )


def diffmean_steer_generate_multikey(
    model,
    tokenizer,
    prompts: list[str],
    refs: list[MLPRef],
    directions: dict[str, np.ndarray],
    beta: float,
    batch_size: int = 8,
    max_new_tokens: int = 48,
) -> list[str]:
    """Steer along per-key DiffMean directions on MULTIPLE layers at once.

    directions: {key: [N] vector}; adds beta * direction at each key.
    """
    hooks: dict[str, Any] = {}
    for key, direction in directions.items():
        idx = np.arange(direction.shape[0])
        hooks[key] = VectorAddHook(idx.tolist(), (beta * direction).tolist())
    from ..models.hooks import intervene

    device = next(model.parameters()).device
    outs: list[str] = []
    with torch.inference_mode():
        with intervene(refs, hooks):
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new = gen[:, enc["input_ids"].shape[1] :]
                outs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    return outs


def steer_generate(
    model,
    tokenizer,
    prompts: list[str],
    refs: list[MLPRef],
    key: str,
    indices: list[int] | np.ndarray,
    alpha: float,
    batch_size: int = 8,
    max_new_tokens: int = 48,
) -> list[str]:
    return generate_with_hook(
        model,
        tokenizer,
        prompts,
        refs,
        key,
        AddHook(indices, alpha),
        batch_size,
        max_new_tokens,
    )


def steer_nll(
    model,
    tokenizer,
    texts: list[str],
    refs: list[MLPRef],
    key: str,
    indices: list[int] | np.ndarray,
    alpha: float,
    batch_size: int = 4,
    max_length: int = 256,
    baseline: float | None = None,
) -> dict[str, Any]:
    base = base_nll(model, tokenizer, texts, batch_size, max_length) if baseline is None else baseline
    steered = nll_with_hook(
        model, tokenizer, texts, refs, key, AddHook(indices, alpha), batch_size, max_length
    )
    return {
        "key": key,
        "indices": list(map(int, indices)),
        "alpha": alpha,
        "base_nll": base,
        "steered_nll": steered,
        "delta_nll": steered - base,
    }
