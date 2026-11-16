"""Neuron knockout (necessity) experiments."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models.hooks import KnockoutHook
from ..models.modules import MLPRef
from .common import base_nll, generate_with_hook, nll_with_hook


def random_indices(n: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(k, n), replace=False))


def knockout_nll(
    model,
    tokenizer,
    texts: list[str],
    refs: list[MLPRef],
    key: str,
    indices: list[int] | np.ndarray,
    batch_size: int = 4,
    max_length: int = 256,
    baseline: float | None = None,
) -> dict[str, Any]:
    base = base_nll(model, tokenizer, texts, batch_size, max_length) if baseline is None else baseline
    hook = KnockoutHook(indices)
    knocked = nll_with_hook(model, tokenizer, texts, refs, key, hook, batch_size, max_length)
    return {
        "key": key,
        "indices": list(map(int, indices)),
        "n_neurons": len(indices),
        "base_nll": base,
        "knocked_nll": knocked,
        "delta_nll": knocked - base,
    }


def knockout_generate(
    model,
    tokenizer,
    prompts: list[str],
    refs: list[MLPRef],
    key: str,
    indices: list[int] | np.ndarray,
    batch_size: int = 8,
    max_new_tokens: int = 48,
) -> list[str]:
    return generate_with_hook(
        model, tokenizer, prompts, refs, key, KnockoutHook(indices), batch_size, max_new_tokens
    )
