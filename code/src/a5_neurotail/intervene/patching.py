"""Donor patching: replace target neurons with donor mean activations."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models.hooks import PatchHook
from ..models.modules import MLPRef
from .common import base_nll, generate_with_hook, nll_with_hook


def donor_mean_values(
    donor_stats: dict[str, dict[str, np.ndarray]], key: str, indices: list[int] | np.ndarray
) -> np.ndarray:
    if key not in donor_stats:
        raise KeyError(f"donor stats missing key {key}")
    return donor_stats[key]["mean"][np.asarray(indices, dtype=int)]


def donor_patch_generate(
    model,
    tokenizer,
    prompts: list[str],
    refs: list[MLPRef],
    key: str,
    target_indices: list[int] | np.ndarray,
    donor_stats: dict[str, dict[str, np.ndarray]],
    batch_size: int = 8,
    max_new_tokens: int = 48,
) -> list[str]:
    values = donor_mean_values(donor_stats, key, target_indices)
    hook = PatchHook(target_indices, values)
    return generate_with_hook(
        model, tokenizer, prompts, refs, key, hook, batch_size, max_new_tokens
    )


def donor_patch_nll(
    model,
    tokenizer,
    texts: list[str],
    refs: list[MLPRef],
    key: str,
    target_indices: list[int] | np.ndarray,
    donor_stats: dict[str, dict[str, np.ndarray]],
    batch_size: int = 4,
    max_length: int = 256,
    baseline: float | None = None,
) -> dict[str, Any]:
    base = base_nll(model, tokenizer, texts, batch_size, max_length) if baseline is None else baseline
    values = donor_mean_values(donor_stats, key, target_indices)
    hook = PatchHook(target_indices, values)
    patched = nll_with_hook(model, tokenizer, texts, refs, key, hook, batch_size, max_length)
    return {
        "key": key,
        "indices": list(map(int, target_indices)),
        "base_nll": base,
        "patched_nll": patched,
        "delta_nll": patched - base,
        "donor_values": values.tolist(),
    }
