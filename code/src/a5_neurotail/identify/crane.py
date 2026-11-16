"""CRANE-style causal relevance: loss increase under neuron masking."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.hooks import KnockoutHook, intervene
from ..models.modules import MLPRef, ref_map


def _nll(model, tokenizer, texts: list[str], batch_size: int, max_length: int) -> float:
    device = next(model.parameters()).device
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:].clone()
            labels[enc["attention_mask"][:, 1:] == 0] = -100
            total += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
    return total


def crane_relevance(
    model,
    tokenizer,
    texts: list[str],
    refs: list[MLPRef],
    key: str,
    candidates: list[int] | np.ndarray,
    batch_size: int = 4,
    max_length: int = 256,
    max_neurons: int | None = 200,
) -> dict[str, np.ndarray]:
    """Return loss increase for each candidate neuron (0 for non-candidates)."""
    refs_map = ref_map(refs)
    if key not in refs_map:
        raise KeyError(f"unknown MLP key {key}")
    ref = refs_map[key]
    n = ref.intermediate_size
    scores = np.zeros(n, dtype=np.float64)
    cand = list(map(int, candidates))
    if max_neurons is not None:
        cand = cand[:max_neurons]
    base = _nll(model, tokenizer, texts, batch_size, max_length)
    for idx in cand:
        hook = KnockoutHook([idx])
        with intervene([ref], {key: hook}):
            loss = _nll(model, tokenizer, texts, batch_size, max_length)
        scores[idx] = loss - base
    return {"scores": scores, "candidates": np.array(cand, dtype=int), "base_nll": np.array([base])}


def select_crane(scores: np.ndarray, top_frac: float = 0.01) -> np.ndarray:
    n = max(1, int(round(len(scores) * top_frac)))
    order = np.argsort(scores)[::-1]
    return np.sort(order[:n])
