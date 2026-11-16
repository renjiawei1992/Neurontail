"""Shared generation/scoring helpers with a forward hook."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..models.hooks import intervene
from ..models.modules import MLPRef, ref_map


def _ref(refs: list[MLPRef], key: str) -> MLPRef:
    m = ref_map(refs)
    if key not in m:
        raise KeyError(f"unknown MLP key {key}")
    return m[key]


def generate_with_hook(
    model,
    tokenizer,
    prompts: list[str],
    refs: list[MLPRef],
    key: str,
    hook: Any,
    batch_size: int = 8,
    max_new_tokens: int = 48,
) -> list[str]:
    ref = _ref(refs, key)
    device = next(model.parameters()).device
    outs: list[str] = []
    with torch.inference_mode():
        with intervene([ref], {key: hook}):
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


def nll_with_hook(
    model,
    tokenizer,
    texts: list[str],
    refs: list[MLPRef],
    key: str,
    hook: Any,
    batch_size: int = 4,
    max_length: int = 256,
) -> float:
    ref = _ref(refs, key)
    device = next(model.parameters()).device
    total = 0.0
    with torch.inference_mode():
        with intervene([ref], {key: hook}):
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


def base_nll(model, tokenizer, texts: list[str], batch_size: int = 4, max_length: int = 256) -> float:
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
