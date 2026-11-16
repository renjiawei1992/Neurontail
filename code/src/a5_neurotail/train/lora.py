"""LoRA training with optional neuron-targeted gradient masks.

Modes:
  full_lora   - standard LoRA on gate/up/down projections
  random_lora - same, but gradients are masked to a random neuron subset
  neuron_lora - same, but gradients are masked to LAPE/DLN-selected neurons

The neuron mask is applied to the low-rank adapter weights:
  gate_proj/up_proj: lora_B rows  (output dim = intermediate neurons)
  down_proj:        lora_A columns (input dim = intermediate neurons)
This is an approximation of "update only these neurons"; document it in the paper.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from ..models.modules import MLPRef


def build_lora_model(model, cfg: dict[str, Any]):
    lora_cfg = LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("alpha", 32)),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.get("target_modules", ["gate_proj", "up_proj", "down_proj"])),
    )
    return get_peft_model(model, lora_cfg)


def _lora_weight(module, which: str):
    for name in (f"lora_{which}.default.weight", f"lora_{which}.weight"):
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            pass
        parts = name.split(".")
        cur = module
        ok = True
        for p in parts:
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok:
            return cur
    return None


def apply_neuron_gradient_mask(model, neuron_map: dict[str, np.ndarray]) -> int:
    """Register gradient hooks. neuron_map: module path suffix -> selected indices."""
    n_hooks = 0
    for name, module in model.named_modules():
        suffix = name.split(".")[-1]
        if suffix not in ("gate_proj", "up_proj", "down_proj"):
            continue
        # match the longest key that is a suffix of the module path
        key = None
        for k in neuron_map:
            if name.endswith(k):
                key = k
                break
        if key is None:
            continue
        idx = torch.as_tensor(neuron_map[key], dtype=torch.long)
        weight = _lora_weight(module, "B" if suffix in ("gate_proj", "up_proj") else "A")
        if weight is None:
            continue

        def hook(grad, idx=idx, suffix=suffix):
            if suffix in ("gate_proj", "up_proj"):
                mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
                mask[idx.to(grad.device)] = True
                grad[~mask] = 0
            else:
                mask = torch.zeros(grad.shape[1], dtype=torch.bool, device=grad.device)
                mask[idx.to(grad.device)] = True
                grad[:, ~mask] = 0
            return grad

        weight.register_hook(hook)
        n_hooks += 1
    return n_hooks


def neuron_map_from_selection(
    selection: dict[str, list[int]], refs: list[MLPRef]
) -> dict[str, np.ndarray]:
    """selection: MLP key -> neuron indices. Returns suffix -> indices for gate/up/down."""
    by_key = {r.key: r for r in refs}
    out: dict[str, np.ndarray] = {}
    for key, idx in selection.items():
        if key not in by_key or not idx:
            continue
        ref = by_key[key]
        arr = np.asarray(sorted(set(map(int, idx))), dtype=int)
        out[ref.path + ".gate_proj"] = arr
        out[ref.path + ".up_proj"] = arr
        out[ref.path + ".down_proj"] = arr
    return out


def random_neuron_map(
    refs: list[MLPRef], selection: dict[str, list[int]], seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for key, idx in selection.items():
        ref = next((r for r in refs if r.key == key), None)
        if ref is None or not idx:
            continue
        k = len(idx)
        rand = np.sort(rng.choice(ref.intermediate_size, size=k, replace=False))
        out[ref.path + ".gate_proj"] = rand
        out[ref.path + ".up_proj"] = rand
        out[ref.path + ".down_proj"] = rand
    return out


def train_lora(
    model,
    tokenizer,
    texts: list[str],
    output_dir: str | Path,
    mode: str = "full_lora",
    neuron_map: dict[str, np.ndarray] | None = None,
    cfg: dict[str, Any] | None = None,
) -> str:
    cfg = cfg or {}
    model = build_lora_model(model, cfg)
    if mode in ("neuron_lora", "random_lora"):
        if neuron_map is None:
            raise ValueError(f"{mode} requires neuron_map")
        n = apply_neuron_gradient_mask(model, neuron_map)
        if n == 0:
            raise RuntimeError("no LoRA weights matched for gradient masking")
    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=int(cfg.get("max_length", 512))
        ),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 8)),
        learning_rate=float(cfg.get("lr", 1e-4)),
        num_train_epochs=float(cfg.get("epochs", 1)),
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    out = os.path.join(str(output_dir), "adapter")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out
