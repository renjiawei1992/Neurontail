"""Discover MLP modules and define a stable neuron reference."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

_LAYER_RE = re.compile(r"(?:layers|h|block)\.(\d+)")
_EXPERT_RE = re.compile(r"experts\.(\d+)")


@dataclass
class MLPRef:
    key: str
    layer: int | None
    expert: int | None
    path: str
    module: Any
    gate: Any
    up: Any
    down: Any
    act_fn: Callable[[torch.Tensor], torch.Tensor]

    @property
    def intermediate_size(self) -> int:
        return int(self.gate.out_features)


def _find_act_fn(module: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    for name in ("act_fn", "activation_fn", "act"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return torch.nn.functional.silu


def find_mlp_refs(model: Any) -> list[MLPRef]:
    """Find modules with gate_proj/up_proj/down_proj (dense MLP or MoE experts)."""
    refs: list[MLPRef] = []
    for name, module in model.named_modules():
        if not all(hasattr(module, x) for x in ("gate_proj", "up_proj", "down_proj")):
            continue
        lm = _LAYER_RE.search(name)
        em = _EXPERT_RE.search(name)
        layer = int(lm.group(1)) if lm else None
        expert = int(em.group(1)) if em else None
        key = f"L{layer}" if expert is None else f"L{layer}E{expert}"
        refs.append(
            MLPRef(
                key=key,
                layer=layer,
                expert=expert,
                path=name,
                module=module,
                gate=module.gate_proj,
                up=module.up_proj,
                down=module.down_proj,
                act_fn=_find_act_fn(module),
            )
        )
    if not refs:
        raise RuntimeError(
            "No MLP modules with gate_proj/up_proj/down_proj found. "
            "This scaffold supports Llama/Qwen/Gemma/Mistral-style MLPs and OLMoE experts; "
            "fused MLP implementations need a model-specific adapter."
        )
    return refs


def refs_by_layer(refs: list[MLPRef]) -> dict[int, list[MLPRef]]:
    out: dict[int, list[MLPRef]] = {}
    for r in refs:
        if r.layer is None:
            continue
        out.setdefault(r.layer, []).append(r)
    return out


def ref_map(refs: list[MLPRef]) -> dict[str, MLPRef]:
    return {r.key: r for r in refs}


def layer_range(refs: list[MLPRef]) -> tuple[int, int]:
    layers = [r.layer for r in refs if r.layer is not None]
    if not layers:
        raise ValueError("no layer indices found")
    return min(layers), max(layers)


def select_layers(refs: list[MLPRef], layers: list[int] | None) -> list[MLPRef]:
    if not layers:
        return list(refs)
    keep = set(layers)
    return [r for r in refs if r.layer in keep]
