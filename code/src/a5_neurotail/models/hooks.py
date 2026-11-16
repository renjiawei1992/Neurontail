"""Forward hooks for activation statistics and interventions.

Neuron definition: one intermediate-dimension unit of an MLP
(`down_proj(act_fn(gate_proj(x)) * up_proj(x))`). Hooks are placed on
`gate_proj`; the activation is `act_fn(gate_output)`, shape [B, S, N].
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

import numpy as np
import torch

from .modules import MLPRef


class ActivationAccumulator:
    """Streaming per-neuron statistics; no full activation tensors are stored."""

    def __init__(
        self,
        n_neurons: int,
        hist_bins: int | None = 32,
        hist_range: tuple[float, float] = (-5.0, 20.0),
    ) -> None:
        self.n = int(n_neurons)
        self.count = torch.zeros(self.n, dtype=torch.float64)
        self.pos = torch.zeros(self.n, dtype=torch.float64)
        self.total = torch.zeros(self.n, dtype=torch.float64)
        self.sumsq = torch.zeros(self.n, dtype=torch.float64)
        self.hist_bins = hist_bins
        if hist_bins:
            self.edges = torch.linspace(hist_range[0], hist_range[1], hist_bins + 1)
            self.hist = torch.zeros(self.n, hist_bins, dtype=torch.float64)
        else:
            self.edges = None
            self.hist = None

    def update(self, act: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if act.dim() != 3:
            raise ValueError(f"expected [B,S,N], got {tuple(act.shape)}")
        if mask is not None:
            if mask.shape != act.shape[:2]:
                raise ValueError(f"mask {tuple(mask.shape)} != act {tuple(act.shape[:2])}")
            x = act[mask.to(act.device)]
        else:
            x = act.reshape(-1, act.shape[-1])
        if x.numel() == 0:
            return
        x = x.detach().float().cpu()
        self.count += x.shape[0]
        self.pos += (x > 0).sum(dim=0).double()
        self.total += x.sum(dim=0).double()
        self.sumsq += (x * x).sum(dim=0).double()
        if self.hist is not None and self.edges is not None:
            idx = torch.bucketize(x, self.edges) - 1
            idx = idx.clamp(0, self.hist_bins - 1)
            # per-neuron np.bincount: all-C fast path
            idx_np = idx.numpy()  # [M, N]
            hist_np = self.hist.numpy()
            for n in range(idx_np.shape[1]):
                counts = np.bincount(idx_np[:, n], minlength=self.hist_bins)
                hist_np[n] += counts[: self.hist_bins]
            self.hist = torch.from_numpy(hist_np)

    def summary(self) -> dict[str, np.ndarray]:
        count = self.count.clamp_min(1.0)
        out = {
            "count": self.count.numpy(),
            "pos": self.pos.numpy(),
            "sum": self.total.numpy(),
            "sum_sq": self.sumsq.numpy(),
            "p_pos": (self.pos / count).numpy(),
            "mean": (self.total / count).numpy(),
        }
        if self.hist is not None:
            out["hist"] = self.hist.numpy()
            out["hist_edges"] = self.edges.numpy()
        return out


def _tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


class CaptureHook:
    def __init__(self, ref: MLPRef, acc: ActivationAccumulator) -> None:
        self.ref = ref
        self.acc = acc
        self.mask: torch.Tensor | None = None

    def set_mask(self, mask: torch.Tensor | None) -> None:
        self.mask = mask

    def __call__(self, module, inputs, output):
        act = self.ref.act_fn(_tensor_output(output))
        self.acc.update(act, self.mask)
        return None


class KnockoutHook:
    def __init__(self, indices: Iterable[int]) -> None:
        self.indices = torch.as_tensor(list(indices), dtype=torch.long)

    def __call__(self, module, inputs, output):
        out = _tensor_output(output).clone()
        idx = self.indices.to(out.device)
        out[..., idx] = 0.0
        return out if not isinstance(output, tuple) else (out, *output[1:])


class AddHook:
    """Add alpha to selected gate-space neurons (pre-activation)."""

    def __init__(self, indices: Iterable[int], alpha: float) -> None:
        self.indices = torch.as_tensor(list(indices), dtype=torch.long)
        self.alpha = float(alpha)

    def __call__(self, module, inputs, output):
        out = _tensor_output(output).clone()
        idx = self.indices.to(out.device)
        out[..., idx] = out[..., idx] + self.alpha
        return out if not isinstance(output, tuple) else (out, *output[1:])


class VectorAddHook:
    """Add a per-neuron vector (DiffMean direction) to selected gate-space neurons.

    values: [N_sel] vector broadcast over [B,S]; added to output[..., indices].
    """

    def __init__(self, indices: Iterable[int], values: Any):
        self.indices = torch.as_tensor(list(indices), dtype=torch.long)
        if torch.is_tensor(values):
            self.values = values
        else:
            self.values = torch.as_tensor(list(values), dtype=torch.float32)

    def __call__(self, module, inputs, output):
        out = _tensor_output(output).clone()
        idx = self.indices.to(out.device)
        v = self.values.to(device=out.device, dtype=out.dtype)
        out[..., idx] = out[..., idx] + v
        return out if not isinstance(output, tuple) else (out, *output[1:])


class PatchHook:
    """Replace selected gate-space neurons with donor values.

    values: scalar, [N_sel], or broadcastable to [B,S,N_sel].
    """

    def __init__(self, indices: Iterable[int], values: Any) -> None:
        self.indices = torch.as_tensor(list(indices), dtype=torch.long)
        if torch.is_tensor(values):
            self.values = values
        else:
            self.values = torch.as_tensor(values, dtype=torch.float32)

    def __call__(self, module, inputs, output):
        out = _tensor_output(output).clone()
        idx = self.indices.to(out.device)
        v = self.values.to(device=out.device, dtype=out.dtype)
        out[..., idx] = v
        return out if not isinstance(output, tuple) else (out, *output[1:])


@contextmanager
def capture(refs: list[MLPRef], accs: dict[str, ActivationAccumulator]):
    hooks: dict[str, CaptureHook] = {}
    handles = []
    for r in refs:
        h = CaptureHook(r, accs[r.key])
        hooks[r.key] = h
        handles.append(r.gate.register_forward_hook(h))
    try:
        yield hooks
    finally:
        for h in handles:
            h.remove()


@contextmanager
def intervene(refs: list[MLPRef], hooks_by_key: dict[str, Any]):
    handles = []
    for r in refs:
        if r.key in hooks_by_key:
            handles.append(r.gate.register_forward_hook(hooks_by_key[r.key]))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
