"""Causal Specificity Gap and related statistics."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def causal_specificity_gap(
    on_target_drop: float, off_target_drop: float, random_drop: float
) -> float:
    """CSG = on-target damage minus the strongest control damage."""
    return float(on_target_drop - max(off_target_drop, random_drop))


def aggregate_csg(records: Iterable[dict]) -> dict[str, float]:
    vals = [
        causal_specificity_gap(
            r["on_target_drop"], r.get("off_target_drop", 0.0), r.get("random_drop", 0.0)
        )
        for r in records
    ]
    if not vals:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }


def bootstrap_ci(
    values: Iterable[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
